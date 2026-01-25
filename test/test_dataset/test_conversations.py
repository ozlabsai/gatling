"""
Tests for conversation sampling module.

Tests the full pipeline from sampling to mutation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from source.dataset.conversations.sampler import (
    Conversation,
    ConversationTurn,
    ConversationSampler,
)
from source.dataset.conversations.intent_extractor import (
    ActionIntent,
    IntentExtractor,
)
from source.dataset.conversations.plan_transformer import (
    ExecutionPlan,
    PlanTransformer,
)
from source.dataset.conversations.mutator import (
    AdversarialMutator,
    MutationType,
    MutatedPlan,
)


class TestConversationSampler:
    """Tests for ConversationSampler."""

    def test_init(self):
        """Test sampler initialization."""
        sampler = ConversationSampler(seed=42)
        assert sampler.seed == 42
        assert sampler._datasets == {}

    def test_unsupported_dataset(self):
        """Test error on unsupported dataset."""
        sampler = ConversationSampler()
        with pytest.raises(ValueError, match="Unsupported dataset"):
            sampler.load_dataset("unknown_dataset")

    def test_parse_wildchat_example(self):
        """Test parsing WildChat example."""
        sampler = ConversationSampler()
        example = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "conversation_id": "test_123",
        }

        conv = sampler._parse_wildchat_example(example, 0)
        assert conv is not None
        assert conv.conversation_id == "test_123"
        assert len(conv.turns) == 2
        assert conv.source == "wildchat"
        assert conv.turns[0].role == "user"
        assert conv.turns[0].content == "Hello"

    def test_parse_wildchat_missing_data(self):
        """Test parsing WildChat with missing data."""
        sampler = ConversationSampler()
        example = {"invalid": "data"}

        conv = sampler._parse_wildchat_example(example, 0)
        assert conv is None

    def test_parse_lmsys_example(self):
        """Test parsing LMSYS example."""
        sampler = ConversationSampler()
        example = {
            "messages": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Response"},
            ],
            "conversation_id": "lmsys_456",
        }

        conv = sampler._parse_lmsys_example(example, 0)
        assert conv is not None
        assert conv.conversation_id == "lmsys_456"
        assert len(conv.turns) == 2
        assert conv.source == "lmsys"

    def test_conversation_turn_model(self):
        """Test ConversationTurn model."""
        turn = ConversationTurn(
            role="user",
            content="Find my invoice",
            metadata={"turn_idx": 0},
        )
        assert turn.role == "user"
        assert turn.content == "Find my invoice"
        assert turn.metadata["turn_idx"] == 0

    def test_conversation_model(self):
        """Test Conversation model."""
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi!"),
        ]
        conv = Conversation(
            conversation_id="test_conv",
            turns=turns,
            source="wildchat",
        )
        assert conv.conversation_id == "test_conv"
        assert len(conv.turns) == 2
        assert conv.source == "wildchat"


class TestIntentExtractor:
    """Tests for IntentExtractor."""

    def test_init_with_api_key(self):
        """Test extractor initialization with API key."""
        extractor = IntentExtractor(api_key="test_key")
        assert extractor.api_key == "test_key"
        assert extractor.model == "claude-sonnet-4-5-20250929"

    def test_init_without_api_key(self):
        """Test extractor initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                IntentExtractor()

    def test_format_conversation(self):
        """Test conversation formatting for prompts."""
        extractor = IntentExtractor(api_key="test_key")
        turns = [
            ConversationTurn(role="user", content="Find my invoice"),
            ConversationTurn(role="assistant", content="I'll help with that."),
        ]
        conv = Conversation(
            conversation_id="test",
            turns=turns,
            source="wildchat",
        )

        formatted = extractor._format_conversation(conv)
        assert "USER: Find my invoice" in formatted
        assert "ASSISTANT: I'll help with that." in formatted

    def test_parse_extraction_response_valid_json(self):
        """Test parsing valid JSON response."""
        extractor = IntentExtractor(api_key="test_key")
        response_text = """```json
[
  {
    "turn_idx": 0,
    "user_message": "Find my invoice",
    "intent_category": "retrieve",
    "action_description": "Retrieve invoice",
    "inferred_tools": ["finance_api"],
    "scope_hints": {"limit": 1},
    "confidence": 0.9
  }
]
```"""

        turns = [ConversationTurn(role="user", content="Find my invoice")]
        conv = Conversation(
            conversation_id="test",
            turns=turns,
            source="wildchat",
        )

        intents = extractor._parse_extraction_response(response_text, conv)
        assert len(intents) == 1
        assert intents[0].intent_category == "retrieve"
        assert intents[0].confidence == 0.9

    def test_parse_extraction_response_empty(self):
        """Test parsing empty response."""
        extractor = IntentExtractor(api_key="test_key")
        response_text = "[]"

        conv = Conversation(
            conversation_id="test",
            turns=[],
            source="wildchat",
        )

        intents = extractor._parse_extraction_response(response_text, conv)
        assert len(intents) == 0

    def test_action_intent_model(self):
        """Test ActionIntent model."""
        intent = ActionIntent(
            turn_idx=0,
            user_message="Find my invoice",
            intent_category="retrieve",
            action_description="Retrieve recent invoice",
            inferred_tools=["finance_api"],
            scope_hints={"limit": 1},
            confidence=0.9,
        )
        assert intent.intent_category == "retrieve"
        assert intent.confidence == 0.9
        assert "limit" in intent.scope_hints


class TestPlanTransformer:
    """Tests for PlanTransformer."""

    def test_init(self):
        """Test transformer initialization."""
        transformer = PlanTransformer(api_key="test_key")
        assert transformer.api_key == "test_key"

    def test_infer_domain_from_tools(self):
        """Test domain inference from tool hints."""
        transformer = PlanTransformer(api_key="test_key")

        intent = ActionIntent(
            turn_idx=0,
            user_message="Check calendar",
            intent_category="retrieve",
            action_description="View calendar",
            inferred_tools=["calendar_api"],
        )

        domain = transformer._infer_domain(intent)
        assert domain == "Calendar"

    def test_infer_domain_from_category(self):
        """Test domain inference from intent category."""
        transformer = PlanTransformer(api_key="test_key")

        intent = ActionIntent(
            turn_idx=0,
            user_message="Send email",
            intent_category="communicate",
            action_description="Send message",
            inferred_tools=[],
        )

        domain = transformer._infer_domain(intent)
        assert domain == "Email"

    def test_execution_plan_model(self):
        """Test ExecutionPlan model."""
        from source.dataset.models import (
            UserRequest,
            SystemPolicy,
            ToolCallGraph,
            ScopeMetadata,
            SensitivityTier,
        )

        intent = ActionIntent(
            turn_idx=0,
            user_message="Test",
            intent_category="retrieve",
            action_description="Test action",
        )

        user_request = UserRequest(
            request_id="req_1",
            domain="Finance",
            text="Test",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(
                rows_requested=1,
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
        )

        policy = SystemPolicy(
            policy_id="policy_1",
            domain="Finance",
            rules=["Rule 1"],
            scope_limits={"max_results": 100},
        )

        graph = ToolCallGraph(
            graph_id="graph_1",
            calls=[],
            execution_order=[],
        )

        plan = ExecutionPlan(
            plan_id="plan_1",
            conversation_id="conv_1",
            source="wildchat",
            user_request=user_request,
            inferred_policy=policy,
            graph=graph,
            original_intent=intent,
        )

        assert plan.plan_id == "plan_1"
        assert plan.source == "wildchat"


class TestAdversarialMutator:
    """Tests for AdversarialMutator."""

    def test_init(self):
        """Test mutator initialization."""
        mutator = AdversarialMutator(mutation_rate=0.3, seed=42)
        assert mutator.mutation_rate == 0.3
        assert mutator.seed == 42

    def test_scope_blowup_mutation(self):
        """Test scope blow-up mutation."""
        from source.dataset.models import (
            UserRequest,
            SystemPolicy,
            ToolCallGraph,
            ToolCall,
            ScopeMetadata,
            ProvenancePointer,
            SensitivityTier,
            TrustTier,
        )

        # Create a simple execution plan
        intent = ActionIntent(
            turn_idx=0,
            user_message="Test",
            intent_category="retrieve",
            action_description="Test",
        )

        call = ToolCall(
            call_id="call_1",
            tool_id="test.tool",
            arguments={"limit": 5},
            scope=ScopeMetadata(
                rows_requested=5,
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_1",
            ),
        )

        graph = ToolCallGraph(
            graph_id="graph_1",
            calls=[call],
            execution_order=["call_1"],
        )

        plan = ExecutionPlan(
            plan_id="plan_1",
            conversation_id="conv_1",
            source="wildchat",
            user_request=UserRequest(
                request_id="req_1",
                domain="Finance",
                text="Test",
                intent_category="retrieve",
                expected_scope=ScopeMetadata(
                    rows_requested=5,
                    sensitivity_tier=SensitivityTier.INTERNAL,
                ),
            ),
            inferred_policy=SystemPolicy(
                policy_id="policy_1",
                domain="Finance",
                rules=[],
                scope_limits={},
            ),
            graph=graph,
            original_intent=intent,
        )

        mutator = AdversarialMutator()
        mutated = mutator._mutate_scope_blowup(plan)

        assert mutated is not None
        assert mutated.mutation_type == MutationType.SCOPE_BLOWUP
        assert (
            mutated.execution_plan.graph.calls[0].scope.rows_requested
            > 5
        )

    def test_privilege_escalation_mutation(self):
        """Test privilege escalation mutation."""
        from source.dataset.models import (
            UserRequest,
            SystemPolicy,
            ToolCallGraph,
            ToolCall,
            ScopeMetadata,
            ProvenancePointer,
            SensitivityTier,
            TrustTier,
        )

        intent = ActionIntent(
            turn_idx=0,
            user_message="Test",
            intent_category="retrieve",
            action_description="Test",
        )

        call = ToolCall(
            call_id="call_1",
            tool_id="test.tool",
            arguments={},
            scope=ScopeMetadata(
                rows_requested=1,
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_1",
            ),
        )

        graph = ToolCallGraph(
            graph_id="graph_1",
            calls=[call],
            execution_order=["call_1"],
        )

        plan = ExecutionPlan(
            plan_id="plan_1",
            conversation_id="conv_1",
            source="wildchat",
            user_request=UserRequest(
                request_id="req_1",
                domain="Finance",
                text="Test",
                intent_category="retrieve",
                expected_scope=ScopeMetadata(
                    rows_requested=1,
                    sensitivity_tier=SensitivityTier.INTERNAL,
                ),
            ),
            inferred_policy=SystemPolicy(
                policy_id="policy_1",
                domain="Finance",
                rules=[],
                scope_limits={},
            ),
            graph=graph,
            original_intent=intent,
        )

        mutator = AdversarialMutator()
        mutated = mutator._mutate_privilege_escalation(plan)

        assert mutated is not None
        assert mutated.mutation_type == MutationType.PRIVILEGE_ESCALATION
        assert (
            mutated.execution_plan.graph.calls[0].scope.sensitivity_tier
            == SensitivityTier.CONFIDENTIAL
        )

    def test_mutate_plans_rate(self):
        """Test that mutation rate is respected."""
        from source.dataset.models import (
            UserRequest,
            SystemPolicy,
            ToolCallGraph,
            ToolCall,
            ScopeMetadata,
            ProvenancePointer,
            SensitivityTier,
            TrustTier,
        )

        # Create 10 simple plans
        plans = []
        for i in range(10):
            intent = ActionIntent(
                turn_idx=0,
                user_message=f"Test {i}",
                intent_category="retrieve",
                action_description="Test",
            )

            call = ToolCall(
                call_id="call_1",
                tool_id="test.tool",
                arguments={},
                scope=ScopeMetadata(
                    rows_requested=5,
                    sensitivity_tier=SensitivityTier.INTERNAL,
                ),
                provenance=Provenance(
                    source_type=TrustTier.USER,
                    source_id="user_1",
                ),
            )

            graph = ToolCallGraph(
                graph_id=f"graph_{i}",
                calls=[call],
                execution_order=["call_1"],
            )

            plan = ExecutionPlan(
                plan_id=f"plan_{i}",
                conversation_id=f"conv_{i}",
                source="wildchat",
                user_request=UserRequest(
                    request_id=f"req_{i}",
                    domain="Finance",
                    text=f"Test {i}",
                    intent_category="retrieve",
                    expected_scope=ScopeMetadata(
                        rows_requested=5,
                        sensitivity_tier=SensitivityTier.INTERNAL,
                    ),
                ),
                inferred_policy=SystemPolicy(
                    policy_id="policy_1",
                    domain="Finance",
                    rules=[],
                    scope_limits={},
                ),
                graph=graph,
                original_intent=intent,
            )
            plans.append(plan)

        mutator = AdversarialMutator(mutation_rate=0.2, seed=42)
        benign, mutated = mutator.mutate_plans(plans)

        # Should mutate ~20% (2 out of 10)
        assert len(mutated) == 2
        assert len(benign) == 8
        assert len(mutated) + len(benign) == 10

    def test_mutated_plan_model(self):
        """Test MutatedPlan model."""
        from source.dataset.models import (
            UserRequest,
            SystemPolicy,
            ToolCallGraph,
            ScopeMetadata,
            SensitivityTier,
        )

        intent = ActionIntent(
            turn_idx=0,
            user_message="Test",
            intent_category="retrieve",
            action_description="Test",
        )

        plan = ExecutionPlan(
            plan_id="plan_1",
            conversation_id="conv_1",
            source="wildchat",
            user_request=UserRequest(
                request_id="req_1",
                domain="Finance",
                text="Test",
                intent_category="retrieve",
                expected_scope=ScopeMetadata(
                    rows_requested=1,
                    sensitivity_tier=SensitivityTier.INTERNAL,
                ),
            ),
            inferred_policy=SystemPolicy(
                policy_id="policy_1",
                domain="Finance",
                rules=[],
                scope_limits={},
            ),
            graph=ToolCallGraph(
                graph_id="graph_1",
                calls=[],
                execution_order=[],
            ),
            original_intent=intent,
        )

        mutated = MutatedPlan(
            plan_id="mutated_plan_1",
            original_plan_id="plan_1",
            mutation_type=MutationType.SCOPE_BLOWUP,
            mutation_description="Test mutation",
            execution_plan=plan,
        )

        assert mutated.plan_id == "mutated_plan_1"
        assert mutated.original_plan_id == "plan_1"
        assert mutated.is_adversarial is True
        assert mutated.mutation_type == MutationType.SCOPE_BLOWUP
