"""
Tests for trace validators.

Validates the validation logic for gold traces.
"""


from source.dataset.models import (
    GoldTrace,
    ProvenancePointer,
    ScopeMetadata,
    SensitivityTier,
    SystemPolicy,
    ToolCall,
    ToolCallGraph,
    TrustTier,
    UserRequest,
)
from source.dataset.validators.trace_validator import TraceValidator


class TestTraceValidator:
    """Tests for TraceValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TraceValidator()

    def create_valid_trace(self) -> GoldTrace:
        """Helper to create a valid gold trace."""
        request = UserRequest(
            request_id="req1",
            domain="Finance",
            text="Get my latest invoice",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(
                rows_requested=1,
                sensitivity_tier=SensitivityTier.CONFIDENTIAL,
            ),
        )

        policy = SystemPolicy(
            policy_id="finance_policy",
            domain="Finance",
            description="Finance policy",
            rules=["Max 100 results", "No exports"],
            forbidden_operations=["finance.delete"],
            scope_limits={"max_results": 100},
        )

        call = ToolCall(
            call_id="call1",
            tool_id="finance.list_invoices",
            arguments={"limit": 1},
            scope=ScopeMetadata(
                rows_requested=1,
                sensitivity_tier=SensitivityTier.CONFIDENTIAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="req1",
            ),
        )

        graph = ToolCallGraph(
            graph_id="graph1",
            calls=[call],
            execution_order=["call1"],
        )

        return GoldTrace(
            trace_id="trace1",
            request=request,
            policy=policy,
            graph=graph,
            validated=True,
        )

    def test_validate_structure_valid(self):
        """Test structural validation on valid trace."""
        trace = self.create_valid_trace()
        is_valid, errors = self.validator.validate_structure(trace)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_structure_missing_dependency(self):
        """Test structural validation detects missing dependencies."""
        trace = self.create_valid_trace()

        # Add a call with non-existent dependency
        bad_call = ToolCall(
            call_id="call2",
            tool_id="finance.export",
            arguments={},
            scope=ScopeMetadata(),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="req1",
            ),
            dependencies=["nonexistent"],
        )
        trace.graph.calls.append(bad_call)
        trace.graph.execution_order.append("call2")

        is_valid, errors = self.validator.validate_structure(trace)

        assert is_valid is False
        assert any("non-existent" in err.lower() for err in errors)

    def test_validate_structure_wrong_execution_order(self):
        """Test structural validation detects wrong execution order."""
        trace = self.create_valid_trace()

        # Add call2 that depends on call1
        call2 = ToolCall(
            call_id="call2",
            tool_id="finance.get_details",
            arguments={},
            scope=ScopeMetadata(),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="req1",
            ),
            dependencies=["call1"],
        )
        trace.graph.calls.append(call2)

        # Wrong order: call2 before call1
        trace.graph.execution_order = ["call2", "call1"]

        is_valid, errors = self.validator.validate_structure(trace)

        assert is_valid is False
        assert any("before its dependency" in err for err in errors)

    def test_validate_policy_compliance_valid(self):
        """Test policy compliance on valid trace."""
        trace = self.create_valid_trace()
        is_compliant, violations = self.validator.validate_policy_compliance(trace, trace.policy)

        assert is_compliant is True
        assert len(violations) == 0

    def test_validate_policy_forbidden_operation(self):
        """Test policy validation detects forbidden operations."""
        trace = self.create_valid_trace()

        # Use forbidden operation
        trace.graph.calls[0].tool_id = "finance.delete"

        is_compliant, violations = self.validator.validate_policy_compliance(trace, trace.policy)

        assert is_compliant is False
        assert any("forbidden" in v.lower() for v in violations)

    def test_validate_policy_scope_limit_exceeded(self):
        """Test policy validation detects scope limit violations."""
        trace = self.create_valid_trace()

        # Exceed scope limit
        trace.graph.calls[0].arguments["limit"] = 200  # Exceeds max of 100
        trace.graph.calls[0].scope.rows_requested = 200

        is_compliant, violations = self.validator.validate_policy_compliance(trace, trace.policy)

        assert is_compliant is False
        assert any("limit" in v.lower() for v in violations)

    def test_validate_minimal_scope(self):
        """Test minimal scope validation."""
        trace = self.create_valid_trace()

        # This should be fine (actual = expected)
        is_minimal, warnings = self.validator.validate_minimal_scope(trace)
        assert is_minimal is True

        # Now exceed expected scope significantly
        trace.graph.calls[0].scope.rows_requested = 10  # Expected is 1
        is_minimal, warnings = self.validator.validate_minimal_scope(trace)
        assert is_minimal is False
        assert len(warnings) > 0

    def test_validate_dataset_diversity(self):
        """Test dataset diversity metrics."""
        traces = [self.create_valid_trace() for _ in range(5)]

        # Modify some traces for diversity
        traces[1].request.intent_category = "update"
        traces[2].request.domain = "HR"
        traces[3].graph.calls[0].tool_id = "finance.export"

        diversity = self.validator.validate_dataset_diversity(traces)

        assert diversity["total_traces"] == 5
        assert diversity["unique_domains"] >= 1
        assert diversity["unique_intents"] >= 2
        assert "domain_distribution" in diversity
        assert "intent_distribution" in diversity

    def test_validate_trace_overall(self):
        """Test overall trace validation."""
        trace = self.create_valid_trace()

        is_valid, report = self.validator.validate_trace(trace)

        assert is_valid is True
        assert report["overall_valid"] is True
        assert report["checks"]["structure"]["valid"] is True
        assert report["checks"]["policy"]["compliant"] is True
        assert report["checks"]["minimal_scope"]["minimal"] is True
