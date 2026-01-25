"""
Transform action intents into ExecutionPlans.

Converts extracted intents into tool-call graphs compatible with the
Gatling data model.
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel

from source.dataset.conversations.intent_extractor import ActionIntent
from source.dataset.conversations.sampler import Conversation
from source.dataset.models import (
    GoldTrace,
    Provenance,
    ScopeMetadata,
    SensitivityTier,
    SystemPolicy,
    ToolCall,
    ToolCallGraph,
    TrustTier,
    UserRequest,
)
from source.dataset.schemas import DomainRegistry


class ExecutionPlan(BaseModel):
    """
    An execution plan derived from a conversation intent.

    This is similar to GoldTrace but sourced from real conversations
    rather than synthetic generation.
    """

    plan_id: str
    conversation_id: str
    source: str  # "wildchat" or "lmsys"
    user_request: UserRequest
    inferred_policy: SystemPolicy
    graph: ToolCallGraph
    original_intent: ActionIntent


class PlanTransformer:
    """
    Transforms action intents into ExecutionPlans.

    Uses an LLM to convert natural language intents into structured
    tool-call graphs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        """
        Initialize the plan transformer.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use for transformation
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    def transform_intents(
        self,
        conversations: list[Conversation],
        intent_map: dict[str, list[ActionIntent]],
    ) -> list[ExecutionPlan]:
        """
        Transform extracted intents into execution plans.

        Args:
            conversations: List of source conversations
            intent_map: Mapping of conversation_id to extracted intents

        Returns:
            List of execution plans
        """
        plans: list[ExecutionPlan] = []

        for conv in conversations:
            intents = intent_map.get(conv.conversation_id, [])
            if not intents:
                continue

            for intent in intents:
                try:
                    plan = self._transform_intent(conv, intent)
                    if plan:
                        plans.append(plan)
                except Exception as e:
                    print(
                        f"Warning: Failed to transform intent from {conv.conversation_id}: {e}"
                    )
                    continue

        print(f"✓ Transformed {len(plans)} intents into execution plans")
        return plans

    def _transform_intent(
        self, conversation: Conversation, intent: ActionIntent
    ) -> ExecutionPlan | None:
        """
        Transform a single intent into an execution plan.

        Args:
            conversation: Source conversation
            intent: Extracted action intent

        Returns:
            Execution plan or None if transformation fails
        """
        # Map intent to a domain
        domain = self._infer_domain(intent)

        # Get domain-specific tools and policy
        tools = DomainRegistry.get_schemas_for_domain(domain)
        policy = DomainRegistry.get_policy_for_domain(domain)

        # Build tool-call graph using LLM
        graph = self._generate_tool_graph(intent, tools, domain)

        if not graph:
            return None

        # Create user request
        user_request = UserRequest(
            request_id=f"{conversation.conversation_id}_intent_{intent.turn_idx}",
            domain=domain,
            text=intent.user_message,
            intent_category=intent.intent_category,
            expected_scope=ScopeMetadata(
                rows_requested=intent.scope_hints.get("limit", 10),
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
        )

        # Create execution plan
        plan_id = f"{conversation.source}_{conversation.conversation_id}_{intent.turn_idx}"

        return ExecutionPlan(
            plan_id=plan_id,
            conversation_id=conversation.conversation_id,
            source=conversation.source,
            user_request=user_request,
            inferred_policy=policy,
            graph=graph,
            original_intent=intent,
        )

    def _infer_domain(self, intent: ActionIntent) -> str:
        """
        Infer the most appropriate domain for an intent.

        Args:
            intent: Action intent

        Returns:
            Domain name
        """
        # Simple heuristic based on inferred tools
        tool_to_domain = {
            "calendar": "Calendar",
            "email": "Email",
            "file": "FileStorage",
            "database": "DatabaseManagement",
            "api": "APIManagement",
            "invoice": "Finance",
            "hr": "HR",
            "customer": "Sales",
        }

        for tool_hint in intent.inferred_tools:
            for keyword, domain in tool_to_domain.items():
                if keyword in tool_hint.lower():
                    return domain

        # Default to generic domain based on intent category
        category_domain = {
            "retrieve": "DatabaseManagement",
            "create": "FileStorage",
            "update": "DatabaseManagement",
            "delete": "FileStorage",
            "analyze": "BusinessIntelligence",
            "communicate": "Email",
            "configure": "CloudInfrastructure",
            "authenticate": "Security",
        }

        return category_domain.get(intent.intent_category, "APIManagement")

    def _generate_tool_graph(
        self, intent: ActionIntent, tools: list[Any], domain: str
    ) -> ToolCallGraph | None:
        """
        Generate a tool-call graph for an intent.

        Args:
            intent: Action intent
            tools: Available tools for the domain
            domain: Domain name

        Returns:
            Tool call graph or None if generation fails
        """
        # For now, create a simple single-tool graph
        # In a full implementation, this would use LLM to generate complex graphs

        # Find the most relevant tool
        tool = self._select_tool(intent, tools)

        if not tool:
            return None

        # Create a simple tool call
        call = ToolCall(
            call_id=f"call_1",
            tool_id=f"{domain.lower()}.{tool.tool_id}",
            arguments=self._infer_arguments(intent, tool),
            scope=ScopeMetadata(
                rows_requested=intent.scope_hints.get("limit", 10),
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
            provenance=Provenance(
                source_type=TrustTier.USER,
                source_id=f"user_msg_{intent.turn_idx}",
            ),
        )

        graph = ToolCallGraph(
            graph_id=f"graph_{intent.turn_idx}",
            calls=[call],
            execution_order=["call_1"],
        )

        return graph

    def _select_tool(self, intent: ActionIntent, tools: list[Any]) -> Any | None:
        """Select the most relevant tool for an intent."""
        if not tools:
            return None

        # Simple heuristic: match intent category to tool operations
        category_ops = {
            "retrieve": ["list", "get", "search", "find"],
            "create": ["create", "add", "new"],
            "update": ["update", "modify", "edit"],
            "delete": ["delete", "remove"],
            "analyze": ["analyze", "calculate", "compute"],
        }

        ops = category_ops.get(intent.intent_category, [])

        for tool in tools:
            tool_name = tool.tool_id.lower()
            if any(op in tool_name for op in ops):
                return tool

        # Fallback to first tool
        return tools[0] if tools else None

    def _infer_arguments(
        self, intent: ActionIntent, tool: Any
    ) -> dict[str, Any]:
        """
        Infer tool arguments from intent.

        Args:
            intent: Action intent
            tool: Selected tool schema

        Returns:
            Dictionary of inferred arguments
        """
        args: dict[str, Any] = {}

        # Add scope hints as arguments
        if "limit" in intent.scope_hints:
            args["limit"] = intent.scope_hints["limit"]

        if "filter" in intent.scope_hints:
            args["filter"] = intent.scope_hints["filter"]

        # Add default required arguments
        for param in tool.parameters:
            if param.required and param.name not in args:
                # Use default or placeholder
                if param.default is not None:
                    args[param.name] = param.default
                else:
                    # Simple type-based defaults
                    type_defaults = {
                        "string": "default",
                        "integer": 1,
                        "boolean": True,
                        "array": [],
                    }
                    args[param.name] = type_defaults.get(param.type, None)

        return args
