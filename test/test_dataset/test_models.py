"""
Tests for dataset data models.

Validates the Pydantic models used for gold trace generation.
"""

import pytest
from datetime import datetime

from source.dataset.models import (
    GoldTrace,
    ProvenancePointer,
    ScopeMetadata,
    SensitivityTier,
    SystemPolicy,
    ToolCall,
    ToolCallGraph,
    ToolParameter,
    ToolSchema,
    TrustTier,
    UserRequest,
)


class TestToolSchema:
    """Tests for ToolSchema model."""

    def test_create_tool_schema(self):
        """Test creating a valid tool schema."""
        schema = ToolSchema(
            tool_id="test.read_data",
            domain="Testing",
            name="Read Data",
            description="Reads test data",
            parameters=[
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                    default=10,
                )
            ],
            returns="List of data",
            sensitivity=SensitivityTier.INTERNAL,
        )

        assert schema.tool_id == "test.read_data"
        assert schema.domain == "Testing"
        assert len(schema.parameters) == 1
        assert schema.parameters[0].name == "limit"


class TestSystemPolicy:
    """Tests for SystemPolicy model."""

    def test_create_policy(self):
        """Test creating a system policy."""
        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Testing",
            description="Test policy",
            rules=["Rule 1", "Rule 2"],
            forbidden_operations=["test.delete"],
            scope_limits={"max_results": 100},
        )

        assert policy.policy_id == "test_policy"
        assert len(policy.rules) == 2
        assert "test.delete" in policy.forbidden_operations
        assert policy.scope_limits["max_results"] == 100


class TestToolCallGraph:
    """Tests for ToolCallGraph model."""

    def test_validate_dag_no_cycles(self):
        """Test DAG validation with no cycles."""
        call1 = ToolCall(
            call_id="call1",
            tool_id="test.tool",
            arguments={},
            scope=ScopeMetadata(),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user1",
            ),
            dependencies=[],
        )

        call2 = ToolCall(
            call_id="call2",
            tool_id="test.tool",
            arguments={},
            scope=ScopeMetadata(),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user1",
            ),
            dependencies=["call1"],
        )

        graph = ToolCallGraph(
            graph_id="graph1",
            calls=[call1, call2],
            execution_order=["call1", "call2"],
        )

        assert graph.validate_dag() is True

    def test_validate_dag_with_cycle(self):
        """Test DAG validation detects cycles."""
        call1 = ToolCall(
            call_id="call1",
            tool_id="test.tool",
            arguments={},
            scope=ScopeMetadata(),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user1",
            ),
            dependencies=["call2"],  # Circular dependency
        )

        call2 = ToolCall(
            call_id="call2",
            tool_id="test.tool",
            arguments={},
            scope=ScopeMetadata(),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user1",
            ),
            dependencies=["call1"],  # Circular dependency
        )

        graph = ToolCallGraph(
            graph_id="graph1",
            calls=[call1, call2],
            execution_order=["call1", "call2"],
        )

        assert graph.validate_dag() is False


class TestGoldTrace:
    """Tests for GoldTrace model."""

    def test_create_gold_trace(self):
        """Test creating a complete gold trace."""
        request = UserRequest(
            request_id="req1",
            domain="Testing",
            text="Get test data",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(rows_requested=1),
        )

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Testing",
            description="Test policy",
            rules=["Test rule"],
        )

        call = ToolCall(
            call_id="call1",
            tool_id="test.read",
            arguments={"limit": 1},
            scope=ScopeMetadata(rows_requested=1),
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

        trace = GoldTrace(
            trace_id="trace1",
            request=request,
            policy=policy,
            graph=graph,
        )

        assert trace.trace_id == "trace1"
        assert trace.validated is False

    def test_to_training_format(self):
        """Test converting trace to training format."""
        request = UserRequest(
            request_id="req1",
            domain="Testing",
            text="Get test data",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(rows_requested=1),
        )

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Testing",
            description="Test policy",
            rules=["Test rule"],
        )

        call = ToolCall(
            call_id="call1",
            tool_id="test.read",
            arguments={"limit": 1},
            scope=ScopeMetadata(rows_requested=1),
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

        trace = GoldTrace(
            trace_id="trace1",
            request=request,
            policy=policy,
            graph=graph,
        )

        training_data = trace.to_training_format()

        assert training_data["trace_id"] == "trace1"
        assert "request" in training_data
        assert "policy" in training_data
        assert "graph" in training_data
        assert isinstance(training_data["created_at"], str)
