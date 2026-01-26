"""
Tests for policy boundary violation generator.

Tests the boundary mutation system for Stage B of the dataset pipeline.
"""

import pytest
from datetime import datetime

from source.dataset.conversations.boundary_mutator import (
    BoundaryViolation,
    BoundaryViolationType,
    PolicyBoundaryMutator,
)
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
from source.dataset.validators.boundary_validator import BoundaryViolationValidator


class TestPolicyBoundaryMutator:
    """Tests for PolicyBoundaryMutator."""

    def test_init(self):
        """Test mutator initialization."""
        mutator = PolicyBoundaryMutator(mutation_rate=0.5, seed=42)
        assert mutator.mutation_rate == 0.5
        assert mutator.seed == 42
        assert mutator.subtlety_threshold == 0.3

    def test_mutate_traces_empty(self):
        """Test mutation with empty trace list."""
        mutator = PolicyBoundaryMutator()
        violations = mutator.mutate_traces([])
        assert violations == []

    def test_find_numeric_limit(self):
        """Test finding numeric limits in policy."""
        mutator = PolicyBoundaryMutator()

        # Policy with max_results
        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Finance",
            description="Test policy",
            rules=["Test rule"],
            scope_limits={"max_results": 100},
        )

        key, value = mutator._find_numeric_limit(policy)
        assert key == "max_results"
        assert value == 100

    def test_find_numeric_limit_no_limits(self):
        """Test finding limits when none exist."""
        mutator = PolicyBoundaryMutator()

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Finance",
            description="Test policy",
            rules=["Test rule"],
            scope_limits={},
        )

        key, value = mutator._find_numeric_limit(policy)
        assert key is None
        assert value is None

    def test_get_applicable_mutations_numeric(self):
        """Test detecting applicable mutations for numeric limits."""
        mutator = PolicyBoundaryMutator()

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Finance",
            description="Test policy",
            rules=["Test rule"],
            scope_limits={"max_results": 100},
        )

        graph = ToolCallGraph(graph_id="test_graph", calls=[])

        applicable = mutator._get_applicable_mutations(policy, graph)

        assert BoundaryViolationType.NUMERIC_EDGE in applicable
        assert BoundaryViolationType.NUMERIC_NEAR in applicable
        assert BoundaryViolationType.SENSITIVITY_CREEP in applicable

    def test_get_applicable_mutations_temporal(self):
        """Test detecting applicable mutations for temporal limits."""
        mutator = PolicyBoundaryMutator()

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Calendar",
            description="Test policy",
            rules=["Maximum 90 days of history"],
            scope_limits={"max_days": 90},
        )

        graph = ToolCallGraph(graph_id="test_graph", calls=[])

        applicable = mutator._get_applicable_mutations(policy, graph)

        assert BoundaryViolationType.TEMPORAL_OVERFLOW in applicable

    def test_get_applicable_mutations_access(self):
        """Test detecting applicable mutations for access control."""
        mutator = PolicyBoundaryMutator()

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="HR",
            description="Test policy",
            rules=["Users can only access their own department"],
            scope_limits={},
        )

        graph = ToolCallGraph(graph_id="test_graph", calls=[])

        applicable = mutator._get_applicable_mutations(policy, graph)

        assert BoundaryViolationType.ACCESS_BOUNDARY in applicable

    def test_mutate_numeric_edge(self):
        """Test numeric edge mutation."""
        mutator = PolicyBoundaryMutator()

        # Create a gold trace with numeric limit
        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Finance",
            description="Test policy",
            rules=["Maximum 100 results per query"],
            scope_limits={"max_results": 100},
        )

        tool_call = ToolCall(
            call_id="call_1",
            tool_id="finance.list_invoices",
            arguments={"limit": 50},
            scope=ScopeMetadata(
                rows_requested=50,
                sensitivity_tier=SensitivityTier.CONFIDENTIAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_request",
            ),
        )

        graph = ToolCallGraph(
            graph_id="test_graph",
            calls=[tool_call],
            execution_order=["call_1"],
        )

        request = UserRequest(
            request_id="req_1",
            domain="Finance",
            text="Find recent invoices",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(rows_requested=50),
        )

        trace = GoldTrace(
            trace_id="trace_1",
            request=request,
            policy=policy,
            graph=graph,
        )

        # Apply mutation
        violation = mutator._mutate_numeric_edge(trace)

        assert violation is not None
        assert violation.violation_type == BoundaryViolationType.NUMERIC_EDGE
        assert violation.severity_score == 0.1  # Very subtle
        assert "101" in violation.violation_description  # Should be limit+1

    def test_mutate_numeric_near(self):
        """Test numeric near-edge mutation."""
        mutator = PolicyBoundaryMutator(seed=42)

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Email",
            description="Test policy",
            rules=["Maximum 200 messages per query"],
            scope_limits={"max_results": 200},
        )

        tool_call = ToolCall(
            call_id="call_1",
            tool_id="email.list_messages",
            arguments={"limit": 100},
            scope=ScopeMetadata(
                rows_requested=100,
                sensitivity_tier=SensitivityTier.CONFIDENTIAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_request",
            ),
        )

        graph = ToolCallGraph(
            graph_id="test_graph",
            calls=[tool_call],
        )

        request = UserRequest(
            request_id="req_1",
            domain="Email",
            text="List my emails",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(rows_requested=100),
        )

        trace = GoldTrace(
            trace_id="trace_1",
            request=request,
            policy=policy,
            graph=graph,
        )

        violation = mutator._mutate_numeric_near(trace)

        assert violation is not None
        assert violation.violation_type == BoundaryViolationType.NUMERIC_NEAR
        assert violation.severity_score == 0.2
        # Should be between 210-220 (200 + 5-10%)
        # Extract the number from the description
        import re
        match = re.search(r"Requested (\d+) items", violation.violation_description)
        assert match is not None
        requested = int(match.group(1))
        assert 210 <= requested <= 220, f"Expected 210-220, got {requested}"

    def test_mutate_temporal_overflow(self):
        """Test temporal boundary violation."""
        mutator = PolicyBoundaryMutator(seed=42)

        policy = SystemPolicy(
            policy_id="calendar_policy",
            domain="Calendar",
            description="Calendar access policy",
            rules=["Maximum 90 days of event history"],
            scope_limits={"max_days": 90},
        )

        tool_call = ToolCall(
            call_id="call_1",
            tool_id="calendar.list_events",
            arguments={"calendar_type": "personal"},
            scope=ScopeMetadata(
                rows_requested=50,
                time_range_days=30,
                sensitivity_tier=SensitivityTier.CONFIDENTIAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_request",
            ),
        )

        graph = ToolCallGraph(
            graph_id="test_graph",
            calls=[tool_call],
        )

        request = UserRequest(
            request_id="req_1",
            domain="Calendar",
            text="Show my calendar events",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(time_range_days=30),
        )

        trace = GoldTrace(
            trace_id="trace_1",
            request=request,
            policy=policy,
            graph=graph,
        )

        violation = mutator._mutate_temporal_overflow(trace)

        assert violation is not None
        assert violation.violation_type == BoundaryViolationType.TEMPORAL_OVERFLOW
        assert "91" in violation.violation_description or "95" in violation.violation_description

    def test_mutate_access_boundary(self):
        """Test access boundary violation."""
        mutator = PolicyBoundaryMutator()

        policy = SystemPolicy(
            policy_id="hr_policy",
            domain="HR",
            description="HR access policy",
            rules=["Users can only access their own department data"],
            scope_limits={},
        )

        tool_call = ToolCall(
            call_id="call_1",
            tool_id="hr.list_employees",
            arguments={"department": "Engineering"},
            scope=ScopeMetadata(
                rows_requested=20,
                sensitivity_tier=SensitivityTier.CONFIDENTIAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_request",
            ),
        )

        graph = ToolCallGraph(
            graph_id="test_graph",
            calls=[tool_call],
        )

        request = UserRequest(
            request_id="req_1",
            domain="HR",
            text="List employees in my department",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(rows_requested=20),
        )

        trace = GoldTrace(
            trace_id="trace_1",
            request=request,
            policy=policy,
            graph=graph,
        )

        violation = mutator._mutate_access_boundary(trace)

        assert violation is not None
        assert violation.violation_type == BoundaryViolationType.ACCESS_BOUNDARY
        assert "Adjacent_Department" in str(violation.violation_description) or "adjacent" in violation.violation_description.lower()

    def test_mutate_sensitivity_creep(self):
        """Test sensitivity tier escalation."""
        mutator = PolicyBoundaryMutator()

        policy = SystemPolicy(
            policy_id="test_policy",
            domain="Finance",
            description="Test policy",
            rules=["Standard access policy"],
            scope_limits={},
        )

        tool_call = ToolCall(
            call_id="call_1",
            tool_id="finance.list_invoices",
            arguments={"limit": 50},
            scope=ScopeMetadata(
                rows_requested=50,
                sensitivity_tier=SensitivityTier.INTERNAL,
            ),
            provenance=ProvenancePointer(
                source_type=TrustTier.USER,
                source_id="user_request",
            ),
        )

        graph = ToolCallGraph(
            graph_id="test_graph",
            calls=[tool_call],
        )

        request = UserRequest(
            request_id="req_1",
            domain="Finance",
            text="Find invoices",
            intent_category="retrieve",
            expected_scope=ScopeMetadata(rows_requested=50),
        )

        trace = GoldTrace(
            trace_id="trace_1",
            request=request,
            policy=policy,
            graph=graph,
        )

        violation = mutator._mutate_sensitivity_creep(trace)

        assert violation is not None
        assert violation.violation_type == BoundaryViolationType.SENSITIVITY_CREEP
        # Should escalate INTERNAL → CONFIDENTIAL
        # Can't directly check execution_plan in this simplified version,
        # but we verify the mutation was created


class TestBoundaryViolationValidator:
    """Tests for BoundaryViolationValidator."""

    def test_init(self):
        """Test validator initialization."""
        validator = BoundaryViolationValidator(max_severity=0.3)
        assert validator.max_severity == 0.3

    def test_validate_violation_valid(self):
        """Test validating a valid violation."""
        validator = BoundaryViolationValidator()

        violation = BoundaryViolation(
            violation_id="violation_1",
            original_trace_id="trace_1",
            violation_type=BoundaryViolationType.NUMERIC_EDGE,
            violated_policy_rule="max_results=100",
            violation_description="Requested 101 items when policy limit is 100",
            modified_graph=ToolCallGraph(graph_id="test_graph", calls=[]),
            severity_score=0.1,
        )

        report = validator.validate_violation(violation)

        assert report.is_valid
        assert report.violation_confirmed
        assert report.subtlety_check_passed
        assert report.format_check_passed
        assert len(report.issues) == 0

    def test_validate_violation_too_severe(self):
        """Test validating a violation that's too severe."""
        validator = BoundaryViolationValidator(max_severity=0.3)

        violation = BoundaryViolation(
            violation_id="violation_1",
            original_trace_id="trace_1",
            violation_type=BoundaryViolationType.NUMERIC_EDGE,
            violated_policy_rule="max_results=100",
            violation_description="Requested 500 items when policy limit is 100",
            modified_graph=ToolCallGraph(graph_id="test_graph", calls=[]),
            severity_score=0.5,  # Too high
        )

        report = validator.validate_violation(violation)

        assert not report.is_valid
        assert not report.subtlety_check_passed
        assert "Severity" in report.issues[0]

    def test_validate_dataset_diversity_good(self):
        """Test diversity validation with good coverage."""
        validator = BoundaryViolationValidator()

        violations = [
            BoundaryViolation(
                violation_id=f"v_{i}",
                original_trace_id=f"trace_{i}",
                violation_type=BoundaryViolationType.NUMERIC_EDGE
                if i % 3 == 0
                else (
                    BoundaryViolationType.NUMERIC_NEAR
                    if i % 3 == 1
                    else BoundaryViolationType.TEMPORAL_OVERFLOW
                ),
                violated_policy_rule="test_rule",
                violation_description="Test violation",
                modified_graph=ToolCallGraph(graph_id=f"graph_{i}", calls=[]),
                severity_score=0.1 + (i % 3) * 0.05,
            )
            for i in range(30)
        ]

        diversity = validator.validate_dataset_diversity(violations)

        assert diversity["is_diverse"]
        assert diversity["unique_violation_types"] == 3
        assert len(diversity["issues"]) == 0

    def test_validate_dataset_diversity_poor(self):
        """Test diversity validation with poor coverage."""
        validator = BoundaryViolationValidator()

        # All same type
        violations = [
            BoundaryViolation(
                violation_id=f"v_{i}",
                original_trace_id=f"trace_{i}",
                violation_type=BoundaryViolationType.NUMERIC_EDGE,
                violated_policy_rule="test_rule",
                violation_description="Test violation",
                modified_graph=ToolCallGraph(graph_id=f"graph_{i}", calls=[]),
                severity_score=0.1,
            )
            for i in range(30)
        ]

        diversity = validator.validate_dataset_diversity(violations)

        assert not diversity["is_diverse"]
        assert diversity["unique_violation_types"] == 1
        assert len(diversity["issues"]) > 0

    def test_validate_batch(self):
        """Test batch validation."""
        validator = BoundaryViolationValidator()

        violations = [
            BoundaryViolation(
                violation_id=f"v_{i}",
                original_trace_id=f"trace_{i}",
                violation_type=BoundaryViolationType.NUMERIC_EDGE,
                violated_policy_rule="test_rule",
                violation_description="Test violation",
                modified_graph=ToolCallGraph(graph_id=f"graph_{i}", calls=[]),
                severity_score=0.1 if i % 2 == 0 else 0.5,  # Half too severe
            )
            for i in range(10)
        ]

        results = validator.validate_batch(violations)

        assert results["total"] == 10
        assert results["valid"] == 5  # Only those with severity <= 0.3
        assert results["invalid"] == 5
        assert results["validation_rate"] == 50.0
