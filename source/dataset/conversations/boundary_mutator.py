"""
Policy Boundary Violation Generator for Stage B: Policy Boundary Cases.

This module generates "near-safe" plans that violate subtle policy boundaries.
Unlike the hard negatives (scope_blowup, exfiltration), boundary cases are
designed to test the model's ability to enforce precise policy limits.

Target: 2M boundary violation samples from 4M gold traces
Strategy: Apply systematic boundary mutations that are subtle but violating

Boundary Types:
    1. Numeric Scope Boundaries (max_results, limit, etc.)
    2. Access Control Boundaries (department, ownership)
    3. Temporal Boundaries (date ranges, history limits)
    4. Approval Boundaries (operations requiring authorization)
"""

from __future__ import annotations

import copy
import random
from enum import Enum
from typing import Any

from pydantic import BaseModel

from source.dataset.models import (
    GoldTrace,
    ScopeMetadata,
    SensitivityTier,
    SystemPolicy,
    ToolCall,
    ToolCallGraph,
    TrustTier,
)


class BoundaryViolationType(str, Enum):
    """Types of policy boundary violations."""

    NUMERIC_EDGE = "numeric_edge"  # Exactly at limit+1 (max=100 → 101)
    NUMERIC_NEAR = "numeric_near"  # Close to limit (max=100 → 105-110)
    TEMPORAL_OVERFLOW = "temporal_overflow"  # Date range slightly over limit
    ACCESS_BOUNDARY = "access_boundary"  # Adjacent department/user access
    APPROVAL_BYPASS = "approval_bypass"  # Operation requiring approval
    SENSITIVITY_CREEP = "sensitivity_creep"  # One tier above allowed


class BoundaryViolation(BaseModel):
    """
    A boundary violation record.

    Tracks the specific policy boundary that was violated and how.
    """

    violation_id: str
    original_trace_id: str
    violation_type: BoundaryViolationType
    violated_policy_rule: str
    violation_description: str
    modified_graph: ToolCallGraph  # The mutated tool call graph
    severity_score: float = 1.0  # How subtle the violation is (0-1, lower = more subtle)


class PolicyBoundaryMutator:
    """
    Generates policy boundary violations from gold traces.

    Takes compliant gold traces and creates subtle violations that are
    "almost safe" but cross specific policy boundaries.
    """

    def __init__(
        self,
        mutation_rate: float = 0.5,
        seed: int = 42,
        subtlety_threshold: float = 0.3,
    ):
        """
        Initialize the boundary mutator.

        Args:
            mutation_rate: Fraction of traces to attempt mutation (default 50%)
            seed: Random seed for reproducible mutations
            subtlety_threshold: Maximum severity score for violations (lower = more subtle)
        """
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.subtlety_threshold = subtlety_threshold
        random.seed(seed)

        # Statistics tracking
        self.stats = {
            "total_attempts": 0,
            "successful_mutations": 0,
            "failed_mutations": 0,
            "by_violation_type": {},
        }

    def mutate_traces(
        self, gold_traces: list[GoldTrace]
    ) -> list[BoundaryViolation]:
        """
        Apply boundary mutations to gold traces.

        Args:
            gold_traces: List of policy-compliant gold traces

        Returns:
            List of boundary violation samples
        """
        violations = []
        n_mutations = int(len(gold_traces) * self.mutation_rate)

        # Randomly select traces to mutate
        traces_to_mutate = random.sample(
            gold_traces, min(n_mutations, len(gold_traces))
        )

        for trace in traces_to_mutate:
            self.stats["total_attempts"] += 1

            try:
                # Try to apply a boundary mutation
                violation = self._apply_boundary_mutation(trace)

                if violation and violation.severity_score <= self.subtlety_threshold:
                    violations.append(violation)
                    self.stats["successful_mutations"] += 1

                    # Update type statistics
                    vtype = violation.violation_type.value
                    self.stats["by_violation_type"][vtype] = (
                        self.stats["by_violation_type"].get(vtype, 0) + 1
                    )
                else:
                    self.stats["failed_mutations"] += 1

            except Exception as e:
                print(f"Warning: Failed to mutate trace {trace.trace_id}: {e}")
                self.stats["failed_mutations"] += 1
                continue

        success_rate = len(violations) / n_mutations * 100 if n_mutations > 0 else 0
        print(
            f"✓ Created {len(violations)} boundary violations "
            f"(success rate: {success_rate:.1f}%)"
        )

        return violations

    def _apply_boundary_mutation(
        self, trace: GoldTrace
    ) -> BoundaryViolation | None:
        """
        Apply a single boundary mutation to a trace.

        Selects the most appropriate violation type based on the
        policy structure and trace characteristics.
        """
        policy = trace.policy
        graph = trace.graph

        # Determine which mutation types are applicable
        applicable_mutations = self._get_applicable_mutations(policy, graph)

        if not applicable_mutations:
            return None

        # Select a mutation type
        mutation_type = random.choice(applicable_mutations)

        # Apply the specific mutation
        if mutation_type == BoundaryViolationType.NUMERIC_EDGE:
            return self._mutate_numeric_edge(trace)
        elif mutation_type == BoundaryViolationType.NUMERIC_NEAR:
            return self._mutate_numeric_near(trace)
        elif mutation_type == BoundaryViolationType.TEMPORAL_OVERFLOW:
            return self._mutate_temporal_overflow(trace)
        elif mutation_type == BoundaryViolationType.ACCESS_BOUNDARY:
            return self._mutate_access_boundary(trace)
        elif mutation_type == BoundaryViolationType.APPROVAL_BYPASS:
            return self._mutate_approval_bypass(trace)
        elif mutation_type == BoundaryViolationType.SENSITIVITY_CREEP:
            return self._mutate_sensitivity_creep(trace)

        return None

    def _get_applicable_mutations(
        self, policy: SystemPolicy, graph: Any
    ) -> list[BoundaryViolationType]:
        """
        Determine which mutation types are applicable for this trace.

        Based on policy structure and existing tool calls.
        """
        applicable = []

        # Check for numeric scope limits
        if policy.scope_limits and any(
            k in policy.scope_limits
            for k in ["max_results", "max_export_rows", "max_recipients"]
        ):
            applicable.extend(
                [BoundaryViolationType.NUMERIC_EDGE, BoundaryViolationType.NUMERIC_NEAR]
            )

        # Check for temporal limits
        if policy.scope_limits and any(
            k in policy.scope_limits for k in ["max_days", "max_history_days"]
        ):
            applicable.append(BoundaryViolationType.TEMPORAL_OVERFLOW)

        # Check for access control rules
        if any(
            "department" in rule.lower() or "own" in rule.lower()
            for rule in policy.rules
        ):
            applicable.append(BoundaryViolationType.ACCESS_BOUNDARY)

        # Check for approval requirements
        if any(
            "approval" in rule.lower() or "require" in rule.lower()
            for rule in policy.rules
        ):
            applicable.append(BoundaryViolationType.APPROVAL_BYPASS)

        # Sensitivity creep is always applicable
        applicable.append(BoundaryViolationType.SENSITIVITY_CREEP)

        return applicable

    def _mutate_numeric_edge(self, trace: GoldTrace) -> BoundaryViolation | None:
        """
        Mutate by setting numeric limit exactly at boundary+1.

        Example: max_results=100 → request 101 items
        This is the most subtle numeric violation.
        """
        mutated_graph = copy.deepcopy(trace.graph)
        policy = trace.policy

        # Find the most restrictive limit
        limit_key, limit_value = self._find_numeric_limit(policy)
        if not limit_key or not limit_value:
            return None

        # Find a tool call with this limit
        modified = False
        for call in mutated_graph.calls:
            if "limit" in call.arguments:
                # Set to exactly limit+1
                call.arguments["limit"] = limit_value + 1
                call.scope.rows_requested = limit_value + 1
                modified = True
                break

            elif limit_key == "max_recipients" and "to" in call.arguments:
                # For email recipients
                if isinstance(call.arguments["to"], list):
                    # Add one more recipient than allowed
                    call.arguments["to"] = [
                        f"user{i}@example.com" for i in range(limit_value + 1)
                    ]
                    modified = True
                    break

        if not modified:
            return None

        violation_id = f"{trace.trace_id}_boundary_edge"

        return BoundaryViolation(
            violation_id=violation_id,
            original_trace_id=trace.trace_id,
            violation_type=BoundaryViolationType.NUMERIC_EDGE,
            violated_policy_rule=f"{limit_key}={limit_value}",
            violation_description=f"Requested {limit_value + 1} items when policy limit is {limit_value}",
            modified_graph=mutated_graph,
            severity_score=0.1,  # Very subtle - just one over the line
        )

    def _mutate_numeric_near(self, trace: GoldTrace) -> BoundaryViolation | None:
        """
        Mutate by setting numeric limit close to boundary.

        Example: max_results=100 → request 105-110 items
        Slightly less subtle than edge violation.
        """
        mutated_graph = copy.deepcopy(trace.graph)
        policy = trace.policy

        limit_key, limit_value = self._find_numeric_limit(policy)
        if not limit_key or not limit_value:
            return None

        # Set to limit + small overshoot (5-10%)
        overshoot = int(limit_value * random.uniform(0.05, 0.10))
        new_value = limit_value + max(1, overshoot)

        modified = False
        for call in mutated_graph.calls:
            if "limit" in call.arguments:
                call.arguments["limit"] = new_value
                call.scope.rows_requested = new_value
                modified = True
                break

        if not modified:
            return None

        violation_id = f"{trace.trace_id}_boundary_near"

        return BoundaryViolation(
            violation_id=violation_id,
            original_trace_id=trace.trace_id,
            violation_type=BoundaryViolationType.NUMERIC_NEAR,
            violated_policy_rule=f"{limit_key}={limit_value}",
            violation_description=f"Requested {new_value} items when policy limit is {limit_value}",
            modified_graph=mutated_graph,
            severity_score=0.2,  # Slightly more obvious
        )

    def _mutate_temporal_overflow(
        self, trace: GoldTrace
    ) -> BoundaryViolation | None:
        """
        Mutate by extending temporal range beyond policy limit.

        Example: max_days=90 → query for 91-95 days
        """
        mutated_graph = copy.deepcopy(trace.graph)
        policy = trace.policy

        # Check for temporal limits
        max_days = policy.scope_limits.get("max_days") or policy.scope_limits.get(
            "max_history_days"
        )
        if not max_days:
            return None

        # Extend by 1-5 days
        overflow_days = max_days + random.randint(1, 5)

        modified = False
        for call in mutated_graph.calls:
            if call.scope.time_range_days:
                call.scope.time_range_days = overflow_days
                modified = True
                break

            # Also check arguments for date range
            if "start_date" in call.arguments or "end_date" in call.arguments:
                # Update time_range_days in scope
                call.scope.time_range_days = overflow_days
                modified = True
                break

        if not modified:
            return None

        violation_id = f"{trace.trace_id}_boundary_temporal"

        return BoundaryViolation(
            violation_id=violation_id,
            original_trace_id=trace.trace_id,
            violation_type=BoundaryViolationType.TEMPORAL_OVERFLOW,
            violated_policy_rule=f"max_days={max_days}",
            violation_description=f"Queried {overflow_days} days when policy limit is {max_days} days",
            modified_graph=mutated_graph,
            severity_score=0.15,
        )

    def _mutate_access_boundary(self, trace: GoldTrace) -> BoundaryViolation | None:
        """
        Mutate by accessing adjacent/nearby access boundaries.

        Example: "own department only" → access adjacent department
        """
        mutated_graph = copy.deepcopy(trace.graph)
        policy = trace.policy

        # Check for access control rules
        has_department_rule = any(
            "department" in rule.lower() for rule in policy.rules
        )
        has_ownership_rule = any("own" in rule.lower() for rule in policy.rules)

        if not (has_department_rule or has_ownership_rule):
            return None

        # Modify arguments to access adjacent resource
        modified = False
        for call in mutated_graph.calls:
            if "department" in call.arguments:
                # Change to "Adjacent_Department"
                call.arguments["department"] = "Adjacent_Department"
                modified = True
                break

            elif "employee_id" in call.arguments and has_ownership_rule:
                # Access another user's data
                call.arguments["employee_id"] = "other_user_123"
                modified = True
                break

            elif "mailbox" in call.arguments or "folder" in call.arguments:
                # Access someone else's mailbox
                if "mailbox" in call.arguments:
                    call.arguments["mailbox"] = "colleague@example.com"
                    modified = True
                    break

        if not modified:
            return None

        violation_id = f"{trace.trace_id}_boundary_access"

        violated_rule = next(
            (rule for rule in policy.rules if "own" in rule.lower() or "department" in rule.lower()),
            "access control policy"
        )

        return BoundaryViolation(
            violation_id=violation_id,
            original_trace_id=trace.trace_id,
            violation_type=BoundaryViolationType.ACCESS_BOUNDARY,
            violated_policy_rule=violated_rule,
            violation_description="Accessed resource outside authorized boundary (adjacent department/user)",
            modified_graph=mutated_graph,
            severity_score=0.25,
        )

    def _mutate_approval_bypass(self, trace: GoldTrace) -> BoundaryViolation | None:
        """
        Mutate by removing approval requirement metadata.

        Example: Production deployment without approval flag
        """
        mutated_graph = copy.deepcopy(trace.graph)
        policy = trace.policy

        # Check for approval rules
        approval_rules = [
            rule for rule in policy.rules if "approval" in rule.lower()
        ]
        if not approval_rules:
            return None

        # Find a high-privilege tool call and remove approval metadata
        modified = False
        for call in mutated_graph.calls:
            # Look for high-sensitivity operations
            if call.scope.sensitivity_tier in [
                SensitivityTier.CONFIDENTIAL,
                SensitivityTier.RESTRICTED,
            ]:
                # Mark as coming from user without approval
                call.provenance.source_type = TrustTier.USER
                call.provenance.content_snippet = "Direct user request (no approval)"
                modified = True
                break

        if not modified:
            return None

        violation_id = f"{trace.trace_id}_boundary_approval"

        return BoundaryViolation(
            violation_id=violation_id,
            original_trace_id=trace.trace_id,
            violation_type=BoundaryViolationType.APPROVAL_BYPASS,
            violated_policy_rule=approval_rules[0],
            violation_description="Attempted operation requiring approval without authorization",
            modified_graph=mutated_graph,
            severity_score=0.3,
        )

    def _mutate_sensitivity_creep(
        self, trace: GoldTrace
    ) -> BoundaryViolation | None:
        """
        Mutate by increasing data sensitivity by one tier.

        Example: INTERNAL → CONFIDENTIAL
        More subtle than full privilege escalation.
        """
        mutated_graph = copy.deepcopy(trace.graph)

        # Find a call we can escalate by one tier
        modified = False
        for call in mutated_graph.calls:
            current_tier = call.scope.sensitivity_tier

            # Escalate by exactly one tier
            if current_tier == SensitivityTier.PUBLIC:
                call.scope.sensitivity_tier = SensitivityTier.INTERNAL
                modified = True
                break
            elif current_tier == SensitivityTier.INTERNAL:
                call.scope.sensitivity_tier = SensitivityTier.CONFIDENTIAL
                modified = True
                break
            elif current_tier == SensitivityTier.CONFIDENTIAL:
                call.scope.sensitivity_tier = SensitivityTier.RESTRICTED
                modified = True
                break

        if not modified:
            return None

        violation_id = f"{trace.trace_id}_boundary_sensitivity"

        return BoundaryViolation(
            violation_id=violation_id,
            original_trace_id=trace.trace_id,
            violation_type=BoundaryViolationType.SENSITIVITY_CREEP,
            violated_policy_rule="data sensitivity classification",
            violation_description="Accessed data one sensitivity tier above authorized level",
            modified_graph=mutated_graph,
            severity_score=0.2,
        )

    def _find_numeric_limit(
        self, policy: SystemPolicy
    ) -> tuple[str | None, int | None]:
        """
        Find the most relevant numeric limit in the policy.

        Returns:
            Tuple of (limit_key, limit_value) or (None, None)
        """
        if not policy.scope_limits:
            return None, None

        # Priority order for limits
        priority_keys = [
            "max_results",
            "max_export_rows",
            "max_recipients",
            "max_concurrent_deployments",
        ]

        for key in priority_keys:
            if key in policy.scope_limits:
                return key, policy.scope_limits[key]

        # Return first available limit
        if policy.scope_limits:
            key = next(iter(policy.scope_limits))
            return key, policy.scope_limits[key]

        return None, None

    def get_statistics(self) -> dict[str, Any]:
        """Get mutation statistics."""
        return self.stats
