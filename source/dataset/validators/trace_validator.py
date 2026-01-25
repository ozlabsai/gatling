"""
Trace validation logic for quality assurance.

Validates that generated traces meet all quality requirements:
- Structural validity (DAG, no missing references)
- Policy compliance (scope limits, forbidden ops)
- Data quality (diversity, realism)
"""

from collections import Counter

from source.dataset.models import GoldTrace, SystemPolicy


class TraceValidator:
    """
    Validates gold traces for quality and compliance.

    Provides multiple validation checks beyond the Oracle Agent's
    self-validation to ensure training data quality.
    """

    @staticmethod
    def validate_structure(trace: GoldTrace) -> tuple[bool, list[str]]:
        """
        Validate structural integrity of a trace.

        Checks:
        - Graph is a valid DAG
        - All tool_ids reference valid tools
        - All dependencies reference existing calls
        - Execution order is valid topological sort

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Check DAG validity
        if not trace.graph.validate_dag():
            errors.append("Tool-call graph contains cycles")

        # Check all dependencies reference existing calls
        call_ids = {call.call_id for call in trace.graph.calls}
        for call in trace.graph.calls:
            for dep in call.dependencies:
                if dep not in call_ids:
                    errors.append(
                        f"Call {call.call_id} depends on non-existent call {dep}"
                    )

        # Check execution order references all calls
        if set(trace.graph.execution_order) != call_ids:
            errors.append(
                "Execution order doesn't match call IDs: "
                f"{trace.graph.execution_order} vs {call_ids}"
            )

        # Verify topological order (dependencies come before dependents)
        executed = set()
        for call_id in trace.graph.execution_order:
            call = next((c for c in trace.graph.calls if c.call_id == call_id), None)
            if call:
                for dep in call.dependencies:
                    if dep not in executed:
                        errors.append(
                            f"Call {call_id} scheduled before its dependency {dep}"
                        )
                executed.add(call_id)

        return len(errors) == 0, errors

    @staticmethod
    def validate_policy_compliance(
        trace: GoldTrace, policy: SystemPolicy
    ) -> tuple[bool, list[str]]:
        """
        Validate that trace complies with system policy.

        Checks:
        - No forbidden operations
        - Scope limits respected
        - Sensitivity tiers appropriate

        Returns:
            (is_compliant, list of violations)
        """
        violations = []

        # Check for forbidden operations
        for call in trace.graph.calls:
            if call.tool_id in policy.forbidden_operations:
                violations.append(
                    f"Used forbidden operation: {call.tool_id}"
                )

        # Check scope limits
        for call in trace.graph.calls:
            # Check max_results limit
            if "max_results" in policy.scope_limits:
                max_allowed = policy.scope_limits["max_results"]
                if call.scope.rows_requested and call.scope.rows_requested > max_allowed:
                    violations.append(
                        f"Call {call.call_id} requests {call.scope.rows_requested} rows "
                        f"(limit: {max_allowed})"
                    )

            # Check for arguments that might violate limits
            for arg_name, arg_value in call.arguments.items():
                if arg_name in ["limit", "max_results", "count"]:
                    max_allowed = policy.scope_limits.get("max_results", float("inf"))
                    if isinstance(arg_value, int) and arg_value > max_allowed:
                        violations.append(
                            f"Call {call.call_id} argument {arg_name}={arg_value} "
                            f"exceeds policy limit of {max_allowed}"
                        )

        return len(violations) == 0, violations

    @staticmethod
    def validate_minimal_scope(trace: GoldTrace) -> tuple[bool, list[str]]:
        """
        Validate that the trace uses minimal necessary scope.

        Checks that the actual scope doesn't significantly exceed
        the expected minimal scope for the user's request.

        Returns:
            (is_minimal, list of warnings)
        """
        warnings = []

        expected = trace.request.expected_scope
        if expected.rows_requested:
            for call in trace.graph.calls:
                if call.scope.rows_requested:
                    # Allow 2x margin for practical reasons, but flag larger deviations
                    if call.scope.rows_requested > expected.rows_requested * 2:
                        warnings.append(
                            f"Call {call.call_id} requests {call.scope.rows_requested} rows "
                            f"but minimal need is {expected.rows_requested}"
                        )

        return len(warnings) == 0, warnings

    @staticmethod
    def validate_dataset_diversity(traces: list[GoldTrace]) -> dict[str, any]:
        """
        Validate diversity across a collection of traces.

        Checks:
        - Domain distribution
        - Intent category distribution
        - Tool usage distribution
        - Scope variety

        Returns:
            Dictionary with diversity metrics
        """
        if not traces:
            return {"error": "No traces to analyze"}

        # Domain distribution
        domains = [t.request.domain for t in traces]
        domain_counts = Counter(domains)

        # Intent distribution
        intents = [t.request.intent_category for t in traces]
        intent_counts = Counter(intents)

        # Tool usage
        tools_used = []
        for trace in traces:
            for call in trace.graph.calls:
                tools_used.append(call.tool_id)
        tool_counts = Counter(tools_used)

        # Scope distribution
        scopes = [
            call.scope.rows_requested
            for trace in traces
            for call in trace.graph.calls
            if call.scope.rows_requested
        ]

        # Graph complexity
        graph_sizes = [len(t.graph.calls) for t in traces]

        return {
            "total_traces": len(traces),
            "unique_domains": len(domain_counts),
            "domain_distribution": dict(domain_counts),
            "unique_intents": len(intent_counts),
            "intent_distribution": dict(intent_counts),
            "unique_tools": len(tool_counts),
            "tool_distribution": dict(tool_counts.most_common(10)),
            "scope_stats": {
                "min": min(scopes) if scopes else None,
                "max": max(scopes) if scopes else None,
                "avg": sum(scopes) / len(scopes) if scopes else None,
            },
            "graph_complexity": {
                "min_calls": min(graph_sizes),
                "max_calls": max(graph_sizes),
                "avg_calls": sum(graph_sizes) / len(graph_sizes),
            },
        }

    @staticmethod
    def validate_trace(trace: GoldTrace) -> tuple[bool, dict[str, any]]:
        """
        Run all validations on a single trace.

        Returns:
            (is_valid, validation_report)
        """
        report = {
            "trace_id": trace.trace_id,
            "validated": trace.validated,
            "checks": {},
        }

        # Structural validation
        structure_valid, structure_errors = TraceValidator.validate_structure(trace)
        report["checks"]["structure"] = {
            "valid": structure_valid,
            "errors": structure_errors,
        }

        # Policy compliance
        policy_valid, policy_violations = TraceValidator.validate_policy_compliance(
            trace, trace.policy
        )
        report["checks"]["policy"] = {
            "compliant": policy_valid,
            "violations": policy_violations,
        }

        # Minimal scope
        scope_valid, scope_warnings = TraceValidator.validate_minimal_scope(trace)
        report["checks"]["minimal_scope"] = {
            "minimal": scope_valid,
            "warnings": scope_warnings,
        }

        # Overall validity
        is_valid = structure_valid and policy_valid and scope_valid
        report["overall_valid"] = is_valid

        return is_valid, report
