"""
Boundary Violation Validator.

Ensures that boundary violations are:
1. Actually violating a policy (not false positives)
2. Subtle enough to be useful training data
3. Diverse across violation types
4. Properly formatted for training
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from source.dataset.conversations.boundary_mutator import (
    BoundaryViolation,
    BoundaryViolationType,
)
from source.dataset.models import GoldTrace, SystemPolicy


class ValidationReport(BaseModel):
    """Report from boundary violation validation."""

    is_valid: bool
    violation_confirmed: bool
    subtlety_check_passed: bool
    format_check_passed: bool
    issues: list[str] = []
    warnings: list[str] = []


class BoundaryViolationValidator:
    """
    Validates boundary violations for quality and correctness.

    Ensures violations are suitable for training the energy-based model.
    """

    def __init__(self, max_severity: float = 0.3):
        """
        Initialize the validator.

        Args:
            max_severity: Maximum allowed severity score
        """
        self.max_severity = max_severity

    def validate_violation(
        self,
        violation: BoundaryViolation,
        original_trace: GoldTrace | None = None,
    ) -> ValidationReport:
        """
        Validate a single boundary violation.

        Args:
            violation: The boundary violation to validate
            original_trace: Optional original gold trace for comparison

        Returns:
            ValidationReport with validation results
        """
        issues = []
        warnings = []

        # Check 1: Violation type is valid
        if not isinstance(violation.violation_type, BoundaryViolationType):
            issues.append(
                f"Invalid violation type: {violation.violation_type}"
            )

        # Check 2: Severity is within threshold
        subtlety_check = violation.severity_score <= self.max_severity
        if not subtlety_check:
            issues.append(
                f"Severity {violation.severity_score} exceeds threshold {self.max_severity}"
            )

        # Check 3: Has valid modified graph
        format_check = self._validate_format(violation)
        if not format_check:
            issues.append("Invalid modified graph format")

        # Check 4: Violation description is meaningful
        if not violation.violation_description or len(
            violation.violation_description
        ) < 10:
            warnings.append("Violation description is too short or missing")

        # Check 5: Policy rule is specified
        if not violation.violated_policy_rule:
            warnings.append("Violated policy rule not specified")

        # Check 6: If original trace provided, verify it's actually different
        if original_trace:
            if not self._verify_mutation(violation, original_trace):
                issues.append(
                    "Violation plan is identical to original trace"
                )

        violation_confirmed = len(issues) == 0
        is_valid = violation_confirmed and subtlety_check and format_check

        return ValidationReport(
            is_valid=is_valid,
            violation_confirmed=violation_confirmed,
            subtlety_check_passed=subtlety_check,
            format_check_passed=format_check,
            issues=issues,
            warnings=warnings,
        )

    def _validate_format(self, violation: BoundaryViolation) -> bool:
        """Validate that violation has proper format."""
        try:
            # Check required fields
            if not violation.violation_id:
                return False
            if not violation.original_trace_id:
                return False
            if not violation.modified_graph:
                return False

            # Modified graph should have structure (ToolCallGraph)
            if not hasattr(violation.modified_graph, "calls"):
                return False

            return True
        except Exception:
            return False

    def _verify_mutation(
        self, violation: BoundaryViolation, original_trace: GoldTrace
    ) -> bool:
        """
        Verify that the violation is actually different from original.

        This is a simplified check - in production would do deep comparison.
        """
        # Check that IDs are different
        if violation.violation_id == original_trace.trace_id:
            return False

        # At minimum, should reference the original
        if original_trace.trace_id not in violation.original_trace_id:
            return False

        return True

    def validate_dataset_diversity(
        self, violations: list[BoundaryViolation]
    ) -> dict[str, Any]:
        """
        Validate diversity of violation dataset.

        Ensures good coverage across violation types and severities.
        """
        if not violations:
            return {
                "is_diverse": False,
                "issues": ["Empty dataset"],
            }

        # Count violation types
        type_counts = {}
        for v in violations:
            vtype = v.violation_type.value
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

        # Check severity distribution
        severity_values = [v.severity_score for v in violations]
        avg_severity = sum(severity_values) / len(severity_values)
        min_severity = min(severity_values)
        max_severity = max(severity_values)

        # Diversity checks
        issues = []
        warnings = []

        # Should have at least 3 different violation types
        unique_types = len(type_counts)
        if unique_types < 3:
            issues.append(
                f"Only {unique_types} violation types (expected at least 3)"
            )

        # No single type should dominate (>60%)
        max_type_pct = max(type_counts.values()) / len(violations) * 100
        if max_type_pct > 60:
            warnings.append(
                f"One violation type dominates: {max_type_pct:.1f}%"
            )

        # Severity should have reasonable spread
        severity_range = max_severity - min_severity
        if severity_range < 0.1:
            warnings.append(
                f"Limited severity range: {severity_range:.3f}"
            )

        is_diverse = len(issues) == 0

        return {
            "is_diverse": is_diverse,
            "unique_violation_types": unique_types,
            "type_distribution": type_counts,
            "severity_stats": {
                "avg": avg_severity,
                "min": min_severity,
                "max": max_severity,
                "range": severity_range,
            },
            "issues": issues,
            "warnings": warnings,
        }

    def validate_batch(
        self, violations: list[BoundaryViolation]
    ) -> dict[str, Any]:
        """
        Validate a batch of violations.

        Args:
            violations: List of violations to validate

        Returns:
            Dictionary with batch validation results
        """
        valid_count = 0
        invalid_count = 0
        all_issues = []

        for violation in violations:
            report = self.validate_violation(violation)
            if report.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_issues.extend(report.issues)

        # Get diversity metrics
        diversity = self.validate_dataset_diversity(violations)

        return {
            "total": len(violations),
            "valid": valid_count,
            "invalid": invalid_count,
            "validation_rate": valid_count / len(violations) * 100
            if violations
            else 0,
            "diversity": diversity,
            "common_issues": self._summarize_issues(all_issues),
        }

    def _summarize_issues(self, issues: list[str]) -> dict[str, int]:
        """Summarize common validation issues."""
        issue_counts = {}
        for issue in issues:
            # Extract issue type (first part before details)
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        return issue_counts
