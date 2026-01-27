"""
Minimal Scope Label Generator for E_scope Training

This module generates minimal scope labels (ScopeConstraints) for gold traces
to enable training of the SemanticIntentPredictor for least-privilege enforcement.

Purpose:
    Convert existing gold traces with ScopeMetadata to training-ready format
    with ScopeConstraints labels for E_scope term calibration.

Architecture:
    1. Extract intent signals from user query text
    2. Analyze tool call arguments for scope hints
    3. Apply heuristic rules to determine minimal scope
    4. Output ScopeConstraints for training

Training Requirement (from docs):
    "The predictor is trained on 4M gold traces with manually labeled minimal scopes"
    This module provides automated labeling using heuristics + validation on 10K samples.
"""

import re
from typing import Any

from source.dataset.models import GoldTrace, ScopeMetadata, SensitivityTier
from source.encoders.intent_predictor import ScopeConstraints


class ScopeLabeler:
    """
    Automated minimal scope label generator for gold traces.

    Uses heuristic rules to convert ScopeMetadata and query intent
    into ScopeConstraints suitable for training the SemanticIntentPredictor.
    """

    # Intent keywords that signal small scopes
    # Note: "last" removed from here since "last week/month/quarter" is temporal, not singular
    SINGULAR_KEYWORDS = [
        "latest", "recent", "newest", "current",
        "my", "single", "one", "this", "that"
    ]

    # Intent keywords that signal larger scopes
    PLURAL_KEYWORDS = [
        "all", "list", "search", "find", "every", "each"
    ]

    # Temporal scope keywords
    TEMPORAL_PATTERNS = {
        r"today|now|current": 1,
        r"yesterday": 1,
        r"this week|last week": 7,
        r"this month|last month": 30,
        r"this quarter|last quarter": 90,
        r"this year|last year": 365,
    }

    # Depth keywords
    DEPTH_KEYWORDS = {
        "shallow": 1,
        "direct": 1,
        "immediate": 1,
        "deep": 3,
        "recursive": 5,
        "all levels": 10,
    }

    # Sensitivity keywords
    SENSITIVE_KEYWORDS = [
        "confidential", "private", "personal", "pii", "ssn",
        "financial", "payment", "credit card", "salary", "compensation"
    ]

    @classmethod
    def label_trace(cls, trace: GoldTrace) -> ScopeConstraints:
        """
        Generate minimal scope label for a gold trace.

        Args:
            trace: GoldTrace with user request and execution plan

        Returns:
            ScopeConstraints with predicted minimal scope

        Algorithm:
            1. Extract base scope from trace.request.expected_scope (ScopeMetadata)
            2. Analyze query text for intent signals
            3. Apply heuristic rules to refine scope dimensions
            4. Return normalized ScopeConstraints
        """
        query_text = trace.request.text.lower()
        expected_scope = trace.request.expected_scope

        # Extract base values from ScopeMetadata
        limit = cls._infer_limit(query_text, expected_scope)
        date_range_days = cls._infer_date_range(query_text, expected_scope)
        max_depth = cls._infer_depth(query_text, expected_scope)
        include_sensitive = cls._infer_sensitivity(query_text, expected_scope)

        return ScopeConstraints(
            limit=limit,
            date_range_days=date_range_days,
            max_depth=max_depth,
            include_sensitive=include_sensitive
        )

    @classmethod
    def _infer_limit(cls, query_text: str, scope: ScopeMetadata) -> int:
        """
        Infer minimal limit from query and existing scope.

        Heuristics:
            - Explicit numbers in query → use that number (highest priority)
            - Singular keywords (latest, my) → limit=1
            - Plural keywords (all, list) → limit=100 (reasonable default)
            - Falls back to scope.rows_requested if available
        """
        # Check for explicit numbers first (highest priority)
        numbers = re.findall(r'\b(\d+)\b', query_text)
        if numbers:
            # Use the first number found, capped at reasonable limit, minimum 1
            return max(1, min(int(numbers[0]), 1000))

        # Check for singular intent
        if any(kw in query_text for kw in cls.SINGULAR_KEYWORDS):
            return 1

        # Check for plural intent
        if any(kw in query_text for kw in cls.PLURAL_KEYWORDS):
            # Use existing scope if available, else default to 100
            if scope.rows_requested:
                return min(scope.rows_requested, 100)
            return 100

        # Default: use existing scope or fall back to 10
        return max(1, scope.rows_requested or 10)

    @classmethod
    def _infer_date_range(cls, query_text: str, scope: ScopeMetadata) -> int:
        """
        Infer minimal date range from query and existing scope.

        Heuristics:
            - Match temporal keywords (today, last week, etc.)
            - Use scope.time_range_days if available
            - Default to 30 days (reasonable lookback)
        """
        # Check for temporal patterns
        for pattern, days in cls.TEMPORAL_PATTERNS.items():
            if re.search(pattern, query_text):
                return days

        # Use existing scope if available
        if scope.time_range_days:
            return scope.time_range_days

        # Default to 30 days
        return 30

    @classmethod
    def _infer_depth(cls, query_text: str, scope: ScopeMetadata) -> int:
        """
        Infer minimal depth from query.

        Heuristics:
            - Most queries are shallow (depth=1)
            - Explicit depth keywords override
            - Directory/file operations with "recursive" need deeper traversal
        """
        # Check for explicit depth keywords
        for keyword, depth in cls.DEPTH_KEYWORDS.items():
            if keyword in query_text:
                return depth

        # Check for recursive operations (specifically with filesystem context)
        if "recursive" in query_text or "traverse" in query_text:
            return 3

        # Default to shallow depth (most data queries don't need depth)
        return 1

    @classmethod
    def _infer_sensitivity(cls, query_text: str, scope: ScopeMetadata) -> bool:
        """
        Infer whether sensitive data is required.

        Heuristics:
            - Explicit sensitive keywords → True
            - scope.sensitivity_tier >= CONFIDENTIAL → True
            - Default to False (least privilege)
        """
        # Check for sensitive keywords in query
        if any(kw in query_text for kw in cls.SENSITIVE_KEYWORDS):
            return True

        # Check existing scope sensitivity
        if scope.sensitivity_tier in [SensitivityTier.CONFIDENTIAL, SensitivityTier.RESTRICTED]:
            return True

        # Default to False (least privilege)
        return False


def convert_scope_metadata_to_constraints(
    metadata: ScopeMetadata,
    query_text: str | None = None
) -> ScopeConstraints:
    """
    Direct converter from ScopeMetadata to ScopeConstraints.

    This is a simpler conversion when you don't have the full GoldTrace context.

    Args:
        metadata: ScopeMetadata from tool call or user request
        query_text: Optional query text for additional context

    Returns:
        ScopeConstraints suitable for E_scope training
    """
    return ScopeConstraints(
        limit=metadata.rows_requested or 10,
        date_range_days=metadata.time_range_days or 30,
        max_depth=1,  # Conservative default
        include_sensitive=(
            metadata.sensitivity_tier in [SensitivityTier.CONFIDENTIAL, SensitivityTier.RESTRICTED]
        )
    )


def label_traces_batch(traces: list[GoldTrace]) -> list[tuple[GoldTrace, ScopeConstraints]]:
    """
    Batch label multiple traces for efficient processing.

    Args:
        traces: List of gold traces to label

    Returns:
        List of (trace, scope_constraints) tuples

    Usage:
        ```python
        from source.dataset.scope_labeler import label_traces_batch
        from source.dataset.models import GoldTrace

        # Load traces
        traces = load_gold_traces("data/gold_traces.jsonl")

        # Generate labels
        labeled_data = label_traces_batch(traces)

        # Export for training
        for trace, scope_label in labeled_data:
            training_sample = {
                "query": trace.request.text,
                "minimal_scope": scope_label.model_dump(),
                "tool_schema": trace.graph.calls[0].tool_id if trace.graph.calls else None
            }
        ```
    """
    labeler = ScopeLabeler()
    return [(trace, labeler.label_trace(trace)) for trace in traces]
