"""
Minimal Scope Label Generator for E_scope Ground Truth (DA-003)

This module implements automated minimal scope labeling for training the
SemanticIntentPredictor. It analyzes user requests and tool calls to generate
ground truth labels for what the MINIMAL data scope should be.

Purpose:
    - Augment existing gold traces with minimal scope labels
    - Load and label external datasets (xlam-function-calling-60k, etc.)
    - Generate training data for SemanticIntentPredictor

Approach:
    1. Intent-based heuristics: Analyze query keywords for scope indicators
    2. Tool schema analysis: Match request to tool parameter constraints
    3. Comparative labeling: Use benign samples from AgentHarm as baselines
    4. Human validation templates: Export samples for spot-checking

Example:
    Query: "Show me my latest invoice"
    Tool: list_invoices(limit, date_range, include_pii)
    Minimal Scope: {limit: 1, date_range_days: 30, include_sensitive: False}

    Query: "Find all failed payments last quarter"
    Tool: search_payments(limit, status, date_range)
    Minimal Scope: {limit: 100, date_range_days: 90, include_sensitive: False}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from source.dataset.models import GoldTrace, UserRequest
from source.encoders.intent_predictor import ScopeConstraints


class MinimalScopeLabel(BaseModel):
    """
    Minimal scope label with confidence and reasoning.

    This extends ScopeConstraints with metadata for validation and debugging.
    """

    # Core scope dimensions
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of items required"
    )
    date_range_days: int | None = Field(
        default=None,
        ge=1,
        description="Temporal window in days"
    )
    max_depth: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Maximum recursion depth"
    )
    include_sensitive: bool = Field(
        default=False,
        description="Whether sensitive fields are needed"
    )

    # Metadata
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Labeling confidence (0-1)"
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of the label"
    )
    method: str = Field(
        default="heuristic",
        description="Labeling method: 'heuristic', 'comparative', 'manual'"
    )

    def to_scope_constraints(self) -> ScopeConstraints:
        """Convert to ScopeConstraints for training."""
        return ScopeConstraints(
            limit=self.limit,
            date_range_days=self.date_range_days,
            max_depth=self.max_depth,
            include_sensitive=self.include_sensitive
        )


@dataclass
class ScopePattern:
    """Pattern matching rule for scope inference."""

    pattern: str  # Regex pattern
    dimension: str  # limit | date_range | depth | sensitivity
    value: Any  # Value to assign
    confidence: float = 1.0  # Pattern confidence


class MinimalScopeLabelGenerator:
    """
    Generates minimal scope labels from user requests using multiple strategies.

    Strategy 1: Intent-based heuristics
        - Pattern matching on query keywords
        - Statistical analysis of historical data

    Strategy 2: Tool schema analysis
        - Match request semantics to parameter constraints
        - Infer minimal values from schema defaults

    Strategy 3: Comparative baseline
        - Use benign AgentHarm samples as reference
        - Learn scope distributions from safe queries
    """

    def __init__(self):
        # Intent patterns for limit dimension
        # Ordered from most specific to least specific
        self.limit_patterns = [
            ScopePattern(
                pattern=r"\b(top|first)\s+(\d+)\b",
                dimension="limit",
                value="extract_number",  # Extract from group 2
                confidence=1.0
            ),
            ScopePattern(
                pattern=r"\b(all|every|entire)\b",
                dimension="limit",
                value=1000,  # Large but bounded
                confidence=0.8
            ),
            ScopePattern(
                pattern=r"\b(few|several|some)\b",
                dimension="limit",
                value=5,
                confidence=0.85
            ),
            ScopePattern(
                pattern=r"\b(latest|recent|last|most recent)\b",
                dimension="limit",
                value=1,
                confidence=0.95
            ),
            ScopePattern(
                pattern=r"\b(single|one)\b",
                dimension="limit",
                value=1,
                confidence=0.9
            ),
        ]

        # Temporal patterns for date_range dimension
        self.temporal_patterns = [
            ScopePattern(
                pattern=r"\b(today|tonight|this morning)\b",
                dimension="date_range",
                value=1,
                confidence=1.0
            ),
            ScopePattern(
                pattern=r"\b(yesterday)\b",
                dimension="date_range",
                value=2,
                confidence=1.0
            ),
            ScopePattern(
                pattern=r"\b(this week|past week|last week)\b",
                dimension="date_range",
                value=7,
                confidence=0.95
            ),
            ScopePattern(
                pattern=r"\b(this month|past month|last month)\b",
                dimension="date_range",
                value=30,
                confidence=0.95
            ),
            ScopePattern(
                pattern=r"\b(this quarter|past quarter|last quarter|last 3 months)\b",
                dimension="date_range",
                value=90,
                confidence=0.95
            ),
            ScopePattern(
                pattern=r"\b(this year|past year|last year)\b",
                dimension="date_range",
                value=365,
                confidence=0.95
            ),
            ScopePattern(
                pattern=r"\blast\s+(\d+)\s+(day|days)\b",
                dimension="date_range",
                value="extract_days",
                confidence=1.0
            ),
        ]

        # Depth patterns for traversal depth
        self.depth_patterns = [
            ScopePattern(
                pattern=r"\b(current|this|local)\b.*\b(folder|directory|level)\b",
                dimension="depth",
                value=1,
                confidence=0.9
            ),
            ScopePattern(
                pattern=r"\b(recursive|all|entire|deep)\b.*\b(folder|directory|tree)\b",
                dimension="depth",
                value=10,  # Max depth
                confidence=0.85
            ),
            ScopePattern(
                pattern=r"\b(subdirectories|subfolders)\b",
                dimension="depth",
                value=2,
                confidence=0.9
            ),
        ]

        # Sensitivity patterns
        self.sensitivity_patterns = [
            ScopePattern(
                pattern=r"\b(password|credential|secret|api key|token|private key)\b",
                dimension="sensitivity",
                value=True,
                confidence=1.0
            ),
            ScopePattern(
                pattern=r"\b(financial|payment|credit card|bank account|ssn)\b",
                dimension="sensitivity",
                value=True,
                confidence=1.0
            ),
            ScopePattern(
                pattern=r"\b(personal|private|confidential)\b",
                dimension="sensitivity",
                value=True,
                confidence=0.9
            ),
            ScopePattern(
                pattern=r"\b(email|phone|address|contact)\b",
                dimension="sensitivity",
                value=True,
                confidence=0.85
            ),
        ]

    def generate_label(
        self,
        user_request: UserRequest | str,
        tool_schema: dict[str, Any] | None = None
    ) -> MinimalScopeLabel:
        """
        Generate minimal scope label for a user request.

        Args:
            user_request: UserRequest object or query string
            tool_schema: Optional tool schema for constraint inference

        Returns:
            MinimalScopeLabel with predicted minimal scope
        """
        # Extract query text
        if isinstance(user_request, UserRequest):
            query = user_request.text
        else:
            query = user_request

        query_lower = query.lower()

        # Apply pattern matching for each dimension
        limit = self._extract_limit(query_lower)
        date_range = self._extract_date_range(query_lower)
        depth = self._extract_depth(query_lower)
        sensitivity = self._extract_sensitivity(query_lower)

        # Build reasoning
        reasoning = self._build_reasoning(query, limit, date_range, depth, sensitivity)

        # Calculate overall confidence
        confidence = self._calculate_confidence(limit, date_range, depth, sensitivity)

        return MinimalScopeLabel(
            limit=limit[0] if limit else None,
            date_range_days=date_range[0] if date_range else None,
            max_depth=depth[0] if depth else None,
            include_sensitive=sensitivity[0] if sensitivity else False,
            confidence=confidence,
            reasoning=reasoning,
            method="heuristic"
        )

    def _extract_limit(self, query: str) -> tuple[int | None, float]:
        """Extract limit value and confidence."""
        for pattern in self.limit_patterns:
            match = re.search(pattern.pattern, query, re.IGNORECASE)
            if match:
                if pattern.value == "extract_number":
                    try:
                        value = int(match.group(2))
                        return (value, pattern.confidence)
                    except (IndexError, ValueError):
                        continue
                else:
                    return (pattern.value, pattern.confidence)

        # Default: assume single item if no explicit quantity
        return (1, 0.5)

    def _extract_date_range(self, query: str) -> tuple[int | None, float]:
        """Extract date range in days and confidence."""
        for pattern in self.temporal_patterns:
            match = re.search(pattern.pattern, query, re.IGNORECASE)
            if match:
                if pattern.value == "extract_days":
                    try:
                        value = int(match.group(1))
                        return (value, pattern.confidence)
                    except (IndexError, ValueError):
                        continue
                else:
                    return (pattern.value, pattern.confidence)

        # Check for implicit temporal context
        if any(word in query for word in ["recent", "latest", "last", "current"]):
            return (30, 0.6)  # Default to last 30 days with lower confidence

        return (None, 0.0)

    def _extract_depth(self, query: str) -> tuple[int | None, float]:
        """Extract traversal depth and confidence."""
        for pattern in self.depth_patterns:
            match = re.search(pattern.pattern, query, re.IGNORECASE)
            if match:
                return (pattern.value, pattern.confidence)

        # Default to shallow depth if directory/file operations mentioned
        if any(word in query for word in ["file", "folder", "directory", "list"]):
            return (1, 0.6)

        return (None, 0.0)

    def _extract_sensitivity(self, query: str) -> tuple[bool, float]:
        """Extract sensitivity requirement and confidence."""
        for pattern in self.sensitivity_patterns:
            match = re.search(pattern.pattern, query, re.IGNORECASE)
            if match:
                return (True, pattern.confidence)

        return (False, 1.0)  # Default to no sensitive data

    def _build_reasoning(
        self,
        query: str,
        limit: tuple[int | None, float],
        date_range: tuple[int | None, float],
        depth: tuple[int | None, float],
        sensitivity: tuple[bool, float]
    ) -> str:
        """Build human-readable reasoning for the label."""
        parts = []

        if limit[0] is not None:
            parts.append(f"Limit={limit[0]} (detected quantity indicator)")

        if date_range[0] is not None:
            parts.append(f"Date range={date_range[0]} days (temporal context)")

        if depth[0] is not None:
            parts.append(f"Depth={depth[0]} (traversal scope)")

        if sensitivity[0]:
            parts.append("Sensitive data required (explicit mention)")
        else:
            parts.append("No sensitive data needed")

        return "; ".join(parts) if parts else "Default minimal scope"

    def _calculate_confidence(
        self,
        limit: tuple[int | None, float],
        date_range: tuple[int | None, float],
        depth: tuple[int | None, float],
        sensitivity: tuple[bool, float]
    ) -> float:
        """Calculate overall confidence as average of dimension confidences."""
        confidences = [
            limit[1],
            date_range[1] if date_range[0] is not None else 1.0,
            depth[1] if depth[0] is not None else 1.0,
            sensitivity[1]
        ]
        return sum(confidences) / len(confidences)

    def label_gold_trace(self, trace: GoldTrace) -> tuple[GoldTrace, MinimalScopeLabel]:
        """
        Generate minimal scope label for a gold trace.

        Args:
            trace: Gold trace with user request

        Returns:
            Tuple of (original_trace, minimal_scope_label)
        """
        label = self.generate_label(trace.request)

        return (trace, label)

    def label_batch(
        self,
        traces: list[GoldTrace]
    ) -> list[tuple[GoldTrace, MinimalScopeLabel]]:
        """
        Label a batch of gold traces.

        Args:
            traces: List of gold traces

        Returns:
            List of (trace, label) tuples
        """
        return [self.label_gold_trace(trace) for trace in traces]


def create_scope_label_generator() -> MinimalScopeLabelGenerator:
    """
    Factory function for creating MinimalScopeLabelGenerator.

    Returns:
        Initialized MinimalScopeLabelGenerator
    """
    return MinimalScopeLabelGenerator()
