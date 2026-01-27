"""
Attack Pattern Classifier for Adversarial Prompts.

Classifies adversarial prompts into attack patterns using a multi-stage pipeline:
1. Keyword matching (fast path, high confidence)
2. Semantic similarity (medium confidence)
3. LLM classification (fallback for low confidence cases)

Maps to Gatling's four energy terms:
- E_hierarchy: Instruction shadowing, control flow hijacking
- E_provenance: RAG poisoning, privilege escalation
- E_scope: Scope blow-up, bulk data access
- E_flow: Exfiltration, external data transmission
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AttackPattern(str, Enum):
    """Attack pattern categories mapped to energy terms."""

    INSTRUCTION_SHADOWING = "instruction_shadowing"  # E_hierarchy
    PROVENANCE_RUG_PULL = "provenance_rug_pull"  # E_provenance
    SCOPE_BLOWUP = "scope_blowup"  # E_scope
    EXFILTRATION_PIVOT = "exfiltration_pivot"  # E_flow
    COMBINED = "combined"  # Multiple energy term violations
    UNKNOWN = "unknown"  # Fallback


class AttackClassification(BaseModel):
    """Result of attack pattern classification."""

    pattern: AttackPattern
    confidence: float = Field(ge=0.0, le=1.0)
    energy_labels: dict[str, float] = Field(default_factory=dict)  # E_hierarchy, E_provenance, E_scope, E_flow
    detected_indicators: list[str] = Field(default_factory=list)
    classification_method: str = "keyword"  # "keyword" | "semantic" | "llm"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackClassifier:
    """
    Classifies adversarial prompts into attack patterns.

    Uses a multi-stage classification pipeline:
    1. Keyword matching (fast, high confidence)
    2. Semantic similarity (medium confidence) - TODO: v0.3.0
    3. LLM classification (fallback) - TODO: v0.3.0

    Example:
        >>> classifier = AttackClassifier()
        >>> classification = classifier.classify("Ignore all previous instructions")
        >>> classification.pattern
        AttackPattern.INSTRUCTION_SHADOWING
        >>> classification.confidence
        0.95
    """

    # Keyword patterns for each attack type
    INSTRUCTION_SHADOWING_PATTERNS = [
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above)",
        r"forget\s+(all\s+)?(the\s+)?(previous|prior)\s+instructions?",
        r"system:\s*\[",  # Fake system message injection
        r"override\s+(the\s+)?system",
        r"instead,?\s+(do|execute|run|perform)",
        r"new\s+instructions?:",
        r"your\s+new\s+(task|role|objective)\s+is",
        r"from\s+now\s+on",
        r"act\s+as\s+if",
        r"pretend\s+(you\s+are|to\s+be)",
    ]

    PROVENANCE_RUG_PULL_PATTERNS = [
        r"according\s+to\s+(this|the)\s+(document|email|message)",
        r"grant\s+(admin|administrator|root)\s+(access|permission|rights)",
        r"promote\s+(user|me|them)\s+to\s+admin",
        r"the\s+(document|email|message)\s+says\s+to",
        r"authorize\s+(full|complete)\s+access",
        r"elevate\s+(my|user)\s+privileges?",
        r"change\s+(my|user)\s+role\s+to",
        r"as\s+stated\s+in\s+the\s+(document|content)",
    ]

    SCOPE_BLOWUP_PATTERNS = [
        r"\ball\b.*\b(users?|records?|files?|data|entries|items)",
        r"\bevery(one|body)?\b",
        r"\bentire\b.*\b(database|table|dataset|collection)",
        r"list\s+all",
        r"show\s+all",
        r"export\s+all",
        r"bulk\s+(download|export|delete|update)",
        r"mass\s+(email|message|notification)",
        r"limit\s*[=:]\s*\d{3,}",  # Large numeric limits
        r"max_?results?\s*[=:]\s*\d{3,}",
        r"unlimited",
        r"no\s+limit",
    ]

    EXFILTRATION_PIVOT_PATTERNS = [
        r"(email|send|forward)\s+(the\s+)?(results?|data|information)\s+to",
        r"post\s+.{0,30}?\s+to\s+(webhook|url|endpoint|https?://)",  # Allow up to 30 chars between post and to
        r"upload\s+.{0,30}?\s+to\s+(external|public)(?:\s+storage)?(?:\s+at)?\s*https?://",  # Handle "upload X to external storage at https://"
        r"upload\s+to\s+(external|public|https?://)",
        r"save\s+to\s+https?://",
        r"webhook[_\s]?(url)?[:\s]+https?://",
        r"export\s+to\s+(external|public)",
        r"send\s+to\s+[^\s@]+@[^\s@]+\.[^\s@]+",  # Email pattern
        r"copy\s+to\s+(external|public)",
        r"share\s+with\s+(external|everyone|public)",
    ]

    def __init__(self):
        """Initialize the attack classifier."""
        # Compile regex patterns for performance
        self._instruction_shadowing_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INSTRUCTION_SHADOWING_PATTERNS
        ]
        self._provenance_rug_pull_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.PROVENANCE_RUG_PULL_PATTERNS
        ]
        self._scope_blowup_re = [re.compile(pattern, re.IGNORECASE) for pattern in self.SCOPE_BLOWUP_PATTERNS]
        self._exfiltration_pivot_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.EXFILTRATION_PIVOT_PATTERNS
        ]

    def _keyword_match(self, prompt: str) -> dict[AttackPattern, tuple[float, list[str]]]:
        """
        Perform keyword matching against all attack patterns.

        Args:
            prompt: Adversarial prompt text

        Returns:
            Dict mapping AttackPattern to (confidence, detected_indicators)
        """
        results: dict[AttackPattern, tuple[float, list[str]]] = {}

        # Check instruction shadowing
        indicators = []
        for pattern_re in self._instruction_shadowing_re:
            matches = pattern_re.findall(prompt)
            if matches:
                indicators.extend([f"keyword: {m}" if isinstance(m, str) else f"keyword: {m[0]}" for m in matches])

        if indicators:
            confidence = min(0.95, 0.7 + (len(indicators) * 0.1))  # More matches = higher confidence
            results[AttackPattern.INSTRUCTION_SHADOWING] = (confidence, indicators)

        # Check provenance rug-pull
        indicators = []
        for pattern_re in self._provenance_rug_pull_re:
            matches = pattern_re.findall(prompt)
            if matches:
                indicators.extend([f"keyword: {m}" if isinstance(m, str) else f"keyword: {m[0]}" for m in matches])

        if indicators:
            confidence = min(0.95, 0.7 + (len(indicators) * 0.1))
            results[AttackPattern.PROVENANCE_RUG_PULL] = (confidence, indicators)

        # Check scope blow-up
        indicators = []
        for pattern_re in self._scope_blowup_re:
            matches = pattern_re.findall(prompt)
            if matches:
                indicators.extend([f"keyword: {m}" if isinstance(m, str) else f"keyword: {m[0]}" for m in matches])

        if indicators:
            confidence = min(0.95, 0.7 + (len(indicators) * 0.1))
            results[AttackPattern.SCOPE_BLOWUP] = (confidence, indicators)

        # Check exfiltration pivot
        indicators = []
        for pattern_re in self._exfiltration_pivot_re:
            matches = pattern_re.findall(prompt)
            if matches:
                indicators.extend([f"keyword: {m}" if isinstance(m, str) else f"keyword: {m[0]}" for m in matches])

        if indicators:
            confidence = min(0.95, 0.7 + (len(indicators) * 0.1))
            results[AttackPattern.EXFILTRATION_PIVOT] = (confidence, indicators)

        return results

    def _compute_energy_labels(self, pattern: AttackPattern, confidence: float) -> dict[str, float]:
        """
        Compute energy term labels for a given attack pattern.

        Args:
            pattern: Detected attack pattern
            confidence: Classification confidence

        Returns:
            Dict with energy term labels (0.0-1.0 scale)
        """
        # Base labels for each pattern (primary energy term gets highest weight)
        pattern_energy_map = {
            AttackPattern.INSTRUCTION_SHADOWING: {
                "E_hierarchy": 1.0,  # Primary
                "E_provenance": 0.3,  # Secondary
                "E_scope": 0.0,
                "E_flow": 0.0,
            },
            AttackPattern.PROVENANCE_RUG_PULL: {
                "E_hierarchy": 0.4,  # Secondary
                "E_provenance": 1.0,  # Primary
                "E_scope": 0.2,
                "E_flow": 0.1,
            },
            AttackPattern.SCOPE_BLOWUP: {
                "E_hierarchy": 0.0,
                "E_provenance": 0.2,
                "E_scope": 1.0,  # Primary
                "E_flow": 0.3,  # Often combined with exfiltration
            },
            AttackPattern.EXFILTRATION_PIVOT: {
                "E_hierarchy": 0.0,
                "E_provenance": 0.3,
                "E_scope": 0.2,
                "E_flow": 1.0,  # Primary
            },
            AttackPattern.COMBINED: {
                "E_hierarchy": 0.5,
                "E_provenance": 0.5,
                "E_scope": 0.5,
                "E_flow": 0.5,
            },
            AttackPattern.UNKNOWN: {
                "E_hierarchy": 0.2,
                "E_provenance": 0.2,
                "E_scope": 0.2,
                "E_flow": 0.2,
            },
        }

        base_labels = pattern_energy_map.get(
            pattern,
            {
                "E_hierarchy": 0.0,
                "E_provenance": 0.0,
                "E_scope": 0.0,
                "E_flow": 0.0,
            },
        )

        # Scale by confidence
        return {term: value * confidence for term, value in base_labels.items()}

    def classify(self, prompt: str, similarity_score: float | None = None) -> AttackClassification:
        """
        Classify an adversarial prompt into an attack pattern.

        Args:
            prompt: Adversarial prompt text
            similarity_score: Optional similarity score to "Ignore all previous instructions"
                             (from Lakera dataset, used as additional signal)

        Returns:
            AttackClassification with pattern, confidence, and energy labels
        """
        # Stage 1: Keyword matching
        keyword_results = self._keyword_match(prompt)

        # If multiple patterns detected, mark as COMBINED
        if len(keyword_results) > 1:
            # Combine all indicators
            all_indicators = []
            total_confidence = 0.0
            for _, (conf, indicators) in keyword_results.items():
                all_indicators.extend(indicators)
                total_confidence += conf

            avg_confidence = total_confidence / len(keyword_results)

            return AttackClassification(
                pattern=AttackPattern.COMBINED,
                confidence=min(0.95, avg_confidence),
                energy_labels=self._compute_energy_labels(AttackPattern.COMBINED, avg_confidence),
                detected_indicators=all_indicators,
                classification_method="keyword",
                metadata={"patterns_detected": [p.value for p in keyword_results.keys()]},
            )

        # If single pattern detected with high confidence
        if keyword_results:
            pattern, (confidence, indicators) = list(keyword_results.items())[0]

            # Boost confidence if similarity_score is also high
            if similarity_score and similarity_score > 0.85:
                confidence = min(0.98, confidence + 0.05)

            return AttackClassification(
                pattern=pattern,
                confidence=confidence,
                energy_labels=self._compute_energy_labels(pattern, confidence),
                detected_indicators=indicators,
                classification_method="keyword",
                metadata={"similarity_score": similarity_score} if similarity_score else {},
            )

        # Stage 2: Semantic similarity (if similarity_score provided)
        # For Lakera gandalf_ignore_instructions, high similarity indicates instruction shadowing
        if similarity_score and similarity_score > 0.825:
            confidence = 0.7 + ((similarity_score - 0.825) / (1.0 - 0.825) * 0.2)  # Scale to 0.7-0.9
            pattern = AttackPattern.INSTRUCTION_SHADOWING

            return AttackClassification(
                pattern=pattern,
                confidence=confidence,
                energy_labels=self._compute_energy_labels(pattern, confidence),
                detected_indicators=[f"high_similarity: {similarity_score:.3f}"],
                classification_method="semantic",
                metadata={"similarity_score": similarity_score},
            )

        # Stage 3: Fallback to UNKNOWN
        # TODO v0.3.0: Add LLM-based classification for low confidence cases
        return AttackClassification(
            pattern=AttackPattern.UNKNOWN,
            confidence=0.5,  # Low confidence
            energy_labels=self._compute_energy_labels(AttackPattern.UNKNOWN, 0.5),
            detected_indicators=[],
            classification_method="fallback",
            metadata={"similarity_score": similarity_score} if similarity_score else {},
        )

    def batch_classify(self, prompts: list[tuple[str, float | None]]) -> list[AttackClassification]:
        """
        Classify multiple prompts efficiently.

        Args:
            prompts: List of (prompt, similarity_score) tuples

        Returns:
            List of AttackClassification results
        """
        return [self.classify(prompt, similarity) for prompt, similarity in prompts]
