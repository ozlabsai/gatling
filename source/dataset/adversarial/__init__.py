"""
Adversarial Dataset Integration with Context Synthesis.

This module provides loaders for adversarial security datasets (Lakera, etc.)
with automated context synthesis for missing policy, tool, and provenance metadata.
"""

from source.dataset.adversarial.attack_classifier import (
    AttackClassification,
    AttackClassifier,
    AttackPattern,
)
from source.dataset.adversarial.lakera_loader import LakeraAdversarialLoader

__all__ = [
    "AttackClassification",
    "AttackClassifier",
    "AttackPattern",
    "LakeraAdversarialLoader",
]
