"""
Gatling Encoders Module

This module implements the JEPA (Joint-Embedding Predictive Architecture) encoders
for Project Gatling's energy-based security system.
"""

from .governance_encoder import GovernanceEncoder

__all__ = ["GovernanceEncoder"]
