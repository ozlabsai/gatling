"""
Oracle Agent for gold trace generation.

This module implements the high-quality AI agent that generates
policy-compliant tool-use traces for Stage A of the SID pipeline.
"""

from source.dataset.oracle.agent import OracleAgent
from source.dataset.oracle.prompts import OraclePromptBuilder

__all__ = ["OracleAgent", "OraclePromptBuilder"]
