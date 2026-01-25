"""
Dataset generation module for Gatling.

This module implements Stage A: Seed Trajectory Generation from the
Synthetic Integrity Dataset (SID) workstream. It generates 4M benign
tool-use traces using Oracle Agents across 50+ domains.
"""

from source.dataset.models import (
    ToolCall,
    ToolCallGraph,
    UserRequest,
    GoldTrace,
    SystemPolicy,
    ToolSchema,
)

__all__ = [
    "ToolCall",
    "ToolCallGraph",
    "UserRequest",
    "GoldTrace",
    "SystemPolicy",
    "ToolSchema",
]
