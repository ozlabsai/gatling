"""
Multi-Agent System for Project Gatling

This package contains the autonomous agent infrastructure for parallel
development across research workstreams.
"""

from .base_agent import GatlingAgent, TaskStatus
from .research_planner import ResearchPlannerAgent

__all__ = ["GatlingAgent", "TaskStatus", "ResearchPlannerAgent"]
