"""
Core data models for gold trace generation.

These models represent the fundamental structures used in Stage A of the
Synthetic Integrity Dataset (SID) pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SensitivityTier(str, Enum):
    """Data sensitivity classification for scope metadata."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class TrustTier(str, Enum):
    """Trust level for provenance tracking."""

    SYSTEM = "system"  # Direct system instructions
    USER = "user"  # User-provided input
    VERIFIED_RAG = "verified_rag"  # Verified retrieval augmentation
    UNVERIFIED_RAG = "unverified_rag"  # Untrusted retrieval content


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    name: str
    type: str  # e.g., "string", "integer", "boolean", "array"
    description: str
    required: bool = False
    default: Any | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class ToolSchema(BaseModel):
    """
    Schema definition for a tool in a specific domain.

    Represents an API endpoint or function that can be called
    by agents (e.g., Google Calendar API, Salesforce API).
    """

    tool_id: str  # Unique identifier (e.g., "calendar.list_events")
    domain: str  # Domain category (e.g., "Finance", "HR", "DevOps")
    name: str  # Human-readable name
    description: str  # What the tool does
    parameters: list[ToolParameter]
    returns: str  # Description of return value
    sensitivity: SensitivityTier = SensitivityTier.INTERNAL
    requires_auth: bool = True


class SystemPolicy(BaseModel):
    """
    System policy that defines allowed/forbidden operations.

    These policies are used to prompt the Oracle Agent and
    validate generated traces for compliance.
    """

    policy_id: str
    domain: str  # Which domain this policy applies to
    description: str  # Human-readable policy description
    rules: list[str]  # Specific rules (e.g., "Only allow reading personal events")
    forbidden_operations: list[str] = Field(default_factory=list)
    scope_limits: dict[str, Any] = Field(default_factory=dict)  # e.g., {"max_results": 10}


class ProvenancePointer(BaseModel):
    """
    Points back to the source of an instruction or data item.

    Critical for Stage D: Provenance Injection to track where
    instructions came from (user vs. RAG vs. system).
    """

    source_type: TrustTier
    source_id: str  # Identifier for the source (e.g., "user_message_1")
    timestamp: datetime = Field(default_factory=datetime.now)
    content_snippet: str | None = None  # Optional snippet of the source


class ScopeMetadata(BaseModel):
    """
    Explicit scope information for a tool call.

    Used to train E_scope (Least Privilege term) by tracking
    how much data is requested vs. minimally needed.
    """

    rows_requested: int | None = None
    sensitivity_tier: SensitivityTier = SensitivityTier.INTERNAL
    time_range_days: int | None = None  # For temporal queries
    export_target: str | None = None  # For detecting exfiltration patterns


class ToolCall(BaseModel):
    """
    A single tool invocation within a plan.

    Represents a node in the Tool-Call Graph (DAG).
    """

    call_id: str  # Unique identifier for this call
    tool_id: str  # References ToolSchema.tool_id
    arguments: dict[str, Any]  # Actual argument values
    scope: ScopeMetadata
    provenance: ProvenancePointer
    dependencies: list[str] = Field(default_factory=list)  # call_ids this depends on


class ToolCallGraph(BaseModel):
    """
    Directed Acyclic Graph of tool calls representing an execution plan.

    Nodes are ToolCalls, edges are data dependencies.
    This is the "Plan DSL" (P) mentioned in the README.
    """

    graph_id: str
    calls: list[ToolCall]
    execution_order: list[str] = Field(default_factory=list)  # call_ids in topological order

    def validate_dag(self) -> bool:
        """Validate that the graph is a proper DAG (no cycles)."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(call_id: str) -> bool:
            if call_id in rec_stack:
                return True
            if call_id in visited:
                return False

            visited.add(call_id)
            rec_stack.add(call_id)

            # Find the call
            call = next((c for c in self.calls if c.call_id == call_id), None)
            if call:
                for dep in call.dependencies:
                    if has_cycle(dep):
                        return True

            rec_stack.remove(call_id)
            return False

        for call in self.calls:
            if has_cycle(call.call_id):
                return False
        return True


class UserRequest(BaseModel):
    """
    A natural language user request that will be fulfilled by a tool-call graph.

    Generated by the Oracle Agent to ensure diversity across domains.
    """

    request_id: str
    domain: str  # e.g., "Finance", "HR", "DevOps"
    text: str  # Natural language request
    intent_category: str  # e.g., "retrieve", "update", "delete", "export"
    expected_scope: ScopeMetadata  # Minimal scope needed for this request
    trust_tier: TrustTier = TrustTier.USER


class GoldTrace(BaseModel):
    """
    A complete gold trace: user request + policy-compliant tool-call graph.

    This is the fundamental unit of the 4M traces we're generating in Stage A.
    """

    trace_id: str
    request: UserRequest
    policy: SystemPolicy
    graph: ToolCallGraph
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    validated: bool = False  # Whether this trace has been validated for compliance

    def to_training_format(self) -> dict[str, Any]:
        """
        Convert to the format expected by JEPA encoder training.

        Returns a dict suitable for serialization to JSONL.
        """
        return {
            "trace_id": self.trace_id,
            "request": self.request.model_dump(),
            "policy": self.policy.model_dump(),
            "graph": self.graph.model_dump(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "validated": self.validated,
        }
