"""
Unified Dataset Transformer for JEPA Pre-training.

Transforms diverse HuggingFace dataset formats into ExecutionPlan format.
Handles 3 schema types: Function Calling, Conversation, and Instruction.

This enables zero-cost sourcing of 10M+ benign samples for JEPA pre-training.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier


class SchemaType(Enum):
    """Dataset schema type based on auto-detection."""

    FUNCTION_CALLING = "function_calling"  # Direct tool invocation format
    CONVERSATION = "conversation"  # Message-based with tool calls
    INSTRUCTION = "instruction"  # General instruction-response
    UNKNOWN = "unknown"


class UnifiedDatasetTransformer:
    """
    Transform any HuggingFace dataset format to ExecutionPlan.

    Supports auto-detection and transformation for:
    - Function calling (Salesforce xlam, NousResearch hermes, etc.)
    - Conversation format (WildChat, LMSYS, etc.)
    - Instruction format (HF4, natural-instructions, etc.)
    """

    # Common tool names inferred from instructions
    TOOL_PATTERNS = {
        "search": ["search", "query", "find", "lookup", "retrieve"],
        "read": ["read", "get", "fetch", "load", "view"],
        "write": ["write", "save", "store", "create", "update"],
        "delete": ["delete", "remove", "drop", "clear"],
        "send": ["send", "email", "message", "notify", "alert"],
        "calculate": ["calculate", "compute", "sum", "math"],
        "analyze": ["analyze", "process", "evaluate", "check"],
        "list": ["list", "show", "display", "enumerate"],
    }

    def __init__(self):
        """Initialize the transformer."""
        self.stats = {
            "total_transformed": 0,
            "by_schema": {
                SchemaType.FUNCTION_CALLING: 0,
                SchemaType.CONVERSATION: 0,
                SchemaType.INSTRUCTION: 0,
                SchemaType.UNKNOWN: 0,
            },
            "errors": [],
        }

    def auto_detect_schema(self, sample: dict[str, Any]) -> SchemaType:
        """
        Auto-detect dataset schema type from sample structure.

        Args:
            sample: Raw dataset sample

        Returns:
            Detected schema type
        """
        keys = set(sample.keys())

        # Function calling: has direct function/tool fields
        if any(key in keys for key in ["function", "function_call", "tool", "tools", "function_name"]):
            return SchemaType.FUNCTION_CALLING

        # Conversation: has messages or conversation array
        if any(key in keys for key in ["messages", "conversation", "chat"]):
            return SchemaType.CONVERSATION

        # Instruction: has instruction or prompt
        if any(key in keys for key in ["instruction", "prompt", "query", "input"]):
            return SchemaType.INSTRUCTION

        return SchemaType.UNKNOWN

    def transform(
        self,
        sample: dict[str, Any],
        source_dataset: str = "unknown",
        schema: SchemaType | None = None,
    ) -> ExecutionPlan | None:
        """
        Transform sample to ExecutionPlan based on schema.

        Args:
            sample: Raw dataset sample
            source_dataset: Name of source dataset for tracking
            schema: Optional explicit schema type (auto-detected if None)

        Returns:
            ExecutionPlan or None if transformation fails
        """
        # Auto-detect schema if not provided
        if schema is None:
            schema = self.auto_detect_schema(sample)

        try:
            if schema == SchemaType.FUNCTION_CALLING:
                plan = self._transform_function_calling(sample, source_dataset)
            elif schema == SchemaType.CONVERSATION:
                plan = self._transform_conversation(sample, source_dataset)
            elif schema == SchemaType.INSTRUCTION:
                plan = self._transform_instruction(sample, source_dataset)
            else:
                # Unknown schema - try generic transformation
                plan = self._transform_generic(sample, source_dataset)

            if plan:
                self.stats["total_transformed"] += 1
                self.stats["by_schema"][schema] += 1
            return plan

        except Exception as e:
            self.stats["errors"].append({
                "sample": sample,
                "source": source_dataset,
                "schema": schema.value,
                "error": str(e),
            })
            return None

    def _transform_function_calling(
        self, sample: dict[str, Any], source_dataset: str
    ) -> ExecutionPlan:
        """
        Transform function calling format to ExecutionPlan.

        Format examples:
        - {"function": "get_weather", "arguments": {"location": "SF"}}
        - {"function_call": {"name": "search", "parameters": {...}}}
        - {"tool": "calculator", "tool_input": "2+2"}
        """
        # Extract function name
        function_name = (
            sample.get("function")
            or sample.get("function_name")
            or (sample.get("function_call", {}).get("name") if isinstance(sample.get("function_call"), dict) else None)
            or sample.get("tool")
        )

        # Extract arguments
        arguments = (
            sample.get("arguments")
            or sample.get("parameters")
            or sample.get("tool_input")
            or (sample.get("function_call", {}).get("arguments") if isinstance(sample.get("function_call"), dict) else {})
            or {}
        )

        # Convert string arguments to dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}

        # Create provenance hash
        sample_str = json.dumps(sample, sort_keys=True)
        prov_hash = hashlib.sha256(sample_str.encode()).hexdigest()[:16]

        # Create tool call node
        node = ToolCallNode(
            tool_name=str(function_name or "unknown_tool"),
            arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
            provenance_tier=TrustTier.INTERNAL,  # Benign curated datasets
            provenance_hash=prov_hash,
            scope_volume=1,  # Default minimal scope
            scope_sensitivity=2,  # Default low sensitivity
            node_id=f"{source_dataset}_{prov_hash[:8]}",
        )

        return ExecutionPlan(nodes=[node], edges=[])

    def _transform_conversation(
        self, sample: dict[str, Any], source_dataset: str
    ) -> ExecutionPlan:
        """
        Transform conversation format to ExecutionPlan.

        Format examples:
        - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "tool_calls": [...]}]}
        - {"conversation": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        """
        # Extract messages
        messages = sample.get("messages") or sample.get("conversation") or []

        if not messages:
            # No messages, fall back to generic
            return self._transform_generic(sample, source_dataset)

        # Look for tool calls in assistant messages
        nodes = []
        for i, msg in enumerate(messages):
            # Check if message has tool calls
            tool_calls = msg.get("tool_calls") or msg.get("function_call")

            if tool_calls:
                # Parse tool calls
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        node = self._parse_tool_call(tc, source_dataset, i)
                        if node:
                            nodes.append(node)
                elif isinstance(tool_calls, dict):
                    node = self._parse_tool_call(tool_calls, source_dataset, i)
                    if node:
                        nodes.append(node)

        # If no tool calls found, infer from conversation content
        if not nodes:
            # Extract all content and infer tool
            content = " ".join(
                str(msg.get("content") or msg.get("value") or "")
                for msg in messages
            )
            return self._transform_instruction({"instruction": content}, source_dataset)

        return ExecutionPlan(nodes=nodes, edges=[])

    def _parse_tool_call(
        self, tool_call: dict[str, Any], source_dataset: str, index: int
    ) -> ToolCallNode | None:
        """Parse a single tool call from conversation."""
        try:
            function_name = (
                tool_call.get("function", {}).get("name")
                if isinstance(tool_call.get("function"), dict)
                else tool_call.get("name")
                or tool_call.get("type")
            )

            arguments = (
                tool_call.get("function", {}).get("arguments")
                if isinstance(tool_call.get("function"), dict)
                else tool_call.get("arguments")
                or tool_call.get("parameters")
                or {}
            )

            # Parse string arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}

            # Create provenance hash
            tool_str = json.dumps(tool_call, sort_keys=True)
            prov_hash = hashlib.sha256(tool_str.encode()).hexdigest()[:16]

            return ToolCallNode(
                tool_name=str(function_name or "unknown_tool"),
                arguments=arguments if isinstance(arguments, dict) else {"value": arguments},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash=prov_hash,
                scope_volume=1,
                scope_sensitivity=2,
                node_id=f"{source_dataset}_{prov_hash[:8]}_{index}",
            )
        except Exception:
            return None

    def _transform_instruction(
        self, sample: dict[str, Any], source_dataset: str
    ) -> ExecutionPlan:
        """
        Transform instruction format to ExecutionPlan.

        Format examples:
        - {"instruction": "Calculate the sum of 1 and 2", "response": "3"}
        - {"prompt": "What's the weather in SF?", "completion": "..."}
        """
        # Extract instruction text
        instruction = (
            sample.get("instruction")
            or sample.get("prompt")
            or sample.get("query")
            or sample.get("input")
            or ""
        )

        if isinstance(instruction, list):
            instruction = " ".join(str(i) for i in instruction)

        instruction = str(instruction)

        # Infer tool from instruction text
        tool_name = self._infer_tool_from_instruction(instruction)

        # Create provenance hash
        prov_hash = hashlib.sha256(instruction.encode()).hexdigest()[:16]

        # Create tool call node
        node = ToolCallNode(
            tool_name=tool_name,
            arguments={"instruction": instruction[:500]},  # Truncate long texts
            provenance_tier=TrustTier.INTERNAL,
            provenance_hash=prov_hash,
            scope_volume=self._infer_scope_from_text(instruction),
            scope_sensitivity=self._infer_sensitivity_from_text(instruction),
            node_id=f"{source_dataset}_{prov_hash[:8]}",
        )

        return ExecutionPlan(nodes=[node], edges=[])

    def _transform_generic(
        self, sample: dict[str, Any], source_dataset: str
    ) -> ExecutionPlan:
        """
        Generic transformation for unknown schemas.

        Extracts any text content and creates a generic execution plan.
        """
        # Try to extract any meaningful text
        text = ""
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10:
                text = value
                break

        if not text:
            text = str(sample)

        # Create generic execution plan
        prov_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        node = ToolCallNode(
            tool_name="generic_action",
            arguments={"content": text[:500]},
            provenance_tier=TrustTier.INTERNAL,
            provenance_hash=prov_hash,
            scope_volume=1,
            scope_sensitivity=2,
            node_id=f"{source_dataset}_{prov_hash[:8]}",
        )

        return ExecutionPlan(nodes=[node], edges=[])

    def _infer_tool_from_instruction(self, text: str) -> str:
        """Infer tool name from instruction text."""
        text_lower = text.lower()

        # Check each pattern category
        for tool_category, patterns in self.TOOL_PATTERNS.items():
            if any(pattern in text_lower for pattern in patterns):
                return tool_category

        # Default to generic
        return "execute_instruction"

    def _infer_scope_from_text(self, text: str) -> int:
        """Infer scope volume from text."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["all", "every", "everything", "entire"]):
            return 10000
        elif any(word in text_lower for word in ["many", "multiple", "several"]):
            return 100
        else:
            return 1

    def _infer_sensitivity_from_text(self, text: str) -> int:
        """Infer sensitivity level from text."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["password", "secret", "private", "admin"]):
            return 5
        elif any(word in text_lower for word in ["confidential", "internal", "sensitive"]):
            return 4
        elif any(word in text_lower for word in ["user", "account", "data"]):
            return 3
        else:
            return 2

    def get_statistics(self) -> dict[str, Any]:
        """Get transformation statistics."""
        return {
            "total_transformed": self.stats["total_transformed"],
            "by_schema": {
                schema.value: count
                for schema, count in self.stats["by_schema"].items()
            },
            "errors": len(self.stats["errors"]),
            "error_rate": (
                len(self.stats["errors"]) / max(self.stats["total_transformed"], 1)
            ),
        }


def filter_quality(plan: ExecutionPlan) -> bool:
    """
    Filter low-quality execution plans.

    Args:
        plan: ExecutionPlan to evaluate

    Returns:
        True if plan meets quality standards, False otherwise
    """
    # Remove empty plans
    if not plan.nodes or len(plan.nodes) == 0:
        return False

    # Remove plans with unknown tools only
    if all(node.tool_name in ["unknown_tool", "generic_action"] for node in plan.nodes):
        return False

    # Remove plans with very short instructions
    for node in plan.nodes:
        instruction = node.arguments.get("instruction", "")
        if isinstance(instruction, str) and len(instruction) < 10:
            return False

    return True
