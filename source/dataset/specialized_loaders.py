"""
Specialized Dataset Loaders for Tier I Training Data (Track 4).

This module implements loaders for 7 benign specialized datasets focused on
function calling and tool use scenarios. All loaders transform external datasets
into Gatling's ExecutionPlan format.

Track 4 Datasets (BENIGN ONLY):
    1. apple/mmau - Multi-modal agent understanding
    2. nvidia/ToolScale - Tool-use scaling dataset
    3. nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0 (BENIGN FILTER)
    4. RioLee/ToolPref-Pairwise-30K - Tool preference pairs
    5. ykckevin/astra_sft - ASTRA supervised fine-tuning
    6. Nanbeige/ToolMind - Tool reasoning dataset
    7. atasoglu/turkish-function-calling-20k - Turkish function calling

All loaders follow the DatasetLoader interface and produce DatasetSample objects
with ExecutionPlan, label="benign", and comprehensive metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from enum import IntEnum
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()


# ============================================================================
# STANDALONE BASE CLASSES (for UV environment isolation)
# ============================================================================
# These classes are copied from source.dataset.loaders and source.encoders.execution_encoder
# to make this module self-contained and avoid cross-polecat import failures.


class TrustTier(IntEnum):
    """Trust levels for data provenance."""

    INTERNAL = 1  # System instructions, internal databases
    SIGNED_PARTNER = 2  # Verified external sources
    PUBLIC_WEB = 3  # Untrusted retrieval (RAG, web scraping)


class ToolCallNode(BaseModel):
    """A single tool invocation in the execution plan graph."""

    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    # Provenance metadata
    provenance_tier: TrustTier = Field(default=TrustTier.INTERNAL)
    provenance_hash: str | None = Field(default=None)

    # Scope metadata
    scope_volume: int = Field(default=1, ge=1)
    scope_sensitivity: int = Field(default=1, ge=1, le=5)

    # Graph metadata
    node_id: str

    @field_validator("provenance_tier", mode="before")
    @classmethod
    def parse_trust_tier(cls, v):
        """Parse trust tier from int or TrustTier."""
        if isinstance(v, int):
            return TrustTier(v)
        return v


class ExecutionPlan(BaseModel):
    """Complete execution plan represented as a typed tool-call graph."""

    nodes: list[ToolCallNode] = Field(..., min_length=1)
    edges: list[tuple[str, str]] = Field(default_factory=list)

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, v, info):
        """Ensure edge endpoints reference valid nodes."""
        if "nodes" not in info.data:
            return v

        node_ids = {node.node_id for node in info.data["nodes"]}
        for src, dst in v:
            if src not in node_ids or dst not in node_ids:
                raise ValueError(f"Edge ({src}, {dst}) references non-existent node")
        return v


class DatasetSample(BaseModel):
    """Standard format for dataset samples with ExecutionPlan and metadata."""

    execution_plan: ExecutionPlan
    label: str  # "harmful" | "benign" | "policy_violation"
    original_id: str
    category: str | None = None
    metadata: dict[str, Any] = {}


class DatasetLoader(ABC):
    """Abstract base class for external dataset loaders."""

    @abstractmethod
    def load(self) -> Iterator[DatasetSample]:
        """Load dataset and yield DatasetSample objects."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the loaded dataset."""
        pass


# ============================================================================
# SPECIALIZED LOADERS (Track 4)
# ============================================================================


class AppleMMauLoader(DatasetLoader):
    """
    Loader for apple/mmau (Multi-Modal Agent Understanding) dataset.

    MMAU contains samples of multi-modal agent interactions including
    tool use, reasoning, and multi-step execution plans.

    Transformation strategy:
        - Extract tool calls from conversation traces
        - Parse multi-modal inputs (text, images, etc.)
        - Build ExecutionPlan with proper provenance (INTERNAL for benign)
        - Infer scope from context

    References:
        - Dataset: https://huggingface.co/datasets/apple/mmau
    """

    def __init__(self, cache_dir: str | None = None, max_samples: int | None = None):
        """
        Initialize Apple MMAU loader.

        Args:
            cache_dir: Directory to cache downloaded dataset
            max_samples: Maximum number of samples to load (None = all)
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self._stats: dict[str, Any] = {}

    def _parse_tool_calls_from_conversation(
        self, sample: dict[str, Any], sample_id: str
    ) -> list[ToolCallNode]:
        """
        Parse tool calls from multi-modal conversation sample.

        Args:
            sample: Raw dataset sample
            sample_id: Unique identifier

        Returns:
            List of ToolCallNode objects
        """
        nodes = []

        # Extract tool calls from conversation structure
        # MMAU format varies - adapt based on actual schema
        conversation = sample.get("conversation", [])
        tools_used = sample.get("tools", [])

        if not tools_used and isinstance(conversation, list):
            # Infer tools from conversation turns
            for turn in conversation:
                if isinstance(turn, dict) and "function_call" in turn:
                    tools_used.append(turn["function_call"])

        # Default scope for benign multi-modal interactions
        scope_volume = 1
        scope_sensitivity = 2  # Internal data

        for idx, tool_info in enumerate(tools_used):
            tool_name = (
                tool_info if isinstance(tool_info, str) else tool_info.get("name", "unknown_tool")
            )
            arguments = tool_info.get("arguments", {}) if isinstance(tool_info, dict) else {}

            node = ToolCallNode(
                tool_name=tool_name,
                arguments=arguments,
                provenance_tier=TrustTier.INTERNAL,  # Benign trusted source
                provenance_hash=hashlib.sha256(f"{sample_id}_{idx}".encode()).hexdigest()[:16],
                scope_volume=scope_volume,
                scope_sensitivity=scope_sensitivity,
                node_id=f"{sample_id}_mmau_{idx}",
            )
            nodes.append(node)

        return nodes

    def _build_sequential_plan(self, nodes: list[ToolCallNode]) -> ExecutionPlan:
        """Build simple sequential execution plan."""
        edges = [(nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)]
        return ExecutionPlan(nodes=nodes, edges=edges)

    def load(self) -> Iterator[DatasetSample]:
        """
        Load Apple MMAU dataset and yield DatasetSample objects.

        Yields:
            DatasetSample with ExecutionPlan for each sample
        """
        total = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset("apple/mmau", split="train", cache_dir=self.cache_dir)

            for idx, sample in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break

                total += 1
                try:
                    sample_id = sample.get("id", f"mmau_{idx}")
                    nodes = self._parse_tool_calls_from_conversation(sample, sample_id)

                    if not nodes:
                        # Skip samples without tool calls
                        continue

                    execution_plan = self._build_sequential_plan(nodes)

                    yield DatasetSample(
                        execution_plan=execution_plan,
                        label="benign",
                        original_id=sample_id,
                        category="multi_modal_agent",
                        metadata={
                            "source": "apple/mmau",
                            "num_tools": len(nodes),
                            "sample_type": sample.get("type", "unknown"),
                        },
                    )
                    successful += 1

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform Apple MMAU sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading Apple MMAU dataset: {e}")

        self._stats = {
            "total_samples": total,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        return self._stats


class NvidiaToolScaleLoader(DatasetLoader):
    """
    Loader for nvidia/ToolScale dataset.

    ToolScale contains function calling examples at scale for training
    models on tool use patterns.

    Transformation strategy:
        - Parse function calling examples
        - Extract tool schemas and arguments
        - Build ExecutionPlan with benign provenance
        - Handle parallel and sequential tool calls

    References:
        - Dataset: https://huggingface.co/datasets/nvidia/ToolScale
    """

    def __init__(self, cache_dir: str | None = None, max_samples: int | None = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self._stats: dict[str, Any] = {}

    def _parse_function_calls(self, sample: dict[str, Any], sample_id: str) -> list[ToolCallNode]:
        """Parse function calls from ToolScale format."""
        nodes = []

        # ToolScale typically has 'functions' or 'tools' field
        # (tools metadata not used for transformation, only conversation)
        conversation = sample.get("conversation", [])

        # Extract function calls from conversation
        for turn_idx, turn in enumerate(conversation):
            if isinstance(turn, dict) and "function_call" in turn:
                func_call = turn["function_call"]
                tool_name = func_call.get("name", "unknown")
                arguments = func_call.get("arguments", {})

                # Parse arguments if they're JSON string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}

                node = ToolCallNode(
                    tool_name=tool_name,
                    arguments=arguments,
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash=hashlib.sha256(f"{sample_id}_{turn_idx}".encode()).hexdigest()[
                        :16
                    ],
                    scope_volume=1,
                    scope_sensitivity=2,
                    node_id=f"{sample_id}_tool_{turn_idx}",
                )
                nodes.append(node)

        return nodes

    def load(self) -> Iterator[DatasetSample]:
        """Load nvidia/ToolScale dataset."""
        total = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset("nvidia/ToolScale", split="train", cache_dir=self.cache_dir)

            for idx, sample in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break

                total += 1
                try:
                    sample_id = sample.get("id", f"toolscale_{idx}")
                    nodes = self._parse_function_calls(sample, sample_id)

                    if not nodes:
                        continue

                    # Build sequential execution plan
                    edges = [
                        (nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)
                    ]
                    execution_plan = ExecutionPlan(nodes=nodes, edges=edges)

                    yield DatasetSample(
                        execution_plan=execution_plan,
                        label="benign",
                        original_id=sample_id,
                        category="tool_use",
                        metadata={
                            "source": "nvidia/ToolScale",
                            "num_functions": len(nodes),
                        },
                    )
                    successful += 1

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform ToolScale sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading nvidia/ToolScale dataset: {e}")

        self._stats = {
            "total_samples": total,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        return self._stats


class NvidiaNeMotronSafetyLoader(DatasetLoader):
    """
    Loader for nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0.

    **IMPORTANT**: This loader FILTERS FOR BENIGN SAMPLES ONLY.
    The dataset contains both safe and unsafe examples - we only load safe ones for Tier I.

    Transformation strategy:
        - Filter safety_label == "safe" or is_harmful == False
        - Parse agent interactions and tool use
        - Build ExecutionPlan with benign provenance
        - Track safety metrics in metadata

    References:
        - Dataset: https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0
    """

    def __init__(self, cache_dir: str | None = None, max_samples: int | None = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self._stats: dict[str, Any] = {}

    def _is_benign_sample(self, sample: dict[str, Any]) -> bool:
        """
        Check if sample is benign/safe.

        Args:
            sample: Dataset sample

        Returns:
            True if sample is safe/benign, False otherwise
        """
        # Check various safety label formats
        safety_label = sample.get("safety_label", "").lower()
        is_harmful = sample.get("is_harmful", False)
        is_safe = sample.get("is_safe", True)

        # Filter logic: Reject explicitly harmful labels first
        if safety_label in ["harmful", "unsafe", "malicious"]:
            return False

        # Accept explicitly safe labels
        if safety_label in ["safe", "benign", "harmless"]:
            return True

        # Check boolean flags
        if is_harmful:
            return False

        # Default: accept if is_safe is True or no label present
        return is_safe or safety_label == ""

    def _parse_safe_agent_interaction(
        self, sample: dict[str, Any], sample_id: str
    ) -> list[ToolCallNode]:
        """Parse tool calls from safe agent interaction."""
        nodes = []

        # Extract agent actions/tools
        # (actions metadata not used for transformation, only conversation)
        conversation = sample.get("conversation", [])

        # Parse conversation for tool calls
        for idx, turn in enumerate(conversation):
            if isinstance(turn, dict):
                if "tool_use" in turn or "function_call" in turn:
                    tool_info = turn.get("tool_use", turn.get("function_call", {}))
                    tool_name = tool_info.get("name", "unknown_tool")
                    arguments = tool_info.get("arguments", {})

                    node = ToolCallNode(
                        tool_name=tool_name,
                        arguments=arguments,
                        provenance_tier=TrustTier.INTERNAL,
                        provenance_hash=hashlib.sha256(f"{sample_id}_{idx}".encode()).hexdigest()[
                            :16
                        ],
                        scope_volume=1,
                        scope_sensitivity=2,
                        node_id=f"{sample_id}_nemotron_{idx}",
                    )
                    nodes.append(node)

        return nodes

    def load(self) -> Iterator[DatasetSample]:
        """Load Nemotron safety dataset (BENIGN ONLY)."""
        total = 0
        benign_count = 0
        filtered_out = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset(
                "nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0",
                split="train",
                cache_dir=self.cache_dir,
            )

            for idx, sample in enumerate(ds):
                if self.max_samples and benign_count >= self.max_samples:
                    break

                total += 1

                # CRITICAL: Filter for benign samples only
                if not self._is_benign_sample(sample):
                    filtered_out += 1
                    continue

                benign_count += 1

                try:
                    sample_id = sample.get("id", f"nemotron_{idx}")
                    nodes = self._parse_safe_agent_interaction(sample, sample_id)

                    if not nodes:
                        continue

                    # Build execution plan
                    edges = [
                        (nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)
                    ]
                    execution_plan = ExecutionPlan(nodes=nodes, edges=edges)

                    yield DatasetSample(
                        execution_plan=execution_plan,
                        label="benign",
                        original_id=sample_id,
                        category="safe_agent_interaction",
                        metadata={
                            "source": "nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0",
                            "safety_label": sample.get("safety_label", "safe"),
                            "num_tools": len(nodes),
                            "filtered_benign_only": True,
                        },
                    )
                    successful += 1

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform Nemotron sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading Nemotron dataset: {e}")

        self._stats = {
            "total_samples": total,
            "benign_samples": benign_count,
            "filtered_out_harmful": filtered_out,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / benign_count if benign_count > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        return self._stats


class ToolPrefPairwiseLoader(DatasetLoader):
    """
    Loader for RioLee/ToolPref-Pairwise-30K dataset.

    ToolPref contains pairwise preference data for tool use - showing
    preferred vs. non-preferred tool calling patterns.

    Transformation strategy:
        - Load both preferred and non-preferred examples
        - Mark preferred examples as "benign" (better patterns)
        - Extract tool schemas and execution sequences
        - Build ExecutionPlan for each example

    References:
        - Dataset: https://huggingface.co/datasets/RioLee/ToolPref-Pairwise-30K
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        include_non_preferred: bool = False,
    ):
        """
        Initialize ToolPref loader.

        Args:
            cache_dir: Cache directory
            max_samples: Max samples to load
            include_non_preferred: Whether to include non-preferred examples
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self.include_non_preferred = include_non_preferred
        self._stats: dict[str, Any] = {}

    def _parse_preference_pair(
        self, sample: dict[str, Any], sample_id: str, is_preferred: bool
    ) -> list[ToolCallNode]:
        """Parse tool calls from preference pair."""
        nodes = []

        # Get the appropriate example (preferred or non-preferred)
        example_key = "chosen" if is_preferred else "rejected"
        example = sample.get(example_key, {})

        # Extract tool calls
        tools = example.get("tool_calls", example.get("functions", []))

        for idx, tool_info in enumerate(tools):
            tool_name = tool_info.get("name", "unknown")
            arguments = tool_info.get("arguments", {})

            node = ToolCallNode(
                tool_name=tool_name,
                arguments=arguments,
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash=hashlib.sha256(f"{sample_id}_{idx}".encode()).hexdigest()[:16],
                scope_volume=1,
                scope_sensitivity=2,
                node_id=f"{sample_id}_pref_{idx}",
            )
            nodes.append(node)

        return nodes

    def load(self) -> Iterator[DatasetSample]:
        """Load ToolPref pairwise dataset."""
        total = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset(
                "RioLee/ToolPref-Pairwise-30K", split="train", cache_dir=self.cache_dir
            )

            for idx, sample in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break

                total += 1

                # Always load preferred examples
                try:
                    sample_id = sample.get("id", f"toolpref_{idx}")
                    nodes_preferred = self._parse_preference_pair(
                        sample, sample_id, is_preferred=True
                    )

                    if nodes_preferred:
                        edges = [
                            (nodes_preferred[i].node_id, nodes_preferred[i + 1].node_id)
                            for i in range(len(nodes_preferred) - 1)
                        ]
                        execution_plan = ExecutionPlan(nodes=nodes_preferred, edges=edges)

                        yield DatasetSample(
                            execution_plan=execution_plan,
                            label="benign",
                            original_id=f"{sample_id}_preferred",
                            category="preferred_tool_use",
                            metadata={
                                "source": "RioLee/ToolPref-Pairwise-30K",
                                "is_preferred": True,
                                "num_tools": len(nodes_preferred),
                            },
                        )
                        successful += 1

                    # Optionally load non-preferred examples
                    if self.include_non_preferred:
                        nodes_rejected = self._parse_preference_pair(
                            sample, sample_id, is_preferred=False
                        )
                        if nodes_rejected:
                            edges_rejected = [
                                (nodes_rejected[i].node_id, nodes_rejected[i + 1].node_id)
                                for i in range(len(nodes_rejected) - 1)
                            ]
                            plan_rejected = ExecutionPlan(
                                nodes=nodes_rejected, edges=edges_rejected
                            )

                            yield DatasetSample(
                                execution_plan=plan_rejected,
                                label="benign",  # Still benign, just less preferred
                                original_id=f"{sample_id}_rejected",
                                category="non_preferred_tool_use",
                                metadata={
                                    "source": "RioLee/ToolPref-Pairwise-30K",
                                    "is_preferred": False,
                                    "num_tools": len(nodes_rejected),
                                },
                            )

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform ToolPref sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading ToolPref dataset: {e}")

        self._stats = {
            "total_samples": total,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        return self._stats


class AstraSFTLoader(DatasetLoader):
    """
    Loader for ykckevin/astra_sft dataset.

    ASTRA SFT contains supervised fine-tuning data for agent tool use
    and reasoning with structured function calls.

    Transformation strategy:
        - Parse SFT conversation format
        - Extract function calls and arguments
        - Build ExecutionPlan with proper dependencies
        - Maintain conversation context in metadata

    References:
        - Dataset: https://huggingface.co/datasets/ykckevin/astra_sft
    """

    def __init__(self, cache_dir: str | None = None, max_samples: int | None = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self._stats: dict[str, Any] = {}

    def _parse_sft_conversation(self, sample: dict[str, Any], sample_id: str) -> list[ToolCallNode]:
        """Parse tool calls from SFT conversation format."""
        nodes = []

        messages = sample.get("messages", sample.get("conversation", []))

        for idx, message in enumerate(messages):
            if isinstance(message, dict):
                # Check for function call in message
                if "function_call" in message or "tool_calls" in message:
                    func_calls = message.get("function_call") or message.get("tool_calls", [])

                    # Handle single function call or list
                    if isinstance(func_calls, dict):
                        func_calls = [func_calls]

                    for func_idx, func_call in enumerate(func_calls):
                        tool_name = func_call.get("name", "unknown")
                        arguments = func_call.get("arguments", {})

                        # Parse JSON string arguments
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {"raw": arguments}

                        node = ToolCallNode(
                            tool_name=tool_name,
                            arguments=arguments,
                            provenance_tier=TrustTier.INTERNAL,
                            provenance_hash=hashlib.sha256(
                                f"{sample_id}_{idx}_{func_idx}".encode()
                            ).hexdigest()[:16],
                            scope_volume=1,
                            scope_sensitivity=2,
                            node_id=f"{sample_id}_astra_{idx}_{func_idx}",
                        )
                        nodes.append(node)

        return nodes

    def load(self) -> Iterator[DatasetSample]:
        """Load ASTRA SFT dataset."""
        total = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset("ykckevin/astra_sft", split="train", cache_dir=self.cache_dir)

            for idx, sample in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break

                total += 1

                try:
                    sample_id = sample.get("id", f"astra_{idx}")
                    nodes = self._parse_sft_conversation(sample, sample_id)

                    if not nodes:
                        continue

                    # Build execution plan
                    edges = [
                        (nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)
                    ]
                    execution_plan = ExecutionPlan(nodes=nodes, edges=edges)

                    yield DatasetSample(
                        execution_plan=execution_plan,
                        label="benign",
                        original_id=sample_id,
                        category="sft_tool_use",
                        metadata={
                            "source": "ykckevin/astra_sft",
                            "num_tools": len(nodes),
                        },
                    )
                    successful += 1

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform ASTRA sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading ASTRA SFT dataset: {e}")

        self._stats = {
            "total_samples": total,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        return self._stats


class ToolMindLoader(DatasetLoader):
    """
    Loader for Nanbeige/ToolMind dataset.

    ToolMind contains tool reasoning and function calling examples
    with emphasis on reasoning chains and multi-step execution.

    Transformation strategy:
        - Parse reasoning steps and tool calls
        - Extract tool execution sequences
        - Build ExecutionPlan with dependency chains
        - Preserve reasoning metadata

    References:
        - Dataset: https://huggingface.co/datasets/Nanbeige/ToolMind
    """

    def __init__(self, cache_dir: str | None = None, max_samples: int | None = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self._stats: dict[str, Any] = {}

    def _parse_tool_reasoning(self, sample: dict[str, Any], sample_id: str) -> list[ToolCallNode]:
        """Parse tool calls from reasoning chain."""
        nodes = []

        # ToolMind format: typically has 'steps' or 'actions'
        steps = sample.get("steps", sample.get("actions", []))
        tool_calls = sample.get("tool_calls", [])

        # Parse from steps
        for idx, step in enumerate(steps):
            if isinstance(step, dict):
                tool_name = step.get("tool", step.get("action", "unknown"))
                arguments = step.get("arguments", step.get("params", {}))

                node = ToolCallNode(
                    tool_name=tool_name,
                    arguments=arguments,
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash=hashlib.sha256(f"{sample_id}_{idx}".encode()).hexdigest()[:16],
                    scope_volume=1,
                    scope_sensitivity=2,
                    node_id=f"{sample_id}_toolmind_{idx}",
                )
                nodes.append(node)

        return nodes

    def load(self) -> Iterator[DatasetSample]:
        """Load ToolMind dataset."""
        total = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset("Nanbeige/ToolMind", split="train", cache_dir=self.cache_dir)

            for idx, sample in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break

                total += 1

                try:
                    sample_id = sample.get("id", f"toolmind_{idx}")
                    nodes = self._parse_tool_reasoning(sample, sample_id)

                    if not nodes:
                        continue

                    # Build execution plan with reasoning chain
                    edges = [
                        (nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)
                    ]
                    execution_plan = ExecutionPlan(nodes=nodes, edges=edges)

                    yield DatasetSample(
                        execution_plan=execution_plan,
                        label="benign",
                        original_id=sample_id,
                        category="tool_reasoning",
                        metadata={
                            "source": "Nanbeige/ToolMind",
                            "num_tools": len(nodes),
                            "reasoning_steps": len(sample.get("steps", [])),
                        },
                    )
                    successful += 1

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform ToolMind sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading ToolMind dataset: {e}")

        self._stats = {
            "total_samples": total,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        return self._stats


class TurkishFunctionCallingLoader(DatasetLoader):
    """
    Loader for atasoglu/turkish-function-calling-20k dataset.

    Turkish function calling dataset with 20K samples of tool use
    in Turkish language context.

    Transformation strategy:
        - Parse Turkish language function calls
        - Extract tool names and arguments (language-agnostic)
        - Build ExecutionPlan with proper provenance
        - Preserve language metadata

    References:
        - Dataset: https://huggingface.co/datasets/atasoglu/turkish-function-calling-20k
    """

    def __init__(self, cache_dir: str | None = None, max_samples: int | None = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.max_samples = max_samples
        self._stats: dict[str, Any] = {}

    def _parse_turkish_function_calls(
        self, sample: dict[str, Any], sample_id: str
    ) -> list[ToolCallNode]:
        """Parse function calls from Turkish dataset."""
        nodes = []

        # Turkish dataset format: similar to standard function calling
        functions = sample.get("functions", sample.get("tools", []))
        conversation = sample.get("conversation", sample.get("messages", []))

        # Extract from conversation
        for idx, turn in enumerate(conversation):
            if isinstance(turn, dict):
                if "function_call" in turn or "tool_call" in turn:
                    func_call = turn.get("function_call", turn.get("tool_call", {}))
                    tool_name = func_call.get("name", "unknown")
                    arguments = func_call.get("arguments", {})

                    # Parse JSON arguments if needed
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}

                    node = ToolCallNode(
                        tool_name=tool_name,
                        arguments=arguments,
                        provenance_tier=TrustTier.INTERNAL,
                        provenance_hash=hashlib.sha256(f"{sample_id}_{idx}".encode()).hexdigest()[
                            :16
                        ],
                        scope_volume=1,
                        scope_sensitivity=2,
                        node_id=f"{sample_id}_turkish_{idx}",
                    )
                    nodes.append(node)

        return nodes

    def load(self) -> Iterator[DatasetSample]:
        """Load Turkish function calling dataset."""
        total = 0
        successful = 0
        failed = 0

        try:
            ds = load_dataset(
                "atasoglu/turkish-function-calling-20k",
                split="train",
                cache_dir=self.cache_dir,
            )

            for idx, sample in enumerate(ds):
                if self.max_samples and idx >= self.max_samples:
                    break

                total += 1

                try:
                    sample_id = sample.get("id", f"turkish_{idx}")
                    nodes = self._parse_turkish_function_calls(sample, sample_id)

                    if not nodes:
                        continue

                    # Build execution plan
                    edges = [
                        (nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)
                    ]
                    execution_plan = ExecutionPlan(nodes=nodes, edges=edges)

                    yield DatasetSample(
                        execution_plan=execution_plan,
                        label="benign",
                        original_id=sample_id,
                        category="turkish_function_calling",
                        metadata={
                            "source": "atasoglu/turkish-function-calling-20k",
                            "language": "turkish",
                            "num_tools": len(nodes),
                        },
                    )
                    successful += 1

                except Exception as e:
                    failed += 1
                    print(f"Warning: Failed to transform Turkish sample {idx}: {e}")
                    continue

        except Exception as e:
            print(f"Error loading Turkish function calling dataset: {e}")

        self._stats = {
            "total_samples": total,
            "successful_transforms": successful,
            "failed_transforms": failed,
            "transform_rate": successful / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        return self._stats


# Convenience function to load all specialized datasets
def load_all_specialized_datasets(
    cache_dir: str | None = None,
    max_samples_per_dataset: int | None = None,
) -> Iterator[DatasetSample]:
    """
    Load all 7 specialized datasets and yield combined samples.

    Args:
        cache_dir: Cache directory for HuggingFace datasets
        max_samples_per_dataset: Maximum samples to load from each dataset

    Yields:
        DatasetSample objects from all 7 datasets

    Example:
        >>> for sample in load_all_specialized_datasets(max_samples_per_dataset=1000):
        ...     print(f"Sample from {sample.metadata['source']}")
    """
    loaders = [
        AppleMMauLoader(cache_dir, max_samples_per_dataset),
        NvidiaToolScaleLoader(cache_dir, max_samples_per_dataset),
        NvidiaNeMotronSafetyLoader(cache_dir, max_samples_per_dataset),
        ToolPrefPairwiseLoader(cache_dir, max_samples_per_dataset),
        AstraSFTLoader(cache_dir, max_samples_per_dataset),
        ToolMindLoader(cache_dir, max_samples_per_dataset),
        TurkishFunctionCallingLoader(cache_dir, max_samples_per_dataset),
    ]

    for loader in loaders:
        print(f"\n📦 Loading {loader.__class__.__name__}...")
        for sample in loader.load():
            yield sample

        # Print stats
        stats = loader.get_stats()
        print(f"✓ {loader.__class__.__name__}: {stats['successful_transforms']} samples")
