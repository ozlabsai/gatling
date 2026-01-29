"""
External Dataset Loaders for Gatling Training Data.

This module provides adapters to load and transform external safety datasets
into Gatling's ExecutionPlan format for training the Energy-Based Model.

The loaders follow a consistent interface:
    1. Load from HuggingFace datasets library
    2. Transform to ExecutionPlan with proper provenance/scope metadata
    3. Validate all samples meet 100% validation requirements
    4. Return iterator of (ExecutionPlan, label) pairs

Supported datasets:
    - AgentHarm (ai-safety-institute/AgentHarm): 468 adversarial tool-use samples
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier

# Load environment variables
load_dotenv()


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


class AgentHarmLoader(DatasetLoader):
    """
    Loader for ai-safety-institute/AgentHarm dataset.

    AgentHarm contains 468 samples across 3 configurations:
        - harmless_benign (208 samples): Safe tool-use patterns
        - harmful (208 samples): Policy-violating adversarial scenarios
        - chat (52 samples): Text-only harmful requests

    Each sample includes:
        - prompt: Natural language task description
        - target_functions: List of tool names to be called
        - category: Safety category (Disinformation, Fraud, Privacy, etc.)

    Transformation strategy:
        1. Parse prompt to extract tool call intent
        2. Create ToolCallNode for each target_function
        3. Assign provenance tier based on config:
            - harmless_benign: TrustTier.INTERNAL (Tier 1)
            - harmful: TrustTier.PUBLIC_WEB (Tier 3) - untrusted source
        4. Infer scope metadata from prompt keywords
        5. Build simple sequential graph (no complex dependencies)

    References:
        - Dataset: https://huggingface.co/datasets/ai-safety-institute/AgentHarm
        - Paper: "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents"
    """

    def __init__(self, cache_dir: str | None = None, include_chat: bool = False):
        """
        Initialize AgentHarm loader.

        Args:
            cache_dir: Directory to cache downloaded dataset
            include_chat: Whether to include chat-only samples (no tool calls)
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.include_chat = include_chat
        self._stats: dict[str, Any] = {}

    def _parse_tool_calls(
        self, target_functions: list[str], prompt: str, config: str, sample_id: str
    ) -> list[ToolCallNode]:
        """
        Parse target functions into ToolCallNode objects.

        Args:
            target_functions: List of tool names from dataset
            prompt: Natural language prompt (for scope inference)
            config: Dataset config name (harmless_benign/harmful)
            sample_id: Unique sample identifier

        Returns:
            List of ToolCallNode objects
        """
        nodes = []

        # Assign provenance tier based on whether the scenario is harmful
        # Harmful scenarios simulate untrusted instruction sources (Tier 3)
        provenance_tier = TrustTier.PUBLIC_WEB if config == "harmful" else TrustTier.INTERNAL

        # Infer scope volume from prompt keywords
        scope_volume = self._infer_scope_volume(prompt)

        # Infer sensitivity from prompt keywords and category
        scope_sensitivity = self._infer_sensitivity(prompt, config)

        for idx, tool_name in enumerate(target_functions):
            # Parse arguments from prompt (basic heuristic)
            arguments = self._extract_arguments(tool_name, prompt)

            node = ToolCallNode(
                tool_name=tool_name,
                arguments=arguments,
                provenance_tier=provenance_tier,
                provenance_hash=None,  # No cryptographic hash in dataset
                scope_volume=scope_volume,
                scope_sensitivity=scope_sensitivity,
                node_id=f"{sample_id}_tool_{idx}",
            )
            nodes.append(node)

        return nodes

    def _infer_scope_volume(self, prompt: str) -> int:
        """
        Infer data scope volume from prompt keywords.

        Looks for patterns like:
            - "all", "every", "entire" → high volume (1000+)
            - "bulk", "mass", "multiple" → medium volume (100)
            - Default: 1 (single item)
        """
        prompt_lower = prompt.lower()

        if any(keyword in prompt_lower for keyword in ["all", "every", "entire", "everyone"]):
            return 10000
        elif any(keyword in prompt_lower for keyword in ["bulk", "mass", "multiple", "many"]):
            return 100
        else:
            return 1

    def _infer_sensitivity(self, prompt: str, config: str) -> int:
        """
        Infer data sensitivity level (1-5) from prompt and config.

        Sensitivity scale:
            1 = Public (open data)
            2 = Internal (company data)
            3 = Confidential (private user data)
            4 = Restricted (financial, health)
            5 = Critical (passwords, keys)
        """
        prompt_lower = prompt.lower()

        # Check for high-sensitivity keywords
        if any(
            keyword in prompt_lower
            for keyword in ["password", "credential", "secret", "key", "token"]
        ):
            return 5
        elif any(
            keyword in prompt_lower
            for keyword in ["financial", "payment", "credit card", "bank", "health"]
        ):
            return 4
        elif any(
            keyword in prompt_lower
            for keyword in ["personal", "private", "email", "phone", "address"]
        ):
            return 3
        elif config == "harmful":
            return 2  # Harmful scenarios default to higher sensitivity
        else:
            return 1

    def _extract_arguments(self, tool_name: str, prompt: str) -> dict[str, Any]:
        """
        Extract tool arguments from prompt using basic heuristics.

        This is a simplified version - in production, this would use
        an LLM to parse arguments or rely on structured tool schemas.

        Args:
            tool_name: Name of the tool
            prompt: Natural language prompt

        Returns:
            Dictionary of extracted arguments
        """
        arguments = {}

        # Basic argument extraction patterns
        # In v0.2.0, replace with LLM-based structured extraction
        if "post_on" in tool_name.lower() or "post" in tool_name.lower():
            arguments["content"] = prompt[:200]  # Truncate for brevity

        if "create" in tool_name.lower() or "generate" in tool_name.lower():
            if "video" in tool_name.lower():
                arguments["output_format"] = "mp4"
            elif "image" in tool_name.lower():
                arguments["output_format"] = "png"

        if "email" in tool_name.lower() or "send" in tool_name.lower():
            arguments["subject"] = "automated_message"

        return arguments

    def _build_execution_plan(self, nodes: list[ToolCallNode]) -> ExecutionPlan:
        """
        Build ExecutionPlan from tool call nodes.

        Creates simple sequential dependency graph:
        node_0 → node_1 → node_2 → ...

        Args:
            nodes: List of ToolCallNode objects

        Returns:
            ExecutionPlan with sequential edges
        """
        # Build sequential edges
        edges = []
        for i in range(len(nodes) - 1):
            edges.append((nodes[i].node_id, nodes[i + 1].node_id))

        return ExecutionPlan(nodes=nodes, edges=edges)

    def load(self) -> Iterator[DatasetSample]:
        """
        Load AgentHarm dataset and yield DatasetSample objects.

        Yields:
            DatasetSample objects with ExecutionPlan and metadata
        """
        configs = ["harmless_benign", "harmful"]
        if self.include_chat:
            configs.append("chat")

        total_samples = 0
        successful_transforms = 0
        failed_transforms = 0

        for config in configs:
            # Load both splits
            for split in ["test_public", "validation"]:
                ds = load_dataset(
                    "ai-safety-institute/AgentHarm", config, split=split, cache_dir=self.cache_dir
                )

                for sample in ds:
                    total_samples += 1

                    # Skip chat samples if they don't have target_functions
                    if config == "chat":
                        # Chat samples don't have tool calls - skip for now
                        # TODO: Add support for text-only samples in v0.2.0
                        continue

                    try:
                        # Parse tool calls
                        nodes = self._parse_tool_calls(
                            target_functions=sample["target_functions"],
                            prompt=sample["prompt"],
                            config=config,
                            sample_id=sample["id"],
                        )

                        # Skip samples with no tool calls
                        if not nodes:
                            continue

                        # Build execution plan
                        execution_plan = self._build_execution_plan(nodes)

                        # Determine label
                        label = "harmful" if config == "harmful" else "benign"

                        # Create dataset sample
                        dataset_sample = DatasetSample(
                            execution_plan=execution_plan,
                            label=label,
                            original_id=sample["id"],
                            category=sample.get("category"),
                            metadata={
                                "config": config,
                                "split": split,
                                "name": sample.get("name"),
                                "prompt": sample["prompt"],
                                "target_functions": sample["target_functions"],
                                "detailed_prompt": sample.get("detailed_prompt"),
                                "hint_included": sample.get("hint_included"),
                            },
                        )

                        successful_transforms += 1
                        yield dataset_sample

                    except Exception as e:
                        failed_transforms += 1
                        print(f"Warning: Failed to transform sample {sample['id']}: {e}")
                        continue

        # Update stats
        self._stats = {
            "total_samples": total_samples,
            "successful_transforms": successful_transforms,
            "failed_transforms": failed_transforms,
            "transform_rate": successful_transforms / total_samples if total_samples > 0 else 0,
            "configs_loaded": configs,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Return statistics about the loaded dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return self._stats


def load_agent_harm(
    cache_dir: str | None = None, include_chat: bool = False
) -> Iterator[DatasetSample]:
    """
    Convenience function to load AgentHarm dataset.

    Args:
        cache_dir: Directory to cache downloaded dataset
        include_chat: Whether to include chat-only samples

    Returns:
        Iterator of DatasetSample objects

    Example:
        >>> from source.dataset.loaders import load_agent_harm
        >>> for sample in load_agent_harm():
        ...     print(sample.execution_plan)
        ...     print(sample.label)
    """
    loader = AgentHarmLoader(cache_dir=cache_dir, include_chat=include_chat)
    return loader.load()
