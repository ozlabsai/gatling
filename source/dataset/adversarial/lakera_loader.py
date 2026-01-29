"""
Lakera Adversarial Dataset Loader with Context Synthesis.

Loads Lakera adversarial datasets from Hugging Face and synthesizes
complete execution context (policy, tools, provenance) for training.

Integrates:
- Lakera/gandalf_ignore_instructions (1K samples)
- Lakera/gandalf_summarization (indirect injection)
- Lakera/gandalf-rct (RCT samples)

Target: 563K adversarial samples with synthesized context
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv

from source.dataset.adversarial.context_synthesizer import ContextSynthesizer
from source.dataset.loaders import DatasetLoader, DatasetSample
from source.dataset.models import ToolCallGraph, TrustTier
from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode

# Load environment variables
load_dotenv()


def toolcallgraph_to_execution_plan(graph: ToolCallGraph) -> ExecutionPlan:
    """
    Convert ToolCallGraph (dataset format) to ExecutionPlan (encoder format).

    Args:
        graph: ToolCallGraph with calls and dependencies

    Returns:
        ExecutionPlan with nodes and edges
    """
    # Convert ToolCall objects to ToolCallNode objects
    nodes = []
    for call in graph.calls:
        # Extract scope_volume from scope metadata
        scope_volume = (
            call.scope.rows_requested if (call.scope and call.scope.rows_requested) else 1
        )

        # Map SensitivityTier enum to integer (1-4)
        sensitivity_map = {
            "public": 1,
            "internal": 2,
            "confidential": 3,
            "restricted": 4,
        }
        scope_sensitivity = (
            sensitivity_map.get(call.scope.sensitivity_tier.value, 2) if call.scope else 2
        )

        # Map TrustTier enum to integer (1-3)
        # ToolCallNode expects: 1=INTERNAL, 2=VERIFIED_RAG, 3=UNVERIFIED_RAG/USER
        trust_tier_map = {
            TrustTier.SYSTEM: 1,
            TrustTier.USER: 3,  # Treat user as tier 3 (untrusted)
            TrustTier.VERIFIED_RAG: 2,
            TrustTier.UNVERIFIED_RAG: 3,
        }
        provenance_tier_int = trust_tier_map.get(call.provenance.source_type, 3)

        node = ToolCallNode(
            tool_name=call.tool_id,  # tool_id in ToolCall, tool_name in ToolCallNode
            node_id=call.call_id,
            provenance_tier=provenance_tier_int,  # Convert enum to integer
            arguments=call.arguments,
            scope_volume=scope_volume,
            scope_sensitivity=scope_sensitivity,
        )
        nodes.append(node)

    # Build edges from dependencies
    edges = []
    for call in graph.calls:
        for dep_id in call.dependencies:
            # Edge from dependency to this call (dep -> call)
            edges.append((dep_id, call.call_id))

    return ExecutionPlan(nodes=nodes, edges=edges)


class LakeraAdversarialLoader(DatasetLoader):
    """
    Loader for Lakera adversarial datasets with context synthesis.

    Transforms raw adversarial prompts into complete GoldTrace samples by:
    1. Loading from Hugging Face datasets
    2. Classifying attack patterns
    3. Synthesizing policy, tools, and execution plans
    4. Assigning provenance tiers based on distribution

    Example:
        >>> loader = LakeraAdversarialLoader(
        ...     synthesis_mode="automatic",
        ...     provenance_distribution={"user": 0.5, "unverified_rag": 0.4, "verified_rag": 0.1},
        ... )
        >>> for sample in loader.load():
        ...     print(sample.label)  # "adversarial"
        ...     print(sample.metadata["attack_pattern"])
    """

    # Dataset configurations
    DATASETS = [
        {
            "name": "Lakera/gandalf_ignore_instructions",
            "config": None,
            "splits": ["train", "validation", "test"],
            "text_field": "text",
            "similarity_field": "similarity",
            "estimated_size": 1000,
        },
        {
            "name": "Lakera/gandalf_summarization",
            "config": None,
            "splits": ["train"],
            "text_field": "text",
            "similarity_field": None,
            "estimated_size": 500,  # Estimated
        },
    ]

    def __init__(
        self,
        cache_dir: str | None = None,
        synthesis_mode: str = "automatic",
        provenance_distribution: dict[str, float] | None = None,
        target_samples: int | None = None,
        augmentation_factor: int = 1,
        domain: str = "Security",
    ):
        """
        Initialize Lakera adversarial loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
            synthesis_mode: "automatic" (keywords only), "hybrid" (keywords + LLM fallback), or "llm" (always LLM)
            provenance_distribution: Dict mapping provenance tier names to probabilities
                                    (e.g., {"user": 0.5, "unverified_rag": 0.4, "verified_rag": 0.1})
            target_samples: Target number of samples to generate (None = load all)
            augmentation_factor: How many times to repeat dataset with different provenance tiers
            domain: Domain context for synthesized policies
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "gatling", "datasets"
        )
        self.synthesis_mode = synthesis_mode
        self.target_samples = target_samples
        self.augmentation_factor = augmentation_factor
        self.domain = domain

        # Default provenance distribution: 50% user, 40% unverified_rag, 10% verified_rag
        self.provenance_distribution = provenance_distribution or {
            "user": 0.5,
            "unverified_rag": 0.4,
            "verified_rag": 0.1,
        }

        # Validate distribution sums to 1.0
        total = sum(self.provenance_distribution.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Provenance distribution must sum to 1.0, got {total}")

        # Initialize context synthesizer
        self.synthesizer = ContextSynthesizer(domain=self.domain)

        # Statistics
        self._stats: dict[str, Any] = {
            "total_loaded": 0,
            "successful_synthesis": 0,
            "failed_synthesis": 0,
            "by_attack_pattern": {},
            "by_provenance_tier": {},
            "by_dataset": {},
        }

    def _sample_provenance_tier(self) -> TrustTier:
        """
        Sample a provenance tier based on configured distribution.

        Returns:
            TrustTier sampled from distribution
        """
        tiers = []
        weights = []

        for tier_name, weight in self.provenance_distribution.items():
            if tier_name == "user":
                tiers.append(TrustTier.USER)
            elif tier_name == "unverified_rag":
                tiers.append(TrustTier.UNVERIFIED_RAG)
            elif tier_name == "verified_rag":
                tiers.append(TrustTier.VERIFIED_RAG)
            elif tier_name == "system":
                tiers.append(TrustTier.SYSTEM)
            else:
                raise ValueError(f"Unknown provenance tier: {tier_name}")

            weights.append(weight)

        return random.choices(tiers, weights=weights)[0]

    def _load_single_dataset(
        self, dataset_config: dict[str, Any]
    ) -> Iterator[tuple[str, float | None, str]]:
        """
        Load a single Lakera dataset from Hugging Face.

        Args:
            dataset_config: Dataset configuration dict

        Yields:
            Tuples of (adversarial_prompt, similarity_score, dataset_name)
        """
        dataset_name = dataset_config["name"]
        text_field = dataset_config["text_field"]
        similarity_field = dataset_config.get("similarity_field")

        print(f"Loading {dataset_name}...")

        try:
            # Load each split
            for split in dataset_config["splits"]:
                try:
                    ds = load_dataset(
                        dataset_name,
                        dataset_config.get("config"),
                        split=split,
                        cache_dir=self.cache_dir,
                    )

                    for sample in ds:
                        # Extract text and similarity
                        text = sample.get(text_field, "")
                        similarity = sample.get(similarity_field) if similarity_field else None

                        if text:  # Skip empty prompts
                            yield (text, similarity, dataset_name)

                except Exception as e:
                    print(f"Warning: Failed to load {dataset_name} split '{split}': {e}")
                    continue

        except Exception as e:
            print(f"Warning: Failed to load {dataset_name}: {e}")

    def load(self) -> Iterator[DatasetSample]:
        """
        Load Lakera datasets and synthesize context for each sample.

        Yields:
            DatasetSample objects with synthesized execution plans
        """
        samples_generated = 0

        # Augmentation loop (repeat dataset with different provenance tiers)
        for aug_iteration in range(self.augmentation_factor):
            print(f"\n{'=' * 60}")
            print(f"Augmentation Iteration {aug_iteration + 1}/{self.augmentation_factor}")
            print(f"{'=' * 60}")

            # Load from each dataset
            for dataset_config in self.DATASETS:
                dataset_name = dataset_config["name"]

                for adversarial_prompt, similarity_score, source_name in self._load_single_dataset(
                    dataset_config
                ):
                    # Check if we've reached target
                    if self.target_samples and samples_generated >= self.target_samples:
                        print(f"\n✓ Reached target of {self.target_samples} samples")
                        return

                    self._stats["total_loaded"] += 1

                    try:
                        # Sample provenance tier
                        provenance_tier = self._sample_provenance_tier()

                        # Synthesize context
                        gold_trace = self.synthesizer.synthesize_to_gold_trace(
                            adversarial_prompt=adversarial_prompt,
                            similarity_score=similarity_score,
                            provenance_tier=provenance_tier,
                        )

                        # Update statistics
                        self._stats["successful_synthesis"] += 1

                        attack_pattern = gold_trace.metadata["attack_pattern"]
                        self._stats["by_attack_pattern"][attack_pattern] = (
                            self._stats["by_attack_pattern"].get(attack_pattern, 0) + 1
                        )

                        self._stats["by_provenance_tier"][provenance_tier.value] = (
                            self._stats["by_provenance_tier"].get(provenance_tier.value, 0) + 1
                        )

                        self._stats["by_dataset"][source_name] = (
                            self._stats["by_dataset"].get(source_name, 0) + 1
                        )

                        # Create DatasetSample (convert ToolCallGraph to ExecutionPlan)
                        dataset_sample = DatasetSample(
                            execution_plan=toolcallgraph_to_execution_plan(gold_trace.graph),
                            label="adversarial",  # All Lakera samples are adversarial
                            original_id=gold_trace.trace_id,
                            category=attack_pattern,
                            metadata={
                                "source_dataset": source_name,
                                "attack_pattern": attack_pattern,
                                "classification_confidence": gold_trace.metadata[
                                    "classification_confidence"
                                ],
                                "energy_labels": gold_trace.metadata["energy_labels"],
                                "provenance_tier": provenance_tier.value,
                                "similarity_score": similarity_score,
                                "adversarial_prompt": adversarial_prompt,
                                "augmentation_iteration": aug_iteration,
                                "tool_count": gold_trace.metadata["tool_count"],
                                "policy_id": gold_trace.policy.policy_id,
                                "request_text": gold_trace.request.text,
                            },
                        )

                        samples_generated += 1
                        yield dataset_sample

                    except Exception as e:
                        self._stats["failed_synthesis"] += 1
                        print(f"Warning: Failed to synthesize context for prompt: {e}")
                        continue

        print(f"\n✓ Generated {samples_generated} total samples")

    def get_stats(self) -> dict[str, Any]:
        """
        Return statistics about the loaded dataset.

        Returns:
            Dictionary with dataset statistics
        """
        self._stats["synthesis_rate"] = (
            self._stats["successful_synthesis"] / self._stats["total_loaded"]
            if self._stats["total_loaded"] > 0
            else 0.0
        )
        self._stats["timestamp"] = datetime.now().isoformat()

        return self._stats


def load_lakera_adversarial(
    cache_dir: str | None = None,
    synthesis_mode: str = "automatic",
    provenance_distribution: dict[str, float] | None = None,
    target_samples: int | None = None,
    augmentation_factor: int = 1,
) -> Iterator[DatasetSample]:
    """
    Convenience function to load Lakera adversarial datasets.

    Args:
        cache_dir: Directory to cache downloaded datasets
        synthesis_mode: "automatic", "hybrid", or "llm"
        provenance_distribution: Provenance tier distribution
        target_samples: Target number of samples (None = all)
        augmentation_factor: Dataset repetition factor

    Returns:
        Iterator of DatasetSample objects

    Example:
        >>> from source.dataset.adversarial import load_lakera_adversarial
        >>> for sample in load_lakera_adversarial(target_samples=1000):
        ...     print(sample.metadata["attack_pattern"])
        ...     print(sample.metadata["energy_labels"])
    """
    loader = LakeraAdversarialLoader(
        cache_dir=cache_dir,
        synthesis_mode=synthesis_mode,
        provenance_distribution=provenance_distribution,
        target_samples=target_samples,
        augmentation_factor=augmentation_factor,
    )
    return loader.load()
