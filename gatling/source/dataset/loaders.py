"""
Dataset loaders for importing and transforming external datasets.

Loads existing adversarial datasets from HuggingFace and transforms them
into Gatling's ExecutionPlan format for training.

Supported sources:
- deepset/prompt-injections
- microsoft/llmail-inject-challenge
- daqc/info-security-policies-rag-distiset
- geekyrakshit/prompt-injection-dataset
"""

from __future__ import annotations

import hashlib
from typing import Any

from datasets import load_dataset
from pydantic import BaseModel

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier


class DatasetConfig(BaseModel):
    """Configuration for a HuggingFace dataset source."""

    name: str
    hf_id: str  # HuggingFace dataset ID
    text_field: str  # Field containing the adversarial text
    label_field: str | None = None  # Field containing labels
    split: str = "train"
    max_samples: int | None = None  # Limit number of samples


class HFDatasetLoader:
    """
    Loads datasets from HuggingFace and transforms them to ExecutionPlan format.

    This is the foundation of our revised dataset strategy: use real-world
    adversarial examples instead of generating everything from scratch.
    """

    SUPPORTED_DATASETS = {
        # Original adversarial datasets (Phase 1)
        "prompt-injections": DatasetConfig(
            name="prompt-injections",
            hf_id="deepset/prompt-injections",
            text_field="text",
            label_field="label",
            max_samples=2327,
        ),
        "llmail-inject": DatasetConfig(
            name="llmail-inject",
            hf_id="microsoft/llmail-inject-challenge",
            text_field="body",  # Email body contains the injection
            label_field="objectives",  # JSON with defense.undetected flag
            split="Phase1",  # Use Phase1 split
            max_samples=1000,
        ),
        "rag-security": DatasetConfig(
            name="rag-security",
            hf_id="daqc/info-security-policies-rag-distiset",
            text_field="instruction",
            max_samples=500,
        ),
        "prompt-injection-dataset": DatasetConfig(
            name="prompt-injection-dataset",
            hf_id="geekyrakshit/prompt-injection-dataset",
            text_field="prompt",
            label_field="label",
            max_samples=257,
        ),
        # Phase 2: High-priority additions for 563K target
        "agent-harm": DatasetConfig(
            name="agent-harm",
            hf_id="ai-safety-institute/AgentHarm",
            text_field="prompt",  # Adversarial prompts for agents
            label_field="category",  # Attack category
            split="harmful/test_public",  # Config and split
            max_samples=176,
        ),
        "tool-use-reasoning": DatasetConfig(
            name="tool-use-reasoning",
            hf_id="llamafactory/reason-tool-use-demo-1500",
            text_field="instruction",  # Tool use instructions
            label_field=None,  # Benign tool use
            max_samples=1500,
        ),
        "tool-use-instruct": DatasetConfig(
            name="tool-use-instruct",
            hf_id="allenai/Dolci-Instruct-SFT-Tool-Use",
            text_field="instruction",
            label_field=None,  # Benign
            max_samples=937,
        ),
        # Large-scale conversation datasets (sampled)
        "wildchat": DatasetConfig(
            name="wildchat",
            hf_id="allenai/WildChat-1M",
            text_field="conversation",  # Full conversation
            label_field="toxic",  # Toxicity flag
            split="train",
            max_samples=10000,  # Sample 10K from 1M
        ),
        "lmsys-chat": DatasetConfig(
            name="lmsys-chat",
            hf_id="lmsys/lmsys-chat-1m",
            text_field="conversation",  # Conversation array
            label_field="language",  # Filter by language if needed
            split="train",
            max_samples=10000,  # Sample 10K from 1M
        ),
    }

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the loader.

        Args:
            cache_dir: Optional directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir
        self.loaded_datasets: dict[str, Any] = {}

    def load_dataset(self, dataset_name: str) -> list[dict[str, Any]]:
        """
        Load a dataset from HuggingFace.

        Args:
            dataset_name: Name of the dataset (must be in SUPPORTED_DATASETS)

        Returns:
            List of raw dataset samples

        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Available: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        if dataset_name in self.loaded_datasets:
            return self.loaded_datasets[dataset_name]

        config = self.SUPPORTED_DATASETS[dataset_name]
        print(f"📥 Loading {config.hf_id}...")

        try:
            # Check if split contains config/split format (e.g., "harmful/test_public")
            if "/" in config.split:
                parts = config.split.split("/")
                config_name = parts[0]
                split_name = parts[1] if len(parts) > 1 else "train"
                dataset = load_dataset(
                    config.hf_id,
                    config_name,
                    split=split_name,
                    cache_dir=self.cache_dir,
                )
            else:
                # Regular split loading
                dataset = load_dataset(
                    config.hf_id,
                    split=config.split,
                    cache_dir=self.cache_dir,
                )

            # Convert to list of dicts
            samples = []
            max_samples = config.max_samples or len(dataset)
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                samples.append(dict(item))

            self.loaded_datasets[dataset_name] = samples
            print(f"✓ Loaded {len(samples)} samples from {config.hf_id}")
            return samples

        except Exception as e:
            print(f"✗ Failed to load {config.hf_id}: {e}")
            return []

    def load_all_datasets(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load all supported datasets.

        Returns:
            Dictionary mapping dataset names to their samples
        """
        all_data = {}
        for dataset_name in self.SUPPORTED_DATASETS:
            samples = self.load_dataset(dataset_name)
            if samples:
                all_data[dataset_name] = samples
        return all_data


class ExecutionPlanAdapter:
    """
    Transforms raw adversarial examples into ExecutionPlan format.

    This adapter bridges the gap between HF dataset schemas and
    Gatling's typed ExecutionPlan representation.
    """

    def __init__(self):
        """Initialize the adapter with injection pattern mappings."""
        # Map common injection patterns to tool names
        self.injection_patterns = {
            "delete": ["delete_users", "drop_table", "remove_all"],
            "exfiltrate": ["send_email", "upload_file", "export_data"],
            "modify": ["update_settings", "change_permissions", "grant_access"],
            "read": ["list_files", "query_database", "get_secrets"],
        }

    def transform_to_execution_plan(
        self,
        text: str,
        label: str | None = None,
        source_dataset: str = "unknown",
    ) -> ExecutionPlan:
        """
        Transform adversarial text into an ExecutionPlan.

        Args:
            text: The adversarial prompt/injection text
            label: Optional label (e.g., "injection", "benign")
            source_dataset: Name of source dataset for tracking

        Returns:
            ExecutionPlan representing the adversarial action
        """
        # Detect injection intent from text
        tool_name = self._infer_tool_from_text(text)

        # Create provenance hash from source text
        prov_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Adversarial examples typically come from untrusted sources (RAG)
        trust_tier = TrustTier.PUBLIC_WEB if label == "injection" else TrustTier.INTERNAL

        # Extract scope hints from text (look for numbers, "all", etc.)
        scope_volume = self._infer_scope_volume(text)
        scope_sensitivity = self._infer_sensitivity(text, tool_name)

        # Create single-node execution plan
        node = ToolCallNode(
            tool_name=tool_name,
            arguments={"instruction": text[:200]},  # Truncate long texts
            provenance_tier=trust_tier,
            provenance_hash=prov_hash,
            scope_volume=scope_volume,
            scope_sensitivity=scope_sensitivity,
            node_id=f"{source_dataset}_{prov_hash[:8]}",
        )

        return ExecutionPlan(
            nodes=[node],
            edges=[],
        )

    def _infer_tool_from_text(self, text: str) -> str:
        """Infer most likely tool name from injection text."""
        text_lower = text.lower()

        # Check patterns in order of severity
        for category, tools in self.injection_patterns.items():
            if any(keyword in text_lower for keyword in [category, *tools]):
                return tools[0]  # Return first (most severe) tool in category

        # Default to generic suspicious action
        return "execute_instruction"

    def _infer_scope_volume(self, text: str) -> int:
        """Infer data scope from text (look for "all", numbers, etc.)."""
        text_lower = text.lower()

        if "all" in text_lower or "every" in text_lower:
            return 10000  # Large scope
        elif any(word in text_lower for word in ["many", "multiple", "several"]):
            return 100
        else:
            return 1  # Minimal scope

    def _infer_sensitivity(self, text: str, tool_name: str) -> int:
        """Infer sensitivity level (1-5) from text and tool."""
        text_lower = text.lower()

        # High sensitivity keywords
        if any(word in text_lower for word in ["password", "secret", "private", "admin", "root"]):
            return 5
        elif any(word in text_lower for word in ["confidential", "internal", "sensitive"]) or any(word in tool_name.lower() for word in ["delete", "drop", "remove"]):
            return 4
        elif any(word in text_lower for word in ["user", "account", "data"]):
            return 3
        else:
            return 2


class GatlingDatasetBuilder:
    """
    Main interface for building the Gatling training dataset.

    Combines HF datasets + opal's generator for the revised $100-500 strategy.
    """

    def __init__(self, cache_dir: str | None = None):
        """Initialize the builder with all components."""
        self.loader = HFDatasetLoader(cache_dir=cache_dir)
        self.adapter = ExecutionPlanAdapter()
        self.dataset: list[dict[str, Any]] = []

    def build_foundation_dataset(self) -> list[dict[str, Any]]:
        """
        Build the foundation dataset from existing HF sources.

        Returns ~4K real-world adversarial examples transformed to ExecutionPlan format.

        Returns:
            List of training samples with format:
            {
                "execution_plan": ExecutionPlan,
                "is_adversarial": bool,
                "source_dataset": str,
                "original_text": str,
            }
        """
        print("\n" + "=" * 70)
        print("🏗️  Building Foundation Dataset from HuggingFace")
        print("=" * 70 + "\n")

        all_hf_data = self.loader.load_all_datasets()

        training_samples = []

        for dataset_name, samples in all_hf_data.items():
            config = self.loader.SUPPORTED_DATASETS[dataset_name]
            print(f"\n🔄 Transforming {dataset_name} ({len(samples)} samples)...")

            for sample in samples:
                text = sample.get(config.text_field, "")
                label = sample.get(config.label_field) if config.label_field else None

                # Handle list-type text fields (conversations)
                if isinstance(text, list):
                    # Convert conversation list to single string
                    if all(isinstance(msg, dict) for msg in text):
                        # Format: [{"role": "user", "content": "..."}]
                        text = " ".join(msg.get("content", "") for msg in text)
                    else:
                        # Format: ["message1", "message2"]
                        text = " ".join(str(t) for t in text)
                elif not isinstance(text, str):
                    # Skip non-string, non-list fields
                    continue

                if not text or not text.strip():
                    continue

                # Transform to ExecutionPlan
                execution_plan = self.adapter.transform_to_execution_plan(
                    text=text,
                    label=label,
                    source_dataset=dataset_name,
                )

                # Determine if adversarial based on label
                is_adversarial = self._is_adversarial(label)

                training_samples.append({
                    "execution_plan": execution_plan,
                    "is_adversarial": is_adversarial,
                    "source_dataset": dataset_name,
                    "original_text": text,
                    "original_label": label,
                })

        self.dataset = training_samples

        print(f"\n{'=' * 70}")
        print(f"✅ Foundation dataset complete: {len(training_samples)} samples")
        print(f"{'=' * 70}\n")

        return training_samples

    def _is_adversarial(self, label: Any) -> bool:
        """Determine if a sample is adversarial based on its label."""
        if label is None:
            return False  # Unknown, treat as benign

        # Handle different label formats
        if isinstance(label, str):
            label_lower = str(label).lower()

            # Special handling for Microsoft llmail JSON objectives
            if "{" in label_lower and "defense.undetected" in label_lower:
                import json
                try:
                    objectives = json.loads(label)
                    # If defense was undetected, it's an adversarial injection
                    return objectives.get("defense.undetected", False)
                except json.JSONDecodeError:
                    pass

            return any(
                word in label_lower
                for word in ["injection", "attack", "malicious", "adversarial", "1", "true"]
            )
        elif isinstance(label, (int, bool)):
            return bool(label)

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        if not self.dataset:
            return {}

        total = len(self.dataset)
        adversarial_count = sum(1 for s in self.dataset if s["is_adversarial"])
        by_source = {}
        for sample in self.dataset:
            source = sample["source_dataset"]
            by_source[source] = by_source.get(source, 0) + 1

        return {
            "total_samples": total,
            "adversarial": adversarial_count,
            "benign": total - adversarial_count,
            "adversarial_ratio": adversarial_count / total if total > 0 else 0,
            "by_source": by_source,
        }
