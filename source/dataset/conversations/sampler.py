"""
Conversation sampler for real-world data integration.

Samples conversations from public datasets and prepares them for transformation
into ExecutionPlans.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A complete conversation from a dataset."""

    conversation_id: str
    turns: list[ConversationTurn]
    source: str  # "wildchat" or "lmsys"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSampler:
    """
    Samples conversations from WildChat-1M and LMSYS-Chat-1M datasets.

    This class handles downloading datasets from HuggingFace Hub and sampling
    diverse conversations for transformation into ExecutionPlans.
    """

    SUPPORTED_DATASETS = {
        "wildchat": "allenai/WildChat-1M",
        "lmsys": "lmsys/lmsys-chat-1m",
    }

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ):
        """
        Initialize the conversation sampler.

        Args:
            cache_dir: Directory to cache downloaded datasets
            seed: Random seed for reproducible sampling
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.seed = seed
        random.seed(seed)
        self._datasets: dict[str, Any] = {}

    def load_dataset(self, dataset_name: str) -> None:
        """
        Load a dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the dataset ("wildchat" or "lmsys")

        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        if dataset_name in self._datasets:
            return  # Already loaded

        hf_dataset_id = self.SUPPORTED_DATASETS[dataset_name]
        print(f"Loading {dataset_name} from {hf_dataset_id}...")

        try:
            dataset = load_dataset(
                hf_dataset_id,
                split="train",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                streaming=True,  # Use streaming to avoid loading entire dataset
            )
            self._datasets[dataset_name] = dataset
            print(f"✓ Loaded {dataset_name}")
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")
            raise

    def sample_conversations(
        self,
        n_samples: int = 10000,
        wildchat_ratio: float = 0.5,
        min_turns: int = 2,
        max_turns: int | None = None,
    ) -> list[Conversation]:
        """
        Sample conversations from the loaded datasets.

        Args:
            n_samples: Total number of conversations to sample
            wildchat_ratio: Fraction of samples from WildChat (rest from LMSYS)
            min_turns: Minimum number of turns in a conversation
            max_turns: Maximum number of turns (None for no limit)

        Returns:
            List of sampled conversations
        """
        # Load datasets if not already loaded
        for dataset_name in ["wildchat", "lmsys"]:
            if dataset_name not in self._datasets:
                self.load_dataset(dataset_name)

        n_wildchat = int(n_samples * wildchat_ratio)
        n_lmsys = n_samples - n_wildchat

        conversations: list[Conversation] = []

        # Sample from WildChat
        if n_wildchat > 0:
            print(f"Sampling {n_wildchat} conversations from WildChat...")
            wildchat_samples = self._sample_from_dataset(
                "wildchat", n_wildchat, min_turns, max_turns
            )
            conversations.extend(wildchat_samples)

        # Sample from LMSYS
        if n_lmsys > 0:
            print(f"Sampling {n_lmsys} conversations from LMSYS...")
            lmsys_samples = self._sample_from_dataset(
                "lmsys", n_lmsys, min_turns, max_turns
            )
            conversations.extend(lmsys_samples)

        # Shuffle to mix datasets
        random.shuffle(conversations)

        print(f"✓ Sampled {len(conversations)} total conversations")
        return conversations

    def _sample_from_dataset(
        self,
        dataset_name: str,
        n_samples: int,
        min_turns: int,
        max_turns: int | None,
    ) -> list[Conversation]:
        """
        Sample conversations from a specific dataset.

        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples to collect
            min_turns: Minimum number of turns
            max_turns: Maximum number of turns

        Returns:
            List of sampled conversations
        """
        dataset = self._datasets[dataset_name]
        conversations: list[Conversation] = []
        seen_ids: set[str] = set()

        # Iterate through dataset (streaming mode)
        for idx, example in enumerate(dataset):
            if len(conversations) >= n_samples:
                break

            # Parse conversation based on dataset format
            if dataset_name == "wildchat":
                conv = self._parse_wildchat_example(example, idx)
            else:  # lmsys
                conv = self._parse_lmsys_example(example, idx)

            if conv is None:
                continue

            # Apply filters
            n_turns = len(conv.turns)
            if n_turns < min_turns:
                continue
            if max_turns is not None and n_turns > max_turns:
                continue

            # Avoid duplicates
            if conv.conversation_id in seen_ids:
                continue

            seen_ids.add(conv.conversation_id)
            conversations.append(conv)

        return conversations

    def _parse_wildchat_example(
        self, example: dict[str, Any], idx: int
    ) -> Conversation | None:
        """
        Parse a WildChat conversation example.

        WildChat format typically has:
        - 'conversation': list of turns with 'role' and 'content'
        - 'conversation_id': unique identifier
        """
        try:
            # Handle different possible formats
            if "conversation" in example:
                raw_turns = example["conversation"]
            elif "messages" in example:
                raw_turns = example["messages"]
            else:
                return None

            turns = [
                ConversationTurn(
                    role=turn.get("role", "unknown"),
                    content=turn.get("content", ""),
                    metadata={"turn_idx": i},
                )
                for i, turn in enumerate(raw_turns)
            ]

            conv_id = example.get("conversation_id", f"wildchat_{idx}")

            return Conversation(
                conversation_id=conv_id,
                turns=turns,
                source="wildchat",
                metadata={"original_index": idx},
            )
        except Exception as e:
            print(f"Warning: Failed to parse WildChat example {idx}: {e}")
            return None

    def _parse_lmsys_example(
        self, example: dict[str, Any], idx: int
    ) -> Conversation | None:
        """
        Parse an LMSYS conversation example.

        LMSYS format typically has:
        - 'conversation': list of turns
        - 'conversation_id': unique identifier
        """
        try:
            # Handle different possible formats
            if "conversation" in example:
                raw_turns = example["conversation"]
            elif "messages" in example:
                raw_turns = example["messages"]
            else:
                return None

            turns = [
                ConversationTurn(
                    role=turn.get("role", "unknown"),
                    content=turn.get("content", ""),
                    metadata={"turn_idx": i},
                )
                for i, turn in enumerate(raw_turns)
            ]

            conv_id = example.get("conversation_id", f"lmsys_{idx}")

            return Conversation(
                conversation_id=conv_id,
                turns=turns,
                source="lmsys",
                metadata={"original_index": idx},
            )
        except Exception as e:
            print(f"Warning: Failed to parse LMSYS example {idx}: {e}")
            return None

    def save_conversations(
        self, conversations: list[Conversation], output_path: str | Path
    ) -> None:
        """
        Save conversations to JSONL format.

        Args:
            conversations: List of conversations to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for conv in conversations:
                f.write(conv.model_dump_json() + "\n")

        print(f"✓ Saved {len(conversations)} conversations to {output_path}")
