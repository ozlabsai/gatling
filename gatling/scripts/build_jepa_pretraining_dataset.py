"""
Build JEPA Pre-training Dataset from HuggingFace Sources.

Loads 2M+ samples from existing curated datasets and transforms them
to ExecutionPlan format for JEPA encoder pre-training.

Cost: $0 (using existing datasets)
Timeline: 2-4 hours compute

Usage:
    uv run python scripts/build_jepa_pretraining_dataset.py \
        --target 2000000 \
        --output data/jepa_pretraining_2m.jsonl \
        --validate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from source.dataset.unified_transformer import (
    UnifiedDatasetTransformer,
    filter_quality,
)


class JEPADatasetBuilder:
    """Build JEPA pre-training dataset from HuggingFace sources."""

    # Dataset configurations for JEPA pre-training (benign samples)
    JEPA_DATASETS = {
        # Large-scale conversations (2M total)
        "wildchat": {
            "hf_id": "allenai/WildChat-1M",
            "text_field": "conversation",
            "split": "train",
            "target_samples": 1000000,
        },
        "lmsys-chat": {
            "hf_id": "lmsys/lmsys-chat-1m",
            "text_field": "conversation",
            "split": "train",
            "target_samples": 1000000,
        },
        # Function calling datasets (EASY - direct tool format)
        "xlam-function-calling": {
            "hf_id": "Salesforce/xlam-function-calling-60k",
            "text_field": "tools",  # or "conversations"
            "split": "train",
            "target_samples": 60000,
        },
        "hermes-function-calling": {
            "hf_id": "NousResearch/hermes-function-calling-v1",
            "text_field": "conversations",
            "split": "train",
            "target_samples": None,  # Use all
        },
        "hermes-reasoning-tool-use": {
            "hf_id": "interstellarninja/hermes_reasoning_tool_use",
            "text_field": "conversations",
            "split": "train",
            "target_samples": None,
        },
        "fiftyone-function-calling": {
            "hf_id": "Voxel51/fiftyone-function-calling-14k",
            "text_field": "messages",
            "split": "train",
            "target_samples": 14000,
        },
        # Instruction following (general benign)
        "hf4-instruction": {
            "hf_id": "HuggingFaceH4/instruction-dataset",
            "text_field": "prompt",
            "split": "train",
            "target_samples": 100000,
        },
        # Existing tool-use datasets
        "tool-use-reasoning": {
            "hf_id": "llamafactory/reason-tool-use-demo-1500",
            "text_field": "instruction",
            "split": "train",
            "target_samples": 1500,
        },
        "tool-use-instruct": {
            "hf_id": "allenai/Dolci-Instruct-SFT-Tool-Use",
            "text_field": "instruction",
            "split": "train",
            "target_samples": 937,
        },
    }

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the builder.

        Args:
            cache_dir: Optional cache directory for HF datasets
        """
        self.cache_dir = cache_dir
        self.transformer = UnifiedDatasetTransformer()
        self.samples: list[dict[str, Any]] = []

    def build_dataset(
        self,
        target_samples: int = 2000000,
        output_path: str = "data/jepa_pretraining_2m.jsonl",
        validate: bool = False,
    ) -> dict[str, Any]:
        """
        Build the JEPA pre-training dataset.

        Args:
            target_samples: Target number of samples (default 2M)
            output_path: Output JSONL file path
            validate: Whether to validate sample quality

        Returns:
            Statistics about the generated dataset
        """
        print("\n" + "=" * 70)
        print("🏗️  Building JEPA Pre-training Dataset")
        print("=" * 70)
        print(f"Target: {target_samples:,} samples")
        print(f"Output: {output_path}")
        print("=" * 70 + "\n")

        total_collected = 0
        stats_by_source = {}

        # Load and transform each dataset
        for dataset_name, config in self.JEPA_DATASETS.items():
            if total_collected >= target_samples:
                print(f"\n✅ Target reached ({total_collected:,} samples), stopping.")
                break

            remaining = target_samples - total_collected
            max_samples = min(
                config.get("target_samples") or remaining,
                remaining,
            )

            print(f"\n{'─' * 70}")
            print(f"📥 Loading: {config['hf_id']}")
            print(f"   Target: {max_samples:,} samples")
            print(f"   Progress: {total_collected:,}/{target_samples:,}")

            try:
                # Load dataset
                dataset = load_dataset(
                    config["hf_id"],
                    split=config["split"],
                    cache_dir=self.cache_dir,
                    streaming=True,  # Use streaming for large datasets
                )

                # Process samples
                collected = 0
                processed = 0
                for sample in dataset:
                    if collected >= max_samples:
                        break

                    processed += 1

                    # Transform to ExecutionPlan
                    execution_plan = self.transformer.transform(
                        sample=sample,
                        source_dataset=dataset_name,
                    )

                    if execution_plan is None:
                        continue

                    # Filter quality
                    if validate and not filter_quality(execution_plan):
                        continue

                    # Save sample
                    self.samples.append({
                        "execution_plan": execution_plan.dict(),
                        "source_dataset": dataset_name,
                        "label": "benign",  # All JEPA pre-training samples are benign
                    })

                    collected += 1
                    total_collected += 1

                    # Progress updates
                    if collected % 10000 == 0:
                        print(f"   ├─ Collected: {collected:,} / {max_samples:,}")

                stats_by_source[dataset_name] = {
                    "collected": collected,
                    "processed": processed,
                    "acceptance_rate": collected / processed if processed > 0 else 0,
                }

                print(f"   ✅ Completed: {collected:,} samples ({collected/processed*100:.1f}% acceptance)")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                stats_by_source[dataset_name] = {
                    "collected": 0,
                    "processed": 0,
                    "error": str(e),
                }
                continue

        # Save to JSONL
        print(f"\n{'═' * 70}")
        print(f"💾 Saving to {output_path}...")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")

        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"✅ Saved {len(self.samples):,} samples ({file_size_mb:.1f} MB)")

        # Final statistics
        print(f"\n{'═' * 70}")
        print("📊 DATASET STATISTICS")
        print(f"{'═' * 70}")
        print(f"Total Samples: {len(self.samples):,}")
        print(f"Target: {target_samples:,} ({len(self.samples)/target_samples*100:.1f}%)")
        print(f"File Size: {file_size_mb:.1f} MB")
        print(f"\nBy Source:")
        for source, stats in stats_by_source.items():
            collected = stats.get("collected", 0)
            processed = stats.get("processed", 0)
            print(f"  {source:30s}: {collected:8,} samples ({collected/processed*100:.1f}% accept)")

        # Transformation statistics
        transformer_stats = self.transformer.get_statistics()
        print(f"\nTransformation:")
        print(f"  Total Transformed: {transformer_stats['total_transformed']:,}")
        print(f"  Errors: {transformer_stats['errors']} ({transformer_stats['error_rate']*100:.2f}%)")
        print(f"\n  By Schema:")
        for schema, count in transformer_stats['by_schema'].items():
            if count > 0:
                print(f"    {schema:25s}: {count:8,} samples")

        print(f"{'═' * 70}\n")

        return {
            "total_samples": len(self.samples),
            "target_samples": target_samples,
            "file_size_mb": file_size_mb,
            "by_source": stats_by_source,
            "transformer_stats": transformer_stats,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build JEPA pre-training dataset from HuggingFace sources"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2000000,
        help="Target number of samples (default: 2M)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/jepa_pretraining_2m.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable quality validation filtering",
    )

    args = parser.parse_args()

    # Build dataset
    builder = JEPADatasetBuilder(cache_dir=args.cache_dir)
    stats = builder.build_dataset(
        target_samples=args.target,
        output_path=args.output,
        validate=args.validate,
    )

    # Save stats
    stats_path = Path(args.output).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"📊 Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
