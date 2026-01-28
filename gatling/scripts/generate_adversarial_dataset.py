#!/usr/bin/env python3
"""
Generate the full adversarial dataset for Gatling training.

Phase 2B - DG-001: Generate 563K samples from HuggingFace datasets.

This script implements the revised $0-500 strategy instead of $40-60K synthetic generation:
- Load real-world adversarial datasets (prompt injections, agent harm, etc.)
- Sample from large conversation datasets (WildChat 1M, LMSYS 1M)
- Transform all samples to ExecutionPlan format
- Save as JSONL for training
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.dataset.loaders import GatlingDatasetBuilder


def main():
    """Generate the full adversarial dataset."""
    parser = argparse.ArgumentParser(
        description="Generate adversarial dataset for Gatling training"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=563_000,
        help="Target number of samples (default: 563K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/adversarial_training_dataset.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/huggingface",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Sample mode: generate only 10K samples for testing",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50_000,
        help="Save checkpoint every N samples",
    )

    args = parser.parse_args()

    if args.sample_mode:
        print("🧪 SAMPLE MODE: Generating 10K samples for testing")
        args.target = 10_000
        args.output = "data/adversarial_sample_10k.jsonl"

    print(f"\n{'=' * 80}")
    print(f"🚀 Gatling Adversarial Dataset Generation - Phase 2B (DG-001)")
    print(f"{'=' * 80}")
    print(f"Target: {args.target:,} samples")
    print(f"Output: {args.output}")
    print(f"Cache: {args.cache_dir}")
    print(f"{'=' * 80}\n")

    # Initialize builder
    builder = GatlingDatasetBuilder(cache_dir=args.cache_dir)

    # Build foundation dataset (load all configured datasets)
    print("\n📦 Loading foundation datasets from HuggingFace...")
    training_samples = builder.build_foundation_dataset()

    current_count = len(training_samples)
    print(f"\n✓ Foundation complete: {current_count:,} samples")

    # If we need more samples, continue sampling from large datasets
    if current_count < args.target:
        remaining = args.target - current_count
        print(f"\n📊 Need {remaining:,} more samples to reach target...")
        print("   Strategy: Sample additional batches from WildChat/LMSYS")

        # TODO: Implement iterative sampling from large datasets
        # For now, we work with what we have
        print(
            "\n⚠️  Note: Current implementation loads configured max_samples."
        )
        print(
            "    To reach 563K, increase max_samples in loaders.py or implement batch sampling."
        )

    # Save to JSONL
    print(f"\n💾 Saving dataset to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in training_samples:
            # Convert ExecutionPlan to dict for JSON serialization
            sample_dict = {
                "execution_plan": {
                    "nodes": [
                        {
                            "tool_name": node.tool_name,
                            "arguments": node.arguments,
                            "provenance_tier": node.provenance_tier.value,
                            "provenance_hash": node.provenance_hash,
                            "scope_volume": node.scope_volume,
                            "scope_sensitivity": node.scope_sensitivity,
                            "node_id": node.node_id,
                        }
                        for node in sample["execution_plan"].nodes
                    ],
                    "edges": sample["execution_plan"].edges,
                },
                "label": "adversarial" if sample["is_adversarial"] else "benign",
                "source_dataset": sample["source_dataset"],
                "original_text": sample["original_text"][:500],  # Truncate for size
            }
            f.write(json.dumps(sample_dict, default=str) + "\n")

    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = output_path.parent / f"metadata_{timestamp}.json"

    stats = builder.get_statistics()
    metadata = {
        "timestamp": timestamp,
        "target_samples": args.target,
        "actual_samples": len(training_samples),
        "statistics": stats,
        "output_file": str(output_path),
        "cost": "$0 (using HuggingFace datasets)",
        "strategy": "Real-world adversarial + benign samples from HF",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"💾 Saved metadata: {metadata_path}")

    # Print final statistics
    print(f"\n{'=' * 80}")
    print(f"✅ Dataset Generation Complete!")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(training_samples):,}")
    print(f"Adversarial: {stats['adversarial']:,} ({stats['adversarial_ratio']:.1%})")
    print(f"Benign: {stats['benign']:,}")
    print(f"\nBy source:")
    for source, count in sorted(
        stats["by_source"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  - {source}: {count:,}")
    print(f"\n💰 Cost: $0 (vs $40-60K for synthetic generation)")
    print(f"🚀 Ready for JEPA encoder training!")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
