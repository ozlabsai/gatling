#!/usr/bin/env python3
"""
Generate Tier I Training Dataset using 7 Specialized Benign Loaders.

This script loads all 7 specialized datasets (Track 4) and generates
a combined JSONL dataset for Tier I training. All samples are benign
with TrustTier.INTERNAL provenance.

Usage:
    # Generate full dataset
    uv run python scripts/generate_tier1_dataset.py --output data/tier1_training.jsonl

    # Generate sample dataset for testing
    uv run python scripts/generate_tier1_dataset.py --samples-per-dataset 100 --output data/tier1_sample.jsonl

    # With progress tracking
    uv run python scripts/generate_tier1_dataset.py --progress
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add source to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.dataset.specialized_loaders import load_all_specialized_datasets


def main():
    """Generate Tier I dataset from all 7 specialized loaders."""
    parser = argparse.ArgumentParser(
        description="Generate Tier I training dataset from specialized benign loaders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tier1_training.jsonl",
        help="Output JSONL file path (default: data/tier1_training.jsonl)",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=None,
        help="Max samples to load from each dataset (default: None = all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/gatling/datasets)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress indicators during generation",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ExecutionPlan format for each sample (slower)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing output file",
    )

    args = parser.parse_args()

    # Print header
    print(f"\n{'=' * 80}")
    print("Tier I Training Dataset Generation (Track 4)")
    print(f"{'=' * 80}")
    print(f"Output: {args.output}")
    print(f"Samples per dataset: {args.samples_per_dataset or 'all'}")
    print(f"Cache directory: {args.cache_dir or 'default'}")
    print(f"{'=' * 80}\n")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate samples
    samples = []
    total_samples = 0
    failed_samples = 0
    dataset_counts = Counter()
    category_counts = Counter()

    print("🔄 Loading datasets and transforming to ExecutionPlan format...\n")

    for sample in load_all_specialized_datasets(
        cache_dir=args.cache_dir,
        max_samples_per_dataset=args.samples_per_dataset,
    ):
        total_samples += 1

        # Validate if requested
        if args.validate:
            try:
                # Basic validation: check that ExecutionPlan has nodes
                if not sample.execution_plan.nodes:
                    print(f"   ⚠️  Warning: Empty execution plan for {sample.original_id}")
                    failed_samples += 1
                    continue

                # Verify all samples are benign
                if sample.label != "benign":
                    print(f"   ⚠️  Warning: Non-benign sample detected: {sample.original_id}")
                    failed_samples += 1
                    continue

            except Exception as e:
                print(f"   ⚠️  Validation failed for {sample.original_id}: {e}")
                failed_samples += 1
                continue

        samples.append(sample)
        dataset_counts[sample.metadata["source"]] += 1
        if sample.category:
            category_counts[sample.category] += 1

        # Progress indicator
        if args.progress and len(samples) % 1000 == 0:
            print(f"   ✓ Generated {len(samples):,} samples...")

    print(f"\n✓ Loaded {len(samples):,} total samples from {len(dataset_counts)} datasets")

    if failed_samples > 0:
        print(f"⚠️  Skipped {failed_samples} samples due to validation failures")

    # Statistics
    print(f"\n{'=' * 80}")
    print("Dataset Statistics")
    print(f"{'=' * 80}")
    print(f"\nBy Source Dataset:")
    for source, count in dataset_counts.most_common():
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {source:50} {count:>8,} ({pct:>5.1f}%)")

    print(f"\nBy Category:")
    for category, count in category_counts.most_common():
        pct = (count / len(samples) * 100) if samples else 0
        print(f"  {category:50} {count:>8,} ({pct:>5.1f}%)")

    # Label verification
    benign_count = sum(1 for s in samples if s.label == "benign")
    print(f"\nLabel Distribution:")
    print(f"  benign:  {benign_count:>8,} ({benign_count/len(samples)*100:>5.1f}%)")

    # Provenance tier verification
    internal_count = sum(
        1
        for s in samples
        if all(node.provenance_tier == 1 for node in s.execution_plan.nodes)
    )
    print(f"\nProvenance Tier Verification:")
    print(f"  All INTERNAL (Tier 1): {internal_count:>8,} ({internal_count/len(samples)*100:>5.1f}%)")

    # Dry run exit
    if args.dry_run:
        print(f"\n✓ Dry run complete (no output file written)")
        return 0

    # Save to JSONL
    print(f"\n💾 Writing dataset to {output_path}...")

    with output_path.open("w") as f:
        for sample in samples:
            # Convert to training format
            training_sample = {
                "execution_plan": {
                    "nodes": [
                        {
                            "tool_name": node.tool_name,
                            "arguments": node.arguments,
                            "provenance_tier": int(node.provenance_tier),
                            "provenance_hash": node.provenance_hash,
                            "scope_volume": node.scope_volume,
                            "scope_sensitivity": node.scope_sensitivity,
                            "node_id": node.node_id,
                        }
                        for node in sample.execution_plan.nodes
                    ],
                    "edges": sample.execution_plan.edges,
                },
                "label": sample.label,
                "original_id": sample.original_id,
                "category": sample.category,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(training_sample) + "\n")

    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = output_path.parent / f"tier1_metadata_{timestamp}.json"

    metadata = {
        "timestamp": timestamp,
        "total_samples": len(samples),
        "failed_samples": failed_samples,
        "dataset_distribution": dict(dataset_counts),
        "category_distribution": dict(category_counts),
        "output_file": str(output_path),
        "samples_per_dataset_limit": args.samples_per_dataset,
        "all_benign": benign_count == len(samples),
        "all_internal_tier": internal_count == len(samples),
        "generation_config": {
            "cache_dir": args.cache_dir,
            "validated": args.validate,
        },
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved metadata: {metadata_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("✅ Tier I Dataset Generation Complete!")
    print(f"{'=' * 80}")
    print(f"Total samples:        {len(samples):,}")
    print(f"Output file:          {output_path}")
    print(f"File size:            {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"All benign:           {'✓' if benign_count == len(samples) else '✗'}")
    print(f"All INTERNAL tier:    {'✓' if internal_count == len(samples) else '✗'}")
    print(f"\n🚀 Ready for JEPA encoder training!")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
