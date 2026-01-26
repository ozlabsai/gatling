#!/usr/bin/env python3
"""
Script to generate Lakera adversarial dataset with context synthesis.

Integrates Lakera datasets (gandalf_ignore_instructions, gandalf_summarization, etc.)
and synthesizes complete execution context (policy, tools, provenance) for training.

Usage:
    # Generate 563K samples (full dataset with augmentation)
    uv run python scripts/generate_lakera_dataset.py --samples 563000 --output data/lakera_adversarial.jsonl

    # Quick test with 1K samples
    uv run python scripts/generate_lakera_dataset.py --samples 1000 --output data/lakera_test.jsonl --validate
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from source.dataset.adversarial import LakeraAdversarialLoader


def main():
    parser = argparse.ArgumentParser(description="Generate Lakera adversarial dataset with context synthesis")
    parser.add_argument(
        "--samples",
        type=int,
        default=563000,
        help="Target number of samples to generate (default: 563K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/lakera_adversarial.jsonl",
        help="Output file path (default: data/lakera_adversarial.jsonl)",
    )
    parser.add_argument(
        "--synthesis-mode",
        type=str,
        choices=["automatic", "hybrid", "llm"],
        default="automatic",
        help="Context synthesis mode (default: automatic)",
    )
    parser.add_argument(
        "--provenance-user",
        type=float,
        default=0.5,
        help="Proportion of TrustTier.USER samples (default: 0.5)",
    )
    parser.add_argument(
        "--provenance-unverified-rag",
        type=float,
        default=0.4,
        help="Proportion of TrustTier.UNVERIFIED_RAG samples (default: 0.4)",
    )
    parser.add_argument(
        "--provenance-verified-rag",
        type=float,
        default=0.1,
        help="Proportion of TrustTier.VERIFIED_RAG samples (default: 0.1)",
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=None,
        help="Dataset repetition factor (auto-calculated if not provided)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated traces for DAG correctness",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Security",
        help="Domain context for synthesized policies (default: Security)",
    )

    args = parser.parse_args()

    # Validate provenance distribution
    total_provenance = args.provenance_user + args.provenance_unverified_rag + args.provenance_verified_rag
    if abs(total_provenance - 1.0) > 0.01:
        print(f"ERROR: Provenance distribution must sum to 1.0, got {total_provenance}")
        return 1

    provenance_distribution = {
        "user": args.provenance_user,
        "unverified_rag": args.provenance_unverified_rag,
        "verified_rag": args.provenance_verified_rag,
    }

    # Calculate augmentation factor if not provided
    # Base Lakera datasets ~1.5K samples, so to reach 563K we need ~375x augmentation
    if args.augmentation_factor is None:
        estimated_base_samples = 1500  # gandalf_ignore_instructions + gandalf_summarization
        args.augmentation_factor = max(1, args.samples // estimated_base_samples)
        print(f"Auto-calculated augmentation factor: {args.augmentation_factor}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Header
    print(f"\n{'='*60}")
    print(f"Lakera Adversarial Dataset Generation")
    print(f"{'='*60}")
    print(f"Target samples: {args.samples:,}")
    print(f"Synthesis mode: {args.synthesis_mode}")
    print(f"Augmentation factor: {args.augmentation_factor}x")
    print(f"Provenance distribution:")
    print(f"  - USER: {args.provenance_user:.1%}")
    print(f"  - UNVERIFIED_RAG: {args.provenance_unverified_rag:.1%}")
    print(f"  - VERIFIED_RAG: {args.provenance_verified_rag:.1%}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Initialize loader
    loader = LakeraAdversarialLoader(
        synthesis_mode=args.synthesis_mode,
        provenance_distribution=provenance_distribution,
        target_samples=args.samples,
        augmentation_factor=args.augmentation_factor,
        domain=args.domain,
    )

    # Generate dataset
    print("Generating adversarial samples with context synthesis...")
    samples = []
    validation_failed = 0

    for sample in loader.load():
        # Validate if requested
        if args.validate:
            if not sample.execution_plan.validate_dag():
                validation_failed += 1
                print(f"   Warning: DAG validation failed for {sample.original_id}")
                continue

        samples.append(sample)

        # Progress indicator
        if len(samples) % 1000 == 0:
            print(f"   Generated {len(samples):,} samples...")

    # Save to JSONL
    print(f"\nSaving to {output_path}...")
    with output_path.open("w") as f:
        for sample in samples:
            # Convert to training format
            training_data = {
                "execution_plan": sample.execution_plan.model_dump(),
                "label": sample.label,
                "original_id": sample.original_id,
                "category": sample.category,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(training_data) + "\n")

    # Get statistics from loader
    stats = loader.get_stats()

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples):,}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    if validation_failed > 0:
        print(f"\nValidation failures: {validation_failed} ({validation_failed/len(samples)*100:.1f}%)")

    # Statistics
    print(f"\nStatistics:")
    print(f"  - Loaded from Lakera: {stats['total_loaded']:,}")
    print(f"  - Successful synthesis: {stats['successful_synthesis']:,}")
    print(f"  - Failed synthesis: {stats['failed_synthesis']:,}")
    print(f"  - Synthesis rate: {stats['synthesis_rate']:.1%}")

    # Attack pattern distribution
    print(f"\n  Attack Pattern Distribution:")
    pattern_counts = Counter(s.category for s in samples)
    for pattern, count in pattern_counts.most_common():
        print(f"    - {pattern}: {count:,} ({count/len(samples)*100:.1f}%)")

    # Provenance tier distribution
    print(f"\n  Provenance Tier Distribution:")
    tier_counts = Counter(s.metadata["provenance_tier"] for s in samples)
    for tier, count in tier_counts.most_common():
        print(f"    - {tier}: {count:,} ({count/len(samples)*100:.1f}%)")

    # Energy label statistics
    print(f"\n  Energy Label Statistics (average):")
    energy_labels = {
        "E_hierarchy": [],
        "E_provenance": [],
        "E_scope": [],
        "E_flow": [],
    }
    for sample in samples:
        for term, value in sample.metadata["energy_labels"].items():
            energy_labels[term].append(value)

    for term, values in energy_labels.items():
        avg = sum(values) / len(values) if values else 0.0
        print(f"    - {term}: {avg:.3f}")

    # Tool usage statistics
    total_tool_calls = sum(s.metadata["tool_count"] for s in samples)
    avg_tool_calls = total_tool_calls / len(samples) if samples else 0.0
    print(f"\n  Tool Call Statistics:")
    print(f"    - Total tool calls: {total_tool_calls:,}")
    print(f"    - Average per sample: {avg_tool_calls:.2f}")

    # Dataset sources
    print(f"\n  Source Dataset Distribution:")
    source_counts = Counter(s.metadata["source_dataset"] for s in samples)
    for source, count in source_counts.most_common():
        print(f"    - {source}: {count:,} ({count/len(samples)*100:.1f}%)")

    print(f"\n✓ Dataset ready for training!")
    print(f"  Integration points:")
    print(f"    - Stage B (Adversarial Mutation): Use as seed patterns")
    print(f"    - Stage D (Provenance Injection): RAG poisoning scenarios")
    print(f"    - JEPA Encoder Training: source/encoders/")
    print(f"    - Energy Function Training: source/energy/")

    return 0


if __name__ == "__main__":
    exit(main())
