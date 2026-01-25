#!/usr/bin/env python3
"""
Script to generate reasoning dataset integration for Gatling.

Integrates Alibaba Superior-Reasoning and RubricHub datasets,
extracts multi-step reasoning chains, and creates ExecutionPlan graphs.

Usage:
    uv run python scripts/generate_reasoning_dataset.py --samples 15000 --output data/reasoning_dataset.jsonl
"""

import argparse
import json
from pathlib import Path

from source.dataset.reasoning_integration import generate_reasoning_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning dataset for Gatling")
    parser.add_argument(
        "--samples",
        type=int,
        default=15000,
        help="Number of samples to generate (default: 15000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reasoning_dataset.jsonl",
        help="Output file path (default: data/reasoning_dataset.jsonl)",
    )
    parser.add_argument(
        "--alibaba-ratio",
        type=float,
        default=0.7,
        help="Proportion from Alibaba dataset (default: 0.7)",
    )
    parser.add_argument(
        "--rubrichub-ratio",
        type=float,
        default=0.3,
        help="Proportion from RubricHub dataset (default: 0.3)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated traces",
    )

    args = parser.parse_args()

    # Validate ratios
    if abs(args.alibaba_ratio + args.rubrichub_ratio - 1.0) > 0.01:
        print("ERROR: Ratios must sum to 1.0")
        return 1

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print(f"\n{'='*60}")
    print(f"Reasoning Dataset Generation")
    print(f"{'='*60}")

    gold_traces = generate_reasoning_dataset(
        target_count=args.samples,
        alibaba_ratio=args.alibaba_ratio,
        rubrichub_ratio=args.rubrichub_ratio,
    )

    # Validate if requested
    if args.validate:
        print("\n3. Validating traces...")
        valid_count = 0
        for trace in gold_traces:
            if trace.graph.validate_dag():
                trace.validated = True
                valid_count += 1

        print(f"   Valid traces: {valid_count}/{len(gold_traces)} ({valid_count/len(gold_traces)*100:.1f}%)")

    # Save to JSONL
    print(f"\n4. Saving to {output_path}...")
    with output_path.open("w") as f:
        for trace in gold_traces:
            training_data = trace.to_training_format()
            f.write(json.dumps(training_data) + "\n")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"Total traces: {len(gold_traces)}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Sample statistics
    total_tool_calls = sum(len(trace.graph.calls) for trace in gold_traces)
    avg_tool_calls = total_tool_calls / len(gold_traces) if gold_traces else 0

    print(f"\nStatistics:")
    print(f"  - Average tool calls per trace: {avg_tool_calls:.2f}")
    print(f"  - Total tool calls: {total_tool_calls}")
    print(f"  - Source distributions:")

    from collections import Counter

    sources = Counter(trace.metadata.get("source_dataset", "unknown") for trace in gold_traces)
    for source, count in sources.items():
        print(f"    - {source}: {count} ({count/len(gold_traces)*100:.1f}%)")

    print(f"\n✓ Dataset ready for training!")
    print(f"  Use with: source/encoders/governance_encoder.py")
    print(f"           source/encoders/execution_encoder.py")

    return 0


if __name__ == "__main__":
    exit(main())
