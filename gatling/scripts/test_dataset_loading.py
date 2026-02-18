#!/usr/bin/env python3
"""
Test script for validating HuggingFace dataset loading and transformation.

This script demonstrates the revised dataset strategy: using existing
adversarial datasets instead of $40K synthetic generation.
"""

import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.dataset.loaders import GatlingDatasetBuilder


def main():
    """Test the dataset loading pipeline."""
    print("\n" + "=" * 80)
    print("🧪 Testing Gatling Dataset Loading Pipeline")
    print("=" * 80)

    # Initialize builder
    builder = GatlingDatasetBuilder(cache_dir=".cache/huggingface")

    # Build foundation dataset from HF
    print("\n📦 Building foundation dataset...")
    try:
        training_samples = builder.build_foundation_dataset()
    except Exception as e:
        print(f"\n❌ Failed to build dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show statistics
    stats = builder.get_statistics()
    print("\n" + "=" * 80)
    print("📊 Dataset Statistics")
    print("=" * 80)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Adversarial: {stats['adversarial']} ({stats['adversarial_ratio']:.1%})")
    print(f"Benign: {stats['benign']}")
    print("\nBy source:")
    for source, count in stats['by_source'].items():
        print(f"  - {source}: {count}")

    # Show example samples
    print("\n" + "=" * 80)
    print("🔍 Example Samples (first 3)")
    print("=" * 80)

    for i, sample in enumerate(training_samples[:3]):
        print(f"\n[{i+1}] Source: {sample['source_dataset']}")
        print(f"    Adversarial: {sample['is_adversarial']}")
        print(f"    Original: {sample['original_text'][:100]}...")
        print(f"    ExecutionPlan:")
        plan = sample['execution_plan']
        for node in plan.nodes:
            print(f"      - Tool: {node.tool_name}")
            print(f"        Trust: {node.provenance_tier.name}")
            print(f"        Scope: {node.scope_volume} records, sensitivity={node.scope_sensitivity}")

    # Validate ExecutionPlan format
    print("\n" + "=" * 80)
    print("✅ Validation")
    print("=" * 80)

    valid_count = 0
    for sample in training_samples:
        plan = sample['execution_plan']
        # Check that plan has valid structure
        if plan.nodes and all(node.node_id for node in plan.nodes):
            valid_count += 1

    print(f"Valid ExecutionPlans: {valid_count}/{len(training_samples)} ({valid_count/len(training_samples):.1%})")

    if valid_count == len(training_samples):
        print("\n✅ All samples successfully transformed to ExecutionPlan format!")
        print("\n💰 Cost: $0 (using existing HuggingFace datasets)")
        print("🚀 Ready for JEPA encoder training!")
        return 0
    else:
        print(f"\n⚠️  {len(training_samples) - valid_count} samples failed validation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
