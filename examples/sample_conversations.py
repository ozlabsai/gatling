#!/usr/bin/env python3
"""
Example script for sampling conversations and creating execution plans.

This demonstrates the full pipeline:
1. Sample conversations from WildChat-1M and LMSYS-Chat-1M
2. Extract action intents using Claude
3. Transform intents into ExecutionPlans
4. Apply adversarial mutations (20%)
5. Save results for training

Usage:
    # Sample 100 conversations (quick test)
    uv run python examples/sample_conversations.py --sample

    # Sample full 10K conversations
    uv run python examples/sample_conversations.py --num-samples 10000

    # Custom configuration
    uv run python examples/sample_conversations.py \
        --num-samples 5000 \
        --wildchat-ratio 0.6 \
        --mutation-rate 0.25 \
        --output-dir ./outputs/conversations
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from source.dataset.conversations import (
    ConversationSampler,
    IntentExtractor,
    PlanTransformer,
    AdversarialMutator,
)


def main():
    """Run the conversation sampling pipeline."""
    parser = argparse.ArgumentParser(
        description="Sample conversations and create execution plans"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of conversations to sample (default: 100)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Quick sample mode (100 conversations)",
    )
    parser.add_argument(
        "--wildchat-ratio",
        type=float,
        default=0.5,
        help="Ratio of WildChat to LMSYS samples (default: 0.5)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="Fraction of plans to mutate adversarially (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/conversations",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum conversation turns (default: 2)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Override for sample mode
    if args.sample:
        args.num_samples = 100
        print("📦 Sample mode: generating 100 conversations")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("🎯 Gatling Conversation Dataset Sampling")
    print(f"{'='*70}")
    print(f"Target samples: {args.num_samples:,}")
    print(f"WildChat ratio: {args.wildchat_ratio:.1%}")
    print(f"Mutation rate: {args.mutation_rate:.1%}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    # Step 1: Sample conversations
    print("📥 Step 1: Sampling conversations from datasets...")
    print("-" * 70)

    sampler = ConversationSampler(
        cache_dir=output_dir / "cache",
        seed=args.seed,
    )

    conversations = sampler.sample_conversations(
        n_samples=args.num_samples,
        wildchat_ratio=args.wildchat_ratio,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )

    # Save raw conversations
    conversations_file = output_dir / "conversations_raw.jsonl"
    sampler.save_conversations(conversations, conversations_file)

    print(f"\n✓ Sampled {len(conversations)} conversations")
    print(
        f"  - WildChat: {sum(1 for c in conversations if c.source == 'wildchat')}"
    )
    print(
        f"  - LMSYS: {sum(1 for c in conversations if c.source == 'lmsys')}"
    )

    # Step 2: Extract intents
    print(f"\n📝 Step 2: Extracting action intents...")
    print("-" * 70)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "⚠️  Warning: ANTHROPIC_API_KEY not found. Skipping intent extraction."
        )
        print("   Set the environment variable to enable this step.")
        return

    extractor = IntentExtractor(api_key=api_key)

    intent_map = extractor.extract_intents(conversations, batch_size=10)

    total_intents = sum(len(intents) for intents in intent_map.values())
    print(f"\n✓ Extracted {total_intents} intents from {len(intent_map)} conversations")

    # Step 3: Transform to execution plans
    print(f"\n🔄 Step 3: Transforming intents to execution plans...")
    print("-" * 70)

    transformer = PlanTransformer(api_key=api_key)

    plans = transformer.transform_intents(conversations, intent_map)

    print(f"\n✓ Created {len(plans)} execution plans")

    # Step 4: Apply adversarial mutations
    print(f"\n⚡ Step 4: Applying adversarial mutations...")
    print("-" * 70)

    mutator = AdversarialMutator(
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    benign_plans, mutated_plans = mutator.mutate_plans(plans)

    print(f"\n✓ Generated adversarial dataset:")
    print(f"  - Benign plans: {len(benign_plans)}")
    print(f"  - Mutated plans: {len(mutated_plans)}")

    # Step 5: Save results
    print(f"\n💾 Step 5: Saving results...")
    print("-" * 70)

    # Save benign plans
    benign_file = output_dir / "execution_plans_benign.jsonl"
    with open(benign_file, "w") as f:
        for plan in benign_plans:
            f.write(plan.model_dump_json() + "\n")
    print(f"✓ Saved {len(benign_plans)} benign plans to {benign_file}")

    # Save mutated plans
    mutated_file = output_dir / "execution_plans_mutated.jsonl"
    with open(mutated_file, "w") as f:
        for plan in mutated_plans:
            f.write(plan.model_dump_json() + "\n")
    print(f"✓ Saved {len(mutated_plans)} mutated plans to {mutated_file}")

    # Save statistics
    stats = {
        "total_conversations": len(conversations),
        "total_intents": total_intents,
        "total_plans": len(plans),
        "benign_plans": len(benign_plans),
        "mutated_plans": len(mutated_plans),
        "mutation_rate": args.mutation_rate,
        "wildchat_count": sum(
            1 for c in conversations if c.source == "wildchat"
        ),
        "lmsys_count": sum(1 for c in conversations if c.source == "lmsys"),
        "seed": args.seed,
    }

    import json

    stats_file = output_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("✅ Pipeline complete!")
    print(f"{'='*70}")
    print(f"\nGenerated {len(conversations)} conversations:")
    print(f"  → {total_intents} action intents")
    print(f"  → {len(plans)} execution plans")
    print(f"  → {len(benign_plans)} benign + {len(mutated_plans)} adversarial")
    print(f"\nOutput directory: {output_dir}")
    print(
        f"\nNext steps:"
        f"\n  1. Review generated plans in {output_dir}"
        f"\n  2. Use for JEPA encoder training"
        f"\n  3. Validate energy function calibration"
    )
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
