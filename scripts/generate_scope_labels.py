#!/usr/bin/env python3
"""
Generate Minimal Scope Labels for E_scope Training Data (DA-003)

This script demonstrates how to use the MinimalScopeLabelGenerator to create
ground truth labels for training the SemanticIntentPredictor.

Usage:
    # Generate labels for sample queries
    uv run python scripts/generate_scope_labels.py --sample

    # Generate labels for existing gold traces
    uv run python scripts/generate_scope_labels.py --gold-traces

    # Export labeled data for validation
    uv run python scripts/generate_scope_labels.py --export labels.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.dataset.scope_labeling import (
    MinimalScopeLabelGenerator,
    create_scope_label_generator,
)


def generate_sample_labels():
    """Generate labels for sample queries to demonstrate the labeling process."""
    generator = create_scope_label_generator()

    # Sample queries from different domains
    sample_queries = [
        # Finance domain
        "Show me my latest invoice",
        "Find all failed payments last quarter",
        "List the top 10 largest transactions",
        "Export all customer financial records",

        # HR domain
        "Show me recent employee onboarding documents",
        "List all employees in engineering",
        "Get the last 30 days of timesheet data",

        # DevOps domain
        "List files in current directory",
        "Search all files recursively in the repository",
        "Check subdirectories for configuration files",

        # Security domain
        "Retrieve user passwords",
        "Access personal information for user John Doe",
        "Get all API keys and credentials",
        "List public documentation",

        # Analytics domain
        "Show today's sales metrics",
        "Summarize this month's performance",
        "Review this year's budget",
        "Get analytics from the last 14 days",
    ]

    print("Minimal Scope Label Generation Demo")
    print("=" * 80)
    print()

    results = []

    for query in sample_queries:
        label = generator.generate_label(query)

        result = {
            "query": query,
            "minimal_scope": {
                "limit": label.limit,
                "date_range_days": label.date_range_days,
                "max_depth": label.max_depth,
                "include_sensitive": label.include_sensitive
            },
            "confidence": label.confidence,
            "reasoning": label.reasoning,
            "method": label.method
        }

        results.append(result)

        # Print human-readable output
        print(f"Query: {query}")
        print(f"Minimal Scope:")
        print(f"  - Limit: {label.limit}")
        print(f"  - Date Range: {label.date_range_days} days" if label.date_range_days else "  - Date Range: None")
        print(f"  - Max Depth: {label.max_depth}" if label.max_depth else "  - Max Depth: None")
        print(f"  - Include Sensitive: {label.include_sensitive}")
        print(f"Confidence: {label.confidence:.2f}")
        print(f"Reasoning: {label.reasoning}")
        print()

    return results


def export_labels(results: list[dict], output_path: str):
    """Export labeled data to JSONL format for validation and training."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Exported {len(results)} labeled samples to {output_path}")


def generate_statistics(results: list[dict]):
    """Generate statistics about the labeled dataset."""
    print("Dataset Statistics")
    print("=" * 80)
    print()

    total = len(results)
    with_limit = sum(1 for r in results if r["minimal_scope"]["limit"] is not None)
    with_date_range = sum(1 for r in results if r["minimal_scope"]["date_range_days"] is not None)
    with_depth = sum(1 for r in results if r["minimal_scope"]["max_depth"] is not None)
    with_sensitivity = sum(1 for r in results if r["minimal_scope"]["include_sensitive"])

    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

    print(f"Total Samples: {total}")
    print(f"Samples with Limit: {with_limit} ({with_limit/total*100:.1f}%)")
    print(f"Samples with Date Range: {with_date_range} ({with_date_range/total*100:.1f}%)")
    print(f"Samples with Depth: {with_depth} ({with_depth/total*100:.1f}%)")
    print(f"Samples with Sensitive Data: {with_sensitivity} ({with_sensitivity/total*100:.1f}%)")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print()

    # Limit distribution
    limits = [r["minimal_scope"]["limit"] for r in results if r["minimal_scope"]["limit"] is not None]
    if limits:
        print("Limit Distribution:")
        from collections import Counter
        limit_counts = Counter(limits)
        for limit, count in sorted(limit_counts.items()):
            print(f"  {limit}: {count} samples")
        print()

    # Date range distribution
    date_ranges = [r["minimal_scope"]["date_range_days"] for r in results if r["minimal_scope"]["date_range_days"] is not None]
    if date_ranges:
        print("Date Range Distribution:")
        date_range_counts = Counter(date_ranges)
        for days, count in sorted(date_range_counts.items()):
            print(f"  {days} days: {count} samples")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate minimal scope labels for E_scope training data"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate labels for sample queries"
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="OUTPUT_FILE",
        help="Export labeled data to JSONL file"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Generate statistics about the labeled dataset"
    )

    args = parser.parse_args()

    if args.sample:
        results = generate_sample_labels()

        if args.stats:
            generate_statistics(results)

        if args.export:
            export_labels(results, args.export)
    else:
        print("No action specified. Use --sample to generate labels for sample queries.")
        print("Run with --help for more options.")


if __name__ == "__main__":
    main()
