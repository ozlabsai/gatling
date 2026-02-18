"""
HuggingFace Dataset Discovery Script

Investigates candidate datasets for JEPA pre-training:
- Reports actual size
- Shows schema format
- Displays sample data
- Assesses transformation feasibility

Usage:
    uv run python scripts/discover_hf_datasets.py
"""

from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset

# Candidate datasets for JEPA pre-training
CANDIDATE_DATASETS = {
    # Tool-Use / Function Calling (High Priority)
    "function_calling": [
        "Salesforce/xlam-function-calling-60k",
        "nvidia/ToolScale",
        "RioLee/ToolPref-Pairwise-30K",
        "atasoglu/turkish-function-calling-20k",
        "Voxel51/fiftyone-function-calling-14k",
        "twinkle-ai/tw-function-call-reasoning-10k",
        "NousResearch/hermes-function-calling-v1",
        "interstellarninja/hermes_reasoning_tool_use",
        "Nanbeige/ToolMind",
        "dongsheng/DTA-Tool",
        "squeeze-ai-lab/ToolBank",
        "google/mobile-actions",
        "apple/mmau",
        "mercor/apex-agents",
    ],
    # Agent Safety (May contain adversarial labels)
    "agent_safety": [
        "nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0",
    ],
    # Instruction-Following (General Benign)
    "instruction_following": [
        "HuggingFaceH4/instruction-dataset",
        "Muennighoff/natural-instructions",
        "nvidia/Nemotron-RL-instruction_following-structured_outputs",
        "nvidia/Nemotron-Math-v2",
    ],
}


def discover_dataset(dataset_id: str, max_samples: int = 3) -> dict[str, Any]:
    """
    Discover a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID
        max_samples: Number of samples to display

    Returns:
        Discovery report dict
    """
    report = {
        "dataset_id": dataset_id,
        "status": "unknown",
        "size": 0,
        "schema": [],
        "samples": [],
        "error": None,
        "transformation_feasibility": "unknown"
    }

    try:
        print(f"\n{'=' * 70}")
        print(f"📊 Discovering: {dataset_id}")
        print(f"{'=' * 70}")

        # Try loading with different splits
        dataset = None
        for split in ["train", "validation", "test", None]:
            try:
                print(f"   Trying split: {split or 'default'}...")
                dataset = load_dataset(dataset_id, split=split, streaming=True)
                # Convert to list to get size
                samples = list(dataset.take(max_samples + 100))
                dataset = samples  # Use the list
                report["split_used"] = split or "default"
                break
            except Exception as e:
                continue

        if dataset is None:
            raise Exception("Could not load any split")

        # Get size (from first 100 samples, estimate total)
        report["size_sampled"] = len(dataset)
        report["status"] = "success"

        # Get schema
        if len(dataset) > 0:
            report["schema"] = list(dataset[0].keys())

            # Get samples
            for i in range(min(max_samples, len(dataset))):
                report["samples"].append(dataset[i])

            # Assess transformation feasibility
            report["transformation_feasibility"] = assess_transformation(dataset[0])

        # Print report
        print(f"\n✅ SUCCESS")
        print(f"   Size (sampled): {report['size_sampled']} (showing first 100)")
        print(f"   Split: {report['split_used']}")
        print(f"   Schema: {report['schema']}")
        print(f"   Transformation: {report['transformation_feasibility']}")
        print(f"\n   Sample 1:")
        print(f"   {json.dumps(dataset[0], indent=2)[:500]}...")

    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        print(f"\n❌ FAILED: {e}")

    return report


def assess_transformation(sample: dict) -> str:
    """Assess how easy it is to transform this sample to ExecutionPlan."""
    keys = set(sample.keys())

    # Function calling format
    if "function" in keys or "function_call" in keys:
        return "EASY - Direct function calling format"

    # Conversation format
    if "messages" in keys or "conversation" in keys:
        return "MEDIUM - Conversation format, need to extract tools"

    # Instruction format
    if "instruction" in keys or "prompt" in keys:
        return "MEDIUM - Instruction format, need to infer tools"

    # Tool format
    if "tool" in keys or "tools" in keys:
        return "EASY - Tool format detected"

    # Unknown
    return "HARD - Unknown format, need custom parser"


def main():
    """Run discovery on all candidate datasets."""
    print("\n" + "=" * 70)
    print("🔍 HuggingFace Dataset Discovery for JEPA Pre-training")
    print("=" * 70)

    all_reports = {}

    for category, datasets in CANDIDATE_DATASETS.items():
        print(f"\n\n{'#' * 70}")
        print(f"# Category: {category.upper()}")
        print(f"{'#' * 70}")

        category_reports = []
        for dataset_id in datasets:
            report = discover_dataset(dataset_id, max_samples=2)
            category_reports.append(report)
            all_reports[dataset_id] = report

    # Summary
    print("\n\n" + "=" * 70)
    print("📈 DISCOVERY SUMMARY")
    print("=" * 70)

    for category, datasets in CANDIDATE_DATASETS.items():
        print(f"\n{category.upper()}:")
        successful = 0
        failed = 0
        total_samples = 0

        for dataset_id in datasets:
            report = all_reports[dataset_id]
            if report["status"] == "success":
                successful += 1
                total_samples += report.get("size_sampled", 0)
                print(f"  ✅ {dataset_id}")
                print(f"     Size: {report['size_sampled']} (sampled)")
                print(f"     Transform: {report['transformation_feasibility']}")
            else:
                failed += 1
                print(f"  ❌ {dataset_id}")
                print(f"     Error: {report['error']}")

        print(f"\n  Summary: {successful}/{successful+failed} successful")
        print(f"  Total samples (estimated from sample): {total_samples}")

    # Overall totals
    print(f"\n{'=' * 70}")
    total_success = sum(1 for r in all_reports.values() if r["status"] == "success")
    total_datasets = len(all_reports)
    total_samples_estimate = sum(r.get("size_sampled", 0) for r in all_reports.values() if r["status"] == "success")

    print(f"OVERALL: {total_success}/{total_datasets} datasets loaded successfully")
    print(f"ESTIMATED TOTAL SAMPLES (from 100-sample probes): {total_samples_estimate}")
    print(f"PROJECTED FULL SIZE: Could be 10M+ if we scale up")
    print("=" * 70)

    # Save report
    with open("outputs/dataset_discovery_report.json", "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n💾 Full report saved: outputs/dataset_discovery_report.json")


if __name__ == "__main__":
    main()
