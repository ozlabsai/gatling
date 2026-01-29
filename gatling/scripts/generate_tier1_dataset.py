#!/usr/bin/env python3
"""
Generate Tier I (Benign) Dataset for Gatling Training.

This script aggregates benign tool-use samples from 20+ loaders across
multiple polecats (quartz, opal, jasper, topaz, ruby) to create the
4M "Standard Utility" baseline for the Gatling-10M dataset.

Tier I Purpose:
- Establish the "Safe Valley" for legitimate tool usage
- Train the energy model to recognize LOW energy patterns
- Provide contrast for adversarial samples (Tier II-IV)

Dataset Sources by Polecat:
    Quartz (6 function-calling loaders):
        - Salesforce/xlam-function-calling-60k
        - squeeze-ai-lab/ToolBank
        - NousResearch/hermes-function-calling-v1
        - Voxel51/fiftyone-function-calling-14k
        - twinkle-ai/tw-function-call-reasoning-10k
        - google/mobile-actions

    Opal (7 specialized loaders):
        - apple/mmau
        - nvidia/ToolScale
        - nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0 (benign only)
        - RioLee/ToolPref-Pairwise-30K
        - ykckevin/astra_sft
        - Nanbeige/ToolMind
        - atasoglu/turkish-function-calling-20k

    Jasper (9 instruction + reasoning loaders):
        - AgentHarm (benign subset)
        - GSM8K (math reasoning)
        - MATH (advanced math)
        - OpenHermes 2.5 (instruction following)
        - UltraChat 200K (conversational instructions)
        - APIBench (API calling)
        - Berkeley Function-Calling (function schemas)
        - ToolBench (tool use)
        - Helpful Instructions (benign baselines)

    Topaz (2 conversation loaders):
        - lmsys/lmsys-chat-1m (sampled)
        - allenai/WildChat-1M (sampled)

    Ruby (1 loader):
        - AgentHarm (benign subset)

Target: 4M samples (best effort from available loaders)
"""

import argparse
import json
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

# Add source to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def import_loaders():
    """
    Import all loader classes from different polecats.

    Returns:
        Dictionary mapping loader names to loader classes
    """
    loaders = {}

    # Import Quartz loaders (function-calling)
    try:
        # Path: gatling/scripts -> gatling/polecats
        polecats_root = Path(__file__).parent.parent / "polecats"
        sys.path.insert(
            0, str(polecats_root / "quartz" / "gatling")
        )
        from source.dataset.function_calling_loaders import (
            SalesforceXLAMLoader,
            ToolBankLoader,
            HermesFunctionCallingLoader,
            FiftyOneFunctionCallingLoader,
            TwinkleFunctionCallingLoader,
            MobileActionsLoader,
        )

        loaders["salesforce_xlam"] = SalesforceXLAMLoader
        loaders["toolbank"] = ToolBankLoader
        loaders["hermes_function_calling"] = HermesFunctionCallingLoader
        loaders["fiftyone"] = FiftyOneFunctionCallingLoader
        loaders["twinkle"] = TwinkleFunctionCallingLoader
        loaders["mobile_actions"] = MobileActionsLoader
        print("✓ Imported 6 loaders from Quartz")
    except Exception as e:
        print(f"⚠ Warning: Could not import Quartz loaders: {e}")

    # Import Opal loaders (specialized)
    try:
        polecats_root = Path(__file__).parent.parent / "polecats"
        opal_path = str(polecats_root / "opal" / "gatling")
        if opal_path not in sys.path:
            sys.path.insert(0, opal_path)

        # Clear ALL cached source modules to avoid namespace collision
        # This is critical because opal's __init__.py imports from source.dataset.loaders
        # and we need to ensure it uses opal's version, not main gatling's
        for mod in list(sys.modules.keys()):
            if mod.startswith('source'):
                del sys.modules[mod]

        from source.dataset.specialized_loaders import (
            AppleMMauLoader,
            NvidiaToolScaleLoader,
            NvidiaNeMotronSafetyLoader,
            ToolPrefPairwiseLoader,
            AstraSFTLoader,
            ToolMindLoader,
            TurkishFunctionCallingLoader,
        )

        loaders["apple_mmau"] = AppleMMauLoader
        loaders["nvidia_toolscale"] = NvidiaToolScaleLoader
        loaders["nvidia_nemotron"] = NvidiaNeMotronSafetyLoader
        loaders["toolpref"] = ToolPrefPairwiseLoader
        loaders["astra_sft"] = AstraSFTLoader
        loaders["toolmind"] = ToolMindLoader
        loaders["turkish_function_calling"] = TurkishFunctionCallingLoader
        print("✓ Imported 7 loaders from Opal")
    except Exception as e:
        print(f"⚠ Warning: Could not import Opal loaders: {e}")

    # Import Jasper loaders (instruction + reasoning - 9 loaders)
    try:
        polecats_root = Path(__file__).parent.parent / "polecats"
        jasper_path = str(polecats_root / "jasper" / "gatling")
        if jasper_path not in sys.path:
            sys.path.insert(0, jasper_path)

        # Clear cached modules to avoid namespace collision with other polecats
        for mod in list(sys.modules.keys()):
            if mod.startswith('source'):
                del sys.modules[mod]

        from source.dataset.loaders import (
            AgentHarmLoader as JasperAgentHarmLoader,
            GSM8KLoader,
            MATHLoader,
            InstructionDatasetLoader,
            INSTRUCTION_DATASET_CONFIGS,
        )

        # Add math + adversarial loaders
        loaders["jasper_agentharm"] = JasperAgentHarmLoader
        loaders["gsm8k"] = GSM8KLoader
        loaders["math"] = MATHLoader

        # Add instruction dataset loaders (6 datasets via InstructionDatasetLoader)
        # These use the generic InstructionDatasetLoader with dataset name parameter
        loaders["jasper_instruction_configs"] = INSTRUCTION_DATASET_CONFIGS
        loaders["jasper_instruction_loader_class"] = InstructionDatasetLoader

        print("✓ Imported 9 loaders from Jasper (3 direct + 6 instruction datasets)")
    except Exception as e:
        print(f"⚠ Warning: Could not import Jasper loaders: {e}")

    # Import Topaz loaders (conversations)
    try:
        polecats_root = Path(__file__).parent.parent / "polecats"
        topaz_path = str(polecats_root / "topaz" / "gatling")
        if topaz_path not in sys.path:
            sys.path.insert(0, topaz_path)

        # Clear any cached imports to avoid conflicts with other polecats
        # Must clear entire source tree (not just source.dataset or source.dataset.loaders)
        # because parent modules cache metadata about whether child modules are packages or files
        for mod in list(sys.modules.keys()):
            if mod.startswith('source'):
                del sys.modules[mod]

        from source.dataset.loaders.conversations import LMSYSLoader, WildChatLoader

        loaders["lmsys"] = LMSYSLoader
        loaders["wildchat"] = WildChatLoader
        print("✓ Imported 2 loaders from Topaz")
    except Exception as e:
        import traceback
        print(f"⚠ Warning: Could not import Topaz loaders: {e}")
        print("Full traceback:")
        traceback.print_exc()

    # Import Ruby loaders (base)
    try:
        from source.dataset.loaders import AgentHarmLoader

        loaders["ruby_agentharm"] = AgentHarmLoader
        print("✓ Imported 1 loader from Ruby")
    except Exception as e:
        print(f"⚠ Warning: Could not import Ruby loaders: {e}")

    return loaders


def initialize_loader(loader_class, loader_name: str, cache_dir: str, max_samples: int | None):
    """
    Initialize a loader with appropriate parameters.

    Handles different loader signatures (cache_dir, max_samples, include_chat, splits, etc.)

    Args:
        loader_class: The loader class to initialize
        loader_name: Name of the loader for logging
        cache_dir: Cache directory for HuggingFace datasets
        max_samples: Maximum samples to load (None = all)

    Returns:
        Initialized loader instance or None if initialization fails
    """
    try:
        # Check what parameters the loader accepts
        import inspect

        sig = inspect.signature(loader_class.__init__)
        params = set(sig.parameters.keys()) - {"self"}

        # Build kwargs based on supported parameters
        kwargs = {}
        if "cache_dir" in params:
            kwargs["cache_dir"] = cache_dir
        if "max_samples" in params:
            kwargs["max_samples"] = max_samples
        if "include_chat" in params:
            kwargs["include_chat"] = False  # Only tool-use samples for Tier I
        # Note: 'splits' parameter handled separately for ToolBank

        loader = loader_class(**kwargs)
        return loader

    except Exception as e:
        print(f"✗ Failed to initialize {loader_name}: {e}")
        return None


def generate_tier1_dataset(
    target_samples: int = 4_000_000,
    output_file: str = "data/tier1_benign_4m.jsonl",
    cache_dir: str = ".cache/huggingface",
    sample_mode: bool = False,
    track_filter: str = "all",
):
    """
    Generate Tier I benign dataset from all available loaders.

    Args:
        target_samples: Target number of samples (default: 4M)
        output_file: Output JSONL file path
        cache_dir: HuggingFace cache directory
        sample_mode: If True, generate only 1000 samples for testing
        track_filter: Filter by polecat track (all, quartz, opal, jasper, topaz, ruby)
    """
    if sample_mode:
        print("\n🧪 SAMPLE MODE: Generating 1,000 samples for testing")
        target_samples = 1000
        output_file = "data/tier1_sample_1k.jsonl"

    print(f"\n{'=' * 80}")
    print(f"🚀 Gatling Tier I Dataset Generation")
    print(f"{'=' * 80}")
    print(f"Target: {target_samples:,} samples")
    print(f"Output: {output_file}")
    print(f"Cache: {cache_dir}")
    print(f"{'=' * 80}\n")

    # Import all loaders
    print("📦 Importing loaders from polecats...")
    loaders = import_loaders()

    if not loaders:
        print("\n✗ ERROR: No loaders could be imported!")
        return 1

    print(f"\n✓ Successfully imported {len(loaders)} loaders\n")

    # Filter loaders by track if specified
    if track_filter != "all":
        # Map loader names to their polecat tracks
        track_mapping = {
            # Quartz loaders
            "salesforce_xlam": "quartz",
            "toolbank": "quartz",
            "hermes_function_calling": "quartz",
            "fiftyone": "quartz",
            "twinkle": "quartz",
            "mobile_actions": "quartz",
            # Opal loaders
            "apple_mmau": "opal",
            "nvidia_toolscale": "opal",
            "nvidia_nemotron": "opal",
            "toolpref": "opal",
            "astra_sft": "opal",
            "toolmind": "opal",
            "turkish_function_calling": "opal",
            # Jasper loaders
            "jasper_agentharm": "jasper",
            "gsm8k": "jasper",
            "math": "jasper",
            "jasper_instruction_configs": "jasper",
            "jasper_instruction_loader_class": "jasper",
            # Topaz loaders
            "lmsys": "topaz",
            "wildchat": "topaz",
            # Ruby loaders
            "ruby_agentharm": "ruby",
        }

        # Filter loaders
        original_count = len(loaders)
        loaders = {
            name: loader_class
            for name, loader_class in loaders.items()
            if track_mapping.get(name) == track_filter
        }
        print(f"🔍 Filtered to {len(loaders)} loaders for track '{track_filter}' (from {original_count})\n")

    # Initialize loaders
    print("🔧 Initializing loaders...")
    initialized_loaders = {}

    # Extract Jasper instruction dataset configs if present
    instruction_configs = loaders.pop("jasper_instruction_configs", None)
    instruction_loader_class = loaders.pop("jasper_instruction_loader_class", None)

    # Initialize regular loaders
    for name, loader_class in loaders.items():
        # Calculate max_samples per loader (distribute evenly)
        max_per_loader = target_samples // len(loaders) if not sample_mode else 100

        loader = initialize_loader(loader_class, name, cache_dir, max_per_loader)
        if loader:
            initialized_loaders[name] = loader
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")

    # Initialize Jasper instruction datasets (4 working datasets)
    if instruction_configs and instruction_loader_class:
        print("\n  Initializing Jasper instruction datasets...")
        working_datasets = ["openhermes", "ultrachat", "helpful-instructions"]
        max_per_instruction = target_samples // (len(loaders) + len(working_datasets)) if not sample_mode else 100

        for dataset_name in working_datasets:
            try:
                loader = instruction_loader_class(
                    dataset_name=dataset_name,
                    cache_dir=cache_dir,
                    max_samples=max_per_instruction
                )
                initialized_loaders[f"jasper_{dataset_name}"] = loader
                print(f"  ✓ jasper_{dataset_name}")
            except Exception as e:
                print(f"  ✗ jasper_{dataset_name}: {e}")

    print(f"\n✓ Initialized {len(initialized_loaders)} loaders total\n")

    # Collect samples
    print("📊 Loading samples from all datasets...")
    all_samples = []
    loader_stats = {}

    for loader_name, loader in initialized_loaders.items():
        print(f"\n  Loading {loader_name}...")
        try:
            samples_from_loader = []
            for sample in loader.load():
                # Only include benign samples for Tier I
                if sample.label in ["benign", "harmless_benign"]:
                    samples_from_loader.append(sample)

                    # Stop early in sample mode
                    if sample_mode and len(samples_from_loader) >= 100:
                        break

            all_samples.extend(samples_from_loader)
            loader_stats[loader_name] = len(samples_from_loader)
            print(f"    ✓ Loaded {len(samples_from_loader):,} samples")

        except Exception as e:
            print(f"    ✗ Error loading {loader_name}: {e}")
            loader_stats[loader_name] = 0

    print(f"\n✓ Total samples collected: {len(all_samples):,}")

    # Save to JSONL
    print(f"\n💾 Saving dataset to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in all_samples:
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
                        for node in sample.execution_plan.nodes
                    ],
                    "edges": sample.execution_plan.edges,
                },
                "label": sample.label,
                "original_id": sample.original_id,
                "category": sample.category,
                "metadata": {
                    k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in sample.metadata.items()
                },
            }
            f.write(json.dumps(sample_dict, default=str) + "\n")

    print(f"✓ Saved {len(all_samples):,} samples")

    # Save metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = output_path.parent / f"{output_path.stem}.metadata.json"

    metadata = {
        "timestamp": timestamp,
        "tier": "tier_1_standard_utility",
        "target_samples": target_samples,
        "actual_samples": len(all_samples),
        "sample_mode": sample_mode,
        "output_file": str(output_path),
        "cache_dir": cache_dir,
        "loader_statistics": loader_stats,
        "total_loaders_attempted": len(loaders),
        "total_loaders_successful": len(initialized_loaders),
        "cost": "$0 (using HuggingFace datasets)",
        "strategy": "Multi-polecat aggregation of benign tool-use samples",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"💾 Saved metadata: {metadata_path}")

    # Print final statistics
    print(f"\n{'=' * 80}")
    print(f"✅ Tier I Dataset Generation Complete!")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(all_samples):,}")
    print(f"Target: {target_samples:,}")
    print(f"Coverage: {len(all_samples) / target_samples * 100:.1f}%")
    print(f"\nSamples by loader:")
    for loader_name, count in sorted(loader_stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  - {loader_name}: {count:,}")

    print(f"\n💰 Cost: $0 (using HuggingFace datasets)")
    print(f"🚀 Ready for JEPA encoder training!")
    print(f"{'=' * 80}\n")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Tier I benign dataset for Gatling training"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=4_000_000,
        help="Target number of samples (default: 4M)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tier1_benign_4m.jsonl",
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
        help="Sample mode: generate only 1,000 samples for testing",
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["all", "quartz", "opal", "jasper", "topaz", "ruby"],
        default="all",
        help="Filter by polecat track (default: all)",
    )

    args = parser.parse_args()

    return generate_tier1_dataset(
        target_samples=args.target,
        output_file=args.output,
        cache_dir=args.cache_dir,
        sample_mode=args.sample_mode,
        track_filter=args.track,
    )


if __name__ == "__main__":
    sys.exit(main())
