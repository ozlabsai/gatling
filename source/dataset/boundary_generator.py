"""
Policy Boundary Case Generator - Stage B of Gatling-10M Dataset.

Generates 2M "near-safe" plans that violate subtle policy boundaries.
These samples test the model's ability to enforce precise policy limits.

Pipeline:
    1. Load 4M gold traces from Stage A
    2. Apply systematic boundary mutations
    3. Validate violations are subtle but real
    4. Export to JSONL format for training

Target: 2M boundary violation samples
Strategy: 50% mutation rate on 4M gold traces
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from source.dataset.conversations.boundary_mutator import (
    BoundaryViolation,
    PolicyBoundaryMutator,
)
from source.dataset.models import GoldTrace


class BoundaryDatasetGenerator:
    """
    Orchestrates generation of 2M policy boundary violation samples.

    Manages:
    - Loading gold traces from Stage A
    - Applying boundary mutations
    - Quality validation
    - Output formatting and storage
    """

    def __init__(
        self,
        gold_traces_dir: str,
        output_dir: str = "outputs/boundary_cases",
        mutation_rate: float = 0.5,
        subtlety_threshold: float = 0.3,
    ):
        """
        Initialize the boundary generator.

        Args:
            gold_traces_dir: Directory containing gold traces from Stage A
            output_dir: Directory to save boundary violation samples
            mutation_rate: Fraction of traces to mutate (default: 50%)
            subtlety_threshold: Maximum severity score for violations
        """
        self.gold_traces_dir = Path(gold_traces_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mutator = PolicyBoundaryMutator(
            mutation_rate=mutation_rate,
            subtlety_threshold=subtlety_threshold,
        )

        # Track generation statistics
        self.stats = {
            "total_gold_traces_loaded": 0,
            "total_violations_generated": 0,
            "by_violation_type": {},
            "start_time": datetime.now(),
        }

    def load_gold_traces(self, limit: int | None = None) -> list[GoldTrace]:
        """
        Load gold traces from Stage A output.

        Args:
            limit: Optional limit on number of traces to load

        Returns:
            List of GoldTrace objects
        """
        traces = []

        print(f"📂 Loading gold traces from: {self.gold_traces_dir}")

        # Find all JSONL files in the directory
        jsonl_files = list(self.gold_traces_dir.glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL files found in {self.gold_traces_dir}. "
                "Please run Stage A gold trace generation first."
            )

        for jsonl_file in jsonl_files:
            print(f"  Loading: {jsonl_file.name}")

            with open(jsonl_file) as f:
                for line in f:
                    if limit and len(traces) >= limit:
                        break

                    try:
                        trace_data = json.loads(line)
                        # Reconstruct GoldTrace from training format
                        trace = self._reconstruct_gold_trace(trace_data)
                        traces.append(trace)
                    except Exception as e:
                        print(f"  Warning: Failed to load trace: {e}")
                        continue

            if limit and len(traces) >= limit:
                break

        self.stats["total_gold_traces_loaded"] = len(traces)
        print(f"✓ Loaded {len(traces):,} gold traces")

        return traces

    def _reconstruct_gold_trace(self, trace_data: dict[str, Any]) -> GoldTrace:
        """
        Reconstruct GoldTrace object from JSONL training format.

        Args:
            trace_data: Dictionary from JSONL file

        Returns:
            GoldTrace object
        """
        from source.dataset.models import (
            SystemPolicy,
            ToolCall,
            ToolCallGraph,
            UserRequest,
        )

        # Reconstruct nested objects
        request = UserRequest(**trace_data["request"])
        policy = SystemPolicy(**trace_data["policy"])

        # Reconstruct graph
        calls = [ToolCall(**call_data) for call_data in trace_data["graph"]["calls"]]
        graph = ToolCallGraph(
            graph_id=trace_data["graph"]["graph_id"],
            calls=calls,
            execution_order=trace_data["graph"].get("execution_order", []),
        )

        return GoldTrace(
            trace_id=trace_data["trace_id"],
            request=request,
            policy=policy,
            graph=graph,
            metadata=trace_data.get("metadata", {}),
            validated=trace_data.get("validated", False),
        )

    def generate_dataset(
        self,
        target_violations: int = 2_000_000,
        checkpoint_every: int = 100_000,
        sample_mode: bool = False,
    ) -> None:
        """
        Generate the full dataset of boundary violations.

        Args:
            target_violations: Target number of violations (default: 2M)
            checkpoint_every: Save checkpoints every N violations
            sample_mode: If True, generate small sample for testing
        """
        if sample_mode:
            print("🧪 SAMPLE MODE: Generating 1000 violations for testing")
            target_violations = 1000
            checkpoint_every = 500

        print(f"\n{'=' * 70}")
        print("🎯 Gatling Policy Boundary Case Generation - Stage B")
        print(f"{'=' * 70}")
        print(f"Target: {target_violations:,} boundary violations")
        print("Strategy: Systematic policy boundary mutations")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 70}\n")

        all_violations = []
        checkpoint_counter = 0

        # Load gold traces in batches to manage memory
        batch_size = 100_000 if not sample_mode else 2000
        traces_loaded = 0

        while len(all_violations) < target_violations:
            # Load next batch of gold traces
            print(f"\n📥 Loading batch of {batch_size:,} gold traces...")
            gold_traces = self.load_gold_traces(limit=batch_size)

            if not gold_traces:
                print("⚠️  No more gold traces available")
                break

            traces_loaded += len(gold_traces)

            # Generate boundary violations
            print("🔄 Applying boundary mutations...")
            batch_violations = self.mutator.mutate_traces(gold_traces)

            all_violations.extend(batch_violations)
            self.stats["total_violations_generated"] = len(all_violations)

            # Update type statistics
            mutator_stats = self.mutator.get_statistics()
            self.stats["by_violation_type"] = mutator_stats.get("by_violation_type", {})

            print(
                f"  ✓ Generated {len(batch_violations):,} violations "
                f"(Total: {len(all_violations):,}/{target_violations:,})"
            )

            # Checkpoint if needed
            if len(all_violations) >= checkpoint_counter + checkpoint_every:
                self._save_checkpoint(all_violations, checkpoint_counter)
                checkpoint_counter += checkpoint_every

            # Stop if we've reached target
            if len(all_violations) >= target_violations:
                break

            # In sample mode, only process one batch
            if sample_mode:
                break

        # Final save
        self._save_final_dataset(all_violations)
        self._print_final_stats(all_violations)

    def _save_checkpoint(self, violations: list[BoundaryViolation], checkpoint_num: int) -> None:
        """Save a checkpoint of violations generated so far."""
        checkpoint_path = self.output_dir / f"checkpoint_{checkpoint_num:07d}.jsonl"
        print(f"\n💾 Saving checkpoint: {checkpoint_path}")

        with open(checkpoint_path, "w") as f:
            for violation in violations[checkpoint_num:]:
                f.write(json.dumps(violation.model_dump()) + "\n")

    def _save_final_dataset(self, violations: list[BoundaryViolation]) -> None:
        """Save the final complete dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / f"boundary_violations_{len(violations)}_{timestamp}.jsonl"

        print(f"\n💾 Saving final dataset: {final_path}")

        with open(final_path, "w") as f:
            for violation in violations:
                f.write(json.dumps(violation.model_dump()) + "\n")

        # Also save metadata
        metadata_path = self.output_dir / f"metadata_{timestamp}.json"
        metadata = {
            "total_violations": len(violations),
            "generation_stats": self.stats,
            "mutator_stats": self.mutator.get_statistics(),
            "violation_type_distribution": self._compute_distribution(violations),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"💾 Saved metadata: {metadata_path}")

    def _compute_distribution(self, violations: list[BoundaryViolation]) -> dict[str, Any]:
        """Compute distribution statistics for violations."""
        type_counts = {}
        severity_buckets = {"very_subtle": 0, "subtle": 0, "moderate": 0}

        for v in violations:
            # Count by type
            vtype = v.violation_type.value
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

            # Bucket by severity
            if v.severity_score <= 0.15:
                severity_buckets["very_subtle"] += 1
            elif v.severity_score <= 0.25:
                severity_buckets["subtle"] += 1
            else:
                severity_buckets["moderate"] += 1

        return {
            "by_type": type_counts,
            "by_severity": severity_buckets,
            "avg_severity": sum(v.severity_score for v in violations) / len(violations)
            if violations
            else 0,
        }

    def _print_final_stats(self, violations: list[BoundaryViolation]) -> None:
        """Print final generation statistics."""
        duration = datetime.now() - self.stats["start_time"]
        distribution = self._compute_distribution(violations)

        print(f"\n{'=' * 70}")
        print("✅ Boundary Case Generation Complete!")
        print(f"{'=' * 70}")
        print(f"Total violations generated: {len(violations):,}")
        print(f"Gold traces processed: {self.stats['total_gold_traces_loaded']:,}")
        print(
            f"Success rate: {len(violations) / self.stats['total_gold_traces_loaded'] * 100:.1f}%"
        )
        print(f"Duration: {duration}")
        print(f"Rate: {len(violations) / duration.total_seconds():.2f} violations/sec")

        print("\nViolation type distribution:")
        for vtype, count in sorted(
            distribution["by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {vtype}: {count:,} ({count / len(violations) * 100:.1f}%)")

        print("\nSeverity distribution:")
        for severity, count in distribution["by_severity"].items():
            print(f"  {severity}: {count:,} ({count / len(violations) * 100:.1f}%)")
        print(f"  Average severity: {distribution['avg_severity']:.3f}")

        print(f"{'=' * 70}\n")


def main():
    """Main entry point for boundary case generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate policy boundary violation cases for Gatling Stage B"
    )
    parser.add_argument(
        "--gold-traces-dir",
        type=str,
        default="outputs/gold_traces",
        help="Directory containing gold traces from Stage A",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/boundary_cases",
        help="Output directory for boundary violations",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2_000_000,
        help="Target number of violations to generate (default: 2M)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="Save checkpoint every N violations",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.5,
        help="Fraction of traces to mutate (default: 0.5)",
    )
    parser.add_argument(
        "--subtlety-threshold",
        type=float,
        default=0.3,
        help="Maximum severity score for violations (default: 0.3)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: generate only 1000 violations for testing",
    )

    args = parser.parse_args()

    generator = BoundaryDatasetGenerator(
        gold_traces_dir=args.gold_traces_dir,
        output_dir=args.output_dir,
        mutation_rate=args.mutation_rate,
        subtlety_threshold=args.subtlety_threshold,
    )

    generator.generate_dataset(
        target_violations=args.target,
        checkpoint_every=args.checkpoint_every,
        sample_mode=args.sample,
    )


if __name__ == "__main__":
    main()
