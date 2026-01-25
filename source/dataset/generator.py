"""
Main generator for 4M gold traces.

Orchestrates the generation of the Gatling-10M dataset Stage A:
4M "Standard Utility" traces across 50+ domains.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from source.dataset.models import GoldTrace
from source.dataset.oracle.agent import OracleAgent
from source.dataset.schemas.registry import DomainRegistry
from source.dataset.validators.trace_validator import TraceValidator


class GoldTraceGenerator:
    """
    Orchestrates generation of 4M gold traces for Stage A.

    Manages:
    - Domain allocation (how many traces per domain)
    - Batch processing for API efficiency
    - Quality validation
    - Output formatting and storage
    """

    def __init__(self, output_dir: str = "outputs/gold_traces", api_key: str | None = None):
        """
        Initialize the generator.

        Args:
            output_dir: Directory to save generated traces
            api_key: Anthropic API key (optional, uses env var if not provided)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.oracle = OracleAgent(api_key=api_key)
        self.validator = TraceValidator()

        # Track generation statistics
        self.stats = {
            "total_attempted": 0,
            "total_generated": 0,
            "total_validated": 0,
            "failed_validation": 0,
            "by_domain": {},
            "start_time": datetime.now(),
        }

    def generate_dataset(
        self,
        total_traces: int = 4_000_000,
        checkpoint_every: int = 10_000,
        sample_mode: bool = False,
    ) -> None:
        """
        Generate the full dataset of gold traces.

        Args:
            total_traces: Total number of traces to generate (default: 4M)
            checkpoint_every: Save checkpoints every N traces
            sample_mode: If True, generate small sample for testing (100 traces)
        """
        if sample_mode:
            print("🧪 SAMPLE MODE: Generating 100 traces across 5 domains for testing")
            total_traces = 100
            checkpoint_every = 50

        domains = DomainRegistry.get_all_domains()
        traces_per_domain = total_traces // len(domains)

        print(f"\n{'=' * 70}")
        print(f"🚀 Gatling Gold Trace Generation - Stage A")
        print(f"{'=' * 70}")
        print(f"Target: {total_traces:,} traces")
        print(f"Domains: {len(domains)}")
        print(f"Traces per domain: ~{traces_per_domain:,}")
        print(f"Output: {self.output_dir}")
        print(f"{'=' * 70}\n")

        all_traces = []
        checkpoint_counter = 0

        for domain_idx, domain in enumerate(domains):
            print(f"\n[{domain_idx + 1}/{len(domains)}] Processing: {domain}")

            try:
                # Generate traces for this domain
                domain_traces = self.oracle.generate_traces_for_domain(
                    domain=domain,
                    num_traces=traces_per_domain if not sample_mode else 20,
                    batch_size=10,
                )

                # Validate traces
                validated_traces = []
                for trace in domain_traces:
                    is_valid, report = self.validator.validate_trace(trace)
                    if is_valid:
                        validated_traces.append(trace)
                        self.stats["total_validated"] += 1
                    else:
                        self.stats["failed_validation"] += 1
                        print(f"  ✗ Trace {trace.trace_id} failed validation")

                all_traces.extend(validated_traces)
                self.stats["total_generated"] += len(validated_traces)
                self.stats["by_domain"][domain] = len(validated_traces)

                print(
                    f"  ✓ Generated {len(validated_traces)} traces for {domain} "
                    f"(Total: {len(all_traces):,})"
                )

                # Checkpoint if needed
                if len(all_traces) >= checkpoint_counter + checkpoint_every:
                    self._save_checkpoint(all_traces, checkpoint_counter)
                    checkpoint_counter += checkpoint_every

            except Exception as e:
                print(f"  ✗ Error processing {domain}: {e}")
                continue

            # In sample mode, only process first 5 domains
            if sample_mode and domain_idx >= 4:
                break

            # Stop if we've reached the target
            if len(all_traces) >= total_traces:
                break

        # Final save
        self._save_final_dataset(all_traces)
        self._print_final_stats(all_traces)

    def _save_checkpoint(self, traces: list[GoldTrace], checkpoint_num: int) -> None:
        """Save a checkpoint of traces generated so far."""
        checkpoint_path = self.output_dir / f"checkpoint_{checkpoint_num:07d}.jsonl"
        print(f"\n💾 Saving checkpoint: {checkpoint_path}")

        with open(checkpoint_path, "w") as f:
            for trace in traces[checkpoint_num:]:
                f.write(json.dumps(trace.to_training_format()) + "\n")

    def _save_final_dataset(self, traces: list[GoldTrace]) -> None:
        """Save the final complete dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / f"gold_traces_{len(traces)}_{timestamp}.jsonl"

        print(f"\n💾 Saving final dataset: {final_path}")

        with open(final_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace.to_training_format()) + "\n")

        # Also save metadata
        metadata_path = self.output_dir / f"metadata_{timestamp}.json"
        metadata = {
            "total_traces": len(traces),
            "generation_stats": self.stats,
            "diversity_metrics": self.validator.validate_dataset_diversity(traces),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"💾 Saved metadata: {metadata_path}")

    def _print_final_stats(self, traces: list[GoldTrace]) -> None:
        """Print final generation statistics."""
        duration = datetime.now() - self.stats["start_time"]

        print(f"\n{'=' * 70}")
        print(f"✅ Generation Complete!")
        print(f"{'=' * 70}")
        print(f"Total traces generated: {len(traces):,}")
        print(f"Total validated: {self.stats['total_validated']:,}")
        print(f"Failed validation: {self.stats['failed_validation']:,}")
        print(f"Duration: {duration}")
        print(f"Rate: {len(traces) / duration.total_seconds():.2f} traces/sec")
        print(f"\nDomain breakdown:")
        for domain, count in sorted(
            self.stats["by_domain"].items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {domain}: {count:,} traces")

        # Diversity metrics
        diversity = self.validator.validate_dataset_diversity(traces)
        print(f"\nDiversity metrics:")
        print(f"  Unique domains: {diversity['unique_domains']}")
        print(f"  Unique intents: {diversity['unique_intents']}")
        print(f"  Unique tools: {diversity['unique_tools']}")
        print(f"  Avg calls per graph: {diversity['graph_complexity']['avg_calls']:.2f}")
        print(f"{'=' * 70}\n")


def main():
    """Main entry point for gold trace generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate gold traces for Gatling Stage A"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=4_000_000,
        help="Total number of traces to generate (default: 4M)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gold_traces",
        help="Output directory for traces",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save checkpoint every N traces",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: generate only 100 traces for testing",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)",
    )

    args = parser.parse_args()

    generator = GoldTraceGenerator(
        output_dir=args.output_dir,
        api_key=args.api_key,
    )

    generator.generate_dataset(
        total_traces=args.total,
        checkpoint_every=args.checkpoint_every,
        sample_mode=args.sample,
    )


if __name__ == "__main__":
    main()
