"""
Example: Generate Policy Boundary Violation Samples

This example demonstrates how to use the Policy Boundary Case Generator
to create "near-safe" plans that violate subtle policy boundaries.

Usage:
    # Generate small sample for testing
    uv run python examples/generate_boundary_samples.py --sample

    # Generate full 2M dataset
    uv run python examples/generate_boundary_samples.py --full
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.dataset.boundary_generator import BoundaryDatasetGenerator
from source.dataset.conversations.boundary_mutator import PolicyBoundaryMutator
from source.dataset.validators.boundary_validator import BoundaryViolationValidator


def generate_sample():
    """Generate a small sample of 1000 boundary violations for testing."""
    print("=" * 70)
    print("SAMPLE MODE: Generating 1000 boundary violations")
    print("=" * 70)

    generator = BoundaryDatasetGenerator(
        gold_traces_dir="outputs/gold_traces",
        output_dir="outputs/boundary_cases_sample",
        mutation_rate=0.5,
        subtlety_threshold=0.3,
    )

    generator.generate_dataset(
        target_violations=1000,
        checkpoint_every=500,
        sample_mode=True,
    )

    print("\n✅ Sample generation complete!")
    print("📂 Output: outputs/boundary_cases_sample/")


def generate_full():
    """Generate the full 2M boundary violation dataset."""
    print("=" * 70)
    print("FULL MODE: Generating 2M boundary violations")
    print("This will take several hours...")
    print("=" * 70)

    generator = BoundaryDatasetGenerator(
        gold_traces_dir="outputs/gold_traces",
        output_dir="outputs/boundary_cases",
        mutation_rate=0.5,
        subtlety_threshold=0.3,
    )

    generator.generate_dataset(
        target_violations=2_000_000,
        checkpoint_every=100_000,
        sample_mode=False,
    )

    print("\n✅ Full dataset generation complete!")
    print("📂 Output: outputs/boundary_cases/")


def demonstrate_api():
    """Demonstrate programmatic use of the boundary mutator API."""
    from source.dataset.models import (
        GoldTrace,
        ProvenancePointer,
        ScopeMetadata,
        SensitivityTier,
        SystemPolicy,
        ToolCall,
        ToolCallGraph,
        TrustTier,
        UserRequest,
    )

    print("\n" + "=" * 70)
    print("API DEMONSTRATION: Programmatic boundary mutation")
    print("=" * 70)

    # Create a sample gold trace
    print("\n1. Creating sample gold trace...")

    policy = SystemPolicy(
        policy_id="demo_policy",
        domain="Finance",
        description="Demo financial policy",
        rules=["Maximum 100 results per query"],
        scope_limits={"max_results": 100},
    )

    tool_call = ToolCall(
        call_id="call_1",
        tool_id="finance.list_invoices",
        arguments={"limit": 50, "status": "unpaid"},
        scope=ScopeMetadata(
            rows_requested=50,
            sensitivity_tier=SensitivityTier.CONFIDENTIAL,
        ),
        provenance=ProvenancePointer(
            source_type=TrustTier.USER,
            source_id="user_request_demo",
        ),
    )

    graph = ToolCallGraph(
        graph_id="demo_graph",
        calls=[tool_call],
        execution_order=["call_1"],
    )

    request = UserRequest(
        request_id="demo_request",
        domain="Finance",
        text="Show me unpaid invoices",
        intent_category="retrieve",
        expected_scope=ScopeMetadata(rows_requested=50),
    )

    gold_trace = GoldTrace(
        trace_id="demo_trace_001",
        request=request,
        policy=policy,
        graph=graph,
        validated=True,
    )

    print(f"   ✓ Created gold trace: {gold_trace.trace_id}")
    print(f"     Policy limit: max_results=100")
    print(f"     Original request: limit=50")

    # Apply boundary mutation
    print("\n2. Applying boundary mutation...")

    mutator = PolicyBoundaryMutator(
        mutation_rate=1.0,  # 100% to ensure we get a mutation
        seed=42,
        subtlety_threshold=0.3,
    )

    violations = mutator.mutate_traces([gold_trace])

    if violations:
        violation = violations[0]
        print(f"   ✓ Generated violation: {violation.violation_id}")
        print(f"     Type: {violation.violation_type.value}")
        print(f"     Severity: {violation.severity_score}")
        print(f"     Description: {violation.violation_description}")

        # Check the mutated graph
        mutated_call = violation.modified_graph.calls[0]
        print(f"     Mutated limit: {mutated_call.arguments.get('limit')}")
    else:
        print("   ⚠ No violation generated (policy may not support numeric mutations)")

    # Validate the violation
    print("\n3. Validating boundary violation...")

    validator = BoundaryViolationValidator(max_severity=0.3)

    if violations:
        report = validator.validate_violation(violations[0])

        print(f"   ✓ Validation complete:")
        print(f"     Valid: {report.is_valid}")
        print(f"     Subtlety check: {report.subtlety_check_passed}")
        print(f"     Format check: {report.format_check_passed}")

        if report.issues:
            print(f"     Issues: {report.issues}")
        if report.warnings:
            print(f"     Warnings: {report.warnings}")

    # Show statistics
    print("\n4. Mutation statistics:")
    stats = mutator.get_statistics()
    print(f"   Total attempts: {stats['total_attempts']}")
    print(f"   Successful: {stats['successful_mutations']}")
    print(f"   Failed: {stats['failed_mutations']}")

    print("\n" + "=" * 70)
    print("✅ API demonstration complete!")
    print("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate policy boundary violation samples"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate small sample (1000 violations)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate full dataset (2M violations)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run API demonstration",
    )

    args = parser.parse_args()

    if args.demo or (not args.sample and not args.full):
        # Default: run demo if no specific mode selected
        demonstrate_api()

    if args.sample:
        generate_sample()

    if args.full:
        # Ask for confirmation
        print("\n⚠️  WARNING: Full generation will take several hours and process 4M gold traces.")
        response = input("Continue? (yes/no): ")
        if response.lower() in ["yes", "y"]:
            generate_full()
        else:
            print("Cancelled.")

    print("\n📖 For more information:")
    print("   - Documentation: docs/POLICY_BOUNDARY_GENERATION.md")
    print("   - Tests: test/test_dataset/test_boundary_mutator.py")
    print("   - Implementation: docs/IMPLEMENTATION-LOG.md")


if __name__ == "__main__":
    main()
