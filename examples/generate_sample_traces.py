"""
Example: Generate sample gold traces.

This script demonstrates how to use the gold trace generation system
to create a small sample dataset for development and testing.
"""

import os
from pathlib import Path

from source.dataset.generator import GoldTraceGenerator
from source.dataset.oracle.agent import OracleAgent
from source.dataset.schemas.registry import DomainRegistry
from source.dataset.validators.trace_validator import TraceValidator


def example_1_generate_sample_dataset():
    """Example 1: Generate a small sample dataset across multiple domains."""
    print("=" * 70)
    print("Example 1: Generate Sample Dataset (100 traces)")
    print("=" * 70)

    # Initialize generator
    generator = GoldTraceGenerator(
        output_dir="outputs/examples/sample_traces",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Generate 100 traces in sample mode
    generator.generate_dataset(
        total_traces=100,
        checkpoint_every=50,
        sample_mode=True,
    )

    print("\n✓ Sample dataset generated in outputs/examples/sample_traces/")


def example_2_single_domain():
    """Example 2: Generate traces for a single domain."""
    print("\n" + "=" * 70)
    print("Example 2: Generate Traces for Finance Domain")
    print("=" * 70)

    oracle = OracleAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Generate 20 Finance traces
    traces = oracle.generate_traces_for_domain(
        domain="Finance",
        num_traces=20,
        batch_size=5,
    )

    print(f"\n✓ Generated {len(traces)} Finance traces")

    # Save traces
    output_path = "outputs/examples/finance_traces.jsonl"
    oracle.save_traces(traces, output_path)

    print(f"✓ Saved to {output_path}")


def example_3_explore_domains():
    """Example 3: Explore available domains and their tools."""
    print("\n" + "=" * 70)
    print("Example 3: Explore Domains")
    print("=" * 70)

    # Get all domains
    domains = DomainRegistry.get_all_domains()
    print(f"\nTotal domains: {len(domains)}")
    print(f"First 10 domains: {domains[:10]}")

    # Explore Finance domain
    print("\n--- Finance Domain ---")
    finance_tools = DomainRegistry.get_schemas_for_domain("Finance")
    print(f"Available tools: {len(finance_tools)}")
    for tool in finance_tools:
        print(f"  - {tool.tool_id}: {tool.description}")

    finance_policy = DomainRegistry.get_policy_for_domain("Finance")
    print(f"\nPolicy: {finance_policy.description}")
    print("Rules:")
    for rule in finance_policy.rules:
        print(f"  - {rule}")


def example_4_validate_traces():
    """Example 4: Generate and validate traces."""
    print("\n" + "=" * 70)
    print("Example 4: Generate and Validate Traces")
    print("=" * 70)

    oracle = OracleAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))
    validator = TraceValidator()

    # Generate traces for HR domain
    traces = oracle.generate_traces_for_domain(
        domain="HR",
        num_traces=10,
        batch_size=5,
    )

    print(f"\n✓ Generated {len(traces)} HR traces")

    # Validate each trace
    print("\nValidating traces...")
    validated = []
    for trace in traces:
        is_valid, report = validator.validate_trace(trace)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {trace.trace_id}: {'Valid' if is_valid else 'Invalid'}")

        if is_valid:
            validated.append(trace)
        else:
            print(f"    Errors: {report}")

    print(f"\nValidation rate: {len(validated)}/{len(traces)} traces passed")

    # Check dataset diversity
    if validated:
        diversity = validator.validate_dataset_diversity(validated)
        print("\nDiversity metrics:")
        print(f"  Unique intents: {diversity['unique_intents']}")
        print(f"  Unique tools: {diversity['unique_tools']}")
        print(f"  Intent distribution: {diversity['intent_distribution']}")


def example_5_trace_structure():
    """Example 5: Examine trace structure."""
    print("\n" + "=" * 70)
    print("Example 5: Examine Trace Structure")
    print("=" * 70)

    oracle = OracleAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Generate one trace
    traces = oracle.generate_traces_for_domain(
        domain="Email",
        num_traces=1,
        batch_size=1,
    )

    if traces:
        trace = traces[0]

        print("\n--- Trace Structure ---")
        print(f"Trace ID: {trace.trace_id}")
        print(f"\nUser Request:")
        print(f"  Domain: {trace.request.domain}")
        print(f"  Text: \"{trace.request.text}\"")
        print(f"  Intent: {trace.request.intent_category}")
        print(f"  Expected scope: {trace.request.expected_scope.rows_requested} rows")

        print(f"\nTool Call Graph:")
        print(f"  Number of calls: {len(trace.graph.calls)}")
        for call in trace.graph.calls:
            print(f"\n  Call: {call.call_id}")
            print(f"    Tool: {call.tool_id}")
            print(f"    Arguments: {call.arguments}")
            print(f"    Scope: {call.scope.rows_requested} rows, "
                  f"{call.scope.sensitivity_tier.value} sensitivity")
            print(f"    Provenance: {call.provenance.source_type.value}")
            print(f"    Dependencies: {call.dependencies or 'None'}")

        print(f"\nExecution order: {trace.graph.execution_order}")
        print(f"Validated: {trace.validated}")

        # Show training format
        print("\n--- Training Format (excerpt) ---")
        training_data = trace.to_training_format()
        print(f"Keys: {list(training_data.keys())}")
        print(f"Trace ID: {training_data['trace_id']}")
        print(f"Request text: {training_data['request']['text']}")


def main():
    """Run all examples."""
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export ANTHROPIC_API_KEY='your_key_here'")
        return

    # Create output directory
    Path("outputs/examples").mkdir(parents=True, exist_ok=True)

    # Run examples
    try:
        # Start with exploration (no API calls)
        example_3_explore_domains()

        # Then generate traces (requires API)
        example_2_single_domain()
        example_4_validate_traces()
        example_5_trace_structure()

        # Optionally run full sample generation
        # example_1_generate_sample_dataset()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
