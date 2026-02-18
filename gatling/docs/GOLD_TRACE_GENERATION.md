# Gold Trace Generation - Stage A Implementation

This document describes the implementation of **Stage A: Seed Trajectory Generation** from the Synthetic Integrity Dataset (SID) workstream. The system generates 4M benign tool-use traces across 50+ domains using Oracle Agents.

## Overview

The gold trace generation system produces high-quality, policy-compliant tool-use traces that serve as training data for the JEPA encoders. These traces establish the "Safe Valley" baseline for the energy-based integrity model.

## Architecture

### Core Components

1. **Data Models** (`source/dataset/models.py`)
   - `GoldTrace`: Complete trace (request + policy + graph)
   - `ToolCallGraph`: DAG of tool invocations
   - `ToolCall`: Individual tool execution
   - `UserRequest`: Natural language request
   - `SystemPolicy`: Domain-specific policies
   - `ToolSchema`: Tool/API definitions

2. **Oracle Agent** (`source/dataset/oracle/agent.py`)
   - High-quality AI agent (Claude Sonnet 4.5)
   - Three-phase generation:
     - Phase 1: Generate diverse user requests
     - Phase 2: Generate tool-call graphs
     - Phase 3: Validate policy compliance

3. **Schema Registry** (`source/dataset/schemas/registry.py`)
   - 50+ domain definitions
   - Tool schemas for each domain
   - Policy definitions

4. **Validators** (`source/dataset/validators/`)
   - Structural validation (DAG, dependencies)
   - Policy compliance checks
   - Minimal scope verification
   - Dataset diversity metrics

5. **Generator** (`source/dataset/generator.py`)
   - Orchestrates full 4M trace generation
   - Checkpoint management
   - Statistics tracking

## Usage

### Quick Start: Generate Sample Dataset

Generate a small sample (100 traces) for testing:

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your_key_here"

# Generate sample dataset
uv run python source/dataset/generator.py --sample
```

This generates ~100 traces across 5 domains and saves to `outputs/gold_traces/`.

### Generate Full 4M Dataset

Generate the complete dataset:

```bash
# Generate 4M traces
uv run python source/dataset/generator.py --total 4000000

# With custom output directory
uv run python source/dataset/generator.py \
  --total 4000000 \
  --output-dir ./datasets/gold_traces

# With checkpoints every 5K traces
uv run python source/dataset/generator.py \
  --total 4000000 \
  --checkpoint-every 5000
```

### Programmatic Usage

```python
from source.dataset.generator import GoldTraceGenerator

# Initialize generator
generator = GoldTraceGenerator(
    output_dir="outputs/gold_traces",
    api_key="your_anthropic_key"
)

# Generate dataset
generator.generate_dataset(
    total_traces=100,
    checkpoint_every=50,
    sample_mode=True
)
```

### Generate for Specific Domain

```python
from source.dataset.oracle.agent import OracleAgent

oracle = OracleAgent(api_key="your_key")

# Generate 1000 traces for Finance domain
traces = oracle.generate_traces_for_domain(
    domain="Finance",
    num_traces=1000,
    batch_size=10
)

# Save traces
oracle.save_traces(traces, "finance_traces.jsonl")
```

## Supported Domains

The system supports 50+ domains including:

**Core Business:** Finance, HR, Sales, Marketing, Legal

**Technical:** DevOps, Cloud Infrastructure, Database Management, API Management, Security

**Productivity:** Email, Calendar, File Storage, Project Management, Documentation

**Communication:** Messaging, Video Conferencing, Team Collaboration, Customer Support

**Data & Analytics:** Business Intelligence, Data Warehousing, Analytics, Reporting

**Specialized:** Healthcare, Education, E-commerce, Supply Chain, Manufacturing

**Emerging:** AI/ML Operations, IoT Management, Blockchain, AR/VR, Robotics

See `DomainRegistry.get_all_domains()` for the complete list.

## Output Format

Traces are saved as JSONL (JSON Lines), one trace per line:

```json
{
  "trace_id": "finance_00000001",
  "request": {
    "request_id": "finance_req_0",
    "domain": "Finance",
    "text": "Find my most recent unpaid invoice",
    "intent_category": "retrieve",
    "expected_scope": {
      "rows_requested": 1,
      "sensitivity_tier": "confidential"
    }
  },
  "policy": {
    "policy_id": "finance_policy_v1",
    "domain": "Finance",
    "rules": ["Users can only access invoices for their own department", ...],
    "scope_limits": {"max_results": 100}
  },
  "graph": {
    "graph_id": "graph_finance_req_0",
    "calls": [
      {
        "call_id": "call_1",
        "tool_id": "finance.list_invoices",
        "arguments": {"limit": 1, "status": "unpaid"},
        "scope": {
          "rows_requested": 1,
          "sensitivity_tier": "confidential"
        },
        "provenance": {
          "source_type": "user",
          "source_id": "finance_req_0"
        }
      }
    ],
    "execution_order": ["call_1"]
  },
  "validated": true,
  "created_at": "2026-01-25T18:00:00"
}
```

## Validation

Every trace undergoes multiple validation checks:

1. **Structural Validation**
   - Graph is a valid DAG (no cycles)
   - All dependencies reference existing calls
   - Execution order is topologically valid

2. **Policy Compliance**
   - No forbidden operations
   - Scope limits respected
   - All policy rules followed

3. **Minimal Scope**
   - Actual scope doesn't significantly exceed expected minimal scope
   - Flags over-fetching patterns

4. **Dataset Diversity**
   - Domain distribution
   - Intent category coverage
   - Tool usage variety
   - Scope variation

## Quality Metrics

The generator tracks and reports:

- Total traces generated
- Validation pass/fail rates
- Domain distribution
- Intent category distribution
- Tool usage statistics
- Average graph complexity
- Generation rate (traces/sec)

Example output:

```
✅ Generation Complete!
======================================================================
Total traces generated: 4,000,000
Total validated: 3,950,000
Failed validation: 50,000
Duration: 8:32:15
Rate: 130.5 traces/sec

Domain breakdown:
  Finance: 85,000 traces
  HR: 82,000 traces
  DevOps: 80,000 traces
  ...

Diversity metrics:
  Unique domains: 50
  Unique intents: 6
  Unique tools: 245
  Avg calls per graph: 1.8
```

## Testing

Run the test suite:

```bash
# Run all dataset tests
uv run pytest test/test_dataset/ -v

# Run with coverage
uv run pytest test/test_dataset/ --cov=source/dataset --cov-report=html

# Run specific test file
uv run pytest test/test_dataset/test_models.py -v
```

## Integration with Training Pipeline

The generated traces are used for JEPA encoder training:

1. **Governance Encoder** learns to map policies to latent space
2. **Execution Encoder** learns to map tool-call graphs to latent space
3. **Energy Functions** are calibrated on gold traces to define the "Safe Valley"

The traces establish baselines for:
- **E_scope**: Minimal necessary data access
- **E_hierarchy**: Trusted instruction sources
- **E_provenance**: Trust tier handling
- **E_flow**: Normal data flow patterns

## Architecture Decisions

### Why Three-Phase Generation?

1. **Phase 1 (High Temperature)**: Generates diverse, creative user requests
2. **Phase 2 (Low Temperature)**: Generates precise, policy-compliant plans
3. **Phase 3 (Zero Temperature)**: Validates consistency and compliance

This ensures both diversity and accuracy.

### Why Self-Validation?

The Oracle Agent validates its own outputs because:
- Ensures 100% policy compliance
- Catches edge cases in plan generation
- Provides quality feedback loop
- Reduces human validation burden

### Why Batch Processing?

- Optimizes API usage costs
- Maintains diversity within batches
- Allows parallel generation across domains
- Enables checkpoint recovery

## Cost Estimation

Generating 4M traces:

- **API Calls**: ~800K calls (batch size 10, 2 phases + validation)
- **Tokens**: ~8B input tokens, ~4B output tokens
- **Estimated Cost**: $40,000-$60,000 (based on Claude Sonnet 4.5 pricing)
- **Time**: ~8-10 hours at 130 traces/sec

Use `--sample` mode extensively during development to minimize costs.

## Troubleshooting

### "Could not parse JSON from response"

The Oracle Agent occasionally returns malformed JSON. The system retries automatically. If persistent:
- Lower batch size (`--batch-size 5`)
- Check API key validity
- Verify model availability

### "Failed validation: scope limit exceeded"

The Oracle Agent generated a non-compliant trace. This is expected and filtered out. If rate > 10%:
- Review policy definitions
- Check prompt clarity
- Verify tool schema accuracy

### Out of memory

For large-scale generation:
- Increase checkpoint frequency
- Run domain-by-domain
- Use distributed generation

## Next Steps

After generating gold traces:

1. **Stage B**: Implement Adversarial Mutation (Corrupter Agent)
2. **Stage C**: Implement Intent Mapping (Minimal Scope extraction)
3. **Stage D**: Implement Provenance Injection (Multi-tier retrieval)
4. **Training**: Use traces to train JEPA encoders

See `docs/DATASET-WORKSTREAM.md` for the complete pipeline.

## Contributing

When adding new domains:

1. Add to `DomainRegistry.get_all_domains()`
2. Implement `_get_{domain}_schemas()`
3. Implement `_get_{domain}_policy()`
4. Add tests in `test/test_dataset/test_schemas.py`

When modifying validation logic:

1. Update `TraceValidator` methods
2. Add tests in `test/test_dataset/test_validators.py`
3. Update this documentation

## References

- [DATASET-WORKSTREAM.md](./DATASET-WORKSTREAM.md) - Complete SID pipeline
- [README.md](../README.md) - Project overview
- [models.py](../source/dataset/models.py) - Data model definitions
- [agent.py](../source/dataset/oracle/agent.py) - Oracle Agent implementation
