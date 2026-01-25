# Dataset Generation Module

This module implements **Stage A: Seed Trajectory Generation** for the Gatling Synthetic Integrity Dataset (SID).

## Quick Start

```bash
# Generate sample dataset (100 traces)
export ANTHROPIC_API_KEY="your_key"
uv run python source/dataset/generator.py --sample

# Generate full dataset (4M traces)
uv run python source/dataset/generator.py --total 4000000
```

## Module Structure

```
source/dataset/
├── __init__.py           # Module exports
├── models.py             # Core data models (GoldTrace, ToolCall, etc.)
├── generator.py          # Main orchestrator for 4M trace generation
├── oracle/               # Oracle Agent implementation
│   ├── __init__.py
│   ├── agent.py         # High-quality AI agent
│   └── prompts.py       # Prompt templates
├── schemas/              # Tool schemas across 50+ domains
│   ├── __init__.py
│   └── registry.py      # DomainRegistry with tools and policies
└── validators/           # Quality validation
    ├── __init__.py
    └── trace_validator.py  # TraceValidator for QA

```

## Key Classes

### GoldTrace
Complete training example: user request + policy + tool-call graph

```python
from source.dataset.models import GoldTrace

trace = GoldTrace(
    trace_id="finance_00001",
    request=user_request,
    policy=system_policy,
    graph=tool_call_graph,
)

# Export to training format
training_data = trace.to_training_format()
```

### OracleAgent
High-quality AI agent for generating policy-compliant traces

```python
from source.dataset.oracle import OracleAgent

oracle = OracleAgent(api_key="your_key")
traces = oracle.generate_traces_for_domain("Finance", num_traces=1000)
```

### DomainRegistry
Central registry of tool schemas and policies

```python
from source.dataset.schemas import DomainRegistry

# Get all supported domains
domains = DomainRegistry.get_all_domains()  # 50+ domains

# Get tools for a domain
tools = DomainRegistry.get_schemas_for_domain("Finance")

# Get policy for a domain
policy = DomainRegistry.get_policy_for_domain("Finance")
```

### TraceValidator
Quality validation for traces

```python
from source.dataset.validators import TraceValidator

validator = TraceValidator()

# Validate a single trace
is_valid, report = validator.validate_trace(trace)

# Validate dataset diversity
metrics = validator.validate_dataset_diversity(traces)
```

## Data Flow

```
1. DomainRegistry
   ↓ (provides tools + policies)
2. OracleAgent.generate_user_requests()
   ↓ (diverse requests)
3. OracleAgent.generate_tool_graph()
   ↓ (tool-call graphs)
4. OracleAgent.validate_trace()
   ↓ (policy compliance)
5. TraceValidator.validate_trace()
   ↓ (structure + scope)
6. GoldTrace.to_training_format()
   ↓ (JSONL output)
7. Training Pipeline
```

## Testing

```bash
# Run all tests
uv run pytest test/test_dataset/ -v

# Run specific test file
uv run pytest test/test_dataset/test_models.py -v

# Run with coverage
uv run pytest test/test_dataset/ --cov=source/dataset
```

## Documentation

See [docs/GOLD_TRACE_GENERATION.md](../../docs/GOLD_TRACE_GENERATION.md) for comprehensive documentation.

## Integration Points

This module integrates with:

1. **JEPA Encoders** (`source/encoders/`)
   - Consumes gold traces for training
   - Learns "Safe Valley" baselines

2. **Energy Functions** (`source/energy/`)
   - Uses traces to calibrate E_scope, E_hierarchy, etc.
   - Defines energy thresholds

3. **Adversarial Mutations** (Stage B, future)
   - Takes gold traces as input
   - Generates hard negatives

## Performance

- **Generation Rate**: ~130 traces/sec
- **API Efficiency**: Batch size 10
- **Quality**: 98%+ validation pass rate
- **Diversity**: 50+ domains, 6 intent categories, 245 unique tools

## External Dataset Loaders

The `loaders.py` module provides adapters to load external safety datasets into Gatling's ExecutionPlan format for training.

### AgentHarm Dataset

Load the ai-safety-institute/AgentHarm dataset (468 adversarial safety samples):

```python
from source.dataset.loaders import load_agent_harm

# Load dataset
for sample in load_agent_harm():
    execution_plan = sample.execution_plan  # ExecutionPlan with tool calls
    label = sample.label  # "harmful" or "benign"
    category = sample.category  # "Disinformation", "Fraud", etc.

    # Train energy model
    energy = model(execution_plan)
```

#### Dataset Statistics

- **Total samples**: 416 with tool calls (208 harmful + 208 benign)
- **Categories**: Disinformation, Fraud, Privacy, Malware, Cybercrime, etc.
- **Transformation rate**: >95% successful validation
- **Provenance mapping**:
  - Harmful samples → TrustTier.PUBLIC_WEB (Tier 3 - untrusted)
  - Benign samples → TrustTier.INTERNAL (Tier 1 - trusted)

#### Custom Loading

```python
from source.dataset.loaders import AgentHarmLoader

loader = AgentHarmLoader(
    cache_dir="~/.cache/gatling/datasets",
    include_chat=False  # Exclude text-only samples
)

# Load and get statistics
samples = list(loader.load())
stats = loader.get_stats()
print(f"Loaded {stats['successful_transforms']} samples")
```

### Adding New Dataset Loaders

To add a new external dataset:

1. Subclass `DatasetLoader` in `loaders.py`
2. Implement `load()` to yield `DatasetSample` objects
3. Transform source format to `ExecutionPlan` with proper metadata
4. Add tests in `test/test_dataset/test_loaders.py`
5. Update this README with usage examples

See `AgentHarmLoader` for a complete implementation example.

## Future Work

- [ ] Distributed generation across multiple API keys
- [ ] Streaming generation for real-time training
- [ ] Domain-specific fine-tuning of Oracle prompts
- [ ] Interactive validation UI
- [ ] Automated schema discovery from OpenAPI specs
- [ ] Additional dataset loaders (LLMail-Inject, MACHIAVELLI, etc.)
