## Conversation Dataset Sampling

This module implements sampling and transformation of real-world conversations from public datasets into ExecutionPlans for training the Gatling JEPA encoders.

## Overview

The conversation sampling pipeline bridges synthetic gold traces with real-world data by:

1. **Sampling** 10K conversations from WildChat-1M and LMSYS-Chat-1M
2. **Extracting** actionable intents from user messages using Claude
3. **Transforming** intents into structured ExecutionPlans
4. **Mutating** 20% of plans adversarially to create hard negatives

This provides diverse training data that combines human naturalness with structured execution semantics.

## Quick Start

### Generate Sample Dataset (100 conversations)

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your_key_here"

# Run sample mode
uv run python examples/sample_conversations.py --sample
```

### Generate Full 10K Dataset

```bash
uv run python examples/sample_conversations.py --num-samples 10000
```

### Custom Configuration

```bash
uv run python examples/sample_conversations.py \
  --num-samples 5000 \
  --wildchat-ratio 0.6 \
  --mutation-rate 0.25 \
  --output-dir ./data/conversations \
  --min-turns 2 \
  --max-turns 10
```

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Dataset Sources                           │
│  ┌──────────────────┐        ┌──────────────────┐           │
│  │  WildChat-1M     │        │  LMSYS-Chat-1M   │           │
│  │  (HuggingFace)   │        │  (HuggingFace)   │           │
│  └────────┬─────────┘        └────────┬─────────┘           │
└───────────┼─────────────────────────────┼────────────────────┘
            │                             │
            └──────────┬──────────────────┘
                       │
            ┌──────────▼──────────┐
            │ ConversationSampler │  Step 1: Sample & Filter
            │  - Streaming load   │
            │  - Turn filtering   │
            │  - De-duplication   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  IntentExtractor    │  Step 2: Extract Intents
            │  - Claude Sonnet    │
            │  - Action detection │
            │  - Scope inference  │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  PlanTransformer    │  Step 3: Create Plans
            │  - Domain mapping   │
            │  - Tool selection   │
            │  - Graph generation │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │ AdversarialMutator  │  Step 4: Create Hard Negatives
            │  - 20% mutation     │
            │  - 4 mutation types │
            │  - Energy violations│
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   Training Data     │
            │  - Benign plans     │
            │  - Mutated plans    │
            │  - JSONL output     │
            └─────────────────────┘
```

## Module Components

### 1. ConversationSampler

Samples conversations from HuggingFace datasets with streaming for memory efficiency.

**Features:**
- Streaming dataset loading (no memory explosion)
- Configurable dataset ratios (WildChat vs LMSYS)
- Turn-based filtering (min/max conversation length)
- De-duplication by conversation ID
- Preserves source metadata

**Example:**
```python
from source.dataset.conversations import ConversationSampler

sampler = ConversationSampler(cache_dir="./cache", seed=42)

conversations = sampler.sample_conversations(
    n_samples=10000,
    wildchat_ratio=0.5,  # 50% WildChat, 50% LMSYS
    min_turns=2,
    max_turns=None,
)

sampler.save_conversations(conversations, "conversations.jsonl")
```

### 2. IntentExtractor

Extracts actionable intents from user messages using Claude Sonnet 4.5.

**Features:**
- 8 intent categories (retrieve, create, update, delete, analyze, communicate, configure, authenticate)
- Confidence scoring
- Tool inference (what APIs would be needed)
- Scope hint extraction (limit, filters, date ranges)
- Batch processing for efficiency

**Intent Categories:**
- **retrieve**: Fetching data
- **create**: Creating new resources
- **update**: Modifying existing resources
- **delete**: Removing resources
- **analyze**: Computing/analyzing data
- **communicate**: Sending messages/notifications
- **configure**: Changing settings
- **authenticate**: Auth-related actions

**Example:**
```python
from source.dataset.conversations import IntentExtractor

extractor = IntentExtractor(api_key="your_key")

intent_map = extractor.extract_intents(conversations, batch_size=10)

# intent_map: dict[conversation_id, list[ActionIntent]]
for conv_id, intents in intent_map.items():
    for intent in intents:
        print(f"Intent: {intent.intent_category}")
        print(f"  Action: {intent.action_description}")
        print(f"  Tools: {intent.inferred_tools}")
        print(f"  Scope: {intent.scope_hints}")
        print(f"  Confidence: {intent.confidence}")
```

### 3. PlanTransformer

Transforms action intents into structured ExecutionPlans compatible with Gatling data models.

**Features:**
- Domain inference from intent context
- Tool selection based on intent category
- Argument inference from scope hints
- Provenance tracking (all USER trust tier)
- Policy assignment per domain

**Domain Mapping Heuristics:**
```python
Tool hints → Domain:
  "calendar" → Calendar
  "email" → Email
  "file" → FileStorage
  "database" → DatabaseManagement
  "invoice" → Finance

Intent category → Domain (fallback):
  "retrieve" → DatabaseManagement
  "communicate" → Email
  "analyze" → BusinessIntelligence
```

**Example:**
```python
from source.dataset.conversations import PlanTransformer

transformer = PlanTransformer(api_key="your_key")

plans = transformer.transform_intents(conversations, intent_map)

for plan in plans:
    print(f"Plan: {plan.plan_id}")
    print(f"  Domain: {plan.user_request.domain}")
    print(f"  Tool calls: {len(plan.graph.calls)}")
```

### 4. AdversarialMutator

Applies adversarial mutations to create hard negatives for energy-based model training.

**Mutation Types:**

1. **Scope Blow-up** (E_scope violation)
   - Expands data access limits: `limit=5 → limit=500`
   - Tests least privilege detection

2. **Privilege Escalation** (E_scope violation)
   - Escalates sensitivity: `INTERNAL → RESTRICTED`
   - Tests data sensitivity boundaries

3. **Provenance Corruption** (E_provenance violation)
   - Changes trust tier: `USER → UNVERIFIED_RAG`
   - Tests trust boundary enforcement

4. **Exfiltration** (E_flow violation)
   - Adds external data export step
   - Tests data flow anomaly detection

**Example:**
```python
from source.dataset.conversations import AdversarialMutator

mutator = AdversarialMutator(mutation_rate=0.2, seed=42)

benign_plans, mutated_plans = mutator.mutate_plans(plans)

print(f"Benign: {len(benign_plans)}")
print(f"Mutated: {len(mutated_plans)}")

for mutated in mutated_plans:
    print(f"Mutation: {mutated.mutation_type}")
    print(f"  Description: {mutated.mutation_description}")
```

## Output Format

### Conversation (Raw)

```json
{
  "conversation_id": "wildchat_00001",
  "source": "wildchat",
  "turns": [
    {
      "role": "user",
      "content": "Find my most recent invoice",
      "metadata": {"turn_idx": 0}
    },
    {
      "role": "assistant",
      "content": "I'll help you find your invoice.",
      "metadata": {"turn_idx": 1}
    }
  ],
  "metadata": {"original_index": 42}
}
```

### Execution Plan (Benign)

```json
{
  "plan_id": "wildchat_00001_intent_0",
  "conversation_id": "wildchat_00001",
  "source": "wildchat",
  "user_request": {
    "request_id": "wildchat_00001_intent_0",
    "domain": "Finance",
    "text": "Find my most recent invoice",
    "intent_category": "retrieve",
    "expected_scope": {
      "rows_requested": 1,
      "sensitivity_tier": "confidential"
    }
  },
  "inferred_policy": {
    "policy_id": "finance_policy_v1",
    "domain": "Finance",
    "rules": ["Users can only access invoices for their own department"],
    "scope_limits": {"max_results": 100}
  },
  "graph": {
    "graph_id": "graph_0",
    "calls": [
      {
        "call_id": "call_1",
        "tool_id": "finance.list_invoices",
        "arguments": {"limit": 1, "status": "recent"},
        "scope": {
          "rows_requested": 1,
          "sensitivity_tier": "confidential"
        },
        "provenance": {
          "source_type": "user",
          "source_id": "user_msg_0"
        }
      }
    ],
    "execution_order": ["call_1"]
  },
  "original_intent": {
    "turn_idx": 0,
    "user_message": "Find my most recent invoice",
    "intent_category": "retrieve",
    "action_description": "Retrieve most recent invoice",
    "inferred_tools": ["finance_api"],
    "scope_hints": {"limit": 1},
    "confidence": 0.95
  }
}
```

### Mutated Plan (Adversarial)

```json
{
  "plan_id": "wildchat_00001_intent_0_mutated_scope",
  "original_plan_id": "wildchat_00001_intent_0",
  "mutation_type": "scope_blowup",
  "mutation_description": "Expanded data scope from 1 to 500 rows",
  "is_adversarial": true,
  "execution_plan": {
    // Same structure as benign, but with mutated values
    "graph": {
      "calls": [{
        "scope": {
          "rows_requested": 500,  // ← Blown up from 1
          "sensitivity_tier": "confidential"
        }
      }]
    }
  }
}
```

## Integration with Training Pipeline

### Usage in JEPA Encoder Training

The conversation-derived plans complement synthetic gold traces:

```python
from source.dataset.conversations import ConversationSampler

# 1. Load conversation-based plans
conversation_plans = load_jsonl("outputs/conversations/execution_plans_benign.jsonl")
conversation_mutated = load_jsonl("outputs/conversations/execution_plans_mutated.jsonl")

# 2. Load synthetic gold traces
gold_traces = load_jsonl("outputs/gold_traces/traces.jsonl")

# 3. Combine for training
training_positive = conversation_plans + gold_traces  # All benign
training_negative = conversation_mutated  # Adversarial

# 4. Train JEPA encoders with InfoNCE loss
train_jepa(positive_samples=training_positive, negative_samples=training_negative)
```

### Energy Function Calibration

The adversarial mutations directly map to energy terms:

| Mutation Type | Energy Term | Purpose |
|---------------|-------------|---------|
| Scope Blow-up | E_scope | Calibrate data access limits |
| Privilege Escalation | E_scope | Calibrate sensitivity boundaries |
| Provenance Corruption | E_provenance | Calibrate trust tier handling |
| Exfiltration | E_flow | Calibrate data flow detection |

## Testing

```bash
# Run all conversation tests
uv run pytest test/test_dataset/test_conversations.py -v

# Run specific test class
uv run pytest test/test_dataset/test_conversations.py::TestConversationSampler -v

# Run with coverage
uv run pytest test/test_dataset/test_conversations.py --cov=source/dataset/conversations
```

## Cost Estimation

For 10K conversations with intent extraction:

- **API Calls**: ~1,000 calls (batch size 10)
- **Tokens**: ~10M input tokens, ~5M output tokens
- **Estimated Cost**: $100-$150 (Claude Sonnet 4.5 pricing)
- **Time**: ~10-15 minutes

Use `--sample` mode during development to minimize costs.

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-samples` | 100 | Total conversations to sample |
| `--sample` | False | Quick sample mode (100 conversations) |
| `--wildchat-ratio` | 0.5 | Fraction from WildChat (rest LMSYS) |
| `--mutation-rate` | 0.2 | Fraction of plans to mutate |
| `--output-dir` | `outputs/conversations` | Output directory |
| `--min-turns` | 2 | Minimum conversation turns |
| `--max-turns` | None | Maximum conversation turns |
| `--seed` | 42 | Random seed for reproducibility |

## Troubleshooting

### Dataset Loading Fails

**Problem**: `Failed to load wildchat: ...`

**Solutions**:
1. Check HuggingFace Hub status
2. Verify internet connection
3. Try alternative dataset mirror
4. Use cached version if available

### Intent Extraction Slow

**Problem**: Intent extraction taking too long

**Solutions**:
1. Reduce `--num-samples` for testing
2. Use `--sample` mode during development
3. Increase batch size (may increase API errors)
4. Check API rate limits

### Mutation Rate Not Respected

**Problem**: Getting more/fewer mutations than expected

**Cause**: Small sample size with integer rounding

**Solution**: Mutation count is `int(n * rate)`, so with n=5 and rate=0.2, you get 1 mutation (20%)

## Future Enhancements

- [ ] Multi-turn intent threading (track intent dependencies across turns)
- [ ] LLM-based graph generation (replace heuristic tool selection)
- [ ] Custom mutation strategies per domain
- [ ] Real-world attack pattern injection
- [ ] Cross-dataset conversation fusion
- [ ] Intent confidence thresholding
- [ ] Distributed sampling across API keys
- [ ] Streaming plan generation

## References

- [DATASET-WORKSTREAM.md](./DATASET-WORKSTREAM.md) - Complete SID pipeline
- [GOLD_TRACE_GENERATION.md](./GOLD_TRACE_GENERATION.md) - Synthetic trace generation
- [README.md](../README.md) - Project overview
- [sampler.py](../source/dataset/conversations/sampler.py) - Sampler implementation
- [intent_extractor.py](../source/dataset/conversations/intent_extractor.py) - Intent extractor
- [plan_transformer.py](../source/dataset/conversations/plan_transformer.py) - Plan transformer
- [mutator.py](../source/dataset/conversations/mutator.py) - Adversarial mutator
