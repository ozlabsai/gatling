# JEPA Pre-training: Dataset Research & Sourcing

**Date**: 2026-01-28
**Goal**: Source 10M+ benign samples from existing HuggingFace datasets
**Cost Target**: $0-500 (vs $2000+ for generation)

## Strategy: Leverage Existing Curated Datasets

Instead of generating from scratch, we'll:
1. Load existing tool-use & instruction-following datasets
2. Transform to ExecutionPlan format
3. Use for JEPA pre-training (benign logs)

## Candidate Datasets Analysis

### Category 1: Tool-Use / Function Calling (High Priority)

These datasets contain tool invocations - **directly maps to ExecutionPlan format**.

| Dataset | Size (Estimate) | Focus | Priority | Notes |
|---------|----------------|-------|----------|-------|
| **Salesforce/xlam-function-calling-60k** | 60K | Function calling | ⭐⭐⭐ | Large, high-quality function calling |
| **nvidia/ToolScale** | Unknown | Tool scaling | ⭐⭐⭐ | NVIDIA quality |
| **RioLee/ToolPref-Pairwise-30K** | 30K | Tool preferences | ⭐⭐⭐ | Pairwise comparisons |
| **atasoglu/turkish-function-calling-20k** | 20K | Function calling | ⭐⭐ | Turkish language |
| **Voxel51/fiftyone-function-calling-14k** | 14K | Function calling | ⭐⭐ | Computer vision tools |
| **twinkle-ai/tw-function-call-reasoning-10k** | 10K | Reasoning + tools | ⭐⭐⭐ | Has reasoning chains |
| **NousResearch/hermes-function-calling-v1** | Unknown | Function calling | ⭐⭐⭐ | Nous quality |
| **interstellarninja/hermes_reasoning_tool_use** | Unknown | Reasoning + tools | ⭐⭐⭐ | Multi-step reasoning |
| **Nanbeige/ToolMind** | Unknown | Tool understanding | ⭐⭐ | Chinese-focused |
| **dongsheng/DTA-Tool** | Unknown | Tool dataset | ⭐⭐ | Unknown quality |
| **squeeze-ai-lab/ToolBank** | Unknown | Tool bank | ⭐⭐ | Need to investigate |
| **google/mobile-actions** | Unknown | Mobile actions | ⭐ | Specialized domain |
| **apple/mmau** | Unknown | Mobile actions | ⭐ | Specialized domain |
| **mercor/apex-agents** | Unknown | Agent examples | ⭐⭐ | Need to investigate |

**Estimated Total**: ~150K-500K samples

### Category 2: Agent Safety (Contains Both Benign + Adversarial)

These may have labeled adversarial examples we can separate.

| Dataset | Size (Estimate) | Focus | Priority | Notes |
|---------|----------------|-------|----------|-------|
| **nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0** | Unknown | Agent safety | ⭐⭐⭐ | May have adversarial labels! |

**Potential**: Could provide adversarial examples for DG-002

### Category 3: Instruction-Following (General Benign Behavior)

For learning prompt understanding and general compliance.

| Dataset | Size (Estimate) | Focus | Priority | Notes |
|---------|----------------|-------|----------|-------|
| **HuggingFaceH4/instruction-dataset** | Unknown | Instructions | ⭐⭐⭐ | HF official |
| **Muennighoff/natural-instructions** | Large | Natural instructions | ⭐⭐⭐ | Natural language |
| **nvidia/Nemotron-RL-instruction_following-structured_outputs** | Unknown | Structured output | ⭐⭐ | NVIDIA quality |
| **nvidia/Nemotron-Math-v2** | Unknown | Math reasoning | ⭐ | Specialized |
| **allenai/Dolci-Instruct-SFT** | 937 | Instruction SFT | ✅ | Already using |

**Estimated Total**: 100K-1M+ samples

### Category 4: Already In Use

| Dataset | Size | Status |
|---------|------|--------|
| allenai/Dolci-Instruct-SFT-Tool-Use | 937 | ✅ Used in DG-001 |
| llamafactory/reason-tool-use-demo-1500 | 1,500 | ✅ Used in DG-001 |
| allenai/WildChat-1M | 270K (of 1M) | ✅ Used in DG-001 |
| lmsys/lmsys-chat-1m | 270K (of 1M) | ✅ Used in DG-001 |

## Sourcing Strategy: 3-Tier Approach

### Tier 1: Quick Wins (This Week)

**Goal**: Get to 1M samples fast with minimal transformation

1. **Salesforce/xlam-function-calling-60k** (60K)
   - Direct function calling format
   - Easy transformation to ExecutionPlan

2. **RioLee/ToolPref-Pairwise-30K** (30K)
   - Tool preferences with reasoning
   - Extract tool invocations

3. **HuggingFaceH4/instruction-dataset** (100K+)
   - General instruction following
   - Map instructions to tool calls

4. **Scale existing** (600K)
   - WildChat: 270K → 500K
   - LMSYS: 270K → 500K

**Tier 1 Total**: ~1.2M samples, $0 cost

### Tier 2: Deep Dive (Week 2)

**Goal**: Add high-quality tool-use datasets

5. **nvidia/ToolScale** (investigate size)
6. **NousResearch/hermes-function-calling-v1** (investigate)
7. **interstellarninja/hermes_reasoning_tool_use** (investigate)
8. **twinkle-ai/tw-function-call-reasoning-10k** (10K)
9. **Voxel51/fiftyone-function-calling-14k** (14K)

**Tier 2 Total**: +200K-500K samples

### Tier 3: Full Coverage (Week 3)

**Goal**: Reach 10M target with comprehensive coverage

10. **Muennighoff/natural-instructions** (investigate - could be massive)
11. **nvidia/Nemotron-RL-instruction_following** (investigate)
12. **Scale WildChat/LMSYS further** (if needed)
13. **Generate remaining with DG-003** (if needed)

## Transformation Pipeline

### Step 1: Schema Detection

Each dataset has different formats:

```python
# Example formats we'll encounter:

# Format A: Function calling (xlam)
{
    "function": "get_weather",
    "arguments": {"location": "San Francisco"},
    "reasoning": "User asked for weather in SF"
}

# Format B: Conversation with tools (hermes)
{
    "messages": [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "function_call": {...}}
    ]
}

# Format C: Instruction-response (HF4)
{
    "instruction": "Calculate the sum of 1 and 2",
    "response": "The sum is 3"
}
```

### Step 2: Unified Transformation

```python
class UnifiedDatasetTransformer:
    """Transform any dataset format → ExecutionPlan."""

    def auto_detect_schema(self, sample: dict) -> str:
        """Detect dataset format."""
        if "function" in sample or "function_call" in sample:
            return "function_calling"
        elif "messages" in sample:
            return "conversation"
        elif "instruction" in sample:
            return "instruction"
        else:
            return "unknown"

    def transform(self, sample: dict, schema: str) -> ExecutionPlan:
        """Transform to ExecutionPlan based on schema."""
        if schema == "function_calling":
            return self._transform_function_calling(sample)
        elif schema == "conversation":
            return self._transform_conversation(sample)
        elif schema == "instruction":
            return self._transform_instruction(sample)
```

### Step 3: Quality Filtering

```python
def filter_quality(plan: ExecutionPlan) -> bool:
    """Filter low-quality samples."""
    # Remove:
    # - Empty plans
    # - Single-word instructions
    # - Malformed tool calls
    # - Duplicate samples

    if len(plan.nodes) == 0:
        return False

    if len(plan.nodes[0].arguments.get("instruction", "")) < 10:
        return False

    return True
```

## Implementation Plan

### Task 1: Dataset Discovery Script (1 hour)

**File**: `scripts/discover_hf_datasets.py`

```python
"""
Load each candidate dataset and report:
- Actual size
- Schema format
- Sample quality
- Transformation feasibility
"""

CANDIDATE_DATASETS = [
    "Salesforce/xlam-function-calling-60k",
    "nvidia/ToolScale",
    "RioLee/ToolPref-Pairwise-30K",
    # ... all candidates
]

for dataset_id in CANDIDATE_DATASETS:
    try:
        dataset = load_dataset(dataset_id, split="train")
        print(f"{dataset_id}: {len(dataset)} samples")
        print(f"Schema: {dataset[0].keys()}")
        print(f"Sample: {dataset[0]}")
    except Exception as e:
        print(f"{dataset_id}: FAILED - {e}")
```

### Task 2: Unified Transformer (2 hours)

**File**: `source/dataset/unified_transformer.py`

Implement transformation for each schema type.

### Task 3: Batch Loading & Filtering (1 hour)

**File**: `scripts/build_jepa_pretraining_dataset.py`

```python
"""
Load all Tier 1 datasets, transform, filter, save.

Output: data/jepa_pretraining_10m.jsonl
"""

# Load Tier 1
datasets = [
    load_and_transform("Salesforce/xlam-function-calling-60k"),
    load_and_transform("RioLee/ToolPref-Pairwise-30K"),
    load_and_transform("HuggingFaceH4/instruction-dataset"),
    scale_existing("allenai/WildChat-1M", target=500000),
    scale_existing("lmsys/lmsys-chat-1m", target=500000),
]

# Combine & filter
combined = []
for dataset in datasets:
    combined.extend([s for s in dataset if filter_quality(s)])

# Save
save_jsonl(combined, "data/jepa_pretraining_10m.jsonl")
```

## Expected Outcomes

### Dataset Composition (10M Target)

```
JEPA Pre-training Dataset (10M samples):
├─ Function Calling (1.5M)     15%
│  ├─ Salesforce xlam          60K
│  ├─ RioLee ToolPref          30K
│  ├─ Hermes function-calling  50K
│  ├─ ToolScale               100K
│  └─ Other function datasets 1.26M
│
├─ Tool Reasoning (500K)        5%
│  ├─ hermes_reasoning        100K
│  ├─ tw-function-reasoning    10K
│  └─ Other reasoning         390K
│
├─ Instruction Following (2M)  20%
│  ├─ HF4 instruction         500K
│  ├─ Natural instructions   1000K
│  └─ Nemotron instruction    500K
│
├─ Conversations (6M)          60%
│  ├─ WildChat scaled         3M
│  └─ LMSYS scaled            3M
│
└─ TOTAL: 10M samples
```

### Cost Comparison

| Approach | Cost | Timeline |
|----------|------|----------|
| **Original (100M synthetic)** | $50-100K | 12+ weeks |
| **Revised (10M synthetic)** | $2,000 | 4 weeks |
| **HF Sourcing (10M curated)** | **$0-500** | **2 weeks** |

**Savings**: $2,000 → $500 (75% reduction!)

## Next Steps

### This Week
1. ✅ Run discovery script on all 20 candidates (1 hour)
2. ✅ Implement unified transformer (2 hours)
3. ✅ Load Tier 1 datasets (1M samples, 4 hours compute)
4. ✅ Generate statistics & validate quality

### Next Week
5. Load Tier 2 datasets (2M samples)
6. Combine with DG-003 gold traces (4M)
7. Finalize 10M pre-training dataset
8. Begin LSA-004 implementation

### Week 3
9. Start JEPA pre-training on 10M benign logs
10. Parallelize with DG-002 fine-tuning dataset generation

---

**Status**: Research phase
**Priority**: P0 (critical path for training)
**Owner**: Research → Implementation
**ETA**: 2 weeks to 10M dataset
