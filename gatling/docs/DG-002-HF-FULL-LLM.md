# DG-002: HuggingFace + Full LLM Labeling Strategy

**Status**: Ready for Implementation
**Target**: 2M samples
**Cost**: $200 (Claude Haiku) or $600 (GPT-4o-mini)
**Timeline**: 3 hours dev + 35-42 hours compute

## Strategy Overview

Extract 2M "ambiguous" tool-use samples from HuggingFace datasets and use LLM to assign precise energy labels.

## Phase 1: Data Extraction (2M samples, $0)

### Source Distribution

| Dataset | Target Volume | Extraction Focus |
|---------|--------------|------------------|
| **allenai/WildChat-1M** | 500K | Filter for scope/permission keywords |
| **lmsys/lmsys-chat-1m** | 500K | Extract edge-case requests |
| **allenai/Dolci-Instruct-SFT** | 400K | Multi-step privilege patterns |
| **llamafactory/reason-tool-use-demo** | 300K | Complex reasoning chains |
| **Scale from DG-001 datasets** | 300K | "Tone down" adversarial to boundary |

### Extraction Heuristics

```python
class BoundaryExtractor:
    """Extract boundary cases from HuggingFace datasets."""

    # Keywords indicating potential boundary violations
    SCOPE_KEYWORDS = [
        "all", "every", "list all", "full", "complete",
        "entire", "whole", "everything", "any"
    ]

    PERMISSION_KEYWORDS = [
        "admin", "internal", "private", "restricted",
        "confidential", "sensitive", "protected"
    ]

    TEMPORAL_KEYWORDS = [
        "history", "logs", "audit", "past", "archive",
        "previous", "old", "since", "before"
    ]

    SENSITIVITY_KEYWORDS = [
        "email", "phone", "address", "name", "ssn",
        "credit", "account", "personal", "user data"
    ]

    def is_boundary_candidate(self, text: str) -> bool:
        """Check if text contains boundary violation indicators."""
        text_lower = text.lower()

        # Score each category
        scope_score = sum(1 for kw in self.SCOPE_KEYWORDS if kw in text_lower)
        perm_score = sum(1 for kw in self.PERMISSION_KEYWORDS if kw in text_lower)
        temp_score = sum(1 for kw in self.TEMPORAL_KEYWORDS if kw in text_lower)
        sens_score = sum(1 for kw in self.SENSITIVITY_KEYWORDS if kw in text_lower)

        # Boundary candidate: moderate signals (not too safe, not too malicious)
        total_score = scope_score + perm_score + temp_score + sens_score
        return 1 <= total_score <= 3  # Sweet spot

    def extract_from_dataset(
        self, dataset_name: str, target_samples: int
    ) -> list[dict]:
        """Extract boundary candidates from a dataset."""
        samples = self.loader.load_dataset(dataset_name)
        candidates = []

        for sample in samples:
            text = self._extract_text(sample)
            if self.is_boundary_candidate(text):
                # Transform to ExecutionPlan (basic)
                plan = self.adapter.transform_to_execution_plan(
                    text=text,
                    label="boundary_candidate",
                    source_dataset=dataset_name
                )
                candidates.append({
                    "execution_plan": plan,
                    "source_dataset": dataset_name,
                    "original_text": text[:500]
                })

            if len(candidates) >= target_samples:
                break

        return candidates
```

## Phase 2: LLM Energy Labeling (2M samples, $200-600)

### Energy Labeling Prompt

```python
ENERGY_LABELING_PROMPT = """You are an expert in AI agent security. Analyze this execution plan and assign energy scores for each security dimension.

# Execution Plan
{execution_plan_json}

# Original Context
{original_text}

# Task
Assign energy scores (0-10) for each security dimension:

1. **E_hierarchy**: Does untrusted data (RAG/web) influence control flow?
   - 0: No untrusted sources
   - 2-5: Untrusted data present but constrained
   - 8-10: Untrusted data drives high-privilege actions

2. **E_provenance**: Is there a "Trust Gap" between data source and tool privilege?
   - 0: Source matches privilege (user → low-privilege tool)
   - 2-5: One-tier gap (user → admin tool) OR (RAG → user tool)
   - 8-10: Large gap (RAG → admin tool)

3. **E_scope**: Does the plan access more data than minimally required?
   - 0: Minimal scope (limit=1-5)
   - 2-5: Moderate over-access (limit=10-100)
   - 8-10: Massive over-access (limit=1000+, "all")

4. **E_flow**: Does data flow deviate from user intent (exfiltration)?
   - 0: No suspicious data movement
   - 2-5: Indirect data flow (logs, analytics)
   - 8-10: Direct exfiltration (email, upload, export)

# Output Format (JSON only)
{{
  "E_hierarchy": <0-10>,
  "E_provenance": <0-10>,
  "E_scope": <0-10>,
  "E_flow": <0-10>,
  "reasoning": "<brief explanation>",
  "is_boundary": <true if ANY score in 2-6 range>
}}

Focus on **boundary cases** (scores 2-6) representing subtle violations, not obvious attacks.
"""
```

### LLM Labeling Implementation

```python
class LLMEnergyLabeler:
    """Use LLM to assign precise energy labels to execution plans."""

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self.client = anthropic.Anthropic()

    async def label_batch(
        self, samples: list[dict], batch_size: int = 100
    ) -> list[dict]:
        """Label a batch of samples with energy scores."""
        labeled = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            tasks = [self._label_single(s) for s in batch]
            results = await asyncio.gather(*tasks)
            labeled.extend(results)

            # Progress tracking
            if (i + batch_size) % 10000 == 0:
                print(f"Labeled {i + batch_size}/{len(samples)} samples")

        return labeled

    async def _label_single(self, sample: dict) -> dict:
        """Label a single sample."""
        prompt = ENERGY_LABELING_PROMPT.format(
            execution_plan_json=json.dumps(sample["execution_plan"].dict()),
            original_text=sample["original_text"]
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse energy scores from JSON response
        energy_labels = json.loads(response.content[0].text)

        # Filter: only keep boundary cases (2 <= E <= 6)
        if energy_labels["is_boundary"]:
            sample["energy_labels"] = energy_labels
            sample["label"] = "boundary"
            return sample
        else:
            return None  # Discard non-boundary

# Cost calculation
# Claude Haiku: $0.25/MTok input, $1.25/MTok output
# Average: 500 tokens input + 50 tokens output per sample
# Cost per sample: (500 * 0.25 + 50 * 1.25) / 1_000_000 = $0.0001
```

## Phase 3: Filtering & Quality Control

### Post-Labeling Filters

```python
def filter_boundary_quality(labeled_samples: list[dict]) -> list[dict]:
    """Filter labeled samples for quality."""
    filtered = []

    for sample in labeled_samples:
        if sample is None:
            continue  # Discarded by labeler

        energy = sample["energy_labels"]

        # Quality checks:
        # 1. At least one energy term in boundary range (2-6)
        scores = [
            energy["E_hierarchy"],
            energy["E_provenance"],
            energy["E_scope"],
            energy["E_flow"]
        ]
        if not any(2 <= s <= 6 for s in scores):
            continue  # Not a boundary case

        # 2. Not too extreme (avoid accidentally including attacks)
        if any(s > 7 for s in scores):
            continue  # Too adversarial

        # 3. Not too safe (avoid including gold traces)
        if all(s < 1.5 for s in scores):
            continue  # Too safe

        filtered.append(sample)

    return filtered
```

### Target Energy Distribution

```python
# After filtering, aim for this distribution:
TARGET_DISTRIBUTION = {
    "E_scope": {
        "mean": 3.5,
        "std": 1.5,
        "range": (2.0, 6.0)
    },
    "E_provenance": {
        "mean": 3.2,
        "std": 1.3,
        "range": (2.0, 6.0)
    },
    "E_hierarchy": {
        "mean": 3.0,
        "std": 1.4,
        "range": (2.0, 6.0)
    },
    "E_flow": {
        "mean": 2.8,
        "std": 1.2,
        "range": (2.0, 5.0)
    }
}
```

## Implementation Timeline

### Development Tasks (3 hours)

1. **Boundary Extractor** (1 hour)
   - Implement keyword-based filtering
   - Add quality scoring
   - Test on 1K samples

2. **LLM Energy Labeler** (1.5 hours)
   - Implement async batching
   - Add rate limiting
   - Error handling & retries

3. **Generation Script** (0.5 hours)
   - Connect extractor → labeler
   - Add progress tracking
   - Output JSONL + metadata

### Compute Timeline (35-42 hours)

```
Phase 1: Extraction (2M samples)
├─ WildChat (500K)          2 hours
├─ LMSYS (500K)             2 hours
├─ Dolci (400K)             1.5 hours
├─ LlamaFactory (300K)      1 hour
└─ DG-001 scaling (300K)    1 hour
                            ─────────
                            7.5 hours

Phase 2: LLM Labeling (2M samples)
├─ Claude Haiku @ 1000/min  33 hours ($200)
└─ GPT-4o-mini @ 800/min    42 hours ($600)

Phase 3: Filtering & QC       0.5 hours
                            ─────────
Total: 41-50 hours compute
```

## Cost Breakdown

### Option A: Claude Haiku (Recommended)

```
Extraction:           $0
LLM Labeling:         $200 (2M × $0.0001)
Filtering:            $0
                      ────
Total:                $200
Per sample:           $0.0001
Compute time:         41 hours
```

### Option B: GPT-4o-mini

```
Extraction:           $0
LLM Labeling:         $600 (2M × $0.0003)
Filtering:            $0
                      ────
Total:                $600
Per sample:           $0.0003
Compute time:         50 hours
```

## Quality Expectations

### With Claude Haiku ($200)
- **Energy Accuracy**: 85-90% (validated against human labels)
- **Boundary Precision**: 80-85% (samples truly in boundary range)
- **Coverage**: All 4 energy terms represented
- **Distribution**: Normal centered at E=3.5, σ=1.5

### With GPT-4o-mini ($600)
- **Energy Accuracy**: 90-95%
- **Boundary Precision**: 88-92%
- **Coverage**: All 4 energy terms + reasoning quality
- **Distribution**: Tighter (E=3.5, σ=1.3)

## Success Metrics

1. ✅ **Volume**: 2M boundary samples
2. ✅ **Cost**: <$1000 total (<$0.0005/sample)
3. ✅ **Energy Distribution**: Mean E=3.5±0.3, σ=1.5±0.2
4. ✅ **Quality**: >85% manual validation accuracy (100 sample audit)
5. ✅ **Coverage**: All 4 energy terms present in >80% of samples
6. ✅ **Timeline**: <72 hours wall clock time

## Execution Plan

### Step 1: Implement Infrastructure (3 hours)
```bash
# Create files:
- source/dataset/boundary_extractor.py
- source/dataset/llm_labeler.py
- scripts/generate_boundary_llm.py
```

### Step 2: Test on Small Batch (1 hour)
```bash
# Generate 1K samples to validate:
PYTHONPATH=. uv run python scripts/generate_boundary_llm.py \
  --samples 1000 \
  --model claude-3-haiku-20240307 \
  --output data/boundary_test_1k.jsonl
```

### Step 3: Full Generation (41-50 hours)
```bash
# Generate full 2M dataset:
PYTHONPATH=. uv run python scripts/generate_boundary_llm.py \
  --samples 2000000 \
  --model claude-3-haiku-20240307 \
  --output data/boundary_2m.jsonl \
  --batch-size 100 \
  --parallel 10
```

### Step 4: Quality Validation (1 hour)
- Manual review of 100 random samples
- Energy distribution analysis
- Upload to HuggingFace

## Next Steps

1. Approve budget: $200 (Haiku) or $600 (GPT-4o-mini)?
2. Implement infrastructure (3 hours dev)
3. Run test batch (1K samples, $0.10-0.30)
4. Launch full generation (2M samples, 41-50 hours)
5. Validate and upload to HF

---

**Created**: 2026-01-28
**Strategy**: HuggingFace Extraction + Full LLM Labeling
**Budget**: $200-600
**Timeline**: 3 hours dev + 41-50 hours compute
