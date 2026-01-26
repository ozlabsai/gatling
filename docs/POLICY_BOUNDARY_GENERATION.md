# Policy Boundary Case Generation - Stage B Documentation

## Overview

This document describes the **Policy Boundary Case Generator**, which produces 2M "near-safe" plans that violate subtle policy boundaries for training the Gatling Energy-Based Model.

## Table of Contents

1. [Purpose](#purpose)
2. [Architecture](#architecture)
3. [Violation Types](#violation-types)
4. [Usage](#usage)
5. [Technical Details](#technical-details)
6. [Quality Assurance](#quality-assurance)

## Purpose

### What are Policy Boundary Cases?

Policy boundary cases are execution plans that are "almost safe" but cross specific policy boundaries in subtle ways. Unlike the hard negatives from adversarial mutations (which dramatically alter plans), boundary cases:

- **Test precise limit enforcement** (max_results=100 → requesting 101)
- **Identify edge cases** in policy interpretation
- **Create the "margin data"** needed to establish stable safety thresholds

### Why 2M Samples?

The Gatling-10M dataset composition requires:
- 4M Standard Utility (safe valley)
- **2M Policy Boundary** (margin data) ← This component
- 2M RAG-Injection (hierarchy violations)
- 2M Data Flow/Exfil (long-horizon attacks)

Boundary cases represent 20% of the dataset and are critical for:
1. **Precise threshold calibration** (θ_safe)
2. **Reducing false positives** on legitimate edge cases
3. **Training the model to enforce exact policy limits**

## Architecture

### Components

```
source/dataset/
├── conversations/
│   └── boundary_mutator.py       # Core mutation logic
├── boundary_generator.py          # Orchestration & batch processing
└── validators/
    └── boundary_validator.py      # Quality validation
```

### Data Flow

```
┌─────────────────┐
│ Gold Traces     │  4M policy-compliant traces from Stage A
│ (Stage A Output)│
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Boundary Mutator    │  Apply systematic boundary violations
│ - 50% mutation rate │
│ - 6 violation types │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Validation Filter   │  Ensure subtlety (severity ≤ 0.3)
│ - Format checks     │
│ - Diversity checks  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 2M Boundary Cases   │  JSONL output for training
│ (.jsonl + metadata) │
└─────────────────────┘
```

## Violation Types

### 1. Numeric Edge (NUMERIC_EDGE)

**Subtlety: 0.1 (Very Subtle)**

Exactly one unit over the policy limit.

```python
# Policy: max_results = 100
# Original: limit=50
# Mutated:  limit=101  # Exactly at boundary+1
```

**Use Case**: Tests the model's ability to enforce precise numeric boundaries.

### 2. Numeric Near (NUMERIC_NEAR)

**Subtlety: 0.2 (Subtle)**

Close to the boundary (5-10% over limit).

```python
# Policy: max_results = 100
# Original: limit=50
# Mutated:  limit=105-110  # Near but clearly over
```

**Use Case**: Tests robustness around the boundary region.

### 3. Temporal Overflow (TEMPORAL_OVERFLOW)

**Subtlety: 0.15 (Very Subtle)**

Date range slightly exceeds policy limit.

```python
# Policy: max_days = 90
# Original: time_range_days=30
# Mutated:  time_range_days=91-95
```

**Use Case**: Calendar/audit log access violations.

### 4. Access Boundary (ACCESS_BOUNDARY)

**Subtlety: 0.25 (Subtle)**

Access adjacent/nearby resource outside authorization.

```python
# Policy: "Users can only access their own department"
# Original: department="Engineering"
# Mutated:  department="Adjacent_Department"
```

**Use Case**: Tests horizontal access control enforcement.

### 5. Approval Bypass (APPROVAL_BYPASS)

**Subtlety: 0.3 (Moderate)**

Operation requiring approval attempted without authorization.

```python
# Policy: "Production deployments require approval"
# Original: TrustTier.SYSTEM (with approval)
# Mutated:  TrustTier.USER (direct request, no approval)
```

**Use Case**: Tests privilege escalation detection.

### 6. Sensitivity Creep (SENSITIVITY_CREEP)

**Subtlety: 0.2 (Subtle)**

Data access escalated by exactly one sensitivity tier.

```python
# Sensitivity Ladder: PUBLIC → INTERNAL → CONFIDENTIAL → RESTRICTED
# Original: INTERNAL
# Mutated:  CONFIDENTIAL  # One tier up
```

**Use Case**: Tests granular data sensitivity enforcement.

## Usage

### Command-Line Interface

```bash
# Generate 2M boundary violations from gold traces
uv run python -m source.dataset.boundary_generator \
    --gold-traces-dir outputs/gold_traces \
    --output-dir outputs/boundary_cases \
    --target 2000000 \
    --checkpoint-every 100000

# Sample mode (1000 violations for testing)
uv run python -m source.dataset.boundary_generator \
    --gold-traces-dir outputs/gold_traces \
    --output-dir outputs/boundary_cases \
    --sample

# Custom mutation parameters
uv run python -m source.dataset.boundary_generator \
    --mutation-rate 0.6 \
    --subtlety-threshold 0.25 \
    --gold-traces-dir outputs/gold_traces
```

### Python API

```python
from source.dataset.boundary_generator import BoundaryDatasetGenerator

# Initialize generator
generator = BoundaryDatasetGenerator(
    gold_traces_dir="outputs/gold_traces",
    output_dir="outputs/boundary_cases",
    mutation_rate=0.5,
    subtlety_threshold=0.3,
)

# Generate dataset
generator.generate_dataset(
    target_violations=2_000_000,
    checkpoint_every=100_000,
    sample_mode=False,
)
```

### Programmatic Mutation

```python
from source.dataset.conversations.boundary_mutator import PolicyBoundaryMutator

# Initialize mutator
mutator = PolicyBoundaryMutator(
    mutation_rate=0.5,
    seed=42,
    subtlety_threshold=0.3,
)

# Apply mutations to gold traces
violations = mutator.mutate_traces(gold_traces)

# Get statistics
stats = mutator.get_statistics()
print(f"Success rate: {stats['successful_mutations'] / stats['total_attempts'] * 100:.1f}%")
```

## Technical Details

### Mutation Algorithm

1. **Selection**: Randomly sample `mutation_rate` fraction of gold traces
2. **Analysis**: Determine applicable violation types based on policy structure
3. **Application**: Apply selected mutation while preserving graph structure
4. **Validation**: Check subtlety score ≤ threshold
5. **Output**: Return BoundaryViolation with modified ToolCallGraph

### Subtlety Scoring

Violations are scored on a scale of 0-1:
- **0.0-0.15**: Very subtle (edge cases)
- **0.15-0.25**: Subtle (near boundaries)
- **0.25-0.35**: Moderate (clear but not egregious)
- **>0.35**: Too obvious (rejected)

### Output Format

Violations are saved as JSONL with the following structure:

```json
{
  "violation_id": "trace_00123_boundary_edge",
  "original_trace_id": "finance_00000123",
  "violation_type": "numeric_edge",
  "violated_policy_rule": "max_results=100",
  "violation_description": "Requested 101 items when policy limit is 100",
  "modified_graph": {
    "graph_id": "test_graph",
    "calls": [...],
    "execution_order": [...]
  },
  "severity_score": 0.1
}
```

### Memory Management

For large-scale generation:
- **Batch processing**: Load 100K traces at a time
- **Checkpointing**: Save every 100K violations
- **Streaming**: JSONL format allows incremental loading

## Quality Assurance

### Validation Checks

The `BoundaryViolationValidator` performs:

1. **Format Validation**: Ensure required fields and proper structure
2. **Subtlety Check**: severity_score ≤ max_severity threshold
3. **Mutation Verification**: Modified graph differs from original
4. **Diversity Analysis**: Check violation type distribution

### Expected Metrics

For a healthy 2M sample dataset:

```
Violation Type Distribution:
  numeric_edge:        ~400K (20%)
  numeric_near:        ~400K (20%)
  temporal_overflow:   ~300K (15%)
  access_boundary:     ~300K (15%)
  approval_bypass:     ~300K (15%)
  sensitivity_creep:   ~300K (15%)

Severity Distribution:
  very_subtle (0.0-0.15): ~40%
  subtle (0.15-0.25):     ~40%
  moderate (0.25-0.3):    ~20%

Success Rate: 45-55% (from 4M gold traces → 2M violations)
```

### Testing

Run the test suite:

```bash
# Run all boundary mutator tests
uv run pytest test/test_dataset/test_boundary_mutator.py -v

# Run with coverage
uv run pytest test/test_dataset/test_boundary_mutator.py --cov=source.dataset.conversations.boundary_mutator --cov-report=html
```

## Integration with Training Pipeline

### Loading Boundary Cases

```python
from source.dataset.loaders import load_boundary_violations

# Load violations for training
for violation in load_boundary_violations("outputs/boundary_cases"):
    # violation.modified_graph: ToolCallGraph
    # violation.severity_score: float
    # Train E(z_g, z_e) to assign high energy to this plan
```

### Training Strategy

Boundary cases are used for:

1. **Margin Learning**: Train model to enforce precise boundaries
   - Positive sample: Original gold trace (low energy)
   - Negative sample: Boundary violation (high energy)
   - Margin: E(violation) - E(gold) ≥ δ_sec

2. **Threshold Calibration**: Determine θ_safe
   - Plot energy distribution for boundary cases
   - Set θ_safe to achieve 99.9% recall on violations

3. **False Positive Reduction**: Fine-tune around boundaries
   - Use hard negatives where E(violation) < θ_safe
   - Increase energy for these samples

## Troubleshooting

### Low Mutation Success Rate (<40%)

**Cause**: Gold traces may not have numeric limits or applicable policies.

**Solution**:
- Check policy structure in gold traces
- Increase mutation_rate to 0.6-0.7
- Verify scope_limits are present in SystemPolicy

### Poor Violation Diversity

**Cause**: Policies may be skewed toward certain types.

**Solution**:
- Stratify gold traces by domain before mutation
- Use weighted sampling for violation types
- Generate additional traces for under-represented domains

### High Severity Scores

**Cause**: Mutations may be too aggressive.

**Solution**:
- Lower subtlety_threshold to 0.2
- Adjust mutation parameters (smaller overshoots)
- Filter post-generation using validator

## Future Enhancements

### Planned for v0.3.0

1. **Compositional Violations**: Combine multiple boundary violations
2. **Adversarial Refinement**: Use EBM feedback to generate harder negatives
3. **Domain-Specific Mutations**: Custom violation types per industry
4. **Temporal Evolution**: Simulate policy changes over time

### Research Directions

1. **Active Learning**: Identify high-curvature regions in energy landscape
2. **Synthetic Policy Generation**: Create new policy boundaries automatically
3. **Cross-Domain Transfer**: Apply learned boundaries to new domains

## References

- [DATASET-WORKSTREAM.md](DATASET-WORKSTREAM.md) - Full pipeline documentation
- [CATALOG-PLAN.md](CATALOG-PLAN.md) - Adversarial mutation catalog
- [PRD.md](PRD.md) - Product requirements and architecture

## Contact

For questions or issues related to policy boundary generation:
- Open an issue on the project repository
- Refer to the main CLAUDE.md for development guidelines
