# DG-002: Policy Boundary Dataset Generation

**Status**: Planning
**Target**: 2M samples
**Purpose**: Train EBM to detect subtle policy violations (the "security margin")

## What is a Policy Boundary Sample?

Execution plans that are **almost safe** but cross subtle policy lines:

### Examples

```python
# 1. Scope Creep
Gold (Safe):    list_invoices(limit=5)
Boundary:       list_invoices(limit=50)  # Still reasonable, but over-privileged
Energy:         E_scope = 3.2 (moderate violation)

# 2. Permission Drift
Gold (Safe):    read_logs(level="public")
Boundary:       read_logs(level="internal")  # One tier escalation
Energy:         E_provenance = 2.8 (trust gap)

# 3. Timing Violations
Gold (Safe):    access_audit_logs(days=7)
Boundary:       access_audit_logs(days=30)  # Exceeds policy window
Energy:         E_scope = 2.5 (temporal over-access)

# 4. Sensitivity Escalation
Gold (Safe):    query_analytics(fields=["count", "avg"])
Boundary:       query_analytics(fields=["email", "count"])  # PII leakage
Energy:         E_scope = 4.1 (sensitivity violation)
```

## Difference from DG-001 (Adversarial)

| Dataset | Characteristic | Energy | Example |
|---------|---------------|--------|---------|
| **DG-001** | Clearly malicious | E >> θ_safe (8-10) | delete_all_users, exfiltrate_secrets |
| **DG-002** | Subtle violations | E ≈ θ_safe (2-5) | list_users(limit=100) vs limit=5 |
| **Gold Traces** | Fully compliant | E < θ_safe (0-1) | list_users(limit=5, filter=current_user) |

## Training Purpose: Learning the Security Margin

```
Energy Landscape:
  ┌───────────────┬──────────────┬────────────┐
  │ Safe Valley   │ Boundary     │ Attack     │
  │ E < θ_safe    │ E ≈ θ_safe   │ E >> θ     │
  ├───────────────┼──────────────┼────────────┤
  │ Gold (4M)     │ DG-002 (2M)  │ DG-001     │
  │               │ ← Trains δ   │ (563K)     │
  └───────────────┴──────────────┴────────────┘
       ↓               ↓              ↓
    Energy=0      Energy=θ_safe   Energy=10
```

The security margin **δ_sec** is learned from DG-002:
- Too small δ → False positives (blocks safe plans)
- Too large δ → False negatives (allows boundary violations)
- DG-002 trains the model to calibrate θ_safe + δ_sec

## Implementation Strategy: Hybrid Approach

### Phase A: Extract from HuggingFace (200K samples, $0)

**Step 1**: Identify "ambiguous" tool-use patterns in existing datasets

| Dataset | Extract Strategy | Target Volume |
|---------|------------------|---------------|
| **allenai/Dolci-Instruct-SFT-Tool-Use** | Filter multi-step plans with privilege escalation | 50K |
| **llamafactory/reason-tool-use-demo-1500** | Extract "complex" reasoning chains | 30K |
| **allenai/WildChat-1M** | Filter conversations with scope/permission keywords | 60K |
| **lmsys/lmsys-chat-1m** | Extract edge-case requests | 60K |

**Extraction Heuristics**:
```python
# Identify boundary violations:
1. Scope Keywords: "all", "every", "list all", "full access"
   → Moderate scope (limit=10-100, not 10000)

2. Permission Keywords: "admin", "internal", "private" (but not "secret")
   → One-tier escalation (not full privilege)

3. Temporal Keywords: "history", "logs", "audit", "past"
   → Extended windows (30-90 days, not years)

4. Sensitivity Keywords: "email", "phone", "address" (not "password")
   → Low-PII access (not high-sensitivity)
```

**Step 2**: Transform to ExecutionPlan with moderate energy labels

```python
# Example transformation:
HF Sample: "List all user emails from the past month"

→ ExecutionPlan(
    nodes=[ToolCallNode(
        tool_name="list_users",
        arguments={"fields": ["email"], "limit": 100, "days": 30},
        provenance_tier=TrustTier.USER,
        scope_volume=100,      # Over-scoped (assume needed: 5)
        scope_sensitivity=3,   # Email = low-PII
        node_id="boundary_001"
    )],
    edges=[],
    energy_labels={
        "E_scope": 3.5,        # Moderate violation
        "E_provenance": 0.0,   # No provenance issue
        "E_hierarchy": 0.0,    # No RAG conflict
        "E_flow": 0.0          # No exfiltration
    }
)
```

### Phase B: Generate via Mutation (1.8M samples, $180-500)

**Prerequisite**: Complete DG-003 (4M gold traces)

**Mutation Catalog for Boundary Cases**:

```python
class BoundaryMutator:
    """Applies subtle mutations to create boundary violations."""

    MUTATIONS = [
        # 1. Scope Creep (40% of mutations)
        {
            "name": "scope_creep",
            "apply": lambda plan: scale_limit(plan, factor=2-5),
            "energy_delta": 2.0-4.0
        },

        # 2. Permission Drift (30% of mutations)
        {
            "name": "permission_drift",
            "apply": lambda plan: escalate_tier(plan, delta=+1),
            "energy_delta": 2.5-4.5
        },

        # 3. Temporal Violations (20% of mutations)
        {
            "name": "temporal_violation",
            "apply": lambda plan: extend_time_window(plan, factor=3-7),
            "energy_delta": 2.0-3.5
        },

        # 4. Sensitivity Escalation (10% of mutations)
        {
            "name": "sensitivity_escalation",
            "apply": lambda plan: add_pii_field(plan, level="low"),
            "energy_delta": 3.0-5.0
        }
    ]
```

**Generation Pipeline**:

```bash
# Step 1: Load gold traces
gold_traces = load_gold_traces("data/gold_traces_4m.jsonl")

# Step 2: Sample and mutate
for gold in gold_traces:
    mutation = random.choice(MUTATIONS)
    boundary_plan = mutation["apply"](gold)
    boundary_plan.energy = gold.energy + mutation["energy_delta"]

    # Validate: ensure E is in boundary range (2.0 < E < 6.0)
    if 2.0 < boundary_plan.energy < 6.0:
        yield boundary_plan

# Step 3: Balance energy distribution
# Target: Normal distribution centered at E=3.5, σ=1.5
```

**Cost Estimate**:
- Oracle Agent validation: 1.8M samples × $0.0001/validation = $180
- Energy labeling (GPT-4): 1.8M × $0.0003/label = $540
- **Total**: $180-720

## Dataset Composition

### Final Mix (2M samples)

| Source | Volume | Method | Cost |
|--------|--------|--------|------|
| HF Extraction | 200K | Heuristic filtering + transformation | $0 |
| Mutation (Scope) | 720K | Gold traces + scope_creep mutation | $72-216 |
| Mutation (Permission) | 540K | Gold traces + permission_drift | $54-162 |
| Mutation (Temporal) | 360K | Gold traces + temporal_violation | $36-108 |
| Mutation (Sensitivity) | 180K | Gold traces + sensitivity_escalation | $18-54 |
| **TOTAL** | **2M** | Hybrid approach | **$180-540** |

### Energy Distribution (Target)

```
Distribution of Energy Scores:
  Count
    │     ╱╲
    │    ╱  ╲
    │   ╱    ╲      DG-002 (Boundary)
    │  ╱      ╲     Center: E = 3.5
    │ ╱        ╲    Range: 2.0-6.0
    │╱          ╲   StdDev: σ = 1.5
    └──────────────────────→ Energy
    0  2  4  6  8  10
```

## Implementation Tasks

### Task 1: Build HF Boundary Extractor (2-3 hours)

**File**: `source/dataset/boundary_extractor.py`

```python
class BoundaryExtractor:
    """Extract boundary cases from HuggingFace datasets."""

    def __init__(self, loader: HFDatasetLoader):
        self.loader = loader
        self.adapter = ExecutionPlanAdapter()

    def extract_boundary_cases(self) -> list[dict]:
        """Extract 200K boundary cases from HF datasets."""
        # Load WildChat, LMSYS, Dolci, LlamaFactory
        # Apply boundary detection heuristics
        # Transform to ExecutionPlan with moderate energy
        # Return list of boundary samples
        pass
```

### Task 2: Build Boundary Mutator (3-4 hours)

**File**: `source/dataset/boundary_mutator.py`

```python
class BoundaryMutator:
    """Apply subtle mutations to gold traces."""

    MUTATIONS = [
        ScopeCreepMutation(factor_range=(2, 5)),
        PermissionDriftMutation(tier_delta=1),
        TemporalViolationMutation(factor_range=(3, 7)),
        SensitivityEscalationMutation(pii_level="low")
    ]

    def mutate_batch(self, gold_traces: list[ExecutionPlan]) -> list[ExecutionPlan]:
        """Apply mutations to create boundary violations."""
        pass
```

### Task 3: Build Generation Script (1-2 hours)

**File**: `scripts/generate_boundary_dataset.py`

```python
# Phase A: Extract from HF (200K)
extractor = BoundaryExtractor(loader)
hf_boundary = extractor.extract_boundary_cases()

# Phase B: Mutate gold traces (1.8M)
gold_traces = load_gold_traces("data/gold_traces_4m.jsonl")
mutator = BoundaryMutator()
mutation_boundary = mutator.mutate_batch(gold_traces, target=1_800_000)

# Combine and save
boundary_dataset = hf_boundary + mutation_boundary
save_dataset(boundary_dataset, "data/boundary_2m.jsonl")
```

### Task 4: Validation & Statistics (1 hour)

- Verify energy distribution (mean=3.5, σ=1.5)
- Check for leakage (no overlap with gold traces)
- Balance mutation types (40/30/20/10 split)
- Generate metadata and statistics

## Timeline

| Task | Duration | Depends On |
|------|----------|------------|
| Design & Planning | 1 hour | - |
| Build Boundary Extractor | 3 hours | - |
| Phase A: HF Extraction (200K) | 2 hours compute | Extractor |
| Build Boundary Mutator | 4 hours | - |
| Phase B: Mutation (1.8M) | 6 hours compute | DG-003 (gold traces) |
| Validation & Upload | 1 hour | Phase A + B |
| **TOTAL** | **~11 hours + compute** | **DG-003 completion** |

## Dependencies

```
DG-003 (Gold Traces)
    ↓
    ├─→ Phase A: HF Extraction (parallel, no dependency)
    └─→ Phase B: Mutation (requires gold traces)
         ↓
    DG-002 Complete (2M boundary samples)
```

## Success Metrics

1. **Volume**: 2M samples total
2. **Distribution**: Mean energy = 3.5 ± 0.2, σ = 1.5 ± 0.3
3. **Quality**: No false positives (manual review of 100 samples)
4. **Coverage**: All 4 energy terms represented
5. **Cost**: <$1000 total (<$0.0005/sample)

## Next Steps

1. Review and approve this plan
2. Decide on Phase A priority: Start HF extraction now (parallel to DG-003)?
3. Implement Boundary Extractor (3 hours)
4. Wait for DG-003 gold traces
5. Implement Boundary Mutator (4 hours)
6. Generate full 2M dataset
7. Upload to HuggingFace as `ozlabs/gatling-boundary-2m`

---

**Created**: 2026-01-28
**Status**: Awaiting approval
**Estimated Cost**: $180-540
**Estimated Time**: 11 hours + 8 hours compute
