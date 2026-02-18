# JEPA Training Status & Integration Plan

**Date**: 2026-01-28
**Current Status**: LSA-004 (JEPA Training) ready, not yet implemented
**Blocker**: Waiting for dataset completion

## What is JEPA Training?

**JEPA (Joint-Embedding Predictive Architecture)** is the core learning framework for Gatling's dual-encoder system.

### The Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    JEPA Dual-Encoder System                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐              ┌────────────────────┐    │
│  │  Governance     │              │  Execution         │    │
│  │  Encoder        │              │  Encoder           │    │
│  │                 │              │                    │    │
│  │  Input:         │              │  Input:            │    │
│  │  - Policy       │              │  - ExecutionPlan   │    │
│  │  - User Role    │              │  - Provenance      │    │
│  │  - Context      │              │  - Scope          │    │
│  │                 │              │                    │    │
│  │  Output:        │              │  Output:           │    │
│  │  z_g ∈ R^1024   │              │  z_e ∈ R^1024      │    │
│  └────────┬────────┘              └─────────┬──────────┘    │
│           │                                  │               │
│           └───────────┐      ┌───────────────┘               │
│                       ↓      ↓                               │
│              ┌────────────────────┐                          │
│              │  Energy Function   │                          │
│              │  E(z_g, z_e)       │                          │
│              │                    │                          │
│              │  E = Σ E_term      │                          │
│              │    E_hierarchy     │                          │
│              │  + E_provenance    │                          │
│              │  + E_scope         │                          │
│              │  + E_flow          │                          │
│              └────────────────────┘                          │
│                       ↓                                      │
│                  Security Score                              │
│                  (Low = Safe, High = Malicious)              │
└──────────────────────────────────────────────────────────────┘
```

### Training Objective: InfoNCE Loss

**Goal**: Learn embeddings where safe (policy, execution) pairs have low energy, violations have high energy.

```python
# InfoNCE Contrastive Loss
def info_nce_loss(z_g, z_e_positive, z_e_negatives, temperature=0.07):
    """
    Maximize similarity between:
    - z_g and z_e_positive (compliant execution)

    Minimize similarity between:
    - z_g and z_e_negatives (violating executions)
    """
    # Positive pair energy
    pos_energy = energy_fn(z_g, z_e_positive)  # Should be LOW

    # Negative pairs energy
    neg_energies = [energy_fn(z_g, z_e_neg) for z_e_neg in z_e_negatives]  # Should be HIGH

    # InfoNCE: pull positives together, push negatives apart
    loss = -log(exp(-pos_energy / temp) / (exp(-pos_energy / temp) + Σ exp(-neg_energies / temp)))

    return loss
```

## Current Implementation Status

### ✅ Completed Components

1. **Governance Encoder** (LSA-001) ✅
   - File: `source/encoders/governance_encoder.py`
   - Status: Implemented
   - Features:
     - Structure-aware attention for JSON/YAML policies
     - Sparse attention (O(n*w) complexity)
     - Outputs z_g ∈ R^1024

2. **Execution Encoder** (LSA-002) ✅
   - File: `source/encoders/execution_encoder.py`
   - Status: Implemented
   - Features:
     - Graph Neural Network for tool-call dependency
     - Provenance-aware attention
     - Outputs z_e ∈ R^1024

3. **Energy Functions** ✅
   - Directory: `source/energy/`
   - Files:
     - `hierarchy.py` - E_hierarchy (RAG influence detection)
     - `provenance.py` - E_provenance (trust gap measurement)
     - `scope.py` - E_scope (over-access detection)
     - `flow.py` - E_flow (exfiltration pattern detection)
     - `composite.py` - E_total = Σ E_terms
   - Status: Implemented

### ❌ Missing Components

1. **Training Script** (LSA-004) ❌
   - File: Should be `scripts/train_jepa_encoders.py`
   - Status: **NOT IMPLEMENTED**
   - Needed:
     - DataLoader for ExecutionPlan dataset
     - InfoNCE loss implementation
     - Training loop with hard negative mining
     - Evaluation metrics
     - Checkpoint saving

2. **Training Integration** (TI-001) ❌
   - Status: Blocked by LSA-004 + datasets (DG-001, DG-002, DG-003)
   - Needed:
     - Connect datasets to training pipeline
     - Batch processing
     - Train/val/test splits

## Dataset Requirements for JEPA Training

### Current Status

| Dataset | Target | Status | Progress |
|---------|--------|--------|----------|
| **DG-001: Adversarial** | 563K | ✅ COMPLETE | 541K samples generated |
| **DG-002: Boundary** | 2M | 🔄 PLANNING | Strategy designed, not started |
| **DG-003: Gold Traces** | 4M | ⏸️ PENDING | Not started |

### Training Data Mix (Target: 6.56M samples)

```
Training Distribution:
┌─────────────────────────────────────────────────────┐
│ Gold Traces (4M)        - Positive examples (E<2)  │ 61%
│ Boundary (2M)           - Hard negatives (E=2-6)   │ 30%
│ Adversarial (563K)      - Clear attacks (E>8)      │  9%
├─────────────────────────────────────────────────────┤
│ Total: 6.56M samples                                │
└─────────────────────────────────────────────────────┘
```

### Why This Mix?

1. **Gold Traces (61%)**: Teach "what safe looks like"
   - Low energy baseline (E < 2)
   - Establish the "safe valley" in embedding space

2. **Boundary Cases (30%)**: Calibrate security margin δ_sec
   - Moderate energy (E = 2-6)
   - Train model to detect subtle violations
   - Most important for real-world deployment

3. **Adversarial (9%)**: Recognize obvious attacks
   - High energy (E > 8)
   - Prevent model from being fooled by clear malicious intent

## LSA-004: Implementation Plan

### Training Pipeline Architecture

```python
# Pseudo-code for JEPA training

class JEPATrainingPipeline:
    def __init__(self):
        self.gov_encoder = GovernanceEncoder()
        self.exec_encoder = ExecutionEncoder()
        self.energy_fn = CompositeEnergyFunction()
        self.optimizer = AdamW()

    def train_step(self, batch):
        """Single training iteration."""
        policy, positive_plan, negative_plans = batch

        # Encode
        z_g = self.gov_encoder(policy)
        z_e_pos = self.exec_encoder(positive_plan)
        z_e_negs = [self.exec_encoder(neg) for neg in negative_plans]

        # Compute energies
        E_pos = self.energy_fn(z_g, z_e_pos)
        E_negs = [self.energy_fn(z_g, z_e_neg) for z_e_neg in z_e_negs]

        # InfoNCE loss
        loss = info_nce_loss(E_pos, E_negs, temperature=0.07)

        # Backprop
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader):
        """Full pass through dataset."""
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
        return total_loss / len(dataloader)
```

### Implementation Tasks

#### Task 1: DataLoader (2 hours)

**File**: `source/training/dataset.py`

```python
class ExecutionPlanDataset(torch.utils.data.Dataset):
    """PyTorch dataset for JEPA training."""

    def __init__(self, jsonl_path: str):
        """Load dataset from JSONL files."""
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __getitem__(self, idx):
        """Return (policy, positive_plan, negative_plans, energy_label)."""
        sample = self.samples[idx]
        return {
            "policy": sample["policy"],
            "positive_plan": sample["execution_plan"],
            "negative_plans": sample.get("hard_negatives", []),
            "energy_label": sample.get("energy_labels", {})
        }

    def __len__(self):
        return len(self.samples)
```

#### Task 2: InfoNCE Loss (1 hour)

**File**: `source/training/losses.py`

```python
def info_nce_loss(
    energy_pos: torch.Tensor,
    energies_neg: list[torch.Tensor],
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss for energy-based models.

    Args:
        energy_pos: Energy of positive (compliant) pair
        energies_neg: Energies of negative (violating) pairs
        temperature: Temperature scaling parameter

    Returns:
        Contrastive loss value
    """
    # Convert energies to similarities (negative energy = high similarity)
    logits_pos = -energy_pos / temperature
    logits_neg = torch.stack([-e / temperature for e in energies_neg])

    # Softmax denominator: positive + all negatives
    logsumexp = torch.logsumexp(
        torch.cat([logits_pos.unsqueeze(0), logits_neg]), dim=0
    )

    # Loss: -log(P(positive))
    loss = -logits_pos + logsumexp

    return loss
```

#### Task 3: Training Loop (2 hours)

**File**: `scripts/train_jepa_encoders.py`

```python
def train_jepa(
    train_path: str,
    val_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4
):
    """Train JEPA dual-encoder system."""

    # Load datasets
    train_dataset = ExecutionPlanDataset(train_path)
    val_dataset = ExecutionPlanDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize models
    gov_encoder = GovernanceEncoder()
    exec_encoder = ExecutionEncoder()
    energy_fn = CompositeEnergyFunction()

    optimizer = AdamW([
        *gov_encoder.parameters(),
        *exec_encoder.parameters()
    ], lr=lr)

    # Training loop
    for epoch in range(epochs):
        gov_encoder.train()
        exec_encoder.train()

        train_loss = 0
        for batch in train_loader:
            # Forward pass
            z_g = gov_encoder(batch["policy"])
            z_e_pos = exec_encoder(batch["positive_plan"])
            z_e_negs = [exec_encoder(neg) for neg in batch["negative_plans"]]

            # Compute energies
            E_pos = energy_fn(z_g, z_e_pos)
            E_negs = [energy_fn(z_g, z_e_neg) for z_e_neg in z_e_negs]

            # Loss
            loss = info_nce_loss(E_pos, E_negs)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        val_loss = validate(gov_encoder, exec_encoder, energy_fn, val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(gov_encoder, exec_encoder, epoch)
```

#### Task 4: Evaluation Metrics (1 hour)

**File**: `source/training/metrics.py`

```python
def evaluate_energy_separation(
    gov_encoder, exec_encoder, energy_fn, dataset
):
    """
    Measure energy separation between safe and malicious plans.

    Metrics:
    - Mean energy (gold traces) should be < 2.0
    - Mean energy (adversarial) should be > 8.0
    - Mean energy (boundary) should be 2.0-6.0
    - Margin δ_sec = E_adversarial - E_boundary
    """
    energies_gold = []
    energies_boundary = []
    energies_adversarial = []

    for sample in dataset:
        z_g = gov_encoder(sample["policy"])
        z_e = exec_encoder(sample["execution_plan"])
        energy = energy_fn(z_g, z_e)

        if sample["label"] == "gold":
            energies_gold.append(energy)
        elif sample["label"] == "boundary":
            energies_boundary.append(energy)
        elif sample["label"] == "adversarial":
            energies_adversarial.append(energy)

    return {
        "E_gold_mean": torch.mean(energies_gold),
        "E_boundary_mean": torch.mean(energies_boundary),
        "E_adversarial_mean": torch.mean(energies_adversarial),
        "delta_sec": torch.mean(energies_adversarial) - torch.mean(energies_boundary)
    }
```

## Timeline & Dependencies

```
Current State (2026-01-28):
├─ ✅ Encoders implemented
├─ ✅ Energy functions implemented
├─ ✅ DG-001 (563K adversarial) - COMPLETE
├─ 🔄 DG-002 (2M boundary) - PLANNING
└─ ⏸️ DG-003 (4M gold traces) - NOT STARTED

Next Steps:
1. Complete DG-002 & DG-003 (datasets)
   ↓
2. Implement LSA-004 (training script) - 6 hours dev
   ↓
3. TI-001 (integrate datasets) - 6 hours
   ↓
4. Run initial training (10 epochs) - 12-24 hours compute
   ↓
5. Evaluate & iterate
```

### Critical Path

**Week 1**: Dataset completion
- DG-002: 3h dev + 41h compute ($200)
- DG-003: 2h dev + 6h compute ($100-500)

**Week 2**: Training implementation
- LSA-004: 6h dev (implement training script)
- TI-001: 6h integration (connect datasets)

**Week 3**: First training run
- Run JEPA training on 6.56M samples
- 10 epochs × 2 hours/epoch = 20 hours compute
- Evaluate energy separation metrics

**Week 4**: Iteration & validation
- Analyze results
- Tune hyperparameters
- Generate hard negatives (Corrupter Agent)

## Success Metrics

### Training Convergence
- ✅ InfoNCE loss decreases consistently
- ✅ Validation loss plateaus (no overfitting)
- ✅ Energy gap δ_sec > 2.0

### Energy Separation
```
Target Energy Distribution:
  Gold Traces:     E < 2.0  (mean ≈ 0.5)
  Boundary Cases:  2 < E < 6 (mean ≈ 3.5)
  Adversarial:     E > 8.0  (mean ≈ 9.5)

  Security Margin: δ_sec = E_adversarial - E_boundary ≈ 6.0
```

### Embedding Quality
- ✅ Nearest neighbor accuracy > 90%
  - Gold traces cluster together
  - Adversarial traces cluster separately
  - Boundary cases lie in between

## Cost Estimate

| Component | Cost | Timeline |
|-----------|------|----------|
| DG-002 (Boundary Dataset) | $200 | 3h dev + 41h compute |
| DG-003 (Gold Traces) | $100-500 | 2h dev + 6h compute |
| LSA-004 (Training Implementation) | $0 | 6h dev |
| TI-001 (Integration) | $0 | 6h dev |
| Training Compute (AWS p3.2xlarge) | ~$500 | 20-40 hours |
| **TOTAL** | **$800-1200** | **~2 weeks** |

## Next Actions

1. **Immediate**: Finish DG-002 planning & approval
2. **This Week**: Implement DG-002 + DG-003 generation
3. **Next Week**: Implement LSA-004 training script
4. **Week 3**: Run first JEPA training experiment

---

**Status**: Encoders ready, datasets in progress, training blocked
**Blocker**: Need DG-002 (2M) and DG-003 (4M) datasets
**ETA**: 2-3 weeks to first training results
