# Next Tasks Created: 2026-01-26

## Summary
Created 7 new high-priority tasks as the natural next steps after today's infrastructure work.

## Immediate Next Steps (P1) - Dataset Generation

### Phase 2A: Execute What We Built

| ID | Task | Priority | Estimate | Depends On |
|----|------|----------|----------|------------|
| **ga-1bmm** | DG-001: Generate Lakera Dataset (563K) | P1 | 2h | DA-004 merge |
| **ga-bavt** | DG-002: Generate Policy Boundary (2M) | P1 | 3h | DA-002 merge |
| **ga-duqv** | DG-003: Label Gold Traces (4M) | P1 | 2h | DA-003 merge |

**Total Output**: ~6.56M training samples
**Total Time**: ~7 hours of compute
**Blocker**: Wait for merge queue to process (overnight)

---

## Training Integration (P1)

### Phase 2B: Connect to JEPA Training

| ID | Task | Priority | Estimate | Depends On |
|----|------|----------|----------|------------|
| **ga-12ve** | TI-001: Integrate with LSA-004 Pipeline | P1 | 6h | DG-001, DG-002, DG-003 |
| **ga-rib6** | VB-001: End-to-End Validation | P1 | 3h | TI-001 |

**Deliverable**: Working training pipeline with all 3 dataset types
**Total Time**: ~9 hours

---

## Security Infrastructure (P1)

### Phase 2C: Provenance & Repair

| ID | Task | Priority | Estimate | Status |
|----|------|----------|----------|--------|
| **ga-feva** | PA-001: Trust Tier Tagging | P1 | 6h | ALREADY OPEN |
| **ga-b8zr** | PA-002: Repair Engine (Discrete Search) | P1 | 10h | Energy terms complete |

**Deliverable**: Provenance tracking + plan repair capability
**Total Time**: ~16 hours

---

## Hard Negatives (P2)

### Phase 2D: Adversarial Training

| ID | Task | Priority | Estimate | Depends On |
|----|------|----------|----------|------------|
| **ga-i4t8** | TI-002: Corrupter Agent | P2 | 8h | DG-003 |

**Deliverable**: 1M automated hard negatives
**Total Time**: ~8 hours

---

## Execution Strategy

### Week 1 (After Merges Complete):
**Day 1**: Dataset Generation
- Run DG-001, DG-002, DG-003 in parallel (7h compute)
- Validate outputs
- Generate statistics

**Day 2-3**: Training Integration
- TI-001: Connect datasets to LSA-004 (6h)
- VB-001: Validate end-to-end (3h)
- Run first training experiments

### Week 2:
**Day 1-2**: Provenance System
- PA-001: Trust tier tagging (6h)
- Integration with GoldTrace

**Day 3-4**: Repair Engine
- PA-002: Discrete search implementation (10h)
- Testing and benchmarking

### Week 3:
**Day 1-2**: Corrupter Agent
- TI-002: Hard negative generation (8h)
- Integration with training loop

---

## Dependencies Visualization

```
[Merge Queue Processing]
         ↓
    ┌────┴────┬─────┬─────┐
    ↓         ↓     ↓     ↓
 DG-001   DG-002  DG-003  PA-001 (parallel)
 (563K)    (2M)    (4M)    (trust)
    └────┬────┴─────┘      ↓
         ↓                 ↓
      TI-001 ←─────────────┘
    (integrate)
         ↓
      VB-001
   (validate)
         ↓
   ┌────┴────┐
   ↓         ↓
TI-002    PA-002
(corrupt) (repair)
```

---

## Success Metrics

### Dataset Generation:
- ✅ 563K Lakera adversarial samples
- ✅ 2M policy boundary cases
- ✅ 4M scope-labeled traces
- ✅ **Total: ~6.56M training samples**

### Training Integration:
- ✅ All 3 datasets loaded successfully
- ✅ InfoNCE loss implemented
- ✅ Training converges on validation set
- ✅ Energy terms show learning signal

### Security Infrastructure:
- ✅ 3-tier provenance classification
- ✅ Repair engine <200ms latency
- ✅ FIS ≥90% (functional intent preservation)
- ✅ θ_safe calibrated with margin δ_sec

---

## Risk Mitigation

**Risk**: Dataset generation fails
- **Mitigation**: Each generator has comprehensive tests (45, 18, 45 tests respectively)
- **Validation**: Built-in quality checks and statistics

**Risk**: Training integration breaks
- **Mitigation**: VB-001 validation step before full training
- **Validation**: Mini training loop on 1K samples first

**Risk**: Repair engine too slow
- **Mitigation**: <200ms latency requirement, benchmark-driven
- **Fallback**: Fast-path policy distillation for common cases

---

Created: 2026-01-26
Mayor: Clem 🤗
Status: Ready for execution (pending merges)
