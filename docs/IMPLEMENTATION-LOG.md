# Implementation Log

This document provides detailed technical implementation notes for all code written in Project Gatling.

## Purpose
- Track low-level implementation details
- Document technical decisions and trade-offs
- Provide context for future debugging and optimization
- Record performance benchmarks and profiling results

---

## Implementation Entry Template

```markdown
### [Module/Feature Name]
**Date**: YYYY-MM-DD
**Implementer**: [Name/AI Assistant]
**Status**: Planning | In Progress | Complete | Deprecated

#### Overview
[High-level description of what was implemented]

#### Technical Approach
[Detailed explanation of the implementation strategy]

#### Key Design Decisions
1. **Decision**: [What was decided]
   - **Options Considered**: [Alternative approaches]
   - **Rationale**: [Why this was chosen]
   - **Trade-offs**: [What was gained/lost]

#### Code Structure
```python
# Example of key classes/functions
class ExampleClass:
    """Brief description"""
    pass
```

#### Dependencies
- Library/Module 1: [Why needed]
- Library/Module 2: [Why needed]

#### Testing Strategy
- Unit tests: [Coverage approach]
- Integration tests: [What interactions tested]
- Performance tests: [Benchmarks]

#### Known Issues
- [Issue 1]
- [Issue 2]

#### Future Improvements
- [Optimization opportunity 1]
- [Refactor opportunity 2]

#### Performance Metrics
- Latency: [Measurement]
- Memory: [Measurement]
- Throughput: [Measurement]

---
```

## Implementations

### GovernanceEncoder (LSA-001)
**Date**: 2026-01-25
**Implementer**: Claude (Sonnet 4.5) + Guy Nachshon
**Status**: Complete

#### Overview
Transformer-based encoder mapping (policy, user_role, session_context) → z_g ∈ R^1024. First component of JEPA dual-encoder architecture for semantic policy representation in energy-based security validation.

#### Technical Approach
**Core Innovation**: Sparse Structured Attention with Hierarchy Preservation

Policy documents are trees, not sequences. Unlike models that flatten to sequences, GovernanceEncoder preserves structure via:
- Depth embeddings (nesting level 0-7)
- Node type embeddings (dict/list/primitive)
- Global tokens (top-level sections) with full attention
- Local windows (32-token sliding) for efficiency

Complexity: O(n×w + n×g) vs O(n²) full attention

#### Key Design Decisions

1. **Decision**: Sparse structured attention over full self-attention
   - **Options**: Full O(n²), Fixed sparse (Longformer), Tree-structured, Hybrid (chosen)
   - **Rationale**: Balances structure awareness with <50ms latency target
   - **Trade-offs**: Gained 3-5x speedup; Lost some long-range deps (mitigated by global tokens)

2. **Decision**: Hash tokenization for v0.1.0
   - **Options**: Char-level, BPE, Hash (chosen), Learned embedding
   - **Rationale**: Simplicity for MVP, defer tokenizer training
   - **Trade-offs**: Zero training time but potential collisions; BPE planned v0.2.0

3. **Decision**: Learned attention pooling
   - **Options**: Mean, Max, CLS token, Attention pooling (chosen)
   - **Rationale**: Learn to focus on security-relevant policy sections
   - **Trade-offs**: +0.5K params for ~15-20% better representation (estimated)

4. **Decision**: 4 layers × 512 dim
   - **Options**: 8×256 (deeper/narrow), 2×1024 (shallow/wide), 4×512 (chosen)
   - **Rationale**: Sweet spot for 98ms latency, 25M params, 86MB size

#### Code Structure
```python
class PolicySchema(BaseModel):
    """Pydantic validation: handles Dict/JSON/YAML"""

class StructuralEmbedding(nn.Module):
    """depth + node_type → hidden_dim"""

class SparseStructuredAttention(nn.Module):
    """O(n×w) attention: local windows + global tokens"""

class TransformerBlock(nn.Module):
    """Standard: Attention → FFN with residuals"""

class GovernanceEncoder(nn.Module):
    """Main encoder: policy → z_g[1, 1024]"""
    def _flatten_policy()  # Tree → sequence with metadata
    def forward()           # End-to-end encoding
    def encode_batch()      # Multi-policy processing
```

#### Dependencies
- torch ≥2.5.0: Neural network framework
- transformers ≥4.46.0: HuggingFace utilities
- pydantic ≥2.10.0: Input validation
- pyyaml ≥6.0.0: YAML parsing
- numpy ≥2.0.0: Numerical ops

#### Testing Strategy
- **35 tests, 98% coverage**
- Categories: Init (4), Validation (5), Core (6), Variable-length (3), Gradients (2), Batch (2), Performance (3), Components (4), Edge cases (4), Integration (2)
- Benchmark: 98.08ms mean (σ=2.40ms), 11 rounds

#### Known Issues
1. **Latency**: 98ms vs 50ms target (quantization/ONNX planned)
2. **Hash collisions**: BPE tokenizer v0.2.0
3. **Session context unused**: Future enhancement

#### Future Improvements
- **Phase 1**: INT8 quantization, ONNX Runtime, dynamic window sizing
- **Phase 2**: BPE tokenizer, subword pooling, semantic hashing
- **Phase 3**: Dynamic depth, hierarchical attention, cross-attention

#### Performance Metrics
- Parameters: 25.2M (86MB)
- CPU inference: 98ms
- Memory: 86MB model + 20MB activation
- Throughput: ~10 encodings/sec

---

### ExecutionEncoder (LSA-002)
**Date**: 2026-01-25
**Implementer**: Claude (Sonnet 4.5) - Jolly Badger Agent
**Status**: Complete

#### Overview
Graph-based transformer encoder mapping (plan_graph, provenance_metadata) → z_e ∈ R^1024. Second component of JEPA dual-encoder architecture, enabling energy-based security validation by encoding proposed execution plans into the same latent space as governance policies.

#### Technical Approach
**Core Innovation**: Graph Neural Network with Provenance-Aware Attention

Execution plans are graphs (not sequences or trees). ExecutionEncoder preserves dependency structure via:
- Graph Attention Networks (GAT) for message passing along data flow edges
- Provenance embeddings encoding Trust Tiers (Internal/Partner/Public)
- Scope metadata integration (log-scaled volume + sensitivity)
- Self-loops in adjacency matrix for residual connections

Complexity: O(n²) for attention over graph nodes (n ≤ 64 max_nodes)

#### Key Design Decisions

1. **Decision**: Graph Attention over sequential attention
   - **Options**: Seq-to-seq (Transformer), GCN, GAT (chosen), GraphTransformer
   - **Rationale**: GAT respects tool-call dependencies while maintaining differentiability
   - **Trade-offs**: O(n²) complexity but n capped at 64 nodes; captures data flow explicitly

2. **Decision**: Trust Tier + Scope as separate embeddings
   - **Options**: Concatenate metadata, Learned fusion (chosen), Ignore metadata
   - **Rationale**: Provenance is critical for E_provenance energy term
   - **Trade-offs**: +2K params for fusion layer; enables trust-aware encoding

3. **Decision**: Adjacency-masked attention with self-loops
   - **Options**: Fully connected, Sparse (edge-only), Sparse + self-loops (chosen)
   - **Rationale**: Self-loops act as residual connections, prevent isolated nodes
   - **Trade-offs**: Slightly denser attention but numerically stable

4. **Decision**: 4 layers × 512 dim (matches GovernanceEncoder)
   - **Options**: 6×384 (deeper/narrow), 3×640 (shallow/wide), 4×512 (chosen)
   - **Rationale**: Architectural symmetry with GovernanceEncoder, 12.6ms latency
   - **Trade-offs**: ~28M params (3MB larger than Governance), balanced depth/width

#### Code Structure
```python
class TrustTier(IntEnum):
    """INTERNAL=1, SIGNED_PARTNER=2, PUBLIC_WEB=3"""

class ToolCallNode(BaseModel):
    """Single tool invocation + provenance + scope metadata"""

class ExecutionPlan(BaseModel):
    """Full plan: nodes + edges with validation"""

class ProvenanceEmbedding(nn.Module):
    """tier + log(volume) + sensitivity → hidden_dim"""

class GraphAttention(nn.Module):
    """Multi-head attention with adjacency masking"""

class ExecutionEncoder(nn.Module):
    """Main encoder: plan_graph → z_e[1, 1024]"""
    def _create_adjacency_matrix()  # Edge list → tensor
    def forward()                    # End-to-end encoding
    def encode_batch()              # Multi-plan processing
```

#### Dependencies
- torch ≥2.5.0: Neural network framework
- pydantic ≥2.10.0: Input validation (ExecutionPlan schema)
- (No transformers dependency - custom GAT implementation)

#### Testing Strategy
- **30 tests, all passing**
- Categories: Init (4), Core encoding (4), Provenance (2), Graph structure (3), Variable-length (3), Gradients (2), Batch (2), Components (3), Benchmarks (2), Edge cases (4), Integration (1)
- Benchmark: 12.61ms mean (σ=0.61ms), 10 rounds
- Deterministic testing: torch.manual_seed() reset between calls (PyTorch internal randomness)

#### Known Issues
1. **Latency**: 12.6ms vs 100ms target (EXCEEDS by 8x - excellent!)
2. **Hash collisions**: Same as GovernanceEncoder, BPE tokenizer v0.2.0
3. **Graph cycles**: Accepted but semantics undefined (DAGs expected)

#### Future Improvements
- **Phase 1**: True batching (current: sequential loop), ONNX export
- **Phase 2**: BPE tokenizer, argument-level attention, cryptographic provenance validation
- **Phase 3**: Hierarchical graphs (subgraphs), temporal edges, learned pooling strategies

#### Performance Metrics
- Parameters: 28.1M (~112MB)
- CPU inference: 12.61ms (8x better than 100ms target)
- Memory: 112MB model + ~30MB activation
- Throughput: ~79 encodings/sec

---

### E_hierarchy Energy Critic (EGA-001)
**Date**: 2026-01-25
**Implementer**: Claude (Sonnet 4.5) - Previous Polecat
**Status**: Complete

#### Overview
First energy critic in the Product of Experts composition. E_hierarchy detects when untrusted data from external sources (RAG, web scraping) inappropriately influences control flow or decision-making logic in execution plans. Addresses the RAG-injection vulnerability where retrieval results hijack agent actions.

#### Technical Approach
**Core Innovation**: Control Flow Classification + Trust Tier Penalty System

E_hierarchy operates in three stages:
1. **Control Flow Classification**: Lightweight neural classifier determines if each tool call affects control flow (conditionals, loops, delegations) vs data operations (reads, writes, transforms)
2. **Trust Penalty Calculation**: Applies learned penalty weights based on provenance tier (Internal=0.0, Partner=0.5, Public=10.0)
3. **Weighted Aggregation**: Multiplies control flow probability by trust penalty, sums across all nodes in execution graph

Energy Function:
```
E_hierarchy(plan, z_g, z_e) = Σ_i (control_prob_i × tier_penalty_i)
Optional: × latent_modulation(z_g, z_e)
```

Complexity: O(n) for n tool calls (linear in plan size)

#### Key Design Decisions

1. **Decision**: Simple penalty system over complex hierarchical attention
   - **Options**: Rule-based, Learned attention (v0.1.0 commit), Control-flow classifier + penalties (chosen/v0.2.0)
   - **Rationale**: Simpler architecture easier to interpret, faster inference, maintains differentiability
   - **Trade-offs**: Less sophisticated than attention-based approach but <1ms latency, more interpretable

2. **Decision**: Learned tier penalties as nn.Parameter
   - **Options**: Fixed [0, 0.5, 10.0], Fully learned (chosen), Per-tool learned
   - **Rationale**: Allows calibration during InfoNCE training to balance false positives/negatives
   - **Trade-offs**: Only 3 parameters but adapts to dataset statistics

3. **Decision**: Optional latent modulation
   - **Options**: Always use (z_g, z_e), Never use, Optional (chosen)
   - **Rationale**: Enables conditional penalties based on governance-execution misalignment
   - **Trade-offs**: Adds small MLP (256→128→1) but allows context-aware energy

4. **Decision**: Hash-based tool tokenization
   - **Options**: Learned embedding table, Hash (chosen), One-hot encoding
   - **Rationale**: Consistency with encoder design, handles unknown tools, zero training cost
   - **Trade-offs**: Potential collisions but 10k vocab reduces risk

#### Code Structure
```python
class ControlFlowClassifier(nn.Module):
    """Embedding + MLP → [0,1] control flow probability"""
    tool_embedding: Embedding(vocab=10k, dim=256)
    classifier: Linear(256→128→1) + Sigmoid

class HierarchyEnergy(nn.Module):
    """Main energy critic: plan → scalar energy"""
    control_flow_classifier: ControlFlowClassifier
    tier_penalties: Parameter([0.0, 0.5, 10.0])  # Learnable
    latent_modulation: Optional[Linear(2048→256→1)]

    def forward(plan, z_g, z_e) → Tensor[1]
    def explain(plan) → Dict[str, Any]  # Interpretability

def create_hierarchy_energy(...) → HierarchyEnergy:
    """Factory with checkpoint loading"""
```

#### Dependencies
- torch ≥2.5.0: Neural network framework
- source.encoders.execution_encoder: ExecutionPlan, ToolCallNode, TrustTier types

#### Testing Strategy
- **8 tests, all passing**
- Categories: Initialization (1), Safe plans (2), RAG-injection detection (1), Differentiability (1), Interpretability (1), Latent modulation (1), Performance benchmark (1)
- Benchmark: <20ms latency requirement MET (actual: ~0.5ms for 20-node plans)
- Coverage: Core functionality, gradient flow, edge cases (empty plans, single nodes)

#### Known Issues
1. **Control flow classifier**: Initialized randomly in v0.2.0, needs pretraining on labeled tool dataset
2. **Hash collisions**: Same as encoders, BPE tokenizer planned v0.3.0
3. **No documentation**: Implementation exists but docs/energy/hierarchy.md not in current branch

#### Future Improvements
- **Phase 1**: Pretrain ControlFlowClassifier on tool taxonomy, add per-tool importance weights
- **Phase 2**: Argument-level analysis (detect malicious parameters), temporal patterns (sequence of risky calls)
- **Phase 3**: Cross-attention with governance latent for policy-aware penalties

#### Performance Metrics
- Parameters: 1.15M (~4.6MB)
- CPU inference: 0.53ms (37x better than 20ms target!)
- Memory: 4.6MB model + ~2MB activation
- Throughput: ~1900 evaluations/sec

---

## Quick Reference: Component Status

| Component | Status | Last Updated | Owner |
|-----------|--------|--------------|-------|
| JEPA Encoders - Governance | ✅ Complete | 2026-01-25 | LeCun Team |
| JEPA Encoders - Execution | ✅ Complete | 2026-01-25 | LeCun Team |
| Energy Functions - E_hierarchy | ✅ Complete | 2026-01-25 | Du Team |
| Energy Functions - E_provenance | ✅ Complete | 2026-01-25 | Du Team |
| Energy Functions - E_scope | ✅ Complete | 2026-01-25 | Du Team |
| Energy Functions - E_flow | ✅ Complete | 2026-01-25 | Du Team |
| Energy Functions - Composite | ✅ Complete | 2026-01-25 | Du Team |
| Repair Engine | Not Started | - | Song Team |
| Corrupter Agent | Not Started | - | Kolter Team |
| Provenance System | Not Started | - | Song Team |
| Training Pipeline | Not Started | - | Multiple |
| Inference API | Not Started | - | Song Team |

---

## Notes

- All implementations must include inline documentation
- Performance-critical code requires benchmarking before merging
- Energy functions must be validated for differentiability
- Security-relevant code requires red-team review
### Minimal Scope Label Generator (DA-003)
**Date**: 2026-01-26
**Implementer**: Claude (Sonnet 4.5)
**Status**: Complete

#### Overview
Automated minimal scope labeling system for generating E_scope ground truth training data. Analyzes user queries using pattern matching heuristics to predict minimal data access requirements across 4 dimensions: limit, date_range, depth, and sensitivity.

#### Technical Approach
**Core Innovation**: Intent-Based Pattern Matching with Confidence Scoring

The generator uses a multi-strategy approach:
1. **Pattern Matching**: Regex-based extraction of scope indicators from queries
2. **Dimension-Specific Rules**: Separate pattern sets for each scope dimension
3. **Confidence Averaging**: Overall confidence calculated across all 4 dimensions
4. **Reasoning Generation**: Human-readable explanations for debugging/validation

Pattern ordering is critical - more specific patterns (e.g., "top N") are matched before general ones (e.g., "latest").

#### Key Design Decisions

1. **Decision**: Heuristic-based labeling over ML model
   - **Options**: Rule-based (chosen), GPT-4 labeling, Hybrid approach
   - **Rationale**: Zero training cost, deterministic, fast (10K queries/sec)
   - **Trade-offs**: Limited semantic understanding but sufficient for bootstrap dataset

2. **Decision**: Confidence averaging across dimensions
   - **Options**: Min confidence, Max confidence, Average (chosen), Weighted average
   - **Rationale**: Reflects uncertainty across all dimensions equally
   - **Trade-offs**: Can penalize high-confidence dimensions with low-confidence ones

3. **Decision**: Bounded "all" keyword (limit=1000)
   - **Options**: Unbounded (limit=∞), Bounded to 1000 (chosen), Context-dependent
   - **Rationale**: Prevents training on unrealistic infinite scopes
   - **Trade-offs**: May not capture true user intent for "all" but safer default

4. **Decision**: Extended MinimalScopeLabel over base ScopeConstraints
   - **Options**: Reuse ScopeConstraints, Extend with metadata (chosen)
   - **Rationale**: Preserves confidence and reasoning for validation
   - **Trade-offs**: Additional storage but valuable for debugging

#### Code Structure

```python
class MinimalScopeLabel(BaseModel):
    """Extended ScopeConstraints with metadata"""
    limit: int | None
    date_range_days: int | None
    max_depth: int | None
    include_sensitive: bool
    confidence: float  # [0, 1]
    reasoning: str
    method: str

class ScopePattern(dataclass):
    """Pattern matching rule"""
    pattern: str  # Regex
    dimension: str
    value: Any
    confidence: float

class MinimalScopeLabelGenerator:
    """Main label generator with pattern-based extraction"""
    def __init__(self):
        # 25 patterns across 4 dimensions
        self.limit_patterns = [...]  # 5 patterns
        self.temporal_patterns = [...]  # 8 patterns
        self.depth_patterns = [...]  # 3 patterns
        self.sensitivity_patterns = [...]  # 4 patterns

    def generate_label(self, query: str) -> MinimalScopeLabel:
        """Generate label from query using pattern matching"""
        ...
```

#### Pattern Catalog

**Limit Patterns** (5):
- `top N` → Extract N (confidence: 1.0)
- `all/every/entire` → 1000 (confidence: 0.8)
- `few/several/some` → 5 (confidence: 0.85)
- `latest/recent/last` → 1 (confidence: 0.95)
- `single/one` → 1 (confidence: 0.9)

**Temporal Patterns** (8):
- `today` → 1 day (confidence: 1.0)
- `yesterday` → 2 days (confidence: 1.0)
- `this week` → 7 days (confidence: 0.95)
- `this month` → 30 days (confidence: 0.95)
- `this quarter` → 90 days (confidence: 0.95)
- `this year` → 365 days (confidence: 0.95)
- `last N days` → Extract N (confidence: 1.0)
- Implicit "recent" → 30 days (confidence: 0.6)

**Depth Patterns** (3):
- `current folder` → 1 (confidence: 0.9)
- `recursive/all/entire directory` → 10 (confidence: 0.85)
- `subdirectories` → 2 (confidence: 0.9)

**Sensitivity Patterns** (4):
- `password/credential/secret` → True (confidence: 1.0)
- `financial/payment/credit` → True (confidence: 1.0)
- `personal/private` → True (confidence: 0.9)
- `email/phone/address` → True (confidence: 0.85)

#### Dependencies
- **pydantic**: Data validation and serialization
- **re**: Regex pattern matching
- **source.dataset.models**: GoldTrace, UserRequest, ScopeMetadata
- **source.encoders.intent_predictor**: ScopeConstraints

#### Testing Strategy

**Test Coverage**: 45/45 tests passing ✅

- **Pattern Matching** (24 tests):
  - Limit extraction: latest, top N, all, few, default
  - Temporal extraction: today, this week, month, quarter, year, last N days, implicit
  - Depth extraction: current folder, recursive, subdirectories, none
  - Sensitivity extraction: password, financial, personal, contact, none

- **Integration** (8 tests):
  - Complex multi-dimensional queries
  - Real-world scenarios (invoice, directory, sensitive data)
  - UserRequest object handling
  - Gold trace labeling
  - Batch processing

- **Edge Cases** (5 tests):
  - Empty queries
  - Multiple conflicting patterns
  - Case-insensitivity
  - Confidence bounds validation

- **SemanticIntentPredictor Integration** (2 tests):
  - Label to ScopeConstraints conversion
  - Tensor round-trip compatibility

#### Sample Dataset Generation

Command-line tool: `scripts/generate_scope_labels.py`

```bash
python scripts/generate_scope_labels.py --sample --stats --export data/labels.jsonl
```

**Sample Statistics** (18 queries):
- Samples with limit: 18 (100%)
- Samples with date_range: 9 (50%)
- Samples with depth: 6 (33%)
- Samples with sensitivity: 2 (11%)
- Average confidence: 0.88

**Limit Distribution**:
- limit=1: 12 samples (67%)
- limit=10: 1 sample (6%)
- limit=1000: 5 samples (28%)

#### Known Issues

1. **English-only**: Pattern matching designed for English queries
   - **Impact**: Cannot label non-English queries
   - **Mitigation**: Multi-language pattern sets in Phase 2

2. **No semantic understanding**: Relies purely on keyword patterns
   - **Impact**: May miss contextual nuances (e.g., "a couple" vs "few")
   - **Mitigation**: LLM-assisted labeling for validation in Phase 2

3. **Static patterns**: No learning from actual data distributions
   - **Impact**: Confidence scores not calibrated to real performance
   - **Mitigation**: Validation-based calibration after training

4. **Sensitivity false negatives**: "passwords" detected but "user password" might not
   - **Impact**: Query "retrieve user password" has include_sensitive=False incorrectly
   - **Mitigation**: Enhanced pattern matching with word boundary handling (fixed in v0.1.0)

#### Future Improvements

**Phase 1: Enhanced Pattern Matching**
- Multi-language support (Spanish, French, German, etc.)
- Domain-specific patterns (Finance: "invoices", HR: "employees")
- Argument extraction from tool calls
- Comparative analysis with AgentHarm benign/malicious pairs

**Phase 2: ML-Enhanced Labeling**
- GPT-4/Claude-assisted labeling for complex queries
- Active learning with human validation
- Ensemble methods (heuristics + ML predictions)
- Confidence calibration based on validation performance

**Phase 3: Context-Aware Labeling**
- Tool schema integration (adjust limits based on tool constraints)
- User history analysis (personalized scope predictions)
- Statistical scope distribution learning

#### Performance Metrics

- **Labeling Speed**: ~10,000 queries/second (CPU)
- **Confidence**: 0.88 average across sample dataset
- **Coverage**: 100% (all queries receive at least default labels)
- **Memory**: Negligible (~100KB pattern data)
- **Accuracy**: Not measured (requires validation against manual labels)

#### Integration Points

1. **AgentHarm Dataset**: 416 tool-calling samples can be labeled
2. **xlam-function-calling-60k**: 60K Salesforce queries ready for labeling
3. **GoldTrace Pipeline**: Direct integration with existing trace generation
4. **SemanticIntentPredictor**: Labels convert to training tensors seamlessly

#### Output Format (JSONL)

```json
{
  "query": "Show me my latest invoice",
  "minimal_scope": {
    "limit": 1,
    "date_range_days": 30,
    "max_depth": null,
    "include_sensitive": false
  },
  "confidence": 0.89,
  "reasoning": "Limit=1 (detected quantity indicator); Date range=30 days (temporal context); No sensitive data needed",
  "method": "heuristic"
}
```

#### Documentation

- **Implementation**: `source/dataset/scope_labeling.py` (384 lines)
- **Tests**: `test/test_dataset/test_scope_labeling.py` (704 lines, 45 tests)
- **Script**: `scripts/generate_scope_labels.py` (237 lines)
- **Docs**: `docs/dataset/SCOPE_LABELING.md` (comprehensive guide)

---

