# GovernanceEncoder Technical Documentation

**Component**: LSA-001 - Governance Encoder
**Version**: 0.1.0
**Status**: Implemented
**Last Updated**: 2026-01-25

## Overview

The GovernanceEncoder is a transformer-based neural network that maps governance policies, user roles, and session context into a fixed 1024-dimensional latent vector (z_g ∈ R^1024). This latent representation is used by Gatling's energy-based model to measure semantic consistency between policies and proposed agent actions.

## Architecture

### High-Level Design

```
Input: (Policy Document, User Role, Session Context)
  ↓
[Policy Flattening] → Structured sequence with metadata
  ↓
[Token + Position + Structural Embeddings]
  ↓
[Role Embedding (broadcast)]
  ↓
[4x Transformer Blocks with Sparse Structured Attention]
  ↓
[Learned Attention Pooling]
  ↓
[Projection Network]
  ↓
Output: z_g ∈ R^1024
```

### Key Components

#### 1. Policy Flattening
**Module**: `_flatten_policy()`

Converts nested JSON/YAML policy documents into a sequence while preserving structural information:

- **Nodes**: Each key-value pair becomes a sequence element
- **Depth tracking**: Maintains nesting level (0-7)
- **Node types**: Distinguishes dicts (0), lists (1), primitives (2), padding (3)
- **Global tokens**: Top-level sections (depth ≤ 1) marked for global attention

**Example**:
```python
policy = {
    "permissions": {
        "read": ["users", "posts"],
        "write": ["posts"]
    }
}

# Flattened to:
[
    {"token": "permissions", "depth": 0, "node_type": 0, "is_global": True},
    {"token": "permissions.read", "depth": 1, "node_type": 1, "is_global": True},
    {"token": "users", "depth": 2, "node_type": 2, "is_global": False},
    {"token": "posts", "depth": 2, "node_type": 2, "is_global": False},
    ...
]
```

#### 2. Structural Embedding
**Module**: `StructuralEmbedding`

Embeds structural metadata to preserve policy hierarchy:

- **Depth embedding**: 8 possible depth levels
- **Node type embedding**: 16 node type categories
- **Fusion layer**: Combines depth + type → hidden_dim

This ensures the model understands that `permissions.admin.write` is structurally different from `permissions.admin` even if tokens are similar.

#### 3. Sparse Structured Attention
**Module**: `SparseStructuredAttention`

Reduces attention complexity from O(n²) to O(n×w + n×g):

- **Local attention**: Each token attends to window_size=32 neighbors
- **Global attention**: Special "structure tokens" (policy sections) attend to all and receive attention from all
- **Multi-head**: 8 attention heads with 64-dim per head

**Attention Pattern**:
```
        0   1   2   3   4   5   ...
    0  [X   X   X   X   X   X   X ]  ← Global token
    1  [X   X   X   X   .   .   . ]  ← Local window + global
    2  [X   X   X   X   X   .   . ]
    3  [X   X   X   X   X   X   . ]
    4  [X   .   X   X   X   X   X ]
    ...
```

#### 4. Transformer Blocks
**Module**: `TransformerBlock` (4 layers)

Standard transformer encoder architecture:
- Pre-LayerNorm design
- Sparse structured attention
- Feed-forward network (hidden_dim × 4 expansion)
- GELU activation
- Residual connections
- Dropout (0.1)

#### 5. Learned Pooling & Projection
**Modules**: `attention_pool`, `projection`

Converts variable-length sequence to fixed 1024-dim vector:

1. **Attention pooling**: Learns which parts of policy are most important
   ```python
   attn_scores = Linear(hidden_dim → 1)(x)
   weights = softmax(attn_scores)
   pooled = weighted_sum(x, weights)
   ```

2. **Projection network**:
   ```
   Linear(512 → 2048) → GELU → Dropout →
   Linear(2048 → 1024) → LayerNorm
   ```

## Performance Characteristics

### Measured Performance (v0.1.0)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Inference Latency (CPU) | <50ms | ~98ms | 🟡 Optimization needed |
| Model Size | <500MB | ~86MB | ✅ Pass |
| Memory Footprint | <500MB | ~86MB | ✅ Pass |
| Parameter Count | <100M | ~25M | ✅ Pass |
| Test Coverage | >90% | 98% | ✅ Pass |

### Optimization Roadmap for <50ms Target

1. **Model Quantization** (INT8): Expected 2-4x speedup
2. **Knowledge Distillation**: Train smaller student model
3. **ONNX Runtime**: Optimized inference engine
4. **Sparse Attention Tuning**: Reduce window_size for simpler policies
5. **GPU Acceleration**: <10ms on GPU

## Usage

### Basic Usage

```python
from source.encoders.governance_encoder import GovernanceEncoder, PolicySchema

# Initialize encoder
encoder = GovernanceEncoder(latent_dim=1024)
encoder.training = False  # Inference mode

# Define policy
policy = {
    "permissions": {
        "read": ["users", "posts"],
        "write": ["posts"]
    },
    "constraints": {
        "max_records": 100
    }
}

# Encode
import torch
with torch.no_grad():
    z_g = encoder(policy, user_role="analyst")

print(z_g.shape)  # torch.Size([1, 1024])
```

### Using PolicySchema for Validation

```python
from source.encoders.governance_encoder import PolicySchema

# Validates input and handles JSON/YAML parsing
schema = PolicySchema(
    document=policy,  # Can be dict, JSON string, or YAML string
    user_role="analyst",
    session_context={"ip": "192.168.1.1"}
)

z_g = encoder(schema)
```

### Batch Encoding

```python
policies = [
    PolicySchema(document=policy1, user_role="admin"),
    PolicySchema(document=policy2, user_role="user"),
    PolicySchema(document=policy3, user_role="analyst"),
]

z_g_batch = encoder.encode_batch(policies)
print(z_g_batch.shape)  # torch.Size([3, 1024])
```

### Training Mode

```python
# Enable training mode for gradient computation
encoder.train()

z_g = encoder(policy, user_role="analyst")
loss = energy_function(z_g, z_e)  # Compute energy-based loss
loss.backward()  # Gradients flow through encoder

optimizer.step()
```

## Input Specification

### Policy Document Format

**Supported formats**:
- Python dict
- JSON string
- YAML string

**Structure**: Arbitrary nested structure up to depth 7

**Example**:
```yaml
identity:
  role: analyst
  clearance_level: 3

permissions:
  data_access:
    read: [users, logs]
    write: []

  tool_access:
    allowed: [read_file, search_logs]
    forbidden: [delete_file, exec_command]

constraints:
  rate_limits:
    queries_per_minute: 100
  scope:
    max_results: 1000
```

### User Role

**Type**: String
**Examples**: "admin", "analyst", "user", "viewer"
**Encoding**: Mapped to learned role embeddings (supports 32 unique roles)

### Session Context (Optional)

**Type**: Dict[str, Any]
**Purpose**: Additional metadata (currently not used in encoding, reserved for future)
**Examples**:
```python
{
    "timestamp": "2026-01-25T10:00:00Z",
    "source_ip": "192.168.1.1",
    "user_id": "12345"
}
```

## Output Specification

### Governance Latent (z_g)

**Shape**: `[batch_size, 1024]`
**Type**: `torch.Tensor` (float32)
**Range**: Normalized by LayerNorm (approximately zero-mean, unit variance)
**Properties**:
- Deterministic in inference mode (with fixed seed)
- Differentiable for training
- Semantic similarity: Similar policies → nearby vectors in latent space

## Implementation Details

### Tokenization Strategy

**Current (v0.1.0)**: Hash-based tokenization
```python
def _tokenize(text: str) -> int:
    return hash(text) % 10000
```

**Limitations**:
- Hash collisions possible
- No subword tokenization
- Not optimized for semantic similarity

**Future**: Replace with learned BPE tokenizer for better semantic understanding

### Role Vocabulary Management

Roles are dynamically mapped to indices:
```python
role_vocab = {
    "admin": 0,
    "analyst": 1,
    "user": 2,
    ...
}
```

**Capacity**: 32 unique roles
**Overflow handling**: Hash-based fallback for roles beyond capacity

### Sequence Length Handling

**Max sequence length**: 512 tokens

**Truncation strategy**:
1. Keep all global tokens (top-level sections)
2. Truncate non-global tokens if necessary
3. Ensures policy structure is preserved

**Padding**: Sequences <512 tokens are padded with special padding tokens

## Integration with Energy Function

The GovernanceEncoder is designed to work with Gatling's energy-based model:

```python
# Governance encoding
z_g = governance_encoder(policy, user_role, session_context)

# Execution encoding (from ExecutionEncoder, not yet implemented)
z_e = execution_encoder(proposed_plan, provenance)

# Energy computation
E_total = energy_function(z_g, z_e)

if E_total > threshold:
    # High energy → policy violation
    repaired_plan = repair_engine(proposed_plan, z_g)
else:
    # Low energy → safe to execute
    execute(proposed_plan)
```

## Testing

### Test Coverage: 98%

**Test suites**:
- Initialization tests
- PolicySchema validation
- Core encoding functionality
- Variable-length handling
- Gradient flow (differentiability)
- Batch processing
- Performance benchmarks
- Internal components
- Edge cases
- Integration tests

### Running Tests

```bash
# All tests
uv run pytest test/test_encoders/test_governance_encoder.py -v

# With coverage
uv run pytest test/test_encoders/test_governance_encoder.py --cov=source/encoders --cov-report=term-missing

# Benchmarks only
uv run pytest test/test_encoders/test_governance_encoder.py -m benchmark

# Exclude benchmarks
uv run pytest test/test_encoders/test_governance_encoder.py -m "not benchmark"
```

## Known Issues and Limitations

### Current Limitations

1. **Latency**: ~98ms on CPU (target: <50ms)
   - **Impact**: May not meet real-time requirements
   - **Mitigation**: See optimization roadmap above

2. **Tokenization**: Simple hash-based approach
   - **Impact**: Potential hash collisions, no semantic understanding
   - **Mitigation**: Plan to implement BPE tokenizer

3. **Batch Processing**: Sequential processing of batches
   - **Impact**: Not fully utilizing batching potential
   - **Mitigation**: Implement true batched forward pass

4. **Session Context**: Not currently used in encoding
   - **Impact**: Missing potential signal
   - **Mitigation**: Future architectural enhancement

### Future Enhancements

1. **Hierarchical Attention**: Explicit tree-structured attention
2. **Policy-Specific Pretraining**: Pretrain on large policy corpus
3. **Dynamic Architecture**: Adapt model size based on policy complexity
4. **Caching**: Cache governance latents for frequently-used policies

## References

### Research Papers

- **StructFormer**: [Document Structure-based Masked Attention](https://arxiv.org/html/2411.16618v1)
- **ETC**: [Encoding Long and Structured Inputs in Transformers](https://arxiv.org/abs/2004.08483)
- **Longformer**: [Sparse Attention Patterns](https://arxiv.org/abs/2004.05150)

### Related Documentation

- [Project Gatling PRD](../PRD.md)
- [Workstream Distribution](../WORK-DISTRIBUTION.md)
- [Energy Function Specification](../energy/) (coming soon)
- [Training Pipeline](../DATASET_WORK.md)

## Changelog

### v0.1.0 (2026-01-25)
- Initial implementation
- Sparse structured attention mechanism
- PolicySchema validation
- Comprehensive test suite (98% coverage)
- Performance benchmarking
- Factory function for easy instantiation
