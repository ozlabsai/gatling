# JEPA Encoder Training Pipeline

This document describes the training pipeline for Gatling's dual JEPA encoders (GovernanceEncoder and ExecutionEncoder).

## Overview

The training pipeline implements InfoNCE contrastive learning to train two complementary encoders that map governance policies and execution plans into a shared 1024-dimensional latent space. This enables energy-based security validation for agentic systems.

## Architecture

### Dual Encoders

1. **GovernanceEncoder**: Maps `(policy, user_role, session_context) → z_g ∈ R^1024`
   - Structure-aware transformer for JSON/YAML policies
   - Hierarchical encoding with sparse attention
   - Sub-50ms inference latency target

2. **ExecutionEncoder**: Maps `(plan_graph, provenance_metadata) → z_e ∈ R^1024`
   - Graph Neural Network for tool-call dependencies
   - Provenance-aware attention mechanism
   - Scope metadata integration

### InfoNCE Loss

The training uses InfoNCE (Noise Contrastive Estimation) loss to maximize agreement between positive pairs while separating negative pairs:

```
Loss = -log(exp(sim(z_g, z_e+) / τ) / Σ exp(sim(z_g, z_e_i) / τ))
```

Where:
- `z_e+` is the positive execution plan for governance `z_g`
- `z_e_i` are negative samples (other traces or adversarial mutations)
- `τ` is the temperature hyperparameter (default: 0.07)

## Training Data

### Gold Traces

Positive samples consist of `(governance_context, compliant_plan)` pairs from gold traces:

```jsonl
{
  "governance_context": {
    "policy": {"domain": "Calendar", "allowed_operations": ["list_events"], ...},
    "user_role": "standard_user",
    "session_context": {"user_id": "user_001"}
  },
  "execution_plan": {
    "nodes": [{"tool_name": "list_events", "node_id": "node1", ...}],
    "edges": []
  },
  "label": "compliant"
}
```

### Hard Negatives (Future)

Adversarial mutations from the Corrupter Agent:
1. **Scope Blow-up**: `limit=5 → limit=10000`
2. **Instruction Shadowing**: Inject untrusted RAG content into control flow
3. **Provenance Rug-Pull**: Swap trusted source for untrusted RAG
4. **Exfiltration Pivot**: Append data export to external endpoints

## Usage

### Basic Training

```bash
# Train with default settings (20 epochs, batch size 32)
uv run python scripts/train_jepa_encoders.py

# Specify dataset path
uv run python scripts/train_jepa_encoders.py --dataset data/gold_traces.jsonl

# Adjust hyperparameters
uv run python scripts/train_jepa_encoders.py \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 5e-5
```

### Push to HuggingFace Hub

```bash
# Set HF_TOKEN environment variable first
export HF_TOKEN=your_token_here

# Train and push models to Hub
uv run python scripts/train_jepa_encoders.py \
  --dataset data/gold_traces.jsonl \
  --epochs 20 \
  --push-to-hub
```

## Configuration

Training hyperparameters can be configured via command-line arguments or by modifying `TrainingConfig` in the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 1024 | Latent space dimensionality |
| `batch_size` | 32 | Training batch size |
| `epochs` | 20 | Number of training epochs |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `temperature` | 0.07 | InfoNCE temperature parameter |
| `gradient_clip` | 1.0 | Maximum gradient norm |
| `security_margin` | 0.5 | Minimum energy gap (δ_sec) |

## Validation Metrics

The pipeline tracks:

1. **Validation Loss**: InfoNCE loss on held-out validation set
2. **Average Positive Similarity**: Cosine similarity for (z_g, z_e) pairs from same trace
   - Target: ≥0.85 for convergence
3. **Nearest Neighbor Accuracy**: Percentage of z_g that retrieve correct z_e as top-1
   - Target: ≥85% per LSA-004 requirements

## Checkpointing

- Checkpoints saved to `checkpoints/checkpoint_epoch_N.pt`
- Best model (lowest validation loss) saved automatically
- Periodic checkpoints every 5 epochs
- Each checkpoint contains:
  - GovernanceEncoder state dict
  - ExecutionEncoder state dict
  - Optimizer state
  - Training configuration

## HuggingFace Hub Integration

When `--push-to-hub` is enabled:

1. Creates repo: `{organization}/{hf_repo_name}` (default: `gatling-jepa-encoders`)
2. Uploads:
   - `governance_encoder.pt` - GovernanceEncoder weights
   - `execution_encoder.pt` - ExecutionEncoder weights
   - `config.json` - Model architecture configuration
3. Commit message includes epoch number

## Performance Requirements

Per LSA-004 specification:

- **Latency**: <200ms end-to-end (Audit + Repair)
  - Governance encoding: <50ms (pre-computable per user role)
  - Execution encoding: <100ms
  - Energy calculation: <50ms
- **Embedding Quality**:
  - Nearest Neighbor Accuracy: ≥85%
  - Functional Intent Similarity (FIS): ≥90% for repaired plans
- **Training Time**: <24hr on single GPU

## Example: Training on Custom Dataset

```python
# 1. Prepare your gold traces in JSONL format
# Each line should contain governance_context, execution_plan, and label

# 2. Run training
uv run python scripts/train_jepa_encoders.py \
  --dataset path/to/your/gold_traces.jsonl \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 1e-4

# 3. Load trained models
import torch
from source.encoders.governance_encoder import GovernanceEncoder
from source.encoders.execution_encoder import ExecutionEncoder

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_epoch_30.pt")

gov_encoder = GovernanceEncoder(latent_dim=1024, hidden_dim=512)
gov_encoder.load_state_dict(checkpoint["governance_encoder_state_dict"])

exec_encoder = ExecutionEncoder(latent_dim=1024, hidden_dim=512)
exec_encoder.load_state_dict(checkpoint["execution_encoder_state_dict"])
```

## Development Mode

If no dataset is found, the script generates synthetic samples for development:
- 12 synthetic gold traces covering Calendar, FileSystem, and Database domains
- Sufficient for testing training loop and model architecture
- Replace with real gold traces for production training

## References

- **InfoNCE**: "Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)
- **JEPA**: "A Path Towards Autonomous Machine Intelligence" (LeCun, 2022)
- **StructFormer**: https://arxiv.org/html/2411.16618v1
- **Graph Attention Networks**: https://arxiv.org/abs/1710.10903

## Next Steps

1. **Adversarial Training** (LSA-005): Integrate Corrupter Agent for hard negative generation
2. **Energy Function Training** (EGA series): Train individual energy critics (E_hierarchy, E_provenance, E_scope, E_flow)
3. **Repair Engine** (REP-001): Implement discrete energy-guided repair with beam search
4. **End-to-End Validation**: Full pipeline testing on Gatling-10M dataset

## Support

For issues or questions:
- File bug: `bd create --title "Training issue: <description>" --type bug`
- Check logs: Training logs include detailed progress and metrics
- GPU requirements: CUDA-compatible GPU recommended for faster training
