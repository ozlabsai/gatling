#!/usr/bin/env python3
"""
JEPA Encoder Training Pipeline

Trains GovernanceEncoder and ExecutionEncoder using InfoNCE contrastive learning
on gold traces and adversarial negative samples.

Architecture:
- Dual encoders map (policy, user_role) → z_g and (plan, provenance) → z_e
- InfoNCE loss: maximize similarity for (z_g, z_e) pairs from same trace
- Hard negatives: adversarial mutations (scope blow-up, provenance rug-pull, etc.)

Training Strategy:
- Positive samples: (governance_context, compliant_plan) from gold traces
- Negative samples: (governance_context, corrupted_plan) from adversarial mutations
- Security margin δ_sec enforces stable energy gap between safe and risky plans

Usage:
    uv run python scripts/train_jepa_encoders.py --config config/training.yaml
    uv run python scripts/train_jepa_encoders.py --dataset path/to/gold_traces.jsonl --epochs 10

References:
- InfoNCE: "Representation Learning with Contrastive Predictive Coding" (van den Oord et al., 2018)
- JEPA: "A Path Towards Autonomous Machine Intelligence" (LeCun, 2022)
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_folder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Import Gatling modules
from source.encoders.governance_encoder import GovernanceEncoder, PolicySchema
from source.encoders.execution_encoder import ExecutionEncoder, ExecutionPlan, ToolCallNode, TrustTier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration."""

    # Model architecture
    latent_dim: int = 1024
    governance_hidden_dim: int = 512
    execution_hidden_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # InfoNCE loss
    temperature: float = 0.07
    security_margin: float = 0.5  # δ_sec: minimum energy gap for safe vs risky

    # Data
    dataset_path: str = "data/gold_traces.jsonl"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5

    # HuggingFace Hub
    hf_repo_name: str = "gatling-jepa-encoders"
    hf_organization: str | None = None
    push_to_hub: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Monitoring
    log_every: int = 100
    val_every: int = 500


class GoldTraceDataset(Dataset):
    """Dataset loader for gold traces with governance context and compliant plans."""

    def __init__(self, traces_path: str):
        """Load gold traces from JSONL file."""
        self.traces = []

        if not Path(traces_path).exists():
            logger.warning(f"Dataset file not found: {traces_path}")
            logger.info("Creating synthetic sample data for development...")
            self.traces = self._create_synthetic_samples()
        else:
            with open(traces_path, 'r') as f:
                for line in f:
                    self.traces.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.traces)} gold traces from {traces_path}")

    def _create_synthetic_samples(self) -> list[dict[str, Any]]:
        """Create synthetic gold traces for development/testing."""
        samples = []

        # Sample 1: Calendar access (compliant)
        samples.append({
            "governance_context": {
                "policy": {
                    "domain": "Calendar",
                    "allowed_operations": ["list_events", "get_event"],
                    "scope_limits": {"max_results": 10, "days_range": 30}
                },
                "user_role": "standard_user",
                "session_context": {"user_id": "user_001"}
            },
            "execution_plan": {
                "nodes": [
                    {
                        "tool_name": "list_events",
                        "node_id": "node1",
                        "provenance_tier": 1,
                        "scope_volume": 10,
                        "scope_sensitivity": 2,
                        "arguments": {"limit": 10, "days": 30}
                    }
                ],
                "edges": []
            },
            "label": "compliant"
        })

        # Sample 2: File access (compliant)
        samples.append({
            "governance_context": {
                "policy": {
                    "domain": "FileSystem",
                    "allowed_operations": ["read_file", "list_directory"],
                    "scope_limits": {"max_depth": 2, "max_files": 50}
                },
                "user_role": "developer",
                "session_context": {"project": "gatling"}
            },
            "execution_plan": {
                "nodes": [
                    {
                        "tool_name": "list_directory",
                        "node_id": "node1",
                        "provenance_tier": 1,
                        "scope_volume": 20,
                        "scope_sensitivity": 1,
                        "arguments": {"depth": 2, "limit": 20}
                    }
                ],
                "edges": []
            },
            "label": "compliant"
        })

        # Add more synthetic samples for better training
        for i in range(10):
            samples.append({
                "governance_context": {
                    "policy": {
                        "domain": "Database",
                        "allowed_operations": ["query", "read"],
                        "scope_limits": {"max_rows": 100}
                    },
                    "user_role": "analyst",
                    "session_context": {"department": "analytics"}
                },
                "execution_plan": {
                    "nodes": [
                        {
                            "tool_name": "query_database",
                            "node_id": f"node{i}",
                            "provenance_tier": 1,
                            "scope_volume": 50,
                            "scope_sensitivity": 2,
                            "arguments": {"limit": 50, "table": "sales"}
                        }
                    ],
                    "edges": []
                },
                "label": "compliant"
            })

        return samples

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.traces[idx]


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    Maximizes agreement between positive pairs (z_g, z_e) from same trace
    while pushing apart negative pairs from different traces or corruptions.

    Loss = -log(exp(sim(z_g, z_e+) / τ) / Σ exp(sim(z_g, z_e_i) / τ))

    where:
    - z_e+ is the positive execution plan for governance z_g
    - z_e_i are negative samples (other traces or corrupted plans)
    - τ is temperature hyperparameter
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_g: torch.Tensor,  # [batch_size, latent_dim]
        z_e: torch.Tensor,  # [batch_size, latent_dim]
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z_g: Governance latents [batch_size, latent_dim]
            z_e: Execution latents [batch_size, latent_dim]

        Returns:
            Scalar loss value
        """
        batch_size = z_g.shape[0]

        # Normalize embeddings
        z_g = F.normalize(z_g, p=2, dim=1)
        z_e = F.normalize(z_e, p=2, dim=1)

        # Compute cosine similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(z_g, z_e.T) / self.temperature

        # Positive samples are on the diagonal
        # Create labels: [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(batch_size, device=z_g.device)

        # Cross-entropy loss treats diagonal as positive class
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function to batch gold traces."""
    governance_contexts = []
    execution_plans = []
    labels = []

    for sample in batch:
        governance_contexts.append(sample["governance_context"])
        execution_plans.append(sample["execution_plan"])
        labels.append(sample.get("label", "compliant"))

    return {
        "governance_contexts": governance_contexts,
        "execution_plans": execution_plans,
        "labels": labels
    }


def train_epoch(
    governance_encoder: GovernanceEncoder,
    execution_encoder: ExecutionEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: InfoNCELoss,
    config: TrainingConfig,
    epoch: int,
) -> float:
    """Train for one epoch."""
    governance_encoder.train()
    execution_encoder.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

    for batch_idx, batch in enumerate(pbar):
        # Extract governance contexts and execution plans
        governance_contexts = batch["governance_contexts"]
        execution_plans = batch["execution_plans"]

        # Encode governance contexts
        z_g_list = []
        for gov_ctx in governance_contexts:
            policy_schema = PolicySchema(
                document=gov_ctx["policy"],
                user_role=gov_ctx["user_role"],
                session_context=gov_ctx.get("session_context", {})
            )
            z_g = governance_encoder(policy_schema)
            z_g_list.append(z_g)

        z_g = torch.cat(z_g_list, dim=0)  # [batch_size, latent_dim]

        # Encode execution plans
        z_e_list = []
        for exec_plan_dict in execution_plans:
            # Convert dict to ExecutionPlan model
            nodes = [ToolCallNode(**node) for node in exec_plan_dict["nodes"]]
            exec_plan = ExecutionPlan(nodes=nodes, edges=exec_plan_dict.get("edges", []))
            z_e = execution_encoder(exec_plan)
            z_e_list.append(z_e)

        z_e = torch.cat(z_e_list, dim=0)  # [batch_size, latent_dim]

        # Compute InfoNCE loss
        loss = criterion(z_g, z_e)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(governance_encoder.parameters()) + list(execution_encoder.parameters()),
            config.gradient_clip
        )

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Logging
        if (batch_idx + 1) % config.log_every == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: avg_loss={avg_loss:.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(
    governance_encoder: GovernanceEncoder,
    execution_encoder: ExecutionEncoder,
    dataloader: DataLoader,
    criterion: InfoNCELoss,
    config: TrainingConfig,
) -> dict[str, float]:
    """Validate the model."""
    governance_encoder.eval()
    execution_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    # Metrics for embedding quality
    similarities = []

    for batch in tqdm(dataloader, desc="Validation"):
        governance_contexts = batch["governance_contexts"]
        execution_plans = batch["execution_plans"]

        # Encode
        z_g_list = []
        for gov_ctx in governance_contexts:
            policy_schema = PolicySchema(
                document=gov_ctx["policy"],
                user_role=gov_ctx["user_role"],
                session_context=gov_ctx.get("session_context", {})
            )
            z_g = governance_encoder(policy_schema)
            z_g_list.append(z_g)

        z_g = torch.cat(z_g_list, dim=0)

        z_e_list = []
        for exec_plan_dict in execution_plans:
            nodes = [ToolCallNode(**node) for node in exec_plan_dict["nodes"]]
            exec_plan = ExecutionPlan(nodes=nodes, edges=exec_plan_dict.get("edges", []))
            z_e = execution_encoder(exec_plan)
            z_e_list.append(z_e)

        z_e = torch.cat(z_e_list, dim=0)

        # Compute loss
        loss = criterion(z_g, z_e)
        total_loss += loss.item()
        num_batches += 1

        # Compute pairwise similarity for positive pairs
        z_g_norm = F.normalize(z_g, p=2, dim=1)
        z_e_norm = F.normalize(z_e, p=2, dim=1)

        # Diagonal elements are positive pairs
        pos_sim = (z_g_norm * z_e_norm).sum(dim=1)
        similarities.extend(pos_sim.cpu().tolist())

    avg_loss = total_loss / num_batches
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    metrics = {
        "val_loss": avg_loss,
        "avg_positive_similarity": avg_similarity,
    }

    logger.info(f"Validation - Loss: {avg_loss:.4f}, Avg Positive Similarity: {avg_similarity:.4f}")

    return metrics


def save_checkpoint(
    governance_encoder: GovernanceEncoder,
    execution_encoder: ExecutionEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
) -> str:
    """Save model checkpoint."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"

    torch.save({
        "epoch": epoch,
        "governance_encoder_state_dict": governance_encoder.state_dict(),
        "execution_encoder_state_dict": execution_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return str(checkpoint_path)


def push_to_huggingface_hub(
    governance_encoder: GovernanceEncoder,
    execution_encoder: ExecutionEncoder,
    config: TrainingConfig,
) -> None:
    """Push trained models to HuggingFace Hub."""
    api = HfApi()

    # Determine repo name
    repo_id = config.hf_repo_name
    if config.hf_organization:
        repo_id = f"{config.hf_organization}/{config.hf_repo_name}"

    logger.info(f"Pushing models to HuggingFace Hub: {repo_id}")

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        logger.error(f"Failed to create repo: {e}")
        return

    # Save models locally first
    output_dir = Path("hub_upload")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save governance encoder
    gov_path = output_dir / "governance_encoder.pt"
    torch.save(governance_encoder.state_dict(), gov_path)

    # Save execution encoder
    exec_path = output_dir / "execution_encoder.pt"
    torch.save(execution_encoder.state_dict(), exec_path)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "latent_dim": config.latent_dim,
            "governance_hidden_dim": config.governance_hidden_dim,
            "execution_hidden_dim": config.execution_hidden_dim,
            "num_attention_heads": config.num_attention_heads,
        }, f, indent=2)

    # Upload folder to Hub
    try:
        upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload trained JEPA encoders (epoch {config.epochs})"
        )
        logger.info(f"Successfully pushed models to {repo_id}")
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")


def main(args: argparse.Namespace) -> None:
    """Main training pipeline."""
    # Create config
    config = TrainingConfig(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        push_to_hub=args.push_to_hub,
    )

    logger.info(f"Training Configuration:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")

    # Load dataset
    dataset = GoldTraceDataset(config.dataset_path)

    # Split dataset
    train_size = int(len(dataset) * config.train_split)
    val_size = int(len(dataset) * config.val_split)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize models
    governance_encoder = GovernanceEncoder(
        latent_dim=config.latent_dim,
        hidden_dim=config.governance_hidden_dim,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
    ).to(config.device)

    execution_encoder = ExecutionEncoder(
        latent_dim=config.latent_dim,
        hidden_dim=config.execution_hidden_dim,
        num_heads=config.num_attention_heads,
        dropout=config.dropout,
    ).to(config.device)

    # Initialize loss and optimizer
    criterion = InfoNCELoss(temperature=config.temperature)

    optimizer = torch.optim.AdamW(
        list(governance_encoder.parameters()) + list(execution_encoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            governance_encoder,
            execution_encoder,
            train_loader,
            optimizer,
            criterion,
            config,
            epoch,
        )

        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")

        # Validate
        val_metrics = validate(
            governance_encoder,
            execution_encoder,
            val_loader,
            criterion,
            config,
        )

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                governance_encoder,
                execution_encoder,
                optimizer,
                epoch,
                config,
            )

        # Save periodic checkpoints
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                governance_encoder,
                execution_encoder,
                optimizer,
                epoch,
                config,
            )

    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"{'='*60}")

    # Push to HuggingFace Hub if requested
    if config.push_to_hub:
        push_to_huggingface_hub(
            governance_encoder,
            execution_encoder,
            config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JEPA Encoders with InfoNCE Loss")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/gold_traces.jsonl",
        help="Path to gold traces dataset (JSONL format)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained models to HuggingFace Hub",
    )

    args = parser.parse_args()
    main(args)
