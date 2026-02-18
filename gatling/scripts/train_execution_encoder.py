"""
Train Execution Encoder using VICReg (Stage 1: Self-Supervised Pre-training)

This script implements JEPA Stage 1 training on benign samples only.
Uses VICReg loss (Variance-Invariance-Covariance Regularization) which is
ideal for learning representations without negative samples.

VICReg enforces:
- Variance: Representations should be diverse (avoid collapse)
- Invariance: Augmented views of same plan should be similar
- Covariance: Feature dimensions should be decorrelated

Usage:
    # Train on local dataset
    uv run python scripts/train_execution_encoder.py \
        --dataset data/tier1_free_loaders.jsonl \
        --max-samples 50000 \
        --epochs 10

    # Train from HuggingFace
    uv run python scripts/train_execution_encoder.py \
        --dataset OzLabs/gatling-tier1-338k \
        --max-samples 338000 \
        --epochs 30

References:
    - VICReg: https://arxiv.org/abs/2105.04906
    - JEPA: https://arxiv.org/abs/2301.08243
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add source to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from source.encoders import ExecutionEncoder, ExecutionPlan


class ExecutionPlanDataset(Dataset):
    """Dataset of benign execution plans for JEPA training."""

    def __init__(self, plans: list[dict[str, Any]]):
        """
        Initialize dataset.

        Args:
            plans: List of ExecutionPlan dicts
        """
        self.plans = plans

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get plan by index."""
        return self.plans[idx]["execution_plan"]


def collate_plans(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Custom collate function for ExecutionPlan dicts.

    PyTorch's default collate tries to stack tensors, but we have
    variable-length plans. Return list of dicts instead.

    Args:
        batch: List of ExecutionPlan dicts

    Returns:
        Same list (no batching into tensors)
    """
    return batch


class VICRegLoss(nn.Module):
    """
    VICReg loss for self-supervised representation learning.

    Loss = λ * variance_loss + μ * invariance_loss + ν * covariance_loss

    This loss doesn't require negative samples - perfect for Stage 1 training.
    """

    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        eps: float = 1e-4
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """MSE between augmented views."""
        return F.mse_loss(z1, z2)

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Enforce standard deviation > 1 to prevent collapse."""
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.mean(F.relu(1 - std))

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Decorrelate features (off-diagonal of covariance matrix should be 0)."""
        batch_size = z.shape[0]
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (batch_size - 1)

        # Off-diagonal elements
        off_diag = cov.flatten()[:-1].view(cov.shape[0] - 1, cov.shape[1] + 1)
        off_diag = off_diag[:, 1:].flatten()

        return off_diag.pow(2).sum() / z.shape[1]

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute VICReg loss.

        Args:
            z1: Embeddings from view 1 [batch_size, latent_dim]
            z2: Embeddings from view 2 [batch_size, latent_dim]

        Returns:
            loss: Total VICReg loss
            metrics: Dict of individual loss components
        """
        inv_loss = self.invariance_loss(z1, z2)
        var_loss = (self.variance_loss(z1) + self.variance_loss(z2)) / 2
        cov_loss = (self.covariance_loss(z1) + self.covariance_loss(z2)) / 2

        loss = (
            self.inv_weight * inv_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )

        metrics = {
            "inv_loss": inv_loss.item(),
            "var_loss": var_loss.item(),
            "cov_loss": cov_loss.item(),
            "total_loss": loss.item()
        }

        return loss, metrics


def augment_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """
    Create augmented view of execution plan.

    Augmentations that preserve semantic meaning:
    - Permute independent nodes (those not connected by edges)
    - Add small noise to scope_volume (±10%)
    - Randomly drop non-critical metadata

    Args:
        plan: Original ExecutionPlan dict

    Returns:
        Augmented ExecutionPlan dict
    """
    import copy
    import random

    aug_plan = copy.deepcopy(plan)

    # Augmentation 1: Permute independent nodes
    nodes = aug_plan["nodes"]
    edges = aug_plan.get("edges", [])

    if len(nodes) > 1 and len(edges) == 0:
        # Only permute if there are no dependencies
        random.shuffle(nodes)

    # Augmentation 2: Add noise to scope_volume (±10%)
    for node in nodes:
        if "scope_volume" in node:
            noise = random.uniform(0.9, 1.1)
            node["scope_volume"] = max(1, int(node["scope_volume"] * noise))

    return aug_plan


def load_dataset_from_source(
    dataset_path: str,
    max_samples: int | None = None
) -> list[dict[str, Any]]:
    """
    Load dataset from HuggingFace or local file.

    Args:
        dataset_path: HuggingFace dataset ID or local JSONL path
        max_samples: Maximum number of samples to load

    Returns:
        List of ExecutionPlan dicts
    """
    print(f"Loading dataset from: {dataset_path}")

    # Check if it's a local file
    if Path(dataset_path).exists():
        print("  Loading from local file...")
        plans = []
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                plans.append(sample)
        print(f"  ✓ Loaded {len(plans)} samples")
        return plans

    # Otherwise, load from HuggingFace
    print("  Loading from HuggingFace Hub...")
    dataset = load_dataset(dataset_path, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    plans = [sample for sample in dataset]
    print(f"  ✓ Loaded {len(plans)} samples")

    return plans


def train_execution_encoder(
    dataset_path: str,
    output_dir: str = "outputs/execution_encoder",
    max_samples: int | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    save_every: int = 5
) -> None:
    """
    Train Execution Encoder using VICReg loss.

    Args:
        dataset_path: Path to dataset (HuggingFace ID or local file)
        output_dir: Directory to save checkpoints
        max_samples: Maximum samples to use (None = all)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on (cpu/cuda/mps)
        save_every: Save checkpoint every N epochs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("🚀 Gatling Execution Encoder Training (Stage 1: VICReg)")
    print(f"{'=' * 70}\n")

    # Load dataset
    plans = load_dataset_from_source(dataset_path, max_samples)

    # Create dataset and dataloader
    dataset = ExecutionPlanDataset(plans)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Sequential for now
        drop_last=True,  # VICReg requires consistent batch size
        collate_fn=collate_plans  # Custom collate for variable-length plans
    )

    print(f"📊 Dataset: {len(dataset)} samples")
    print(f"📦 Batches per epoch: {len(dataloader)}")
    print(f"🔧 Device: {device}\n")

    # Initialize model
    model = ExecutionEncoder(latent_dim=1024)
    model = model.to(device)
    model.train()

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize loss function
    criterion = VICRegLoss()

    # Training loop
    print("🏋️ Starting training...\n")

    for epoch in range(1, epochs + 1):
        epoch_metrics = {
            "inv_loss": 0.0,
            "var_loss": 0.0,
            "cov_loss": 0.0,
            "total_loss": 0.0
        }

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress:
            # Create two augmented views
            aug1 = [augment_plan(plan) for plan in batch]
            aug2 = [augment_plan(plan) for plan in batch]

            # Encode both views
            try:
                z1 = torch.stack([model(plan) for plan in aug1]).squeeze(1)
                z2 = torch.stack([model(plan) for plan in aug2]).squeeze(1)
            except Exception as e:
                print(f"\n⚠️  Skipping batch due to error: {e}")
                continue

            # Compute VICReg loss
            loss, metrics = criterion(z1, z2)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            for key, value in metrics.items():
                epoch_metrics[key] += value

            # Update progress bar
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "inv": f"{metrics['inv_loss']:.4f}",
                "var": f"{metrics['var_loss']:.4f}",
                "cov": f"{metrics['cov_loss']:.4f}"
            })

        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total Loss: {epoch_metrics['total_loss']:.4f}")
        print(f"  Invariance: {epoch_metrics['inv_loss']:.4f}")
        print(f"  Variance:   {epoch_metrics['var_loss']:.4f}")
        print(f"  Covariance: {epoch_metrics['cov_loss']:.4f}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            checkpoint_path = Path(output_dir) / f"encoder_epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")

        print()

    # Save final model
    final_path = Path(output_dir) / "encoder_final.pt"
    torch.save(model.state_dict(), final_path)

    print(f"{'=' * 70}")
    print("✅ Training Complete!")
    print(f"{'=' * 70}")
    print(f"\n📍 Final model: {final_path}")
    print(f"💡 Load with: model.load_state_dict(torch.load('{final_path}'))\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Execution Encoder (JEPA Stage 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/tier1_free_loaders.jsonl",
        help="Dataset path (HuggingFace ID or local JSONL file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/execution_encoder",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (default: all)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )

    args = parser.parse_args()

    # Load environment variables (for HuggingFace token)
    load_dotenv()

    train_execution_encoder(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()
