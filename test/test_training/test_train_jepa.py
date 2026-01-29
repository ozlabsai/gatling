"""
Tests for JEPA encoder training pipeline.

Validates InfoNCE loss, dataset loading, and training loop components.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from scripts.train_jepa_encoders import (
    GoldTraceDataset,
    InfoNCELoss,
    TrainingConfig,
    collate_fn,
)


class TestInfoNCELoss:
    """Test InfoNCE contrastive loss function."""

    def test_loss_computation(self):
        """InfoNCE loss should compute without errors."""
        criterion = InfoNCELoss(temperature=0.07)

        batch_size = 4
        latent_dim = 1024

        z_g = torch.randn(batch_size, latent_dim)
        z_e = torch.randn(batch_size, latent_dim)

        loss = criterion(z_g, z_e)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0.0

    def test_perfect_alignment_low_loss(self):
        """Identical embeddings should have low loss."""
        criterion = InfoNCELoss(temperature=0.07)

        batch_size = 4
        latent_dim = 1024

        # Same embeddings for governance and execution
        z = torch.randn(batch_size, latent_dim)

        loss = criterion(z, z)

        # Loss should be low (near zero) for perfect alignment
        assert loss.item() < 1.0

    def test_temperature_effect(self):
        """Higher temperature should reduce loss magnitude."""
        batch_size = 4
        latent_dim = 1024

        z_g = torch.randn(batch_size, latent_dim)
        z_e = torch.randn(batch_size, latent_dim)

        loss_low_temp = InfoNCELoss(temperature=0.01)(z_g, z_e)
        loss_high_temp = InfoNCELoss(temperature=1.0)(z_g, z_e)

        # Higher temperature typically gives smaller gradients
        assert loss_high_temp.item() < loss_low_temp.item() * 2


class TestGoldTraceDataset:
    """Test gold trace dataset loader."""

    def test_synthetic_samples(self):
        """Dataset should create synthetic samples if file not found."""
        dataset = GoldTraceDataset("nonexistent_file.jsonl")

        assert len(dataset) > 0
        sample = dataset[0]

        assert "governance_context" in sample
        assert "execution_plan" in sample
        assert "label" in sample

    def test_load_from_jsonl(self):
        """Dataset should load traces from JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write sample gold trace
            trace = {
                "governance_context": {
                    "policy": {"domain": "Test"},
                    "user_role": "tester",
                    "session_context": {},
                },
                "execution_plan": {
                    "nodes": [
                        {
                            "tool_name": "test_tool",
                            "node_id": "node1",
                            "provenance_tier": 1,
                            "scope_volume": 1,
                            "scope_sensitivity": 1,
                            "arguments": {},
                        }
                    ],
                    "edges": [],
                },
                "label": "compliant",
            }
            json.dump(trace, f)
            f.write("\n")
            temp_path = f.name

        try:
            dataset = GoldTraceDataset(temp_path)
            assert len(dataset) == 1

            sample = dataset[0]
            assert sample["governance_context"]["user_role"] == "tester"
            assert sample["label"] == "compliant"
        finally:
            Path(temp_path).unlink()

    def test_getitem(self):
        """Dataset __getitem__ should return valid samples."""
        dataset = GoldTraceDataset("nonexistent.jsonl")

        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert isinstance(sample, dict)
            assert "governance_context" in sample
            assert "execution_plan" in sample


class TestCollateFn:
    """Test batch collation function."""

    def test_collate_batch(self):
        """Collate function should properly batch samples."""
        samples = [
            {
                "governance_context": {"policy": {}, "user_role": "user1"},
                "execution_plan": {"nodes": [], "edges": []},
                "label": "compliant",
            },
            {
                "governance_context": {"policy": {}, "user_role": "user2"},
                "execution_plan": {"nodes": [], "edges": []},
                "label": "compliant",
            },
        ]

        batch = collate_fn(samples)

        assert "governance_contexts" in batch
        assert "execution_plans" in batch
        assert "labels" in batch

        assert len(batch["governance_contexts"]) == 2
        assert len(batch["execution_plans"]) == 2
        assert len(batch["labels"]) == 2


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = TrainingConfig()

        assert config.latent_dim == 1024
        assert config.batch_size == 32
        assert config.epochs == 20
        assert config.learning_rate == 1e-4
        assert config.temperature == 0.07

    def test_custom_config(self):
        """Config should accept custom parameters."""
        config = TrainingConfig(batch_size=64, epochs=50, learning_rate=5e-5)

        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.learning_rate == 5e-5


class TestIntegration:
    """Integration tests for training components."""

    def test_end_to_end_batch_processing(self):
        """Test complete batch processing flow."""
        # Create dataset
        dataset = GoldTraceDataset("nonexistent.jsonl")

        # Create dataloader
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        # Get one batch
        batch = next(iter(loader))

        assert "governance_contexts" in batch
        assert "execution_plans" in batch
        assert len(batch["governance_contexts"]) <= 2

    @pytest.mark.slow
    def test_training_imports(self):
        """Verify all training script imports work."""
        from scripts.train_jepa_encoders import (
            main,
            save_checkpoint,
            train_epoch,
            validate,
        )

        assert callable(main)
        assert callable(save_checkpoint)
        assert callable(train_epoch)
        assert callable(validate)
