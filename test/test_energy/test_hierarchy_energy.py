"""
Tests for HierarchyEnergy (E_hierarchy) energy critic.

Coverage targets:
- Forward pass with various input shapes
- Energy bounds and scaling
- Cross-attention mechanism
- Violation detection
- Component breakdown
- Gradient flow (differentiability)
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn

from source.energy.hierarchy_energy import HierarchyEnergy, create_hierarchy_energy


class TestHierarchyEnergyInit:
    """Test initialization and configuration."""

    def test_default_initialization(self):
        """Test default parameters."""
        model = HierarchyEnergy()
        assert model.latent_dim == 1024
        assert model.hidden_dim == 512
        assert model.temperature == 1.0

    def test_custom_initialization(self):
        """Test custom parameters."""
        model = HierarchyEnergy(
            latent_dim=512,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2,
            temperature=0.5
        )
        assert model.latent_dim == 512
        assert model.hidden_dim == 256
        assert model.temperature == 0.5

    def test_factory_function(self):
        """Test create_hierarchy_energy factory."""
        model = create_hierarchy_energy(latent_dim=1024, device="cpu")
        assert isinstance(model, HierarchyEnergy)
        assert model.latent_dim == 1024
        assert not model.training  # Should be in inference mode


class TestHierarchyEnergyForward:
    """Test forward pass computation."""

    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return HierarchyEnergy(latent_dim=1024)

    @pytest.fixture
    def sample_latents(self):
        """Generate sample latent vectors."""
        batch_size = 4
        latent_dim = 1024
        z_g = torch.randn(batch_size, latent_dim)
        z_e = torch.randn(batch_size, latent_dim)
        return z_g, z_e

    def test_forward_pass_shape(self, model, sample_latents):
        """Test output shape is correct."""
        z_g, z_e = sample_latents
        energy = model(z_g, z_e)

        assert energy.shape == (4, 1)
        assert energy.dtype == torch.float32

    def test_energy_bounds(self, model, sample_latents):
        """Test energy is bounded to [0, 1] due to sigmoid."""
        z_g, z_e = sample_latents
        energy = model(z_g, z_e)

        assert torch.all(energy >= 0.0)
        assert torch.all(energy <= 1.0)

    def test_single_sample(self, model):
        """Test with batch_size=1."""
        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 1024)

        energy = model(z_g, z_e)
        assert energy.shape == (1, 1)

    def test_large_batch(self, model):
        """Test with larger batch size."""
        batch_size = 32
        z_g = torch.randn(batch_size, 1024)
        z_e = torch.randn(batch_size, 1024)

        energy = model(z_g, z_e)
        assert energy.shape == (batch_size, 1)

    def test_return_components(self, model, sample_latents):
        """Test return_components=True provides diagnostic info."""
        z_g, z_e = sample_latents
        result = model(z_g, z_e, return_components=True)

        assert isinstance(result, dict)
        assert 'energy' in result
        assert 'semantic_distance' in result
        assert 'cosine_similarity' in result
        assert 'attention_weighted_exec' in result

        # Check shapes
        assert result['energy'].shape == (4, 1)
        assert result['semantic_distance'].shape == (4, 1)
        assert result['cosine_similarity'].shape == (4, 1)
        assert result['attention_weighted_exec'].shape == (4, 1024)


class TestHierarchyEnergyCrossAttention:
    """Test cross-attention mechanism."""

    @pytest.fixture
    def model(self):
        return HierarchyEnergy(latent_dim=1024)

    def test_attention_output_shape(self, model):
        """Test cross-attention produces correct output shape."""
        z_g = torch.randn(2, 1024)
        z_e = torch.randn(2, 1024)

        attn_output = model._compute_cross_attention(z_g, z_e)
        assert attn_output.shape == (2, 1024)

    def test_attention_differentiable(self, model):
        """Test attention mechanism supports gradient flow."""
        z_g = torch.randn(2, 1024, requires_grad=True)
        z_e = torch.randn(2, 1024, requires_grad=True)

        attn_output = model._compute_cross_attention(z_g, z_e)
        loss = attn_output.sum()
        loss.backward()

        assert z_g.grad is not None
        assert z_e.grad is not None


class TestViolationDetection:
    """Test violation score computation."""

    @pytest.fixture
    def model(self):
        return HierarchyEnergy(latent_dim=1024)

    def test_compute_violation_score(self, model):
        """Test violation detection with default threshold."""
        z_g = torch.randn(4, 1024)
        z_e = torch.randn(4, 1024)

        energy, is_violation = model.compute_violation_score(z_g, z_e)

        assert energy.shape == (4, 1)
        assert is_violation.shape == (4, 1)
        assert is_violation.dtype == torch.bool

    def test_custom_threshold(self, model):
        """Test violation detection with custom threshold."""
        z_g = torch.randn(4, 1024)
        z_e = torch.randn(4, 1024)

        energy, is_violation_low = model.compute_violation_score(z_g, z_e, threshold=0.3)
        _, is_violation_high = model.compute_violation_score(z_g, z_e, threshold=0.9)

        # Lower threshold should flag more violations
        assert is_violation_low.sum() >= is_violation_high.sum()

    def test_violation_energy_finite(self, model):
        """Test that energy values are always finite and bounded."""
        # Test with various input configurations
        test_cases = [
            (torch.randn(5, 1024), torch.randn(5, 1024)),  # Random different
            (torch.zeros(5, 1024), torch.zeros(5, 1024)),  # Identical zeros
            (torch.ones(5, 1024), torch.ones(5, 1024)),   # Identical ones
        ]

        for z_g, z_e in test_cases:
            energy = model(z_g, z_e)

            # Energy should be finite and bounded
            assert torch.all(torch.isfinite(energy))
            assert torch.all(energy >= 0.0)
            assert torch.all(energy <= 1.0)


class TestGradientFlow:
    """Test differentiability for training."""

    @pytest.fixture
    def model(self):
        return HierarchyEnergy(latent_dim=1024)

    def test_backward_pass(self, model):
        """Test gradients flow through the energy function."""
        z_g = torch.randn(2, 1024, requires_grad=True)
        z_e = torch.randn(2, 1024, requires_grad=True)

        energy = model(z_g, z_e)
        loss = energy.sum()
        loss.backward()

        # Check gradients exist
        assert z_g.grad is not None
        assert z_e.grad is not None

        # Check gradients are non-zero
        assert torch.abs(z_g.grad).sum() > 0
        assert torch.abs(z_e.grad).sum() > 0

    def test_model_parameters_trainable(self, model):
        """Test all model parameters require gradients."""
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"

    def test_training_mode(self, model):
        """Test model can switch between train/eval modes."""
        model.train()
        assert model.training

        model.training = False
        assert not model.training


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def model(self):
        return HierarchyEnergy(latent_dim=1024)

    def test_wrong_latent_dim_raises_error(self, model):
        """Test mismatched latent dimensions raise assertion."""
        z_g = torch.randn(2, 512)  # Wrong dimension
        z_e = torch.randn(2, 1024)

        with pytest.raises(AssertionError):
            model(z_g, z_e)

    def test_mismatched_batch_sizes_raises_error(self, model):
        """Test mismatched batch sizes raise assertion."""
        z_g = torch.randn(2, 1024)
        z_e = torch.randn(4, 1024)  # Different batch size

        with pytest.raises(AssertionError):
            model(z_g, z_e)

    def test_zero_vectors(self, model):
        """Test with zero vectors (edge case)."""
        z_g = torch.zeros(1, 1024)
        z_e = torch.zeros(1, 1024)

        energy = model(z_g, z_e)
        assert energy.shape == (1, 1)
        assert torch.isfinite(energy).all()

    def test_extreme_values(self, model):
        """Test with extreme values."""
        z_g = torch.ones(1, 1024) * 1000
        z_e = torch.ones(1, 1024) * -1000

        energy = model(z_g, z_e)
        assert energy.shape == (1, 1)
        assert torch.isfinite(energy).all()
        # Energy should still be bounded [0, 1] due to sigmoid
        assert torch.all(energy >= 0.0) and torch.all(energy <= 1.0)


class TestTemperatureScaling:
    """Test temperature parameter effects."""

    def test_low_temperature_sharper_boundaries(self):
        """Test that lower temperature creates sharper energy boundaries."""
        model_low_temp = HierarchyEnergy(latent_dim=1024, temperature=0.1)
        model_high_temp = HierarchyEnergy(latent_dim=1024, temperature=10.0)

        z_g = torch.randn(10, 1024)
        z_e = torch.randn(10, 1024)

        energy_low = model_low_temp(z_g, z_e)
        energy_high = model_high_temp(z_g, z_e)

        # Low temperature should have higher variance (sharper decisions)
        assert energy_low.var() >= energy_high.var()


class TestIntegration:
    """Integration tests with encoder outputs."""

    def test_with_mock_encoder_outputs(self):
        """Test with realistic encoder-like outputs."""
        # Simulate encoder outputs (normalized embeddings)
        z_g = torch.randn(4, 1024)
        z_g = torch.nn.functional.normalize(z_g, dim=-1)

        z_e = torch.randn(4, 1024)
        z_e = torch.nn.functional.normalize(z_e, dim=-1)

        model = HierarchyEnergy()
        energy = model(z_g, z_e)

        assert energy.shape == (4, 1)
        assert torch.all(torch.isfinite(energy))

    def test_batch_processing(self):
        """Test processing multiple samples efficiently."""
        model = HierarchyEnergy()

        # Process 100 samples
        batch_size = 100
        z_g = torch.randn(batch_size, 1024)
        z_e = torch.randn(batch_size, 1024)

        energy = model(z_g, z_e)

        assert energy.shape == (batch_size, 1)
        assert torch.all(torch.isfinite(energy))


# Performance benchmarks (marked as slow)
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for E_hierarchy."""

    def test_inference_latency(self):
        """Test inference meets <20ms requirement (part of composite)."""
        import time

        model = create_hierarchy_energy(device="cpu")

        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 1024)

        # Warmup
        for _ in range(10):
            _ = model(z_g, z_e)

        # Measure
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            _ = model(z_g, z_e)
        end = time.time()

        avg_latency_ms = (end - start) / iterations * 1000

        # Single energy term should be <10ms on CPU
        # (Full composite of 4 terms must be <20ms total)
        assert avg_latency_ms < 10, f"Latency {avg_latency_ms:.2f}ms exceeds 10ms target"
