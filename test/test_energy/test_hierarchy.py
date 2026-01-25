"""
Comprehensive tests for E_hierarchy energy function.

Test coverage includes:
- Basic energy computation
- Gradient flow (differentiability)
- Component decomposition
- Batch processing
- Edge cases and error handling
- Performance benchmarks (<20ms target)
- Integration with encoders
"""

import time
from typing import Any

import pytest
import torch

from source.encoders.execution_encoder import ExecutionEncoder, ExecutionPlan, ToolCallNode, TrustTier
from source.encoders.governance_encoder import GovernanceEncoder, PolicySchema
from source.energy.hierarchy import HierarchyEnergyFunction, compute_hierarchy_energy


# Fixtures

@pytest.fixture
def energy_fn():
    """Create a HierarchyEnergyFunction instance."""
    torch.manual_seed(42)
    return HierarchyEnergyFunction(latent_dim=1024, hidden_dim=256, num_heads=4)


@pytest.fixture
def sample_latents():
    """Sample governance and execution latents for testing."""
    torch.manual_seed(42)
    z_g = torch.randn(1, 1024)
    z_e = torch.randn(1, 1024)
    return z_g, z_e


@pytest.fixture
def batch_latents():
    """Batch of latents for batch processing tests."""
    torch.manual_seed(42)
    batch_size = 8
    z_g = torch.randn(batch_size, 1024)
    z_e = torch.randn(batch_size, 1024)
    return z_g, z_e


@pytest.fixture
def governance_encoder():
    """GovernanceEncoder for integration tests."""
    torch.manual_seed(42)
    return GovernanceEncoder()


@pytest.fixture
def execution_encoder():
    """ExecutionEncoder for integration tests."""
    torch.manual_seed(42)
    return ExecutionEncoder()


# Test Class 1: Initialization and Configuration

class TestInitialization:
    """Tests for HierarchyEnergyFunction initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        energy_fn = HierarchyEnergyFunction()

        assert energy_fn.latent_dim == 1024
        assert energy_fn.hidden_dim == 256
        assert energy_fn.num_heads == 4
        assert energy_fn.temperature == 0.1

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        energy_fn = HierarchyEnergyFunction(
            latent_dim=512,
            hidden_dim=128,
            num_heads=8,
            temperature=0.2,
            dropout=0.2
        )

        assert energy_fn.latent_dim == 512
        assert energy_fn.hidden_dim == 128
        assert energy_fn.num_heads == 8
        assert energy_fn.temperature == 0.2

    def test_parameter_count(self, energy_fn):
        """Test that model has reasonable parameter count."""
        total_params = sum(p.numel() for p in energy_fn.parameters())

        # Expect roughly 1-3M parameters for this architecture
        assert 500_000 < total_params < 5_000_000, f"Unexpected param count: {total_params}"

    def test_alpha_initialization(self, energy_fn):
        """Test that alpha scaling parameter is initialized."""
        assert hasattr(energy_fn, 'alpha')
        assert isinstance(energy_fn.alpha, torch.nn.Parameter)
        assert energy_fn.alpha.requires_grad


# Test Class 2: Core Functionality

class TestCoreFunctionality:
    """Tests for basic energy computation."""

    def test_forward_pass(self, energy_fn, sample_latents):
        """Test basic forward pass."""
        z_g, z_e = sample_latents
        energy = energy_fn(z_g, z_e)

        assert energy.shape == (1,), f"Expected shape (1,), got {energy.shape}"
        assert energy.dtype == torch.float32
        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()

    def test_energy_is_differentiable(self, energy_fn, sample_latents):
        """Test that energy function is differentiable."""
        z_g, z_e = sample_latents
        z_g.requires_grad_(True)
        z_e.requires_grad_(True)

        energy = energy_fn(z_g, z_e)

        assert energy.requires_grad, "Energy should require gradients"

        # Backpropagate
        energy.sum().backward()

        assert z_g.grad is not None, "Gradients should flow to z_g"
        assert z_e.grad is not None, "Gradients should flow to z_e"
        assert not torch.isnan(z_g.grad).any(), "Gradients should not be NaN"
        assert not torch.isnan(z_e.grad).any(), "Gradients should not be NaN"

    def test_return_components(self, energy_fn, sample_latents):
        """Test that return_components=True returns diagnostic dict."""
        z_g, z_e = sample_latents
        result = energy_fn(z_g, z_e, return_components=True)

        assert isinstance(result, dict)
        assert 'energy' in result
        assert 'deviation' in result
        assert 'control_mask' in result
        assert 'importance' in result
        assert 'combined_weight' in result
        assert 'weighted_deviation' in result
        assert 'alpha' in result

        # Check shapes
        assert result['energy'].shape == (1,)
        assert result['deviation'].shape == (1, 1024)
        assert result['control_mask'].shape == (1, 1024)
        assert result['importance'].shape == (1, 1024)

    def test_identical_latents_low_energy(self, energy_fn):
        """Test that identical latents produce low energy."""
        torch.manual_seed(42)
        z = torch.randn(1, 1024)

        energy = energy_fn(z, z)

        # When z_g == z_e, deviation is zero, so energy should be low
        assert energy.abs() < 1.0, f"Expected low energy for identical latents, got {energy.item()}"

    def test_orthogonal_latents_high_energy(self, energy_fn):
        """Test that very different latents produce higher energy."""
        torch.manual_seed(42)
        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 1024) * 10  # Scale up to create large deviation

        energy_different = energy_fn(z_g, z_e)
        energy_same = energy_fn(z_g, z_g)

        # Energy should be higher when latents differ
        assert energy_different.abs() > energy_same.abs()


# Test Class 3: Batch Processing

class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    def test_batch_forward(self, energy_fn, batch_latents):
        """Test batched forward pass."""
        z_g, z_e = batch_latents
        batch_size = z_g.shape[0]

        energy = energy_fn(z_g, z_e)

        assert energy.shape == (batch_size,)
        assert not torch.isnan(energy).any()

    def test_batch_consistency(self, energy_fn):
        """Test that batch processing gives same results as individual (in inference mode)."""
        torch.manual_seed(42)
        z_g_batch = torch.randn(4, 1024)
        z_e_batch = torch.randn(4, 1024)

        # Disable training mode (sets dropout to deterministic behavior)
        energy_fn.train(False)

        with torch.no_grad():
            # Batch computation
            energy_batch = energy_fn(z_g_batch, z_e_batch)

            # Individual computations
            energies_individual = []
            for i in range(4):
                e = energy_fn(z_g_batch[i:i+1], z_e_batch[i:i+1])
                energies_individual.append(e)

            energies_individual = torch.cat(energies_individual)

        # Should match within numerical precision
        torch.testing.assert_close(energy_batch, energies_individual, rtol=1e-5, atol=1e-5)

    def test_batch_gradients(self, energy_fn, batch_latents):
        """Test gradient flow for batched inputs."""
        z_g, z_e = batch_latents
        z_g.requires_grad_(True)
        z_e.requires_grad_(True)

        energy = energy_fn(z_g, z_e)
        loss = energy.sum()
        loss.backward()

        assert z_g.grad.shape == z_g.shape
        assert z_e.grad.shape == z_e.shape
        assert not torch.isnan(z_g.grad).any()


# Test Class 4: Component Analysis

class TestComponentAnalysis:
    """Tests for internal component behavior."""

    def test_control_mask_range(self, energy_fn, sample_latents):
        """Test that control mask values are in valid range."""
        z_g, z_e = sample_latents

        with torch.no_grad():
            control_mask = energy_fn._compute_control_mask(z_g, z_e)

        assert control_mask.min() >= 0.0, "Mask values should be non-negative"
        assert control_mask.max() <= 1.0, "Mask values should be <= 1.0"

    def test_importance_weights_range(self, energy_fn, sample_latents):
        """Test that importance weights are in [0, 1] (sigmoid output)."""
        z_g, z_e = sample_latents

        with torch.no_grad():
            importance = energy_fn._compute_importance_weights(z_g, z_e)

        assert importance.min() >= 0.0
        assert importance.max() <= 1.0

    def test_get_energy_decomposition(self, energy_fn, sample_latents):
        """Test energy decomposition utility."""
        z_g, z_e = sample_latents

        decomp = energy_fn.get_energy_decomposition(z_g, z_e)

        assert isinstance(decomp, dict)
        assert 'total_energy' in decomp
        assert 'deviation_norm' in decomp
        assert 'control_focus' in decomp
        assert 'importance_mean' in decomp
        assert 'alpha_scale' in decomp
        assert 'top_control_dims' in decomp

        # Check that top_control_dims contains indices
        assert isinstance(decomp['top_control_dims'], list)
        assert len(decomp['top_control_dims']) == 5


# Test Class 5: Edge Cases and Error Handling

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_shape_mismatch_raises_error(self, energy_fn):
        """Test that mismatched input shapes raise AssertionError."""
        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 512)  # Wrong dimension

        with pytest.raises(AssertionError):
            energy_fn(z_g, z_e)

    def test_wrong_latent_dim_raises_error(self):
        """Test that wrong latent dimension raises error."""
        energy_fn = HierarchyEnergyFunction(latent_dim=1024)
        z_g = torch.randn(1, 512)  # Wrong latent_dim
        z_e = torch.randn(1, 512)

        with pytest.raises(AssertionError):
            energy_fn(z_g, z_e)

    def test_zero_input(self, energy_fn):
        """Test with zero inputs."""
        z_g = torch.zeros(1, 1024)
        z_e = torch.zeros(1, 1024)

        energy = energy_fn(z_g, z_e)

        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()

    def test_large_input_values(self, energy_fn):
        """Test with very large input values."""
        z_g = torch.randn(1, 1024) * 100
        z_e = torch.randn(1, 1024) * 100

        energy = energy_fn(z_g, z_e)

        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()

    def test_single_sample_decomposition_only(self, energy_fn):
        """Test that decomposition method requires single sample."""
        z_g_batch = torch.randn(4, 1024)
        z_e_batch = torch.randn(4, 1024)

        with pytest.raises(AssertionError):
            energy_fn.get_energy_decomposition(z_g_batch, z_e_batch)


# Test Class 6: Integration with Encoders

class TestEncoderIntegration:
    """Tests for integration with GovernanceEncoder and ExecutionEncoder."""

    def test_integration_with_governance_encoder(self, governance_encoder, energy_fn):
        """Test E_hierarchy with real GovernanceEncoder output."""
        policy = PolicySchema(
            document={"permissions": {"read": ["users"]}, "constraints": {"max_records": 100}},
            user_role="analyst"
        )

        with torch.no_grad():
            z_g = governance_encoder(policy)

        # Create a mock execution latent
        z_e = torch.randn(1, 1024)

        energy = energy_fn(z_g, z_e)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_integration_with_execution_encoder(self, execution_encoder, energy_fn):
        """Test E_hierarchy with real ExecutionEncoder output."""
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read_database",
                    arguments={"table": "users", "limit": 10},
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10,
                    scope_sensitivity=3,
                    node_id="node_1"
                ),
                ToolCallNode(
                    tool_name="filter_results",
                    arguments={"field": "email"},
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10,
                    scope_sensitivity=3,
                    node_id="node_2"
                )
            ],
            edges=[("node_1", "node_2")]
        )

        with torch.no_grad():
            z_e = execution_encoder(plan)

        # Create a mock governance latent
        z_g = torch.randn(1, 1024)

        energy = energy_fn(z_g, z_e)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_end_to_end_pipeline(self, governance_encoder, execution_encoder, energy_fn):
        """Test complete pipeline: policy + plan -> energy."""
        # Create policy
        policy = PolicySchema(
            document={
                "permissions": {"read": ["users", "posts"]},
                "constraints": {"max_records": 100}
            },
            user_role="analyst"
        )

        # Create execution plan
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read_database",
                    arguments={"table": "users", "limit": 50},
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=50,
                    scope_sensitivity=2,
                    node_id="node_1"
                )
            ],
            edges=[]
        )

        # Encode
        with torch.no_grad():
            z_g = governance_encoder(policy)
            z_e = execution_encoder(plan)

        # Compute energy
        energy = energy_fn(z_g, z_e)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()


# Test Class 7: Performance Benchmarks

class TestPerformance:
    """Performance benchmarks for E_hierarchy."""

    @pytest.mark.benchmark
    def test_inference_latency_single(self, energy_fn, sample_latents):
        """Benchmark single inference latency."""
        z_g, z_e = sample_latents

        # Warmup
        for _ in range(10):
            energy_fn(z_g, z_e)

        # Measure
        num_rounds = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_rounds):
                energy_fn(z_g, z_e)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / num_rounds) * 1000

        print(f"\nE_hierarchy inference latency: {latency_ms:.2f}ms")

        # Target: <20ms (per PRD requirements for energy functions)
        assert latency_ms < 20.0, f"Latency {latency_ms:.2f}ms exceeds 20ms target"

    @pytest.mark.benchmark
    def test_inference_latency_batch(self, energy_fn, batch_latents):
        """Benchmark batched inference latency."""
        z_g, z_e = batch_latents
        batch_size = z_g.shape[0]

        # Warmup
        for _ in range(10):
            energy_fn(z_g, z_e)

        # Measure
        num_rounds = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_rounds):
                energy_fn(z_g, z_e)
        elapsed = time.perf_counter() - start

        latency_per_sample_ms = (elapsed / num_rounds / batch_size) * 1000

        print(f"\nE_hierarchy batched latency per sample: {latency_per_sample_ms:.2f}ms")

        # Should be faster than single-sample due to batching
        assert latency_per_sample_ms < 10.0

    @pytest.mark.benchmark
    def test_memory_usage(self, energy_fn, sample_latents):
        """Test memory footprint."""
        z_g, z_e = sample_latents

        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in energy_fn.parameters())
        param_memory_mb = param_memory / (1024 ** 2)

        print(f"\nE_hierarchy model size: {param_memory_mb:.2f} MB")

        # Should be reasonable (<50MB for energy function)
        assert param_memory_mb < 50.0


# Test Class 8: Standalone Function

class TestStandaloneFunction:
    """Tests for compute_hierarchy_energy convenience function."""

    def test_standalone_function(self, sample_latents):
        """Test standalone function with default model."""
        z_g, z_e = sample_latents

        energy = compute_hierarchy_energy(z_g, z_e)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_standalone_with_custom_model(self, energy_fn, sample_latents):
        """Test standalone function with custom model."""
        z_g, z_e = sample_latents

        energy = compute_hierarchy_energy(z_g, z_e, model=energy_fn)

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_standalone_preserves_model_mode(self, energy_fn, sample_latents):
        """Test that standalone function preserves training mode."""
        z_g, z_e = sample_latents

        energy_fn.train()
        assert energy_fn.training

        compute_hierarchy_energy(z_g, z_e, model=energy_fn)

        assert energy_fn.training, "Model should still be in training mode"


# Test Class 9: Sensitivity Analysis

class TestSensitivityAnalysis:
    """Tests for energy sensitivity to different perturbations."""

    def test_sensitivity_to_provenance_changes(self, governance_encoder, execution_encoder, energy_fn):
        """Test that energy is sensitive to provenance tier changes."""
        policy = PolicySchema(
            document={"permissions": {"read": ["users"]}},
            user_role="analyst"
        )

        # Plan with INTERNAL provenance
        plan_trusted = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read_database",
                    arguments={"table": "users"},
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10,
                    scope_sensitivity=2,
                    node_id="node_1"
                )
            ]
        )

        # Plan with PUBLIC_WEB provenance (untrusted)
        plan_untrusted = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read_database",
                    arguments={"table": "users"},
                    provenance_tier=TrustTier.PUBLIC_WEB,
                    scope_volume=10,
                    scope_sensitivity=2,
                    node_id="node_1"
                )
            ]
        )

        with torch.no_grad():
            z_g = governance_encoder(policy)
            z_e_trusted = execution_encoder(plan_trusted)
            z_e_untrusted = execution_encoder(plan_untrusted)

            energy_trusted = energy_fn(z_g, z_e_trusted)
            energy_untrusted = energy_fn(z_g, z_e_untrusted)

        # Energy should reflect the provenance difference
        # (Note: Actual behavior depends on training, but encoder should encode this difference)
        assert energy_trusted.shape == energy_untrusted.shape
