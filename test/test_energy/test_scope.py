"""
Comprehensive tests for E_scope energy function.

Test coverage includes:
- Basic energy computation with actual vs minimal scope
- Gradient flow (differentiability through intent predictor)
- Scope extraction from execution plans
- Integration with SemanticIntentPredictor
- Edge cases and error handling
- Performance benchmarks
"""

import time

import pytest
import torch

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier
from source.encoders.intent_predictor import SemanticIntentPredictor, ScopeConstraints
from source.energy.scope import ScopeEnergyFunction, compute_scope_energy


# Fixtures

@pytest.fixture
def intent_predictor():
    """Create a SemanticIntentPredictor instance."""
    torch.manual_seed(42)
    return SemanticIntentPredictor(vocab_size=1000, hidden_dim=128, num_encoder_layers=2)


@pytest.fixture
def energy_fn(intent_predictor):
    """Create a ScopeEnergyFunction instance."""
    torch.manual_seed(42)
    return ScopeEnergyFunction(intent_predictor=intent_predictor)


@pytest.fixture
def sample_query_tokens():
    """Sample tokenized query."""
    torch.manual_seed(42)
    return torch.randint(0, 1000, (1, 10))  # [1, seq_len=10]


@pytest.fixture
def sample_schema_features():
    """Sample schema features."""
    torch.manual_seed(42)
    return torch.randn(1, 16, 128)  # [1, max_params=16, hidden_dim=128]


@pytest.fixture
def minimal_scope_plan():
    """Execution plan with minimal scope (limit=10)."""
    return ExecutionPlan(
        nodes=[
            ToolCallNode(
                tool_name="read_database",
                arguments={"table": "users"},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=10,  # Small scope
                scope_sensitivity=2,
                node_id="node_1"
            )
        ]
    )


@pytest.fixture
def excessive_scope_plan():
    """Execution plan with excessive scope (limit=10000)."""
    return ExecutionPlan(
        nodes=[
            ToolCallNode(
                tool_name="read_database",
                arguments={"table": "users"},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=10000,  # Excessive scope
                scope_sensitivity=5,
                node_id="node_1"
            )
        ]
    )


# Test Class 1: Initialization

class TestInitialization:
    """Tests for ScopeEnergyFunction initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        energy_fn = ScopeEnergyFunction()

        assert energy_fn.intent_predictor is not None
        assert energy_fn.temperature == 1.0
        assert energy_fn.alpha.requires_grad

    def test_custom_intent_predictor(self, intent_predictor):
        """Test initialization with custom intent predictor."""
        energy_fn = ScopeEnergyFunction(intent_predictor=intent_predictor)

        assert energy_fn.intent_predictor is intent_predictor

    def test_custom_scope_weights(self):
        """Test initialization with custom scope dimension weights."""
        weights = torch.tensor([2.0, 1.0, 1.0, 3.0])  # Emphasize limit and sensitivity
        energy_fn = ScopeEnergyFunction(scope_weights=weights)

        torch.testing.assert_close(energy_fn.scope_weights, weights)

    def test_alpha_initialization(self, energy_fn):
        """Test that alpha scaling parameter is initialized."""
        assert hasattr(energy_fn, 'alpha')
        assert isinstance(energy_fn.alpha, torch.nn.Parameter)
        assert energy_fn.alpha.requires_grad


# Test Class 2: Scope Extraction

class TestScopeExtraction:
    """Tests for extracting actual scope from execution plans."""

    def test_extract_scope_single_node(self, energy_fn):
        """Test scope extraction from single-node plan."""
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read_database",
                    scope_volume=100,
                    scope_sensitivity=3,
                    node_id="node_1"
                )
            ]
        )

        actual_scope = energy_fn._extract_actual_scope(plan)

        assert actual_scope.shape == (4,)
        assert actual_scope[0] == 100  # limit (from scope_volume)
        assert actual_scope[3] == 3  # sensitivity

    def test_extract_scope_multiple_nodes(self, energy_fn):
        """Test scope extraction with multiple nodes (max pooling)."""
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(tool_name="tool1", scope_volume=50, scope_sensitivity=2, node_id="n1"),
                ToolCallNode(tool_name="tool2", scope_volume=200, scope_sensitivity=4, node_id="n2"),
                ToolCallNode(tool_name="tool3", scope_volume=10, scope_sensitivity=1, node_id="n3")
            ]
        )

        actual_scope = energy_fn._extract_actual_scope(plan)

        # Should use max pooling
        assert actual_scope[0] == 200  # max limit
        assert actual_scope[3] == 4  # max sensitivity

    def test_extract_scope_none_plan(self, energy_fn):
        """Test scope extraction from None plan."""
        actual_scope = energy_fn._extract_actual_scope(None)

        assert torch.all(actual_scope == 0)


# Test Class 3: Core Functionality

class TestCoreFunctionality:
    """Tests for basic energy computation."""

    def test_direct_mode_minimal_scope(self, energy_fn):
        """Test direct mode when actual equals minimal (should be low energy)."""
        actual = torch.tensor([[10.0, 30.0, 1.0, 2.0]])
        minimal = torch.tensor([[10.0, 30.0, 1.0, 2.0]])

        energy = energy_fn(actual_scope=actual, minimal_scope=minimal)

        assert energy.shape == (1,)
        assert energy.item() < 0.01, "Energy should be near zero when actual == minimal"

    def test_direct_mode_excessive_scope(self, energy_fn):
        """Test direct mode when actual exceeds minimal (should be high energy)."""
        actual = torch.tensor([[1000.0, 90.0, 5.0, 4.0]])
        minimal = torch.tensor([[10.0, 30.0, 1.0, 2.0]])

        energy = energy_fn(actual_scope=actual, minimal_scope=minimal)

        assert energy.shape == (1,)
        assert energy.item() > 100.0, "Energy should be high when scope is excessive"

    def test_full_mode_with_plan(self, energy_fn, minimal_scope_plan, sample_query_tokens, sample_schema_features):
        """Test full mode with execution plan."""
        energy = energy_fn(
            plan=minimal_scope_plan,
            query_tokens=sample_query_tokens,
            schema_features=sample_schema_features
        )

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()
        assert not torch.isinf(energy).any()

    def test_energy_increases_with_scope(self, energy_fn, sample_query_tokens, sample_schema_features):
        """Test that energy increases as scope increases."""
        small_plan = ExecutionPlan(
            nodes=[ToolCallNode(tool_name="tool", scope_volume=10, scope_sensitivity=1, node_id="n1")]
        )
        large_plan = ExecutionPlan(
            nodes=[ToolCallNode(tool_name="tool", scope_volume=1000, scope_sensitivity=5, node_id="n1")]
        )

        energy_small = energy_fn(plan=small_plan, query_tokens=sample_query_tokens, schema_features=sample_schema_features)
        energy_large = energy_fn(plan=large_plan, query_tokens=sample_query_tokens, schema_features=sample_schema_features)

        assert energy_large > energy_small


# Test Class 4: Differentiability

class TestDifferentiability:
    """Tests for gradient flow through the energy function."""

    def test_gradient_flow_direct_mode(self, energy_fn):
        """Test gradient flow in direct mode."""
        actual = torch.tensor([[100.0, 60.0, 3.0, 4.0]], requires_grad=True)
        minimal = torch.tensor([[10.0, 30.0, 1.0, 2.0]], requires_grad=True)

        energy = energy_fn(actual_scope=actual, minimal_scope=minimal)

        assert energy.requires_grad
        energy.backward()

        assert actual.grad is not None
        assert minimal.grad is not None
        assert not torch.isnan(actual.grad).any()

    def test_gradient_flow_through_predictor(self, energy_fn, minimal_scope_plan, sample_query_tokens, sample_schema_features):
        """Test gradient flow through SemanticIntentPredictor."""
        sample_query_tokens.requires_grad = False  # Tokens don't have grads
        sample_schema_features.requires_grad_(True)

        energy = energy_fn(
            plan=minimal_scope_plan,
            query_tokens=sample_query_tokens,
            schema_features=sample_schema_features
        )

        energy.backward()

        # Gradients should flow to schema features
        assert sample_schema_features.grad is not None
        assert not torch.isnan(sample_schema_features.grad).any()


# Test Class 5: Batch Processing

class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    def test_batch_direct_mode(self, energy_fn):
        """Test batched computation in direct mode."""
        batch_size = 4
        actual = torch.randn(batch_size, 4).abs() * 100 + 10  # Ensure positive
        minimal = torch.randn(batch_size, 4).abs() * 10 + 1

        energy = energy_fn(actual_scope=actual, minimal_scope=minimal)

        assert energy.shape == (batch_size,)
        assert not torch.isnan(energy).any()

    def test_batch_consistency(self, energy_fn):
        """Test that batch processing is consistent with individual processing."""
        energy_fn.train(False)  # Disable dropout for deterministic behavior

        actual_batch = torch.tensor([[100.0, 60.0, 3.0, 4.0], [50.0, 30.0, 2.0, 3.0]])
        minimal_batch = torch.tensor([[10.0, 30.0, 1.0, 2.0], [10.0, 30.0, 1.0, 2.0]])

        with torch.no_grad():
            energy_batch = energy_fn(actual_scope=actual_batch, minimal_scope=minimal_batch)

            energies_individual = []
            for i in range(2):
                e = energy_fn(actual_scope=actual_batch[i:i+1], minimal_scope=minimal_batch[i:i+1])
                energies_individual.append(e)

            energies_individual = torch.cat(energies_individual)

        torch.testing.assert_close(energy_batch, energies_individual, rtol=1e-5, atol=1e-5)


# Test Class 6: Edge Cases

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_inputs_raises_error(self, energy_fn):
        """Test that missing required inputs raise ValueError."""
        with pytest.raises(ValueError):
            energy_fn()  # No inputs provided

    def test_partial_inputs_raises_error(self, energy_fn, sample_query_tokens):
        """Test that partial inputs raise ValueError."""
        with pytest.raises(ValueError):
            energy_fn(query_tokens=sample_query_tokens)  # Missing plan and schema

    def test_zero_scope(self, energy_fn):
        """Test with zero actual and minimal scope."""
        actual = torch.zeros(1, 4)
        minimal = torch.zeros(1, 4)

        energy = energy_fn(actual_scope=actual, minimal_scope=minimal)

        assert not torch.isnan(energy).any()
        assert energy.item() < 0.01  # Should be near zero with ReLU

    def test_negative_over_privilege(self, energy_fn):
        """Test when actual < minimal (should use soft ReLU to get ~0)."""
        actual = torch.tensor([[5.0, 10.0, 1.0, 1.0]])
        minimal = torch.tensor([[10.0, 30.0, 2.0, 3.0]])  # Minimal > actual

        energy = energy_fn(actual_scope=actual, minimal_scope=minimal)

        # With soft ReLU, this should be very small (not negative)
        assert energy.item() >= 0
        assert energy.item() < 1.0


# Test Class 7: Integration Tests

class TestIntegration:
    """Tests for integration with other components."""

    def test_integration_with_intent_predictor(self, intent_predictor, minimal_scope_plan):
        """Test full integration with SemanticIntentPredictor."""
        energy_fn = ScopeEnergyFunction(intent_predictor=intent_predictor)

        query_tokens = torch.randint(0, 1000, (1, 15))
        schema_features = torch.randn(1, 16, 128)

        energy = energy_fn(
            plan=minimal_scope_plan,
            query_tokens=query_tokens,
            schema_features=schema_features
        )

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_compute_detailed_breakdown(self, energy_fn, excessive_scope_plan, sample_query_tokens, sample_schema_features):
        """Test detailed energy breakdown utility."""
        breakdown = energy_fn.compute_detailed_breakdown(
            plan=excessive_scope_plan,
            query_tokens=sample_query_tokens,
            schema_features=sample_schema_features
        )

        assert 'total_energy' in breakdown
        assert 'actual_scope' in breakdown
        assert 'minimal_scope' in breakdown
        assert 'over_privilege' in breakdown
        assert 'dimension_penalties' in breakdown
        assert 'alpha_scale' in breakdown

        # Check structure
        assert 'limit' in breakdown['actual_scope']
        assert 'limit' in breakdown['over_privilege']


# Test Class 8: Performance Benchmarks

class TestPerformance:
    """Performance benchmarks for E_scope."""

    @pytest.mark.benchmark
    def test_inference_latency_direct_mode(self, energy_fn):
        """Benchmark direct mode inference latency."""
        actual = torch.randn(1, 4).abs() * 100
        minimal = torch.randn(1, 4).abs() * 10

        # Warmup
        for _ in range(10):
            energy_fn(actual_scope=actual, minimal_scope=minimal)

        # Measure
        num_rounds = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_rounds):
                energy_fn(actual_scope=actual, minimal_scope=minimal)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / num_rounds) * 1000

        print(f"\nE_scope direct mode latency: {latency_ms:.3f}ms")

        # Target: <1ms for direct mode (no predictor call)
        assert latency_ms < 1.0

    @pytest.mark.benchmark
    def test_inference_latency_full_mode(self, energy_fn, minimal_scope_plan, sample_query_tokens, sample_schema_features):
        """Benchmark full mode inference latency (includes predictor)."""
        # Warmup
        for _ in range(10):
            energy_fn(plan=minimal_scope_plan, query_tokens=sample_query_tokens, schema_features=sample_schema_features)

        # Measure
        num_rounds = 50
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_rounds):
                energy_fn(plan=minimal_scope_plan, query_tokens=sample_query_tokens, schema_features=sample_schema_features)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / num_rounds) * 1000

        print(f"\nE_scope full mode latency: {latency_ms:.2f}ms")

        # Target: <100ms (includes intent predictor overhead)
        assert latency_ms < 100.0


# Test Class 9: Standalone Function

class TestStandaloneFunction:
    """Tests for compute_scope_energy convenience function."""

    def test_standalone_function(self, minimal_scope_plan):
        """Test standalone function with default predictor."""
        # Use default predictor dimensions: vocab_size=50000, hidden_dim=256
        query_tokens = torch.randint(0, 50000, (1, 15))
        schema_features = torch.randn(1, 16, 256)  # Match default hidden_dim
        
        energy = compute_scope_energy(
            plan=minimal_scope_plan,
            query_tokens=query_tokens,
            schema_features=schema_features
        )

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()

    def test_standalone_with_custom_predictor(self, intent_predictor, minimal_scope_plan, sample_query_tokens, sample_schema_features):
        """Test standalone function with custom predictor."""
        energy = compute_scope_energy(
            plan=minimal_scope_plan,
            query_tokens=sample_query_tokens,
            schema_features=sample_schema_features,
            intent_predictor=intent_predictor
        )

        assert energy.shape == (1,)
        assert not torch.isnan(energy).any()
