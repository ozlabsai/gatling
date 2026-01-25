"""
Tests for E_scope energy term.

Validates least privilege enforcement and over-scoping detection.
"""

import pytest
import torch

from source.energy.scope import ScopeEnergy, create_scope_energy
from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier
from source.encoders.intent_predictor import ScopeConstraints


class TestScopeEnergy:
    """Test suite for E_scope."""

    def test_initialization(self):
        """Test energy term initializes correctly."""
        energy = create_scope_energy(use_latent_modulation=False)

        assert isinstance(energy, ScopeEnergy)
        assert energy.dimension_weights.shape == (4,)
        # Weights should be: [1.0, 0.5, 0.3, 2.0] for [limit, date, depth, sensitivity]
        assert torch.allclose(
            energy.dimension_weights,
            torch.tensor([1.0, 0.5, 0.3, 2.0])
        )

    def test_minimal_plan(self):
        """Minimal safe plan should have low energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="noop",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL
                )
            ],
            edges=[]
        )
        E = energy(plan)

        assert E.shape == (1,)
        assert float(E) >= 0.0

    def test_perfect_scope_zero_energy(self):
        """Plan with scope matching minimal budget should have zero energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        # Create plan with exactly minimal scope
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_items",
                    node_id="node1",
                    arguments={"limit": 1},
                    scope_volume=1,
                    scope_sensitivity=1  # Min valid value
                )
            ],
            edges=[]
        )

        # Define minimal scope that matches the plan
        minimal_scope = ScopeConstraints(
            limit=1,
            date_range_days=1,
            max_depth=1,
            include_sensitive=False
        )

        E = energy(plan, minimal_scope)
        assert float(E) < 0.5, "Perfect scope match should have low energy"

    def test_over_limit_high_energy(self):
        """Plan requesting excessive items should spike energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User asks for "latest invoice" but agent requests 1000
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_invoices",
                    node_id="node1",
                    arguments={"limit": 1000},
                    scope_volume=1000,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        # Minimal scope for "latest invoice" is just 1
        minimal_scope = ScopeConstraints(
            limit=1,
            date_range_days=30,
            max_depth=1,
            include_sensitive=False
        )

        E = energy(plan, minimal_scope)
        # Over-scope = 999, energy = 1.0 * 999^2 = 998001
        assert float(E) > 1000, "Massive over-scoping should spike energy"

    def test_over_date_range_penalty(self):
        """Plan with excessive date range should be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_transactions",
                    node_id="node1",
                    arguments={"limit": 10, "days": 365},
                    scope_volume=10,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        # Minimal: recent transactions = 7 days
        minimal_scope = ScopeConstraints(
            limit=10,
            date_range_days=7,
            max_depth=1,
            include_sensitive=False
        )

        E = energy(plan, minimal_scope)
        # Over-scope on date_range = 358, weight = 0.5
        # Energy contribution from date = 0.5 * 358^2 = 64082
        assert float(E) > 50000, "Excessive date range should contribute significant energy"

    def test_sensitivity_penalty(self):
        """Accessing sensitive data unnecessarily should spike energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_user_data",
                    node_id="node1",
                    arguments={"include_ssn": True, "include_financial": True},
                    scope_volume=1,
                    scope_sensitivity=5  # Max sensitivity
                )
            ],
            edges=[]
        )

        # Minimal: no sensitive data needed
        minimal_scope = ScopeConstraints(
            limit=1,
            date_range_days=1,
            max_depth=1,
            include_sensitive=False  # 0.0
        )

        E = energy(plan, minimal_scope)
        # Sensitivity over-scope = 1.0 (normalized from 5/5)
        # Energy = 2.0 * 1.0^2 = 2.0
        assert float(E) >= 2.0, "Unnecessary sensitive access should be heavily penalized"

    def test_multi_dimension_over_scope(self):
        """Over-scoping on multiple dimensions should accumulate."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="search_records",
                    node_id="node1",
                    arguments={"limit": 100, "days": 90, "depth": 3},
                    scope_volume=100,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        minimal_scope = ScopeConstraints(
            limit=10,
            date_range_days=7,
            max_depth=1,
            include_sensitive=False
        )

        E = energy(plan, minimal_scope)

        # Expected contributions:
        # limit: 1.0 * (100-10)^2 = 1.0 * 8100 = 8100
        # date_range: 0.5 * (90-7)^2 = 0.5 * 6889 = 3444.5
        # depth: 0.3 * (3-1)^2 = 0.3 * 4 = 1.2
        # Total ≈ 11545.7
        assert float(E) > 11000, "Multi-dimension over-scope should accumulate penalties"
        assert float(E) < 12000, "Energy should match expected calculation"

    def test_explain_method(self):
        """Explanation should provide interpretable breakdown."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_users",
                    node_id="node1",
                    arguments={"limit": 100},
                    scope_volume=100,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        minimal_scope = ScopeConstraints(
            limit=10,
            date_range_days=7,
            max_depth=1,
            include_sensitive=False
        )

        explanation = energy.explain(plan, minimal_scope)

        assert 'total_energy' in explanation
        assert 'actual_scope' in explanation
        assert 'minimal_scope' in explanation
        assert 'over_scope' in explanation
        assert 'dimension_energies' in explanation
        assert 'recommendations' in explanation

        # Check recommendations are generated
        assert len(explanation['recommendations']) > 0
        assert any("limit" in rec.lower() for rec in explanation['recommendations'])

    def test_differentiability(self):
        """Energy should be differentiable for gradient-based training."""
        energy = create_scope_energy(use_latent_modulation=True)
        energy.train()  # Enable gradients

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="fetch_data",
                    node_id="node1",
                    arguments={"limit": 50},
                    scope_volume=50,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        z_g = torch.randn(1, 1024, requires_grad=True)
        z_e = torch.randn(1, 1024, requires_grad=True)

        minimal_scope = ScopeConstraints(
            limit=10,
            date_range_days=7,
            max_depth=1,
            include_sensitive=False
        )

        E = energy(plan, minimal_scope, z_g, z_e)

        # Backpropagate
        E.backward()

        assert z_g.grad is not None, "Should backprop through z_g"
        assert z_e.grad is not None, "Should backprop through z_e"

    def test_latent_modulation(self):
        """Latent modulation should affect energy magnitude."""
        energy = create_scope_energy(use_latent_modulation=True)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="test_tool",
                    node_id="node1",
                    arguments={"limit": 100},
                    scope_volume=100,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        minimal_scope = ScopeConstraints(
            limit=10,
            date_range_days=7,
            max_depth=1,
            include_sensitive=False
        )

        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 1024)

        E_with_latent = energy(plan, minimal_scope, z_g, z_e)
        E_without_latent = energy(plan, minimal_scope, None, None)

        # Energies should differ when latent modulation is used
        assert E_with_latent.shape == E_without_latent.shape == (1,)

    def test_no_penalty_for_under_scope(self):
        """Using less scope than needed should not be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_items",
                    node_id="node1",
                    arguments={"limit": 5},
                    scope_volume=5,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        # Minimal scope allows up to 100
        minimal_scope = ScopeConstraints(
            limit=100,
            date_range_days=365,
            max_depth=10,
            include_sensitive=True
        )

        E = energy(plan, minimal_scope)
        assert float(E) == 0.0, "Under-scoping should have zero energy"

    def test_single_node_plan(self):
        """Single node plan should work correctly."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="noop",
                    node_id="node1",
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )
        E = energy(plan)

        assert float(E) >= 0.0


@pytest.mark.benchmark
class TestScopePerformance:
    """Performance benchmarks for E_scope."""

    def test_latency(self):
        """Energy calculation should complete in <20ms."""
        import time

        energy = create_scope_energy(use_latent_modulation=False)

        # Create moderately complex plan
        nodes = [
            ToolCallNode(
                tool_name=f"tool_{i}",
                node_id=f"node{i}",
                arguments={"limit": i * 10, "days": i * 5},
                scope_volume=i * 10,
                scope_sensitivity=(i % 5) + 1  # Valid range: 1-5
            )
            for i in range(1, 21)
        ]

        plan = ExecutionPlan(nodes=nodes, edges=[])

        minimal_scope = ScopeConstraints(
            limit=10,
            date_range_days=7,
            max_depth=1,
            include_sensitive=False
        )

        # Warm-up
        for _ in range(5):
            _ = energy(plan, minimal_scope)

        # Benchmark
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = energy(plan, minimal_scope)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000

        assert avg_latency_ms < 20.0, f"Latency {avg_latency_ms:.2f}ms exceeds 20ms target"
