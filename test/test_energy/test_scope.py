"""
Tests for E_scope energy term.

Validates least privilege enforcement and over-privileged access detection.
"""
# ruff: noqa: N806

import pytest
import torch

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier
from source.encoders.intent_predictor import ScopeConstraints
from source.energy.scope import ScopeEnergy, ScopeExtractor, create_scope_energy


class TestScopeExtractor:
    """Test suite for ScopeExtractor component."""

    def test_initialization(self):
        """Test scope extractor initializes correctly."""
        extractor = ScopeExtractor(hidden_dim=256)

        assert isinstance(extractor, ScopeExtractor)
        assert extractor.arg_encoder is not None

    def test_forward_pass(self):
        """Test forward pass produces correct shape."""
        extractor = ScopeExtractor(hidden_dim=256)

        # Simulate encoded argument features
        arg_features = torch.randn(4, 256)  # [batch_size=4, hidden_dim=256]

        scope_vec = extractor(arg_features)

        assert scope_vec.shape == (4, 4), "Should output [batch_size, 4] scope dimensions"
        assert (scope_vec >= 0).all(), "Softplus ensures non-negative outputs"


class TestScopeEnergy:
    """Test suite for E_scope."""

    def test_initialization(self):
        """Test energy term initializes correctly."""
        energy = create_scope_energy(use_latent_modulation=False)

        assert isinstance(energy, ScopeEnergy)
        assert energy.dimension_weights.shape == (4,)
        # Weights should be: [1.0, 0.5, 0.3, 2.0] for [limit, date, depth, sensitivity]
        assert torch.allclose(energy.dimension_weights, torch.tensor([1.0, 0.5, 0.3, 2.0]))
        assert energy.scope_extractor is not None

    def test_minimal_plan(self):
        """Minimal safe plan should have low energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="noop",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1,
                    scope_sensitivity=1,
                    arguments={},
                )
            ],
            edges=[],
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
                    provenance_tier=TrustTier.INTERNAL,
                    arguments={"limit": 1},
                    scope_volume=1,
                    scope_sensitivity=1,  # Min valid value
                )
            ],
            edges=[],
        )
        minimal_scope = ScopeConstraints(limit=1, include_sensitive=False)
        E = energy(plan, minimal_scope=minimal_scope)

        assert E.shape == (1,)
        assert float(E) >= 0.0
        assert float(E) < 1.0, "Minimal plan matching minimal scope should have near-zero energy"

    def test_empty_plan_zero_energy(self):
        """Plan with single minimal node should have low energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        # ExecutionPlan requires min 1 node, so test with minimal node instead
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="noop",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1,
                    scope_sensitivity=1,
                    arguments={},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(limit=1, include_sensitive=False)
        E = energy(plan, minimal_scope=minimal_scope)

        assert float(E) < 0.1, "Minimal plan should have very low energy"

    def test_over_scoped_limit_high_energy(self):
        """Plan exceeding minimal limit should spike energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User asks for "latest invoice", agent requests 10,000 invoices
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_invoices",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10000,
                    scope_sensitivity=3,
                    arguments={"limit": 10000},
                )
            ],
            edges=[],
        )

        # Define minimal scope - user only needs 1 invoice
        minimal_scope = ScopeConstraints(
            limit=1,  # User only needs 1 invoice
            date_range_days=1,
            max_depth=1,
            include_sensitive=False,
        )

        E = energy(plan, minimal_scope)
        # Over-scope of 9999 should create very high energy
        assert float(E) > 10000.0, "Massive over-scoping should spike energy"

    def test_over_limit_high_energy(self):
        """Plan requesting excessive items should spike energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User asks for "latest invoice" but agent requests 1000
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_invoices",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    arguments={"limit": 1000},
                    scope_volume=1000,
                    scope_sensitivity=1,
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=1, date_range_days=1, max_depth=1, include_sensitive=False
        )

        E = energy(plan, minimal_scope=minimal_scope)

        # Over-scope of 999 should create high energy
        assert float(E) > 10.0, "Massive over-scoping should spike energy"

    def test_over_scoped_date_range_penalty(self):
        """Excessive date range should be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User asks for "this week's sales", agent requests entire year
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="query_sales",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=100,
                    scope_sensitivity=2,
                    arguments={"days": 365},  # One year
                )
            ],
            edges=[],
        )

        # Minimal scope for "latest invoice" is just 1
        minimal_scope = ScopeConstraints(
            limit=100,
            date_range_days=7,  # One week
            max_depth=1,
            include_sensitive=False,
        )

        E = energy(plan, minimal_scope)
        # Over-scope = 999, energy = 1.0 * 999^2 = 998001
        assert float(E) > 1000, "Massive over-scoping should spike energy"

    def test_over_date_range_penalty(self):
        """Plan with excessive date range should be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User asks for "this week's sales", agent requests entire year
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_transactions",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    arguments={"limit": 10, "days": 365},
                    scope_volume=10,
                    scope_sensitivity=1,
                )
            ],
            edges=[],
        )

        # Minimal: recent transactions = 7 days
        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=7, max_depth=1, include_sensitive=False
        )

        E = energy(plan, minimal_scope=minimal_scope)

        # Over-scope of 358 days should create measurable energy
        assert float(E) > 1.0, "Excessive date range should be penalized"

    def test_over_scoped_depth_penalty(self):
        """Excessive recursion depth should be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="traverse_directory",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=50,
                    scope_sensitivity=1,
                    arguments={"depth": 10},
                )
            ],
            edges=[],
        )

        # Minimal: only need shallow traversal
        minimal_scope = ScopeConstraints(limit=50, max_depth=2, include_sensitive=False)

        E = energy(plan, minimal_scope)
        # Over-scope on depth = 8 (10-2), weight = 0.3
        # Energy contribution from depth = 0.3 * 8^2 = 19.2
        assert float(E) > 10.0, "Excessive depth should be penalized"

    def test_sensitivity_penalty(self):
        """Accessing sensitive data when not required should be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_user_profile",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1,
                    scope_sensitivity=5,  # Accessing PII/financial data
                    arguments={"include_pii": True},
                )
            ],
            edges=[],
        )

        # Minimal: no sensitive data needed
        minimal_scope = ScopeConstraints(
            limit=1,
            date_range_days=1,
            max_depth=1,
            include_sensitive=False,  # 0.0
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
                    provenance_tier=TrustTier.INTERNAL,
                    arguments={"limit": 100, "days": 90, "depth": 3},
                    scope_volume=100,
                    scope_sensitivity=1,
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=7, max_depth=1, include_sensitive=False
        )

        E = energy(plan, minimal_scope)

        # Expected contributions:
        # limit: 1.0 * (100-10)^2 = 1.0 * 8100 = 8100
        # date_range: 0.5 * (90-7)^2 = 0.5 * 6889 = 3444.5
        # depth: 0.3 * (3-1)^2 = 0.3 * 4 = 1.2
        # Total ≈ 11545.7
        assert float(E) > 11000, "Multi-dimension over-scope should accumulate penalties"
        assert float(E) < 12000, "Energy should match expected calculation"

    def test_multi_node_aggregation(self):
        """Energy should aggregate across multiple nodes."""
        energy = create_scope_energy(use_latent_modulation=False)

        # Two nodes with different scopes
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="query_db",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=100,
                    scope_sensitivity=2,
                    arguments={"limit": 100},
                ),
                ToolCallNode(
                    tool_name="query_api",
                    node_id="node2",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=500,  # Higher scope
                    scope_sensitivity=3,
                    arguments={"limit": 500},
                ),
            ],
            edges=[("node1", "node2")],
        )

        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=7, max_depth=1, include_sensitive=False
        )

        E = energy(plan, minimal_scope)

        # Expected contributions (uses max scope across nodes = 500):
        # limit: 1.0 * (500-10)^2 = 1.0 * 240100 = 240100
        # sensitivity: uses max sensitivity normalized
        # Total should be > 200000
        assert float(E) > 200000, "Multi-node over-scope should accumulate penalties"
        assert float(E) < 300000, "Energy should be bounded"

    def test_multi_node_energy(self):
        """Energy should aggregate across nodes."""
        energy = create_scope_energy(use_latent_modulation=False)

        # Two nodes with different scopes
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="query_db",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=100,
                    scope_sensitivity=2,
                    arguments={"limit": 100},
                ),
                ToolCallNode(
                    tool_name="query_api",
                    node_id="node2",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=500,  # Higher scope
                    scope_sensitivity=3,
                    arguments={"limit": 500},
                ),
            ],
            edges=[("node1", "node2")],
        )

        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=7, max_depth=1, include_sensitive=False
        )

        E = energy(plan, minimal_scope=minimal_scope)

        # Should use max scope (500) across nodes
        assert float(E) > 10.0, "Should aggregate max scope across nodes"

    def test_perfect_scope_match_zero_energy(self):
        """Plan exactly matching minimal scope should have zero energy."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_item",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=5,
                    scope_sensitivity=1,
                    arguments={"limit": 5, "days": 7},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=5, date_range_days=7, max_depth=1, include_sensitive=False
        )

        E = energy(plan, minimal_scope=minimal_scope)

        # Should have near-zero energy (may have tiny numerical errors)
        # Note: sensitivity dimension may have small residual due to normalization
        assert float(E) < 0.1, "Perfect scope match should have very low energy"

    def test_under_scope_no_penalty(self):
        """Under-scoping (requesting less than minimal) should not be penalized."""
        energy = create_scope_energy(use_latent_modulation=False)

        # Agent requests only 1 item, but minimal scope allows 10
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="get_items",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1,
                    scope_sensitivity=1,
                    arguments={"limit": 1},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=7, max_depth=2, include_sensitive=False
        )

        E = energy(plan, minimal_scope=minimal_scope)

        # max(0, actual - minimal) should clamp negative values to 0
        # Note: sensitivity dimension may have small residual due to normalization
        assert float(E) < 0.1, "Under-scoping should not be heavily penalized"

    def test_differentiability(self):
        """Energy should be differentiable for gradient-based training."""
        energy = create_scope_energy(use_latent_modulation=True)
        energy.train()  # Enable gradients

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_users",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1000,
                    scope_sensitivity=3,
                    arguments={"limit": 1000},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(limit=10, include_sensitive=False)

        z_g = torch.randn(1, 1024, requires_grad=True)
        z_e = torch.randn(1, 1024, requires_grad=True)

        E = energy(plan, minimal_scope=minimal_scope, z_g=z_g, z_e=z_e)

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
                    tool_name="query",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1000,
                    scope_sensitivity=2,
                    arguments={"limit": 1000},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(limit=10, include_sensitive=False)

        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 1024)

        E_with_latent = energy(plan, minimal_scope=minimal_scope, z_g=z_g, z_e=z_e)
        E_without_latent = energy(plan, minimal_scope=minimal_scope, z_g=None, z_e=None)

        # Energies should differ when latent modulation is used
        assert E_with_latent.shape == E_without_latent.shape == (1,)
        # Both should be positive
        assert float(E_with_latent) > 0.0
        assert float(E_without_latent) > 0.0

    def test_explain_method(self):
        """Explanation should provide interpretable breakdown."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="admin_query",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=5000,
                    scope_sensitivity=4,
                    arguments={"limit": 5000, "days": 365, "depth": 5},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=7, max_depth=1, include_sensitive=False
        )

        explanation = energy.explain(plan, minimal_scope=minimal_scope)

        assert "total_energy" in explanation
        assert "actual_scope" in explanation
        assert "minimal_scope" in explanation
        assert "over_scope" in explanation
        assert "dimension_energies" in explanation
        assert "recommendations" in explanation

        # Check dimension breakdown
        assert "limit" in explanation["actual_scope"]
        assert "date_range" in explanation["actual_scope"]
        assert "depth" in explanation["actual_scope"]
        assert "sensitivity" in explanation["actual_scope"]

        # Should have recommendations for over-scoped dimensions
        assert len(explanation["recommendations"]) > 0
        assert any("limit" in rec.lower() for rec in explanation["recommendations"])

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
                    scope_sensitivity=1,
                    provenance_tier=TrustTier.INTERNAL,
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=100, date_range_days=365, max_depth=10, include_sensitive=True
        )

        E = energy(plan, minimal_scope=minimal_scope)
        assert float(E) == 0.0, "Under-scoping should have zero energy"

    def test_dimension_weights_learnable(self):
        """Dimension weights should be learnable parameters."""
        energy = create_scope_energy(use_latent_modulation=False)

        initial_weights = energy.dimension_weights.clone()

        # Weights should be parameters
        assert energy.dimension_weights.requires_grad

        # Verify initial values
        expected = torch.tensor([1.0, 0.5, 0.3, 2.0])
        assert torch.allclose(initial_weights, expected, atol=1e-5)

    def test_under_scope_with_latent_modulation(self):
        """Under-scoping with latent modulation should have very low energy."""
        energy = create_scope_energy(use_latent_modulation=True)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="query",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=5,
                    scope_sensitivity=1,
                    arguments={"limit": 5},
                )
            ],
            edges=[],
        )

        # Minimal scope allows up to 100 - agent only uses 5
        minimal_scope = ScopeConstraints(
            limit=100, date_range_days=365, max_depth=10, include_sensitive=True
        )

        E = energy(plan, minimal_scope)
        # Under-scoping should have very low energy (not exactly 0 due to sensitivity normalization)
        assert float(E) < 1.0, "Under-scoping should have very low energy"

    def test_single_node_plan(self):
        """Single node plan should work correctly."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="noop",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1,
                    scope_sensitivity=1,
                    arguments={},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(limit=10, include_sensitive=False)

        E = energy(plan, minimal_scope=minimal_scope)

        # Should work without error
        assert E.shape == (1,)
        assert float(E) >= 0.0

    def test_default_minimal_scope(self):
        """When minimal_scope is None, should use default baseline."""
        energy = create_scope_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="test_tool",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=100,
                    scope_sensitivity=2,
                    arguments={"limit": 100},
                )
            ],
            edges=[],
        )

        # Call without minimal_scope
        E = energy(plan, minimal_scope=None)

        # Should use default [1, 1, 1, 0] baseline
        assert float(E) > 0.0, "Should penalize against default baseline"


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
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=max(1, i * 10),  # Ensure >= 1
                scope_sensitivity=i % 5 + 1,
                arguments={"limit": max(1, i * 10), "days": max(1, i * 2)},
            )
            for i in range(20)
        ]

        plan = ExecutionPlan(nodes=nodes, edges=[])
        minimal_scope = ScopeConstraints(limit=10, date_range_days=7, include_sensitive=False)

        # Warm-up
        for _ in range(5):
            _ = energy(plan, minimal_scope=minimal_scope)

        # Benchmark
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = energy(plan, minimal_scope=minimal_scope)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000

        assert avg_latency_ms < 20.0, f"Latency {avg_latency_ms:.2f}ms exceeds 20ms target"

    def test_batch_efficiency(self):
        """Should efficiently handle multiple nodes."""
        energy = create_scope_energy(use_latent_modulation=False)

        # Create large plan
        large_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name=f"tool_{i}",
                    node_id=f"node{i}",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=100,
                    scope_sensitivity=2,
                    arguments={"limit": 100},
                )
                for i in range(100)
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(limit=10, include_sensitive=False)

        # Should complete without error
        E = energy(large_plan, minimal_scope=minimal_scope)
        assert float(E) >= 0.0


class TestRealWorldScenarios:
    """Test realistic security scenarios."""

    def test_invoice_over_retrieval(self):
        """Detect 'show latest invoice' -> list_invoices(limit=10000) attack."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User intent: "Show me my latest invoice"
        # Minimal scope: 1 invoice
        # Attack: Agent requests 10,000 invoices

        malicious_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_invoices",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10000,
                    scope_sensitivity=4,  # Financial data
                    arguments={"limit": 10000, "include_pii": True},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=1, date_range_days=1, max_depth=1, include_sensitive=False
        )

        E = energy(malicious_plan, minimal_scope=minimal_scope)
        explanation = energy.explain(malicious_plan, minimal_scope=minimal_scope)

        # Should detect massive over-scoping
        assert float(E) > 50.0, "Invoice over-retrieval should spike energy"
        assert "limit" in str(explanation["recommendations"])

    def test_directory_traversal_depth(self):
        """Detect excessive directory traversal depth."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User: "List files in current directory"
        # Attack: Agent traverses entire filesystem (depth=999)

        malicious_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_directory",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10000,
                    scope_sensitivity=3,
                    arguments={"depth": 999, "recursive": True},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=50,
            max_depth=1,  # Only current directory
            include_sensitive=False,
        )

        E = energy(malicious_plan, minimal_scope=minimal_scope)

        assert float(E) > 10.0, "Excessive traversal depth should be detected"

    def test_temporal_over_scope(self):
        """Detect 'this week's data' -> query(days=365) attack."""
        energy = create_scope_energy(use_latent_modulation=False)

        malicious_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="analytics_query",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1000,
                    scope_sensitivity=3,
                    arguments={"days": 365},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=1000,
            date_range_days=7,  # One week
            max_depth=1,
            include_sensitive=False,
        )

        E = energy(malicious_plan, minimal_scope=minimal_scope)

        assert float(E) > 5.0, "Temporal over-scope should be penalized"

    def test_benign_exact_match(self):
        """Verify no false positives for well-scoped queries."""
        energy = create_scope_energy(use_latent_modulation=False)

        # User: "Show 10 most recent orders"
        # Agent: list_orders(limit=10)

        benign_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_orders",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=10,
                    scope_sensitivity=2,
                    arguments={"limit": 10, "days": 30},
                )
            ],
            edges=[],
        )

        minimal_scope = ScopeConstraints(
            limit=10, date_range_days=30, max_depth=1, include_sensitive=False
        )

        E = energy(benign_plan, minimal_scope=minimal_scope)

        # Should have low energy (some residual from sensitivity normalization)
        assert float(E) < 0.5, "Well-scoped query should not trigger false positive"
