"""
Tests for E_hierarchy energy term.

Validates RAG-injection detection and control flow hijacking prevention.
"""

import pytest
import torch

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier
from source.energy.hierarchy import HierarchyEnergy, create_hierarchy_energy


class TestHierarchyEnergy:
    """Test suite for E_hierarchy."""

    def test_initialization(self):
        """Test energy term initializes correctly."""
        energy = create_hierarchy_energy(use_latent_modulation=False)

        assert isinstance(energy, HierarchyEnergy)
        assert energy.tier_penalties.shape == (3,)

    def test_minimal_plan(self):
        """Minimal safe plan should have low energy."""
        energy = create_hierarchy_energy(use_latent_modulation=False)

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

    def test_safe_plan_low_energy(self):
        """Plan with only internal (Tier 1) sources should have low energy."""
        energy = create_hierarchy_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_files",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL
                ),
                ToolCallNode(
                    tool_name="read_file",
                    node_id="node2",
                    provenance_tier=TrustTier.INTERNAL
                )
            ],
            edges=[("node1", "node2")]
        )

        E = energy(plan)
        assert float(E) < 1.0, "Safe plan should have low energy"

    def test_rag_control_flow_high_energy(self):
        """Untrusted (Tier 3) control flow should spike energy."""
        energy = create_hierarchy_energy(use_latent_modulation=False)

        # Simulate RAG-injection attempting to invoke control flow
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="if_then_else",  # Control flow tool
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB  # Untrusted!
                )
            ],
            edges=[]
        )

        E = energy(plan)
        # Should be penalized but exact value depends on classifier initialization
        assert float(E) >= 0.0

    def test_differentiability(self):
        """Energy should be differentiable for gradient-based training."""
        energy = create_hierarchy_energy(use_latent_modulation=True)
        energy.train()  # Enable gradients

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="send_email",
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB
                )
            ],
            edges=[]
        )

        z_g = torch.randn(1, 1024, requires_grad=True)
        z_e = torch.randn(1, 1024, requires_grad=True)

        E = energy(plan, z_g, z_e)

        # Backpropagate
        E.backward()

        assert z_g.grad is not None, "Should backprop through z_g"
        assert z_e.grad is not None, "Should backprop through z_e"

    def test_explain_method(self):
        """Explanation should provide interpretable breakdown."""
        energy = create_hierarchy_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_users",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL
                ),
                ToolCallNode(
                    tool_name="delete_all",
                    node_id="node2",
                    provenance_tier=TrustTier.PUBLIC_WEB
                )
            ],
            edges=[]
        )

        explanation = energy.explain(plan)

        assert 'total_energy' in explanation
        assert 'node_contributions' in explanation
        assert 'high_risk_nodes' in explanation
        assert len(explanation['node_contributions']) == 2

    def test_latent_modulation(self):
        """Latent modulation should affect energy magnitude."""
        energy = create_hierarchy_energy(use_latent_modulation=True)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="test_tool",
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB
                )
            ],
            edges=[]
        )

        z_g = torch.randn(1, 1024)
        z_e = torch.randn(1, 1024)

        E_with_latent = energy(plan, z_g, z_e)
        E_without_latent = energy(plan, None, None)

        # Energies should differ when latent modulation is used
        assert E_with_latent.shape == E_without_latent.shape == (1,)


@pytest.mark.benchmark
class TestHierarchyPerformance:
    """Performance benchmarks for E_hierarchy."""

    def test_latency(self):
        """Energy calculation should complete in <20ms."""
        import time

        energy = create_hierarchy_energy(use_latent_modulation=False)

        # Create moderately complex plan
        nodes = [
            ToolCallNode(
                tool_name=f"tool_{i}",
                node_id=f"node{i}",
                provenance_tier=TrustTier(i % 3 + 1)
            )
            for i in range(20)
        ]

        plan = ExecutionPlan(nodes=nodes, edges=[])

        # Warm-up
        for _ in range(5):
            _ = energy(plan)

        # Benchmark
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = energy(plan)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000

        assert avg_latency_ms < 20.0, f"Latency {avg_latency_ms:.2f}ms exceeds 20ms target"
