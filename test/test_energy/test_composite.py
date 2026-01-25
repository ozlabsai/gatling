"""
Tests for CompositeEnergy (full Product of Experts).

Validates that all four energy terms compose correctly and produce
interpretable security assessments.
"""

import pytest
import torch

from source.energy import create_composite_energy
from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier
from source.encoders.intent_predictor import ScopeConstraints


class TestCompositeEnergy:
    """Test suite for full energy function."""

    def test_initialization(self):
        """Composite energy should initialize all four terms."""
        energy = create_composite_energy(use_latent_modulation=False)

        assert energy.hierarchy is not None
        assert energy.provenance is not None
        assert energy.scope is not None
        assert energy.flow is not None
        assert energy.weights.shape == (4,)

    def test_safe_plan_low_energy(self):
        """Benign plan should have low total energy."""
        energy = create_composite_energy(use_latent_modulation=False)

        # Safe plan: Internal source, read-only, minimal scope
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_files",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=5,
                    scope_sensitivity=1,
                    arguments={"limit": 5}
                ),
                ToolCallNode(
                    tool_name="display_to_user",
                    node_id="node2",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=5,
                    scope_sensitivity=1
                )
            ],
            edges=[("node1", "node2")]
        )

        minimal_scope = ScopeConstraints(limit=5, include_sensitive=False)

        E = energy(plan, minimal_scope=minimal_scope)

        assert float(E) < 5.0, "Safe plan should have low energy"

    def test_malicious_plan_high_energy(self):
        """Attack plan should spike energy."""
        energy = create_composite_energy(use_latent_modulation=False)

        # Malicious plan: RAG-injection + high-privilege + exfiltration
        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="admin_list_all_users",  # High privilege
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB,  # Untrusted source!
                    scope_volume=10000,  # Over-scoped
                    scope_sensitivity=5,  # Sensitive data
                    arguments={"limit": 10000}
                ),
                ToolCallNode(
                    tool_name="send_to_external_api",  # Exfiltration
                    node_id="node2",
                    provenance_tier=TrustTier.PUBLIC_WEB,
                    scope_volume=10000,
                    scope_sensitivity=5
                )
            ],
            edges=[("node1", "node2")]
        )

        minimal_scope = ScopeConstraints(limit=10, include_sensitive=False)

        E = energy(plan, minimal_scope=minimal_scope)

        # Should trigger multiple energy terms
        assert float(E) > 5.0, "Malicious plan should have high energy"

    def test_explain_provides_breakdown(self):
        """Explanation should detail all energy term contributions."""
        energy = create_composite_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="fetch_sensitive_data",
                    node_id="node1",
                    provenance_tier=TrustTier.SIGNED_PARTNER,
                    scope_volume=100,
                    scope_sensitivity=3,
                    arguments={"limit": 100}
                )
            ],
            edges=[]
        )

        minimal_scope = ScopeConstraints(limit=10)
        explanation = energy.explain(plan, minimal_scope=minimal_scope)

        assert 'total_energy' in explanation
        assert 'term_energies' in explanation
        assert 'weighted_contributions' in explanation
        assert 'risk_assessment' in explanation
        assert 'recommended_actions' in explanation

        # All four terms should be present
        assert 'hierarchy' in explanation['term_energies']
        assert 'provenance' in explanation['term_energies']
        assert 'scope' in explanation['term_energies']
        assert 'flow' in explanation['term_energies']

    def test_term_disabling_for_ablation(self):
        """Should be able to disable individual terms."""
        energy = create_composite_energy(use_latent_modulation=False)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="test_tool",
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB,
                    scope_volume=100,
                    scope_sensitivity=3
                )
            ],
            edges=[]
        )

        # Full energy
        E_full = energy(plan)

        # Disable all terms
        energy.disable_term('hierarchy')
        energy.disable_term('provenance')
        energy.disable_term('scope')
        energy.disable_term('flow')

        E_disabled = energy(plan)

        assert float(E_disabled) == 0.0, "All terms disabled should yield zero energy"

        # Re-enable one term
        energy.enable_term('provenance')
        E_provenance_only = energy(plan)

        assert 0.0 < float(E_provenance_only) < float(E_full)

    def test_learnable_weights(self):
        """Weights should be trainable when learnable_weights=True."""
        energy = create_composite_energy(
            use_latent_modulation=False,
            learnable_weights=True
        )

        assert energy.weights.requires_grad, "Weights should be trainable"

        # Simulate gradient update
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

        E = energy(plan)
        E.backward()

        assert energy.weights.grad is not None, "Weights should receive gradients"

    def test_risk_assessment_categories(self):
        """Risk assessment should categorize energy levels."""
        energy = create_composite_energy(use_latent_modulation=False)

        # Safe plan
        safe_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="list_files",
                    node_id="node1",
                    provenance_tier=TrustTier.INTERNAL,
                    scope_volume=1,
                    scope_sensitivity=1
                )
            ],
            edges=[]
        )

        explanation = energy.explain(safe_plan)
        assert explanation['risk_assessment'] == 'safe'

        # Critical plan
        critical_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="admin_delete_all",
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB,
                    scope_volume=10000,
                    scope_sensitivity=5
                )
            ],
            edges=[]
        )

        explanation = energy.explain(critical_plan)
        # Should be at least 'suspicious', likely 'critical'
        assert explanation['risk_assessment'] in ['suspicious', 'critical']

    def test_differentiability_end_to_end(self):
        """Full energy should backpropagate through all terms."""
        energy = create_composite_energy(
            use_latent_modulation=True,
            learnable_weights=True
        )
        energy.train()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="test_tool",
                    node_id="node1",
                    provenance_tier=TrustTier.PUBLIC_WEB,
                    scope_volume=100,
                    scope_sensitivity=3
                )
            ],
            edges=[]
        )

        z_g = torch.randn(1, 1024, requires_grad=True)
        z_e = torch.randn(1, 1024, requires_grad=True)

        E = energy(plan, z_g, z_e)
        E.backward()

        assert z_g.grad is not None
        assert z_e.grad is not None
        assert energy.weights.grad is not None


@pytest.mark.benchmark
class TestCompositePerformance:
    """Performance benchmarks for composite energy."""

    def test_total_latency_under_20ms(self):
        """Full energy calculation should complete in <20ms (PRD requirement)."""
        import time

        energy = create_composite_energy(use_latent_modulation=False)

        # Realistic plan size
        nodes = [
            ToolCallNode(
                tool_name=f"tool_{i}",
                node_id=f"node{i}",
                provenance_tier=TrustTier(i % 3 + 1),
                scope_volume=10,
                scope_sensitivity=i % 5 + 1
            )
            for i in range(10)
        ]

        edges = [(f"node{i}", f"node{i+1}") for i in range(9)]

        plan = ExecutionPlan(nodes=nodes, edges=edges)
        minimal_scope = ScopeConstraints(limit=5)

        # Warm-up
        for _ in range(10):
            _ = energy(plan, minimal_scope=minimal_scope)

        # Benchmark
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = energy(plan, minimal_scope=minimal_scope)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000

        print(f"\nAverage latency: {avg_latency_ms:.2f}ms")
        assert avg_latency_ms < 20.0, f"Latency {avg_latency_ms:.2f}ms exceeds 20ms target"
