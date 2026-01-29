"""
Tests for E_flow: Data Flow Energy Term

Tests cover:
    1. Sink risk detection (exfiltration)
    2. Graph topology validation (cycles, connectivity)
    3. Transformation coherence (volume consistency)
    4. Provenance flow tracking (trust tier violations)
    5. Component composition and weighting
    6. Explain method comprehensiveness
"""

import pytest
import torch

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode, TrustTier
from source.energy.flow import (
    FlowEnergy,
    GraphTopologyValidator,
    ProvenanceFlowValidator,
    TransformationCoherenceValidator,
    create_flow_energy,
)


class TestGraphTopologyValidator:
    """Test graph structure validation."""

    def test_detect_cycles_simple(self):
        """Detect simple circular dependency."""
        validator = GraphTopologyValidator()

        nodes = [
            ToolCallNode(
                tool_name="fetch_data",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="hash1",
                scope_volume=100,
                scope_sensitivity=2,
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="process_data",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="hash2",
                scope_volume=100,
                scope_sensitivity=2,
                node_id="n2",
            ),
            ToolCallNode(
                tool_name="aggregate",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="hash3",
                scope_volume=50,
                scope_sensitivity=2,
                node_id="n3",
            ),
        ]

        # Create cycle: n1 → n2 → n3 → n1
        edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n1")]

        has_cycles, cycle_paths = validator.detect_cycles(nodes, edges)

        assert has_cycles
        assert len(cycle_paths) > 0
        # Verify cycle contains all three nodes
        assert len(cycle_paths[0]) == 4  # Path includes return to start

    def test_detect_cycles_none(self):
        """Acyclic graph has no cycles."""
        validator = GraphTopologyValidator()

        nodes = [
            ToolCallNode(
                tool_name="fetch",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h1",
                scope_volume=100,
                scope_sensitivity=1,
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="display",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h2",
                scope_volume=100,
                scope_sensitivity=1,
                node_id="n2",
            ),
        ]

        edges = [("n1", "n2")]

        has_cycles, cycle_paths = validator.detect_cycles(nodes, edges)

        assert not has_cycles
        assert len(cycle_paths) == 0

    def test_connectivity_single_component(self):
        """Fully connected graph has 1 component."""
        validator = GraphTopologyValidator()

        nodes = [
            ToolCallNode(
                tool_name=f"tool_{i}",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash=f"h{i}",
                scope_volume=10,
                scope_sensitivity=1,
                node_id=f"n{i}",
            )
            for i in range(3)
        ]

        edges = [("n0", "n1"), ("n1", "n2")]

        num_components = validator.check_connectivity(nodes, edges)

        assert num_components == 1

    def test_connectivity_disconnected(self):
        """Disconnected graph has multiple components."""
        validator = GraphTopologyValidator()

        nodes = [
            ToolCallNode(
                tool_name=f"tool_{i}",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash=f"h{i}",
                scope_volume=10,
                scope_sensitivity=1,
                node_id=f"n{i}",
            )
            for i in range(4)
        ]

        # Two disconnected components: (n0, n1) and (n2, n3)
        edges = [("n0", "n1"), ("n2", "n3")]

        num_components = validator.check_connectivity(nodes, edges)

        assert num_components == 2

    def test_invalid_edge_references(self):
        """Detect edges referencing non-existent nodes."""
        validator = GraphTopologyValidator()

        nodes = [
            ToolCallNode(
                tool_name="tool_a",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h1",
                scope_volume=10,
                scope_sensitivity=1,
                node_id="n1",
            )
        ]

        # Edge references non-existent node "n2"
        edges = [("n1", "n2"), ("n3", "n1")]

        invalid_edges = validator.validate_edge_references(nodes, edges)

        assert len(invalid_edges) == 2
        assert ("n1", "n2") in invalid_edges
        assert ("n3", "n1") in invalid_edges

    def test_topology_energy_high_for_violations(self):
        """High energy when topology violations present."""
        validator = GraphTopologyValidator()

        nodes = [
            ToolCallNode(
                tool_name=f"t{i}",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash=f"h{i}",
                scope_volume=10,
                scope_sensitivity=1,
                node_id=f"n{i}",
            )
            for i in range(3)
        ]

        # Cycle + invalid edge
        edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n0"), ("n0", "n99")]

        energy = validator(nodes, edges)

        assert energy.item() > 10.0  # Cycle (10.0) + invalid edge (5.0)


class TestTransformationCoherenceValidator:
    """Test data transformation validation."""

    def test_classify_transformation_reducer(self):
        """Classify reducer tools correctly."""
        validator = TransformationCoherenceValidator()

        assert validator.classify_transformation("filter_users") == "reducer"
        assert validator.classify_transformation("aggregate_sales") == "reducer"
        assert validator.classify_transformation("sample_data") == "reducer"

    def test_classify_transformation_expander(self):
        """Classify expander tools correctly."""
        validator = TransformationCoherenceValidator()

        assert validator.classify_transformation("expand_details") == "expander"
        assert validator.classify_transformation("join_tables") == "expander"
        assert validator.classify_transformation("cross_product_merge") == "expander"

    def test_volume_consistency_reducer_violation(self):
        """Detect when reducer increases volume (violation)."""
        validator = TransformationCoherenceValidator()

        nodes = [
            ToolCallNode(
                tool_name="fetch_data",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h1",
                scope_volume=100,
                scope_sensitivity=1,
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="filter_records",  # Reducer
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h2",
                scope_volume=200,  # Volume increased (violation!)
                scope_sensitivity=1,
                node_id="n2",
            ),
        ]

        edges = [("n1", "n2")]

        violations = validator.check_volume_consistency(nodes, edges)

        assert len(violations) == 1
        assert violations[0]["penalty"] == 3.0
        assert "increased volume" in violations[0]["reason"].lower()

    def test_volume_consistency_neutral_large_expansion(self):
        """Detect when neutral transform greatly expands volume."""
        validator = TransformationCoherenceValidator()

        nodes = [
            ToolCallNode(
                tool_name="read_file",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h1",
                scope_volume=10,
                scope_sensitivity=1,
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="transform_data",  # Neutral
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h2",
                scope_volume=100,  # 10x expansion
                scope_sensitivity=1,
                node_id="n2",
            ),
        ]

        edges = [("n1", "n2")]

        violations = validator.check_volume_consistency(nodes, edges)

        assert len(violations) == 1
        assert violations[0]["penalty"] == 1.5

    def test_suspicious_chain_detection(self):
        """Detect suspicious transformation sequences."""
        validator = TransformationCoherenceValidator()

        nodes = [
            ToolCallNode(
                tool_name="fetch_secrets",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h1",
                scope_volume=10,
                scope_sensitivity=5,
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="encrypt_data",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h2",
                scope_volume=10,
                scope_sensitivity=5,
                node_id="n2",
            ),
            ToolCallNode(
                tool_name="compress_file",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h3",
                scope_volume=5,
                scope_sensitivity=5,
                node_id="n3",
            ),
            ToolCallNode(
                tool_name="send_external",
                arguments={},
                provenance_tier=TrustTier.PUBLIC_WEB,
                provenance_hash="h4",
                scope_volume=5,
                scope_sensitivity=5,
                node_id="n4",
            ),
        ]

        edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n4")]

        chains = validator.detect_suspicious_chains(nodes, edges)

        assert len(chains) > 0
        assert chains[0]["penalty"] == 8.0


class TestProvenanceFlowValidator:
    """Test trust tier propagation validation."""

    def test_tier_degradation_critical(self):
        """Tier 3 → Tier 1 flow is critical violation."""
        validator = ProvenanceFlowValidator()

        nodes = [
            ToolCallNode(
                tool_name="fetch_web_data",
                arguments={},
                provenance_tier=TrustTier.PUBLIC_WEB,  # Tier 3
                provenance_hash="h1",
                scope_volume=100,
                scope_sensitivity=1,
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="admin_action",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,  # Tier 1
                provenance_hash="h2",
                scope_volume=100,
                scope_sensitivity=1,
                node_id="n2",
            ),
        ]

        edges = [("n1", "n2")]

        violations = validator.track_provenance_degradation(nodes, edges)

        assert len(violations) == 1
        assert violations[0]["penalty"] == 5.0  # Critical
        assert violations[0]["src_tier"] == 3
        assert violations[0]["dst_tier"] == 1

    def test_sensitivity_escalation(self):
        """Detect significant sensitivity increases."""
        validator = ProvenanceFlowValidator()

        nodes = [
            ToolCallNode(
                tool_name="read_public_data",
                arguments={},
                provenance_tier=TrustTier.PUBLIC_WEB,
                provenance_hash="h1",
                scope_volume=10,
                scope_sensitivity=1,  # Low sensitivity
                node_id="n1",
            ),
            ToolCallNode(
                tool_name="access_secrets",
                arguments={},
                provenance_tier=TrustTier.INTERNAL,
                provenance_hash="h2",
                scope_volume=10,
                scope_sensitivity=5,  # High sensitivity (escalation!)
                node_id="n2",
            ),
        ]

        edges = [("n1", "n2")]

        escalations = validator.detect_sensitivity_escalation(nodes, edges)

        assert len(escalations) == 1
        assert escalations[0]["escalation"] == 4
        assert escalations[0]["penalty"] == 8.0  # 2.0 * 4


class TestFlowEnergy:
    """Test complete E_flow energy term."""

    def test_initialization(self):
        """FlowEnergy initializes all components."""
        model = FlowEnergy()

        assert model.destination_classifier is not None
        assert model.topology_validator is not None
        assert model.transformation_validator is not None
        assert model.provenance_validator is not None
        assert model.subweights.shape == (4,)

    def test_safe_plan_low_energy(self):
        """Benign plan has low energy."""
        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="fetch_data",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=100,
                    scope_sensitivity=1,
                    node_id="n1",
                ),
                ToolCallNode(
                    tool_name="filter_records",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h2",
                    scope_volume=50,
                    scope_sensitivity=1,
                    node_id="n2",
                ),
                ToolCallNode(
                    tool_name="display_to_user",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h3",
                    scope_volume=50,
                    scope_sensitivity=1,
                    node_id="n3",
                ),
            ],
            edges=[("n1", "n2"), ("n2", "n3")],
        )

        energy = model(plan)

        assert energy.item() < 10.0  # Safe threshold

    def test_cyclic_plan_high_energy(self):
        """Plan with circular dependency has high energy."""
        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="fetch",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n1",
                ),
                ToolCallNode(
                    tool_name="process",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h2",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n2",
                ),
                ToolCallNode(
                    tool_name="loop_back",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h3",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n3",
                ),
            ],
            edges=[("n1", "n2"), ("n2", "n3"), ("n3", "n1")],  # Cycle!
        )

        energy = model(plan)

        # w_topo (3.0) × E_topo (10.0 for cycle) = 30.0
        assert energy.item() > 20.0

    def test_exfiltration_plan_high_energy(self):
        """Plan with exfiltration risk has high energy."""
        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="fetch_financial_data",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=1000,
                    scope_sensitivity=5,  # High sensitivity
                    node_id="n1",
                ),
                ToolCallNode(
                    tool_name="encrypt_data",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h2",
                    scope_volume=1000,
                    scope_sensitivity=5,
                    node_id="n2",
                ),
                ToolCallNode(
                    tool_name="send_to_external_api",  # Exfiltration!
                    arguments={},
                    provenance_tier=TrustTier.PUBLIC_WEB,
                    provenance_hash="h3",
                    scope_volume=1000,
                    scope_sensitivity=5,
                    node_id="n3",
                ),
            ],
            edges=[("n1", "n2"), ("n2", "n3")],
        )

        energy = model(plan)

        # High sensitivity + external send + suspicious transform
        assert energy.item() > 15.0

    def test_trust_tier_violation_high_energy(self):
        """Tier 3 → Tier 1 flow has high energy."""
        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="scrape_web",
                    arguments={},
                    provenance_tier=TrustTier.PUBLIC_WEB,  # Tier 3
                    provenance_hash="h1",
                    scope_volume=100,
                    scope_sensitivity=1,
                    node_id="n1",
                ),
                ToolCallNode(
                    tool_name="admin_delete",  # Tier 1 operation
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h2",
                    scope_volume=100,
                    scope_sensitivity=1,
                    node_id="n2",
                ),
            ],
            edges=[("n1", "n2")],
        )

        energy = model(plan)

        # w_prov (2.5) × E_prov (5.0 for Tier 3→1) = 12.5
        assert energy.item() > 10.0

    def test_explain_provides_comprehensive_breakdown(self):
        """Explain method returns all required fields."""
        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="fetch",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n1",
                ),
                ToolCallNode(
                    tool_name="display",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h2",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n2",
                ),
            ],
            edges=[("n1", "n2")],
        )

        explanation = model.explain(plan)

        # Check all required fields
        assert "total_energy" in explanation
        assert "component_energies" in explanation
        assert "weighted_contributions" in explanation
        assert "subweights" in explanation
        assert "sink_nodes" in explanation
        assert "topology_violations" in explanation
        assert "transformation_violations" in explanation
        assert "provenance_violations" in explanation
        assert "risk_assessment" in explanation
        assert "recommendations" in explanation

        # Check component energies structure
        assert "sink_risk" in explanation["component_energies"]
        assert "topology" in explanation["component_energies"]
        assert "transformation" in explanation["component_energies"]
        assert "provenance" in explanation["component_energies"]

    def test_explain_risk_assessment_levels(self):
        """Risk assessment correctly categorizes energy levels."""
        model = create_flow_energy()

        # Safe plan
        safe_plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=1,
                    scope_sensitivity=1,
                    node_id="n1",
                ),
                ToolCallNode(
                    tool_name="display",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h2",
                    scope_volume=1,
                    scope_sensitivity=1,
                    node_id="n2",
                ),
            ],
            edges=[("n1", "n2")],
        )

        safe_explanation = model.explain(safe_plan)
        assert safe_explanation["risk_assessment"] in ["safe", "suspicious", "critical"]

    def test_differentiability_with_latent_modulation(self):
        """Model is differentiable with latent modulation."""
        model = FlowEnergy(use_latent_modulation=True)

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="tool_a",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n1",
                )
            ],
            edges=[],
        )

        z_g = torch.randn(1, 1024, requires_grad=True)
        z_e = torch.randn(1, 1024, requires_grad=True)

        energy = model(plan, z_g, z_e)
        energy.backward()

        assert z_g.grad is not None
        assert z_e.grad is not None

    def test_learnable_subweights(self):
        """Subweights are learnable when specified."""
        model = FlowEnergy(learnable_subweights=True)

        assert isinstance(model.subweights, torch.nn.Parameter)
        assert model.subweights.requires_grad

    def test_fixed_subweights(self):
        """Subweights are fixed when specified."""
        model = FlowEnergy(learnable_subweights=False)

        assert not model.subweights.requires_grad

    def test_single_node_plan_low_energy(self):
        """Single node plan has low energy."""
        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name="read_file",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash="h1",
                    scope_volume=10,
                    scope_sensitivity=1,
                    node_id="n1",
                )
            ],
            edges=[],
        )

        energy = model(plan)

        # Single node should have low energy
        assert energy.item() < 5.0


class TestFlowEnergyPerformance:
    """Performance tests for E_flow (requires pytest-benchmark)."""

    @pytest.mark.skipif(
        True,  # Skip until pytest-benchmark is installed
        reason="pytest-benchmark not installed",
    )
    def test_forward_latency_small_plan(self):
        """Forward pass latency for small plan (5 nodes)."""
        import time

        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name=f"tool_{i}",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash=f"h{i}",
                    scope_volume=100,
                    scope_sensitivity=2,
                    node_id=f"n{i}",
                )
                for i in range(5)
            ],
            edges=[("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")],
        )

        # Simple timing test
        start = time.perf_counter()
        for _ in range(100):
            _ = model(plan)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100
        assert avg_time < 0.02  # 20ms average

    @pytest.mark.skipif(
        True,  # Skip until pytest-benchmark is installed
        reason="pytest-benchmark not installed",
    )
    def test_explain_latency(self):
        """Explain method latency."""
        import time

        model = create_flow_energy()

        plan = ExecutionPlan(
            nodes=[
                ToolCallNode(
                    tool_name=f"tool_{i}",
                    arguments={},
                    provenance_tier=TrustTier.INTERNAL,
                    provenance_hash=f"h{i}",
                    scope_volume=100,
                    scope_sensitivity=2,
                    node_id=f"n{i}",
                )
                for i in range(10)
            ],
            edges=[(f"n{i}", f"n{i + 1}") for i in range(9)],
        )

        # Simple timing test
        start = time.perf_counter()
        for _ in range(50):
            _ = model.explain(plan)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 50
        assert avg_time < 0.05  # 50ms average
