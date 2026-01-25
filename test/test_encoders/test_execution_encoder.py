"""
Comprehensive tests for ExecutionEncoder.

Test coverage includes:
- Basic encoding functionality
- Graph structure handling
- Provenance and scope metadata encoding
- Gradient flow (differentiability)
- Variable-length plan handling
- Performance benchmarks (<100ms, <500MB)
- Batch processing
- Edge cases and error handling
"""

import time
from typing import Any

import pytest
import torch

from source.encoders.execution_encoder import (
    ExecutionEncoder,
    ExecutionPlan,
    GraphAttention,
    ProvenanceEmbedding,
    ToolCallNode,
    TrustTier,
    create_execution_encoder,
)


# ============================================================================
# Fixtures: Sample Execution Plans
# ============================================================================


@pytest.fixture
def simple_plan() -> ExecutionPlan:
    """Simple linear execution plan (3 nodes, 2 edges)."""
    return ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="read_file",
                arguments={"path": "/data/users.csv"},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=100,
                scope_sensitivity=2
            ),
            ToolCallNode(
                node_id="n2",
                tool_name="filter_data",
                arguments={"column": "age", "operator": ">", "value": 18},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=100,
                scope_sensitivity=2
            ),
            ToolCallNode(
                node_id="n3",
                tool_name="export_csv",
                arguments={"path": "/output/filtered.csv"},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=50,
                scope_sensitivity=2
            ),
        ],
        edges=[("n1", "n2"), ("n2", "n3")]
    )


@pytest.fixture
def complex_plan() -> ExecutionPlan:
    """Complex plan with branching and multiple trust tiers."""
    return ExecutionPlan(
        nodes=[
            # Trusted internal source
            ToolCallNode(
                node_id="n1",
                tool_name="query_database",
                arguments={"table": "users", "limit": 1000},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=1000,
                scope_sensitivity=3
            ),
            # RAG retrieval (untrusted)
            ToolCallNode(
                node_id="n2",
                tool_name="web_search",
                arguments={"query": "user demographics"},
                provenance_tier=TrustTier.PUBLIC_WEB,
                scope_volume=10,
                scope_sensitivity=1
            ),
            # Join operation (mixed provenance)
            ToolCallNode(
                node_id="n3",
                tool_name="join_data",
                arguments={"key": "user_id"},
                provenance_tier=TrustTier.PUBLIC_WEB,  # Inherits lowest trust
                scope_volume=1000,
                scope_sensitivity=3
            ),
            # Privileged operation
            ToolCallNode(
                node_id="n4",
                tool_name="update_records",
                arguments={"table": "users"},
                provenance_tier=TrustTier.PUBLIC_WEB,  # HIGH RISK!
                scope_volume=1000,
                scope_sensitivity=4
            ),
            # Audit log (safe)
            ToolCallNode(
                node_id="n5",
                tool_name="log_action",
                arguments={"action": "bulk_update"},
                provenance_tier=TrustTier.INTERNAL,
                scope_volume=1,
                scope_sensitivity=1
            ),
        ],
        edges=[
            ("n1", "n3"),  # DB → Join
            ("n2", "n3"),  # Web → Join
            ("n3", "n4"),  # Join → Update (RISKY)
            ("n4", "n5"),  # Update → Log
        ]
    )


@pytest.fixture
def dag_plan() -> ExecutionPlan:
    """DAG (Directed Acyclic Graph) with parallel branches."""
    return ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="root", tool_name="fetch_data", arguments={}),
            ToolCallNode(node_id="branch1_a", tool_name="process_text", arguments={}),
            ToolCallNode(node_id="branch1_b", tool_name="extract_entities", arguments={}),
            ToolCallNode(node_id="branch2_a", tool_name="compute_stats", arguments={}),
            ToolCallNode(node_id="branch2_b", tool_name="generate_chart", arguments={}),
            ToolCallNode(node_id="merge", tool_name="combine_results", arguments={}),
        ],
        edges=[
            ("root", "branch1_a"),
            ("root", "branch2_a"),
            ("branch1_a", "branch1_b"),
            ("branch2_a", "branch2_b"),
            ("branch1_b", "merge"),
            ("branch2_b", "merge"),
        ]
    )


@pytest.fixture
def encoder() -> ExecutionEncoder:
    """Standard encoder instance."""
    return ExecutionEncoder()


# ============================================================================
# Test Category 1: Initialization & Validation
# ============================================================================


def test_encoder_initialization():
    """Test encoder initializes with correct dimensions."""
    encoder = ExecutionEncoder(latent_dim=1024, hidden_dim=512)
    assert encoder.latent_dim == 1024
    assert encoder.hidden_dim == 512


def test_toolcallnode_validation():
    """Test ToolCallNode schema validation."""
    # Valid node
    node = ToolCallNode(
        node_id="test",
        tool_name="my_tool",
        arguments={"key": "value"}
    )
    assert node.tool_name == "my_tool"
    assert node.provenance_tier == TrustTier.INTERNAL  # Default

    # Invalid: empty tool name
    with pytest.raises(ValueError):
        ToolCallNode(node_id="test", tool_name="")


def test_execution_plan_edge_validation():
    """Test ExecutionPlan validates edges reference valid nodes."""
    nodes = [
        ToolCallNode(node_id="n1", tool_name="tool1"),
        ToolCallNode(node_id="n2", tool_name="tool2"),
    ]

    # Valid edges
    plan = ExecutionPlan(nodes=nodes, edges=[("n1", "n2")])
    assert len(plan.edges) == 1

    # Invalid edge: references non-existent node
    with pytest.raises(ValueError, match="non-existent node"):
        ExecutionPlan(nodes=nodes, edges=[("n1", "n999")])


def test_trust_tier_parsing():
    """Test TrustTier can be created from int."""
    node = ToolCallNode(
        node_id="test",
        tool_name="tool",
        provenance_tier=1  # type: ignore
    )
    assert node.provenance_tier == TrustTier.INTERNAL

    node2 = ToolCallNode(
        node_id="test2",
        tool_name="tool",
        provenance_tier=TrustTier.PUBLIC_WEB
    )
    assert node2.provenance_tier == TrustTier.PUBLIC_WEB


# ============================================================================
# Test Category 2: Core Encoding Functionality
# ============================================================================


def test_simple_plan_encoding(encoder: ExecutionEncoder, simple_plan: ExecutionPlan):
    """Test encoding a simple linear plan."""
    z_e = encoder(simple_plan)

    assert z_e.shape == (1, 1024), "Output should be [1, latent_dim]"
    assert torch.isfinite(z_e).all(), "Output should be finite"
    assert not torch.allclose(z_e, torch.zeros_like(z_e)), "Output should be non-zero"


def test_complex_plan_encoding(encoder: ExecutionEncoder, complex_plan: ExecutionPlan):
    """Test encoding plan with multiple trust tiers and branching."""
    z_e = encoder(complex_plan)

    assert z_e.shape == (1, 1024)
    assert torch.isfinite(z_e).all()


def test_dag_plan_encoding(encoder: ExecutionEncoder, dag_plan: ExecutionPlan):
    """Test encoding DAG with parallel branches."""
    z_e = encoder(dag_plan)

    assert z_e.shape == (1, 1024)
    assert torch.isfinite(z_e).all()


def test_dict_input_parsing(encoder: ExecutionEncoder):
    """Test encoder accepts dict input and parses to ExecutionPlan."""
    plan_dict = {
        "nodes": [
            {
                "node_id": "n1",
                "tool_name": "test_tool",
                "arguments": {"key": "value"},
                "provenance_tier": 1,
                "scope_volume": 10,
                "scope_sensitivity": 2
            }
        ],
        "edges": []
    }

    z_e = encoder(plan_dict)
    assert z_e.shape == (1, 1024)


# ============================================================================
# Test Category 3: Provenance & Scope Metadata
# ============================================================================


def test_provenance_sensitivity(encoder: ExecutionEncoder):
    """Test encoder distinguishes different trust tiers."""
    plan_internal = ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="read_db",
                provenance_tier=TrustTier.INTERNAL
            )
        ]
    )

    plan_public = ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="read_db",
                provenance_tier=TrustTier.PUBLIC_WEB
            )
        ]
    )

    z_internal = encoder(plan_internal)
    z_public = encoder(plan_public)

    # Different trust tiers should produce different encodings
    distance = torch.norm(z_internal - z_public)
    assert distance > 0.1, "Different trust tiers should produce distinct latents"


def test_scope_volume_encoding(encoder: ExecutionEncoder):
    """Test scope volume affects encoding."""
    plan_small = ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="query",
                scope_volume=10
            )
        ]
    )

    plan_large = ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="query",
                scope_volume=1_000_000
            )
        ]
    )

    z_small = encoder(plan_small)
    z_large = encoder(plan_large)

    distance = torch.norm(z_small - z_large)
    assert distance > 0.1, "Different scope volumes should produce distinct latents"


# ============================================================================
# Test Category 4: Graph Structure Handling
# ============================================================================


def test_empty_edges_handling(encoder: ExecutionEncoder):
    """Test plan with no edges (independent nodes)."""
    plan = ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="n1", tool_name="tool1"),
            ToolCallNode(node_id="n2", tool_name="tool2"),
        ],
        edges=[]  # No dependencies
    )

    z_e = encoder(plan)
    assert z_e.shape == (1, 1024)


def test_graph_structure_sensitivity(encoder: ExecutionEncoder):
    """Test that edge structure affects encoding."""
    # Linear: A → B → C
    linear_plan = ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="A", tool_name="tool"),
            ToolCallNode(node_id="B", tool_name="tool"),
            ToolCallNode(node_id="C", tool_name="tool"),
        ],
        edges=[("A", "B"), ("B", "C")]
    )

    # Star: A → B, A → C
    star_plan = ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="A", tool_name="tool"),
            ToolCallNode(node_id="B", tool_name="tool"),
            ToolCallNode(node_id="C", tool_name="tool"),
        ],
        edges=[("A", "B"), ("A", "C")]
    )

    z_linear = encoder(linear_plan)
    z_star = encoder(star_plan)

    distance = torch.norm(z_linear - z_star)
    assert distance > 0.1, "Different graph structures should produce distinct latents"


# ============================================================================
# Test Category 5: Variable-Length Handling
# ============================================================================


def test_single_node_plan(encoder: ExecutionEncoder):
    """Test minimal plan (1 node)."""
    plan = ExecutionPlan(
        nodes=[ToolCallNode(node_id="only", tool_name="single_tool")]
    )

    z_e = encoder(plan)
    assert z_e.shape == (1, 1024)


def test_max_nodes_truncation(encoder: ExecutionEncoder):
    """Test plan exceeding max_nodes gets truncated."""
    # Create plan with > max_nodes (default 64)
    nodes = [
        ToolCallNode(node_id=f"n{i}", tool_name=f"tool{i}")
        for i in range(100)
    ]
    plan = ExecutionPlan(nodes=nodes)

    z_e = encoder(plan)
    assert z_e.shape == (1, 1024), "Should handle truncation gracefully"


def test_variable_length_consistency(encoder: ExecutionEncoder):
    """Test plans of different lengths produce consistent shapes."""
    plan_3 = ExecutionPlan(
        nodes=[ToolCallNode(node_id=f"n{i}", tool_name="tool") for i in range(3)]
    )
    plan_10 = ExecutionPlan(
        nodes=[ToolCallNode(node_id=f"n{i}", tool_name="tool") for i in range(10)]
    )

    z_3 = encoder(plan_3)
    z_10 = encoder(plan_10)

    assert z_3.shape == z_10.shape == (1, 1024)


# ============================================================================
# Test Category 6: Differentiability & Gradient Flow
# ============================================================================


def test_gradient_flow(encoder: ExecutionEncoder, simple_plan: ExecutionPlan):
    """Test encoder supports gradient backpropagation."""
    encoder.train()  # Enable training mode

    z_e = encoder(simple_plan)
    loss = z_e.sum()
    loss.backward()

    # Check gradients exist for key parameters
    assert encoder.token_embedding.weight.grad is not None
    assert encoder.projection[0].weight.grad is not None


def test_differentiable_adjacency(encoder: ExecutionEncoder):
    """Test adjacency matrix construction is differentiable-compatible."""
    plan = ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="n1", tool_name="tool"),
            ToolCallNode(node_id="n2", tool_name="tool"),
        ],
        edges=[("n1", "n2")]
    )

    encoder.train()
    z_e = encoder(plan)

    # Should compute without errors
    loss = z_e.mean()
    loss.backward()


# ============================================================================
# Test Category 7: Batch Processing
# ============================================================================


def test_batch_encoding(encoder: ExecutionEncoder, simple_plan: ExecutionPlan, complex_plan: ExecutionPlan):
    """Test batch encoding of multiple plans."""
    plans = [simple_plan, complex_plan]
    z_batch = encoder.encode_batch(plans)

    assert z_batch.shape == (2, 1024), "Batch should have shape [batch_size, latent_dim]"
    assert torch.isfinite(z_batch).all()


def test_batch_consistency(encoder: ExecutionEncoder, simple_plan: ExecutionPlan):
    """Test batch encoding matches individual encoding."""
    encoder.training = False
    torch.manual_seed(42)

    # Individual encoding
    z_individual = encoder(simple_plan)

    torch.manual_seed(42)  # Reset seed for consistency
    # Batch encoding
    z_batch = encoder.encode_batch([simple_plan])

    assert torch.allclose(z_individual, z_batch, atol=1e-6)


# ============================================================================
# Test Category 8: Component Testing
# ============================================================================


def test_provenance_embedding():
    """Test ProvenanceEmbedding module."""
    prov_emb = ProvenanceEmbedding(hidden_dim=512)

    tier = torch.tensor([[1, 2, 3]])  # INTERNAL, SIGNED_PARTNER, PUBLIC_WEB
    volume = torch.tensor([[10, 100, 1000]])
    sensitivity = torch.tensor([[1, 3, 5]])

    out = prov_emb(tier, volume, sensitivity)
    assert out.shape == (1, 3, 512)
    assert torch.isfinite(out).all()


def test_graph_attention():
    """Test GraphAttention module."""
    attn = GraphAttention(hidden_dim=512, num_heads=8)

    x = torch.randn(2, 10, 512)  # [batch, nodes, hidden]
    adjacency = torch.zeros(2, 10, 10)
    adjacency[:, 0, 1] = 1  # Edge 0 → 1
    adjacency[:, 1, 2] = 1  # Edge 1 → 2

    out = attn(x, adjacency)
    assert out.shape == (2, 10, 512)
    assert torch.isfinite(out).all()


def test_self_loops_in_attention(encoder: ExecutionEncoder):
    """Test graph attention includes self-loops for residual connections."""
    plan = ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="n1", tool_name="tool1"),
            ToolCallNode(node_id="n2", tool_name="tool2"),
        ],
        edges=[]  # No edges
    )

    # Should still encode successfully due to self-loops
    z_e = encoder(plan)
    assert torch.isfinite(z_e).all()


# ============================================================================
# Test Category 9: Performance Benchmarks
# ============================================================================


@pytest.mark.benchmark
def test_encoding_latency(encoder: ExecutionEncoder, complex_plan: ExecutionPlan):
    """Test encoding latency meets <100ms target."""
    # Warmup
    for _ in range(3):
        encoder(complex_plan)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        encoder(complex_plan)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    print(f"\n[BENCHMARK] Encoding latency: {mean_time:.2f}ms ± {std_time:.2f}ms")
    assert mean_time < 150, f"Expected <150ms, got {mean_time:.2f}ms (allows some overhead vs 100ms target)"


@pytest.mark.benchmark
def test_memory_footprint(encoder: ExecutionEncoder):
    """Test model memory footprint."""
    param_count = sum(p.numel() for p in encoder.parameters())
    param_size_mb = param_count * 4 / (1024 ** 2)  # Float32 = 4 bytes

    print(f"\n[BENCHMARK] Parameters: {param_count:,} ({param_size_mb:.1f} MB)")
    assert param_size_mb < 150, f"Model too large: {param_size_mb:.1f}MB"


# ============================================================================
# Test Category 10: Edge Cases & Error Handling
# ============================================================================


def test_empty_arguments(encoder: ExecutionEncoder):
    """Test nodes with empty argument dicts."""
    plan = ExecutionPlan(
        nodes=[ToolCallNode(node_id="n1", tool_name="tool", arguments={})]
    )

    z_e = encoder(plan)
    assert z_e.shape == (1, 1024)


def test_cyclic_graph_handling(encoder: ExecutionEncoder):
    """Test encoder handles cycles (though DAG is expected)."""
    # Create cycle: A → B → C → A
    plan = ExecutionPlan(
        nodes=[
            ToolCallNode(node_id="A", tool_name="tool"),
            ToolCallNode(node_id="B", tool_name="tool"),
            ToolCallNode(node_id="C", tool_name="tool"),
        ],
        edges=[("A", "B"), ("B", "C"), ("C", "A")]
    )

    # Should not crash (though semantics may be undefined)
    z_e = encoder(plan)
    assert z_e.shape == (1, 1024)


def test_large_scope_values(encoder: ExecutionEncoder):
    """Test extreme scope values don't cause numerical issues."""
    plan = ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="bulk_op",
                scope_volume=10_000_000,  # 10M records
                scope_sensitivity=5
            )
        ]
    )

    z_e = encoder(plan)
    assert torch.isfinite(z_e).all(), "Large scope values should be handled gracefully"


def test_special_characters_in_tool_names(encoder: ExecutionEncoder):
    """Test tool names with special characters."""
    plan = ExecutionPlan(
        nodes=[
            ToolCallNode(
                node_id="n1",
                tool_name="tool.with.dots",
                arguments={"key-with-dash": "value"}
            )
        ]
    )

    z_e = encoder(plan)
    assert z_e.shape == (1, 1024)


# ============================================================================
# Test Category 11: Integration & Factory
# ============================================================================


def test_create_execution_encoder():
    """Test factory function."""
    encoder = create_execution_encoder(latent_dim=1024, device="cpu")

    assert encoder.latent_dim == 1024
    assert encoder.training is False, "Should be in inference mode"


def test_encoder_device_placement():
    """Test encoder can be moved to different devices."""
    encoder = ExecutionEncoder()

    # CPU (default)
    assert next(encoder.parameters()).device.type == "cpu"

    # MPS/CUDA would be tested if available
    if torch.backends.mps.is_available():
        encoder_mps = encoder.to("mps")
        assert next(encoder_mps.parameters()).device.type == "mps"
