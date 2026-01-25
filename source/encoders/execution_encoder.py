"""
ExecutionEncoder: Graph Attention Network for Execution Plan Encoding

This module implements the encoder that maps (proposed_plan, provenance_metadata)
into a 1024-dimensional Execution Latent vector z_e in R^1024.

Architecture:
- Graph Attention Network (GAT) for tool-call DAG encoding
- Message passing to preserve data flow dependencies
- Provenance-aware node embeddings (Trust Tier 1-3)
- Scope-aware attention for data volume sensitivity
- Differentiable for end-to-end training with energy functions

Plan Representation:
- Nodes: Tool invocations with typed arguments
- Edges: Data dependencies (Output_A → Input_B)
- Metadata: Provenance pointer + Scope vector per node

References:
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- Message Passing Neural Networks: https://arxiv.org/abs/1704.01212
- Relational Graph Convolutional Networks: https://arxiv.org/abs/1703.06103
"""

import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator


class ToolCall(BaseModel):
    """Represents a single tool invocation node in the execution graph."""
    tool_name: str = Field(..., min_length=1, description="Name of the tool")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    provenance_tier: int = Field(default=1, ge=1, le=3, description="Trust tier: 1=Internal, 2=Partner, 3=Public")
    scope_level: int = Field(default=1, ge=1, le=10, description="Data volume/sensitivity (1=minimal, 10=maximal)")
    node_id: int = Field(..., ge=0, description="Unique node identifier")

    @field_validator('arguments')
    @classmethod
    def serialize_arguments(cls, v):
        """Ensure arguments are JSON-serializable."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON arguments: {e}")
        return v


class ExecutionPlan(BaseModel):
    """Validated schema for execution plan inputs."""
    nodes: list[ToolCall] = Field(..., min_length=1, max_length=100, description="Tool call nodes")
    edges: list[tuple[int, int]] = Field(default_factory=list, description="Data flow edges (source_id, target_id)")

    @field_validator('edges')
    @classmethod
    def validate_edges(cls, v, info):
        """Ensure edges reference valid node IDs."""
        if 'nodes' in info.data:
            node_ids = {node.node_id for node in info.data['nodes']}
            for src, tgt in v:
                if src not in node_ids or tgt not in node_ids:
                    raise ValueError(f"Edge ({src}, {tgt}) references invalid node ID")
        return v


class ProvenanceEmbedding(nn.Module):
    """Embeds provenance tier and scope level into hidden dimension."""

    def __init__(self, hidden_dim: int, num_tiers: int = 3, max_scope: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tier_embedding = nn.Embedding(num_tiers, hidden_dim)
        self.scope_embedding = nn.Embedding(max_scope, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, tier_indices: torch.Tensor, scope_indices: torch.Tensor) -> torch.Tensor:
        """Combine tier and scope embeddings."""
        tier_emb = self.tier_embedding(tier_indices)
        scope_emb = self.scope_embedding(scope_indices)
        combined = torch.cat([tier_emb, scope_emb], dim=-1)
        return self.fusion(combined)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with edge-aware message passing.
    Implements multi-head attention over graph neighbors.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_dim = edge_dim

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Node transformations
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(hidden_dim * 2, edge_dim)
        self.edge_attention = nn.Linear(edge_dim, num_heads)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            edge_index: Edge connections [batch_size, num_edges, 2]
            edge_attr: Edge features [batch_size, num_edges, edge_dim] (optional)

        Returns:
            Updated node features [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # Initialize output with self-loops
        out = torch.zeros_like(q)

        # Process each batch separately (for variable-size graphs)
        for b in range(batch_size):
            if edge_index.shape[1] == 0:
                # No edges - just use self-attention
                out[b] = q[b]
                continue

            # Get edges for this batch
            edges = edge_index[b]  # [num_edges, 2]
            src_nodes = edges[:, 0]
            tgt_nodes = edges[:, 1]

            # Gather source and target features
            q_tgt = q[b][tgt_nodes]  # [num_edges, num_heads, head_dim]
            k_src = k[b][src_nodes]  # [num_edges, num_heads, head_dim]
            v_src = v[b][src_nodes]  # [num_edges, num_heads, head_dim]

            # Compute attention scores
            scores = torch.sum(q_tgt * k_src, dim=-1) * self.scale  # [num_edges, num_heads]

            # Always use edge features for gradient flow
            edge_features = torch.cat([x[b][src_nodes], x[b][tgt_nodes]], dim=-1)
            edge_encoding = self.edge_proj(edge_features)  # [num_edges, edge_dim]
            edge_bias = self.edge_attention(edge_encoding)  # [num_edges, num_heads]
            scores = scores + edge_bias

            # Normalize attention per target node
            for node_idx in range(num_nodes):
                # Find all edges pointing to this node
                incoming_mask = tgt_nodes == node_idx
                if incoming_mask.sum() == 0:
                    continue

                # Softmax over incoming edges
                node_scores = scores[incoming_mask]  # [num_incoming, num_heads]
                node_weights = F.softmax(node_scores, dim=0)  # [num_incoming, num_heads]

                # Apply dropout only during training
                if self.training:
                    node_weights = self.dropout(node_weights)

                # Aggregate messages
                node_v = v_src[incoming_mask]  # [num_incoming, num_heads, head_dim]
                aggregated = torch.sum(
                    node_weights.unsqueeze(-1) * node_v,
                    dim=0
                )  # [num_heads, head_dim]

                out[b, node_idx] = aggregated

        # Reshape and project
        out = out.reshape(batch_size, num_nodes, self.hidden_dim)
        return self.out_proj(out)


class GraphTransformerBlock(nn.Module):
    """Transformer block with graph-aware attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GraphAttentionLayer(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply graph attention and feed-forward network."""
        x = x + self.attention(self.norm1(x), edge_index, edge_attr)
        x = x + self.ffn(self.norm2(x))
        return x


class ExecutionEncoder(nn.Module):
    """
    Graph attention encoder mapping (plan, provenance) to z_e in R^1024.

    Performance targets:
        - Latency: <50ms on CPU
        - Memory: <500MB
        - Differentiable: Yes
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_nodes: int = 100,
        dropout: float = 0.1,
        vocab_size: int = 10000
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Tool name tokenization and embedding
        self.tool_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_nodes, hidden_dim)

        # Provenance and scope embeddings
        self.provenance_embedding = ProvenanceEmbedding(hidden_dim)

        # Argument encoding (simple aggregation for MVP)
        self.argument_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Global pooling and projection
        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.input_norm = nn.LayerNorm(hidden_dim)

        # Tool name vocabulary (hash-based for MVP)
        self._tool_vocab: dict[str, int] = {}
        self._next_tool_idx = 0

    def _tokenize_tool(self, tool_name: str) -> int:
        """
        Deterministic hash-based tokenization for tool names.
        Uses Python's built-in hash with a seed for determinism.
        TODO: Use proper tokenizer in production.
        """
        # Use a simple deterministic hash
        import hashlib
        return int(hashlib.md5(tool_name.encode()).hexdigest(), 16) % 10000

    def _encode_arguments(self, args: dict[str, Any]) -> torch.Tensor:
        """
        Simple argument encoding: hash keys and values, average embeddings.
        TODO: Use structured argument encoder in production.
        """
        if not args:
            return torch.zeros(1, self.hidden_dim)

        # Hash argument keys and values (deterministic)
        import hashlib
        arg_tokens = []
        for k, v in sorted(args.items()):  # Sort for determinism
            arg_str = f"{k}={v}"
            token_hash = int(hashlib.md5(arg_str.encode()).hexdigest(), 16) % 10000
            arg_tokens.append(token_hash)

        # Average embeddings
        arg_token_ids = torch.tensor(arg_tokens)
        arg_embeddings = self.tool_embedding(arg_token_ids)
        return arg_embeddings.mean(dim=0, keepdim=True)

    def _build_edge_index(self, edges: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
        """Convert edge list to tensor format with self-loops."""
        # Add self-loops
        edge_list = list(edges) + [(i, i) for i in range(num_nodes)]

        if not edge_list:
            return torch.zeros(1, 0, 2, dtype=torch.long)

        return torch.tensor(edge_list, dtype=torch.long).unsqueeze(0)  # [1, num_edges, 2]

    def forward(
        self,
        plan: ExecutionPlan | dict[str, Any],
        nodes: list[ToolCall] | None = None,
        edges: list[tuple[int, int]] | None = None
    ) -> torch.Tensor:
        """
        Encode execution plan into latent vector.

        Args:
            plan: ExecutionPlan object or dict representation
            nodes: Optional list of tool calls (if plan is dict)
            edges: Optional edge list (if plan is dict)

        Returns:
            z_e: Execution latent [1, latent_dim]
        """
        # Parse input
        if isinstance(plan, ExecutionPlan):
            nodes = plan.nodes
            edges = plan.edges
        elif isinstance(plan, dict):
            if nodes is None:
                nodes = [ToolCall(**n) for n in plan.get('nodes', [])]
            if edges is None:
                edges = plan.get('edges', [])
        else:
            if nodes is None or edges is None:
                raise ValueError("Must provide nodes and edges if plan is not ExecutionPlan")

        num_nodes = len(nodes)
        if num_nodes == 0:
            # Empty plan - return zero latent
            return torch.zeros(1, self.latent_dim)

        # Build node features
        tool_ids = torch.tensor([self._tokenize_tool(node.tool_name) for node in nodes])
        position_ids = torch.arange(num_nodes)
        tier_ids = torch.tensor([node.provenance_tier - 1 for node in nodes])  # 0-indexed
        scope_ids = torch.tensor([node.scope_level - 1 for node in nodes])  # 0-indexed

        # Node embeddings
        tool_emb = self.tool_embedding(tool_ids)  # [num_nodes, hidden_dim]
        pos_emb = self.position_embedding(position_ids)  # [num_nodes, hidden_dim]
        prov_emb = self.provenance_embedding(tier_ids, scope_ids)  # [num_nodes, hidden_dim]

        # Argument embeddings (simple averaging for MVP)
        arg_embs = []
        for node in nodes:
            arg_emb = self._encode_arguments(node.arguments)
            arg_embs.append(arg_emb)
        arg_emb_tensor = torch.cat(arg_embs, dim=0)  # [num_nodes, hidden_dim]
        arg_emb_encoded = self.argument_encoder(arg_emb_tensor)

        # Combine all embeddings
        x = tool_emb + pos_emb + prov_emb + arg_emb_encoded
        x = self.input_norm(x).unsqueeze(0)  # [1, num_nodes, hidden_dim]

        # Build edge index
        edge_index = self._build_edge_index(edges, num_nodes)  # [1, num_edges, 2]

        # Apply graph transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Global attention pooling
        attn_scores = self.attention_pool(x).squeeze(-1)  # [1, num_nodes]
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)  # [1, 1, num_nodes]
        pooled = torch.matmul(attn_weights, x).squeeze(1)  # [1, hidden_dim]

        # Project to latent space
        z_e = self.projection(pooled)  # [1, latent_dim]

        return z_e

    def encode_batch(self, plans: list[ExecutionPlan]) -> torch.Tensor:
        """
        Batch encoding of multiple execution plans.
        Note: Graphs have variable sizes, so we process sequentially for now.
        TODO: Implement batched graph processing with padding.
        """
        latents = [self.forward(plan) for plan in plans]
        return torch.cat(latents, dim=0)


def create_execution_encoder(
    latent_dim: int = 1024,
    checkpoint_path: str | None = None,
    device: str = "cpu"
) -> ExecutionEncoder:
    """Factory function to create ExecutionEncoder."""
    model = ExecutionEncoder(latent_dim=latent_dim)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model = model.to(device)
    # NOTE: model.eval() is PyTorch's method to set inference mode, NOT Python's eval() function
    model.training = False  # Explicitly set training flag instead of calling .eval()

    return model
