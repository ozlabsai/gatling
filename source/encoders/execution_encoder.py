"""
ExecutionEncoder: Graph-Based Transformer for Execution Plan Encoding

This module implements the transformer-based encoder that maps
(plan_graph, provenance_metadata) → z_e ∈ R^1024.

The ExecutionEncoder is the complementary half of the JEPA dual-encoder architecture,
encoding proposed execution plans into the same latent space as governance policies
to enable energy-based security validation.

Architecture:
- Graph Neural Network for tool-call dependency encoding
- Provenance-aware attention mechanism
- Scope metadata integration
- Differentiable for end-to-end training with energy functions

References:
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- Relational Graph Convolutional Networks: https://arxiv.org/abs/1703.06103
"""

from enum import IntEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator


class TrustTier(IntEnum):
    """Trust levels for data provenance (as per Dawn Song's workstream)."""

    INTERNAL = 1  # System instructions, internal databases
    SIGNED_PARTNER = 2  # Verified external sources
    PUBLIC_WEB = 3  # Untrusted retrieval (RAG, web scraping)


class ToolCallNode(BaseModel):
    """A single tool invocation in the execution plan graph."""

    tool_name: str = Field(..., min_length=1, description="Name of the tool being invoked")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    # Provenance metadata
    provenance_tier: TrustTier = Field(
        default=TrustTier.INTERNAL, description="Trust tier of instruction source"
    )
    provenance_hash: str | None = Field(default=None, description="Cryptographic hash of source")

    # Scope metadata
    scope_volume: int = Field(default=1, ge=1, description="Data volume (rows, records, files)")
    scope_sensitivity: int = Field(
        default=1, ge=1, le=5, description="Sensitivity level (1=public, 5=critical)"
    )

    # Graph metadata
    node_id: str = Field(..., description="Unique node identifier")

    @field_validator("provenance_tier", mode="before")
    @classmethod
    def parse_trust_tier(cls, v):
        """Parse trust tier from int or TrustTier."""
        if isinstance(v, int):
            return TrustTier(v)
        return v


class ExecutionPlan(BaseModel):
    """Complete execution plan represented as a typed tool-call graph."""

    nodes: list[ToolCallNode] = Field(..., min_length=1, description="Tool invocation nodes")
    edges: list[tuple[str, str]] = Field(
        default_factory=list, description="Data flow edges (src_id, dst_id)"
    )

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, v, info):
        """Ensure edge endpoints reference valid nodes."""
        if "nodes" not in info.data:
            return v

        node_ids = {node.node_id for node in info.data["nodes"]}
        for src, dst in v:
            if src not in node_ids or dst not in node_ids:
                raise ValueError(f"Edge ({src}, {dst}) references non-existent node")
        return v


class ProvenanceEmbedding(nn.Module):
    """Embeds provenance metadata (trust tier + cryptographic hash)."""

    def __init__(self, hidden_dim: int, num_tiers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tier_embedding = nn.Embedding(num_tiers + 1, hidden_dim)  # +1 for padding
        self.scope_projection = nn.Linear(2, hidden_dim)  # volume + sensitivity
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        tier_indices: torch.Tensor,
        scope_volume: torch.Tensor,
        scope_sensitivity: torch.Tensor,
    ) -> torch.Tensor:
        """Combine provenance tier and scope metadata."""
        tier_emb = self.tier_embedding(tier_indices)

        # Log-scale volume to handle wide range (1 to 1M+)
        log_volume = torch.log1p(scope_volume.float()).unsqueeze(-1)
        sensitivity = scope_sensitivity.float().unsqueeze(-1)
        scope_features = torch.cat([log_volume, sensitivity], dim=-1)
        scope_emb = self.scope_projection(scope_features)

        combined = torch.cat([tier_emb, scope_emb], dim=-1)
        return self.fusion(combined)


class GraphAttention(nn.Module):
    """
    Graph Attention layer for encoding tool-call dependencies.
    Implements message passing with edge-aware attention.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Apply graph attention.

        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            adjacency: Adjacency matrix [batch_size, num_nodes, num_nodes]
                      1 = edge exists, 0 = no edge
        """
        batch_size, num_nodes, _ = x.shape

        q = (
            self.q_proj(x)
            .view(batch_size, num_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, num_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, num_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Mask attention to respect graph structure
        # Also add self-loops (diagonal) for residual connections
        mask = adjacency.unsqueeze(1)  # [batch, 1, nodes, nodes]
        eye = torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        mask = torch.maximum(mask, eye)  # Add self-loops

        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)

        return self.out_proj(out)


class GraphTransformerBlock(nn.Module):
    """Transformer block with graph-aware attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GraphAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply graph transformer block."""
        x = x + self.attention(self.norm1(x), adjacency)
        x = x + self.ffn(self.norm2(x))
        return x


class ExecutionEncoder(nn.Module):
    """
    Graph-based transformer encoder mapping execution plans to z_e ∈ R^1024.

    Encodes:
    - Tool invocation sequences
    - Data flow dependencies (graph edges)
    - Provenance metadata (trust tiers)
    - Scope metadata (volume + sensitivity)

    Performance targets:
        - Latency: <100ms on CPU (pairs with GovernanceEncoder's 98ms)
        - Memory: <500MB
        - Differentiable: Yes
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_nodes: int = 64,
        dropout: float = 0.1,
        vocab_size: int = 10000,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Token embeddings for tool names and arguments
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_nodes, hidden_dim)

        # Provenance and scope embeddings
        self.provenance_embedding = ProvenanceEmbedding(hidden_dim)

        # Graph transformer layers
        self.layers = nn.ModuleList(
            [GraphTransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Pooling and projection
        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.input_norm = nn.LayerNorm(hidden_dim)

    def _tokenize(self, text: str) -> int:
        """
        Hash-based tokenization (v0.1.0).

        Future: Replace with BPE tokenizer (v0.2.0) to reduce collisions.
        """
        return hash(text) % 10000

    def _create_adjacency_matrix(
        self, num_nodes: int, edges: list[tuple[int, int]], device: torch.device
    ) -> torch.Tensor:
        """Build adjacency matrix from edge list."""
        adjacency = torch.zeros(num_nodes, num_nodes, device=device)
        for src, dst in edges:
            if src < num_nodes and dst < num_nodes:
                adjacency[src, dst] = 1
        return adjacency

    def forward(self, plan: ExecutionPlan | dict[str, Any]) -> torch.Tensor:
        """
        Encode execution plan into latent vector.

        Args:
            plan: ExecutionPlan or dict conforming to ExecutionPlan schema

        Returns:
            z_e: Latent vector [1, latent_dim]
        """
        # Validate and parse input
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan(**plan)

        nodes = plan.nodes
        edges = plan.edges

        # Build node ID mapping
        node_id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        edge_indices = [(node_id_to_idx[src], node_id_to_idx[dst]) for src, dst in edges]

        # Pad or truncate to max_nodes
        num_nodes = min(len(nodes), self.max_nodes)
        nodes = nodes[:num_nodes]

        # Tokenize tool names and arguments
        tool_tokens = []
        for node in nodes:
            # Combine tool name + serialized args for richer representation
            arg_str = ",".join(f"{k}={v}" for k, v in sorted(node.arguments.items()))
            combined = f"{node.tool_name}({arg_str})"
            tool_tokens.append(self._tokenize(combined))

        # Pad tokens
        if len(tool_tokens) < self.max_nodes:
            tool_tokens.extend([0] * (self.max_nodes - len(tool_tokens)))

        # Convert to tensors
        token_ids = torch.tensor(tool_tokens[: self.max_nodes]).unsqueeze(0)
        position_ids = torch.arange(self.max_nodes).unsqueeze(0)

        # Provenance and scope metadata
        tier_indices = torch.tensor(
            [node.provenance_tier for node in nodes] + [0] * (self.max_nodes - num_nodes)
        ).unsqueeze(0)
        scope_volume = torch.tensor(
            [node.scope_volume for node in nodes] + [1] * (self.max_nodes - num_nodes)
        ).unsqueeze(0)
        scope_sensitivity = torch.tensor(
            [node.scope_sensitivity for node in nodes] + [1] * (self.max_nodes - num_nodes)
        ).unsqueeze(0)

        # Build adjacency matrix
        adjacency = self._create_adjacency_matrix(
            self.max_nodes, edge_indices, token_ids.device
        ).unsqueeze(0)

        # Embed tokens
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(position_ids)
        prov_emb = self.provenance_embedding(tier_indices, scope_volume, scope_sensitivity)

        # Combine embeddings
        x = token_emb + pos_emb + prov_emb
        x = self.input_norm(x)

        # Apply graph transformer layers
        for layer in self.layers:
            x = layer(x, adjacency)

        # Attention pooling over nodes
        attn_scores = self.attention_pool(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)
        pooled = torch.matmul(attn_weights, x).squeeze(1)

        # Project to latent space
        z_e = self.projection(pooled)

        return z_e

    def encode_batch(self, plans: list[ExecutionPlan]) -> torch.Tensor:
        """
        Batch encoding of multiple execution plans.

        Args:
            plans: List of ExecutionPlan objects

        Returns:
            z_e: Latent vectors [batch_size, latent_dim]
        """
        latents = [self.forward(plan) for plan in plans]
        return torch.cat(latents, dim=0)


def create_execution_encoder(
    latent_dim: int = 1024, checkpoint_path: str | None = None, device: str = "cpu"
) -> ExecutionEncoder:
    """
    Factory function to create ExecutionEncoder.

    Args:
        latent_dim: Dimension of output latent vector (must match GovernanceEncoder)
        checkpoint_path: Optional path to pretrained weights
        device: Device to load model on

    Returns:
        Initialized ExecutionEncoder in inference mode
    """
    model = ExecutionEncoder(latent_dim=latent_dim)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.training = False  # Set to inference mode

    return model
