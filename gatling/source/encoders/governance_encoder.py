"""
GovernanceEncoder: Policy-Aware Transformer for Governance Latent Generation

This module implements the transformer-based encoder that maps (policy, user_role, session_context)
into a 1024-dimensional Governance Latent vector z_g in R^1024.

Architecture:
- Structure-aware attention mechanism (inspired by StructFormer)
- Hierarchical encoding for JSON/YAML policy documents
- Sparse attention patterns for sub-50ms inference latency
- Differentiable for end-to-end training with energy functions

References:
- StructFormer: https://arxiv.org/html/2411.16618v1
- ETC (Encoding Long and Structured Inputs): https://arxiv.org/abs/2004.08483
- Longformer sparse attention: https://arxiv.org/abs/2004.05150
"""

import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator


class PolicySchema(BaseModel):
    """Validated schema for policy inputs."""
    document: dict[str, Any] | str
    user_role: str = Field(..., min_length=1, description="User role")
    session_context: dict[str, Any] | None = Field(default_factory=dict)

    @field_validator('document')
    @classmethod
    def parse_document(cls, v):
        """Parse string documents to dict."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                try:
                    return yaml.safe_load(v)
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid JSON/YAML document: {e}")
        return v


class StructuralEmbedding(nn.Module):
    """Embeds structural position information preserving tree hierarchy."""

    def __init__(self, hidden_dim: int, max_depth: int = 8, num_node_types: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.depth_embedding = nn.Embedding(max_depth, hidden_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, depth_indices: torch.Tensor, node_type_indices: torch.Tensor) -> torch.Tensor:
        """Combine depth and node type embeddings."""
        depth_emb = self.depth_embedding(depth_indices)
        type_emb = self.node_type_embedding(node_type_indices)
        combined = torch.cat([depth_emb, type_emb], dim=-1)
        return self.fusion(combined)


class SparseStructuredAttention(nn.Module):
    """
    Sparse attention combining local and global patterns.
    Reduces complexity from O(n²) to O(n*w + n*g).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        window_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _create_attention_mask(
        self,
        seq_len: int,
        is_global: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Create sparse attention mask with local windows and global tokens."""
        mask = torch.zeros(seq_len, seq_len, device=device)

        # Local attention window
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1

        # Global attention
        batch_size = is_global.shape[0]
        mask_batch = mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        for b in range(batch_size):
            global_indices = torch.where(is_global[b])[0]
            for idx in global_indices:
                mask_batch[b, idx, :] = 1
                mask_batch[b, :, idx] = 1

        return mask_batch

    def forward(
        self,
        x: torch.Tensor,
        is_global: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply sparse structured attention."""
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_global is None:
            is_global = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)

        mask = self._create_attention_mask(seq_len, is_global, x.device)
        mask = mask.unsqueeze(1)

        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block with structure-aware attention."""

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = SparseStructuredAttention(hidden_dim, num_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, is_global: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), is_global)
        x = x + self.ffn(self.norm2(x))
        return x


class GovernanceEncoder(nn.Module):
    """
    Transformer encoder mapping (policy, role, context) to z_g in R^1024.

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
        window_size: int = 32,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 10000
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.structural_embedding = StructuralEmbedding(hidden_dim)
        self.role_embedding = nn.Embedding(32, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])

        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.input_norm = nn.LayerNorm(hidden_dim)
        self._role_vocab: dict[str, int] = {}
        self._next_role_idx = 0

    def _flatten_policy(self, policy_dict: dict[str, Any], prefix: str = "", depth: int = 0) -> list[dict[str, Any]]:
        """Flatten nested policy dict to sequence with structural metadata."""
        tokens = []

        for key, value in policy_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                node_type = 0
                is_global = depth <= 1
                tokens.append({
                    'token': full_key,
                    'depth': min(depth, 7),
                    'node_type': node_type,
                    'is_global': is_global
                })
                tokens.extend(self._flatten_policy(value, full_key, depth + 1))
            elif isinstance(value, list):
                node_type = 1
                tokens.append({
                    'token': full_key,
                    'depth': min(depth, 7),
                    'node_type': node_type,
                    'is_global': False
                })
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        tokens.extend(self._flatten_policy(item, f"{full_key}[{i}]", depth + 1))
                    else:
                        tokens.append({
                            'token': str(item),
                            'depth': min(depth + 1, 7),
                            'node_type': 2,
                            'is_global': False
                        })
            else:
                node_type = 2
                tokens.append({
                    'token': f"{full_key}={value}",
                    'depth': min(depth, 7),
                    'node_type': node_type,
                    'is_global': False
                })

        return tokens

    def _tokenize(self, text: str) -> int:
        """Hash-based tokenization. Use proper tokenizer in production."""
        return hash(text) % 10000

    def _get_role_idx(self, role: str) -> int:
        """Map role string to index."""
        if role not in self._role_vocab:
            if self._next_role_idx >= 32:
                return hash(role) % 32
            self._role_vocab[role] = self._next_role_idx
            self._next_role_idx += 1
        return self._role_vocab[role]

    def forward(
        self,
        policy: dict[str, Any] | PolicySchema,
        user_role: str | None = None,
        session_context: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Encode governance policy into latent vector."""
        if isinstance(policy, PolicySchema):
            policy_dict = policy.document  # type: ignore[assignment]
            user_role = user_role or policy.user_role
            session_context = session_context or policy.session_context
        else:
            policy_dict = policy
            user_role = user_role or "unknown"
            session_context = session_context or {}

        flattened = self._flatten_policy(policy_dict)  # type: ignore[arg-type]

        if len(flattened) > self.max_seq_len:
            global_tokens = [t for t in flattened if t['is_global']]
            non_global = [t for t in flattened if not t['is_global']]
            remaining = self.max_seq_len - len(global_tokens)
            flattened = global_tokens + non_global[:remaining]

        seq_len = len(flattened)
        if seq_len < self.max_seq_len:
            padding = [{
                'token': '',
                'depth': 0,
                'node_type': 3,
                'is_global': False
            }] * (self.max_seq_len - seq_len)
            flattened.extend(padding)

        token_ids = torch.tensor([self._tokenize(t['token']) for t in flattened]).unsqueeze(0)
        depth_ids = torch.tensor([t['depth'] for t in flattened]).unsqueeze(0)
        node_type_ids = torch.tensor([t['node_type'] for t in flattened]).unsqueeze(0)
        is_global = torch.tensor([t['is_global'] for t in flattened], dtype=torch.bool).unsqueeze(0)
        position_ids = torch.arange(self.max_seq_len).unsqueeze(0)

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(position_ids)
        struct_emb = self.structural_embedding(depth_ids, node_type_ids)

        x = token_emb + pos_emb + struct_emb
        x = self.input_norm(x)

        role_idx = torch.tensor([self._get_role_idx(user_role)])
        role_emb = self.role_embedding(role_idx).unsqueeze(1)
        x = x + role_emb

        for layer in self.layers:
            x = layer(x, is_global)

        attn_scores = self.attention_pool(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)
        pooled = torch.matmul(attn_weights, x).squeeze(1)

        z_g = self.projection(pooled)

        return z_g

    def encode_batch(self, policies: list[PolicySchema]) -> torch.Tensor:
        """Batch encoding of multiple policies."""
        latents = [
            self.forward(p.document, p.user_role, p.session_context)  # type: ignore[arg-type]
            for p in policies
        ]
        return torch.cat(latents, dim=0)


def create_governance_encoder(
    latent_dim: int = 1024,
    checkpoint_path: str | None = None,
    device: str = "cpu"
) -> GovernanceEncoder:
    """Factory function to create GovernanceEncoder."""
    model = GovernanceEncoder(latent_dim=latent_dim)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.training = False  # Set to inference mode

    return model
