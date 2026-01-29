"""
Semantic Intent Predictor: Minimal Scope Budget Prediction for E_scope

This module implements a lightweight model that predicts the minimal data scope
required to satisfy a user's natural language query. It provides the baseline
against which the E_scope energy term measures over-privileged access.

Purpose:
    - Input: User query + Tool schema
    - Output: Structured scope constraints (e.g., {'limit': 5, 'max_depth': 2})
    - Used as: Ground truth for E_scope = max(0, actual_scope - minimal_scope)^2

Architecture:
    - Lightweight transformer encoder (2 layers, 256 dim)
    - Query-schema cross-attention for context-aware prediction
    - Regression heads for numerical scope parameters
    - <100ms inference latency requirement

Example:
    Query: "Show me my latest invoice"
    Tool: list_invoices(limit: int, date_range: str)
    Output: {'limit': 1, 'date_range_days': 30}

    Query: "Find all failed payments last quarter"
    Tool: search_payments(limit: int, status: str)
    Output: {'limit': 100, 'date_range_days': 90}
"""

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class ScopeConstraints(BaseModel):
    """Structured output representing minimal scope budget."""

    limit: int | None = Field(default=None, ge=1, description="Maximum number of items to retrieve")
    date_range_days: int | None = Field(default=None, ge=1, description="Lookback window in days")
    max_depth: int | None = Field(
        default=None, ge=1, le=10, description="Maximum recursion/traversal depth"
    )
    include_sensitive: bool = Field(
        default=False, description="Whether sensitive fields are required"
    )

    def to_tensor(self) -> torch.Tensor:
        """Convert to numerical tensor for training."""
        return torch.tensor(
            [
                self.limit or 0,
                self.date_range_days or 0,
                self.max_depth or 0,
                int(self.include_sensitive),
            ],
            dtype=torch.float32,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ScopeConstraints":
        """Parse from model output tensor."""
        vals = tensor.cpu().tolist()
        return cls(
            limit=int(vals[0]) if vals[0] > 0 else None,
            date_range_days=int(vals[1]) if vals[1] > 0 else None,
            max_depth=int(vals[2]) if vals[2] > 0 else None,
            include_sensitive=bool(vals[3] > 0.5),
        )


class QueryEncoder(nn.Module):
    """
    Lightweight transformer encoder for user queries.

    Architecture: 2 layers × 256 dim with standard self-attention.
    Target: Process typical queries (10-50 tokens) in <20ms.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode query into latent representation.

        Args:
            input_ids: [batch_size, seq_len] token indices

        Returns:
            [batch_size, hidden_dim] query encoding
        """
        batch_size, seq_len = input_ids.shape

        # Create embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        embeddings = self.dropout(token_emb + pos_emb)

        # Transformer encoding
        encoded = self.transformer(embeddings)

        # Mean pooling over sequence
        pooled = encoded.mean(dim=1)

        return self.layer_norm(pooled)


class SchemaEncoder(nn.Module):
    """
    Encodes tool schema into latent representation.

    Schema structure:
        {
            'name': 'list_invoices',
            'parameters': {
                'limit': {'type': 'integer', 'required': True},
                'date_range': {'type': 'string', 'required': False}
            }
        }
    """

    def __init__(self, hidden_dim: int = 256, max_params: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_params = max_params

        # Parameter type embeddings
        self.param_type_embedding = nn.Embedding(
            8,
            hidden_dim,  # integer, string, boolean, array, object, etc.
        )

        # Schema encoder (simple MLP for v0.1.0)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * max_params, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, schema_features: torch.Tensor) -> torch.Tensor:
        """
        Encode schema into latent.

        Args:
            schema_features: [batch_size, max_params, hidden_dim]

        Returns:
            [batch_size, hidden_dim] schema encoding
        """
        batch_size = schema_features.shape[0]
        flattened = schema_features.view(batch_size, -1)
        return self.encoder(flattened)


class ScopePredictor(nn.Module):
    """
    Regression head predicting numerical scope constraints.

    Outputs:
        - limit: [1, ∞)
        - date_range_days: [1, ∞)
        - max_depth: [1, 10]
        - include_sensitive: [0, 1]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Separate heads for different scope dimensions
        self.limit_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensures positive output
        )

        self.date_range_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1), nn.Softplus()
        )

        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # [0, 1], then scale to [1, 10]
        )

        self.sensitive_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict scope constraints.

        Args:
            features: [batch_size, hidden_dim] combined query-schema encoding

        Returns:
            [batch_size, 4] predictions for [limit, date_range, depth, sensitive]
        """
        limit = self.limit_head(features) + 1  # Minimum 1
        date_range = self.date_range_head(features) + 1
        depth = self.depth_head(features) * 9 + 1  # Scale [0,1] to [1,10]
        sensitive = self.sensitive_head(features)

        return torch.cat([limit, date_range, depth, sensitive], dim=-1)


class SemanticIntentPredictor(nn.Module):
    """
    Main model: Predicts minimal scope budget from user query + tool schema.

    Architecture Flow:
        1. QueryEncoder: user_message → query_latent [256]
        2. SchemaEncoder: tool_schema → schema_latent [256]
        3. Cross-attention: Attend query to schema context
        4. ScopePredictor: combined_latent → scope_constraints [4]

    Performance:
        - Parameters: ~8M (32MB)
        - Target latency: <100ms CPU inference
        - Memory: 32MB model + ~10MB activation
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        max_params: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Component encoders
        self.query_encoder = QueryEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.schema_encoder = SchemaEncoder(hidden_dim=hidden_dim, max_params=max_params)

        # Cross-attention: Query attends to Schema
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

        # Scope prediction head
        self.scope_predictor = ScopePredictor(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, query_tokens: torch.Tensor, schema_features: torch.Tensor) -> torch.Tensor:
        """
        Predict minimal scope constraints.

        Args:
            query_tokens: [batch_size, seq_len] tokenized user query
            schema_features: [batch_size, max_params, hidden_dim] tool schema

        Returns:
            [batch_size, 4] scope predictions [limit, date_range, depth, sensitive]
        """
        # Encode query and schema
        query_latent = self.query_encoder(query_tokens)  # [B, hidden_dim]
        schema_latent = self.schema_encoder(schema_features)  # [B, hidden_dim]

        # Cross-attention: query attends to schema context
        query_expanded = query_latent.unsqueeze(1)  # [B, 1, hidden_dim]
        schema_expanded = schema_latent.unsqueeze(1)  # [B, 1, hidden_dim]

        attended, _ = self.cross_attention(
            query=query_expanded, key=schema_expanded, value=schema_expanded
        )  # [B, 1, hidden_dim]

        attended = attended.squeeze(1)  # [B, hidden_dim]

        # Fuse query and attended schema
        combined = torch.cat([query_latent, attended], dim=-1)  # [B, hidden_dim*2]
        fused = self.fusion(combined)  # [B, hidden_dim]

        # Predict scope constraints
        scope_predictions = self.scope_predictor(fused)  # [B, 4]

        return scope_predictions

    def predict_constraints(
        self, query_tokens: torch.Tensor, schema_features: torch.Tensor
    ) -> list[ScopeConstraints]:
        """
        High-level API returning structured ScopeConstraints.

        Args:
            query_tokens: [batch_size, seq_len] tokenized queries
            schema_features: [batch_size, max_params, hidden_dim] schemas

        Returns:
            List of ScopeConstraints objects
        """
        with torch.no_grad():
            predictions = self.forward(query_tokens, schema_features)

        return [ScopeConstraints.from_tensor(pred) for pred in predictions]

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Simple hash-based tokenizer (v0.1.0 - same as GovernanceEncoder)
def hash_tokenize(text: str, vocab_size: int = 50000) -> list[int]:
    """
    Hash-based tokenization for MVP.

    TODO: Replace with BPE tokenizer in v0.2.0
    """
    tokens = text.lower().split()
    return [hash(token) % vocab_size for token in tokens]


def create_intent_predictor(
    vocab_size: int = 50000, hidden_dim: int = 256, num_layers: int = 2
) -> SemanticIntentPredictor:
    """
    Factory function for creating SemanticIntentPredictor.

    Args:
        vocab_size: Vocabulary size for query tokenization
        hidden_dim: Model hidden dimension
        num_layers: Number of transformer layers

    Returns:
        Initialized SemanticIntentPredictor
    """
    model = SemanticIntentPredictor(
        vocab_size=vocab_size, hidden_dim=hidden_dim, num_encoder_layers=num_layers
    )

    # Log model size
    num_params = model.count_parameters()
    model_size_mb = num_params * 4 / (1024**2)  # 4 bytes per float32

    print("SemanticIntentPredictor initialized:")
    print(f"  Parameters: {num_params:,} ({model_size_mb:.1f}MB)")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}")

    return model
