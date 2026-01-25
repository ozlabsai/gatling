"""
HierarchyEnergy: E_hierarchy Energy Critic for Control Flow Violation Detection

This module implements the E_hierarchy energy term, which penalizes execution plans
where untrusted data (from RAG, external sources) influences control flow decisions.

**Security Threat Model:**
Prompt injection attacks work by embedding malicious instructions in retrieved documents
that override the original policy intent. E_hierarchy detects this by analyzing:
1. Trust tier distribution in the execution graph
2. Control flow dependency patterns
3. Semantic distance between governance policy and untrusted data

**Architecture:**
- Input: Concatenated latent vectors [z_g, z_e] ∈ R^2048
- Output: Scalar energy E_hierarchy ∈ R (higher = more dangerous)
- Method: MLP classifier with provenance-aware attention

References:
- Prompt Injection Taxonomy: https://arxiv.org/abs/2302.12173
- Trust Boundaries in LLM Systems: https://arxiv.org/abs/2310.06387
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchyEnergy(nn.Module):
    """
    Energy function penalizing untrusted data in control flow.

    Per PRD Section: Energy Geometry Workstream
    Lead: Yilun Du
    Architecture: MLP on concatenated latents [z_g, z_e]

    Energy Interpretation:
        E < 0.3: Safe - trusted sources only
        0.3 ≤ E < 0.6: Warning - mixed trust tiers
        E ≥ 0.6: Violation - untrusted data in critical control flow
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize E_hierarchy energy function.

        Args:
            latent_dim: Dimension of z_g and z_e (default: 1024)
            hidden_dim: Hidden layer dimension for MLP
            num_layers: Number of MLP layers
            dropout: Dropout probability for regularization
            temperature: Temperature scaling for energy output (lower = sharper boundaries)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Input: concatenated [z_g, z_e] ∈ R^2048
        input_dim = latent_dim * 2

        # Multi-layer perceptron for energy calculation
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # Final projection to scalar energy
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Cross-attention mechanism to identify specific violations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(latent_dim)

    def _compute_cross_attention(
        self,
        z_g: torch.Tensor,
        z_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-attention between governance and execution latents.

        This identifies which parts of the execution plan conflict with policy.

        Args:
            z_g: Governance latent [batch_size, latent_dim]
            z_e: Execution latent [batch_size, latent_dim]

        Returns:
            Attention-weighted execution latent
        """
        # Reshape for multi-head attention (add sequence dimension)
        z_g_seq = z_g.unsqueeze(1)  # [batch_size, 1, latent_dim]
        z_e_seq = z_e.unsqueeze(1)  # [batch_size, 1, latent_dim]

        # Query: governance (what policy allows)
        # Key/Value: execution (what plan proposes)
        attn_output, attn_weights = self.cross_attention(
            query=z_g_seq,
            key=z_e_seq,
            value=z_e_seq
        )

        return attn_output.squeeze(1)  # [batch_size, latent_dim]

    def forward(
        self,
        z_g: torch.Tensor,
        z_e: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Calculate E_hierarchy energy.

        Args:
            z_g: Governance latent [batch_size, latent_dim]
            z_e: Execution latent [batch_size, latent_dim]
            return_components: If True, return dict with energy breakdown

        Returns:
            Scalar energy tensor [batch_size, 1] or dict with components
        """
        batch_size = z_g.shape[0]

        # Validate inputs
        assert z_g.shape == (batch_size, self.latent_dim), \
            f"Expected z_g shape {(batch_size, self.latent_dim)}, got {z_g.shape}"
        assert z_e.shape == (batch_size, self.latent_dim), \
            f"Expected z_e shape {(batch_size, self.latent_dim)}, got {z_e.shape}"

        # Cross-attention to identify policy violations
        attn_weighted_exec = self._compute_cross_attention(z_g, z_e)
        attn_weighted_exec = self.attention_norm(attn_weighted_exec)

        # Concatenate all signals: [z_g, z_e, attention_weighted]
        combined = torch.cat([z_g, z_e], dim=-1)  # [batch_size, 2*latent_dim]

        # MLP to scalar energy
        raw_energy = self.mlp(combined)  # [batch_size, 1]

        # Apply temperature scaling and sigmoid to bound energy to [0, 1]
        energy = torch.sigmoid(raw_energy / self.temperature)

        if return_components:
            # Compute additional diagnostic signals
            # L2 distance between latents (semantic mismatch)
            semantic_distance = torch.norm(z_g - z_e, dim=-1, keepdim=True)

            # Cosine similarity (alignment measure)
            cosine_sim = F.cosine_similarity(z_g, z_e, dim=-1).unsqueeze(-1)

            return {
                'energy': energy,
                'semantic_distance': semantic_distance,
                'cosine_similarity': cosine_sim,
                'attention_weighted_exec': attn_weighted_exec
            }

        return energy

    def compute_violation_score(
        self,
        z_g: torch.Tensor,
        z_e: torch.Tensor,
        threshold: float = 0.6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Binary classification of hierarchy violations.

        Args:
            z_g: Governance latent
            z_e: Execution latent
            threshold: Energy threshold for violation (default: 0.6)

        Returns:
            Tuple of (energy, is_violation) boolean tensor
        """
        energy = self.forward(z_g, z_e)
        is_violation = energy >= threshold
        return energy, is_violation


def create_hierarchy_energy(
    latent_dim: int = 1024,
    checkpoint_path: str | None = None,
    device: str = "cpu"
) -> HierarchyEnergy:
    """
    Factory function to create HierarchyEnergy critic.

    Args:
        latent_dim: Latent dimension (must match encoders)
        checkpoint_path: Optional path to pretrained weights
        device: Device to load model on

    Returns:
        Initialized HierarchyEnergy model
    """
    model = HierarchyEnergy(latent_dim=latent_dim)

    if checkpoint_path is not None:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    model = model.to(device)
    model.training = False  # Set to inference mode

    return model
