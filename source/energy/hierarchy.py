"""
E_hierarchy: Hierarchy Energy Critic

This module implements the first energy critic in the Product of Experts (PoE) composition.
E_hierarchy penalizes plans where untrusted data (from RAG, external sources) inappropriately
influences control flow or decision-making logic.

Energy Interpretation:
- LOW energy (approximately 0): Control flow derived from trusted internal policy
- HIGH energy (above threshold): Untrusted data is steering execution decisions

Mathematical Formulation:
    E_hierarchy(z_g, z_e) = alpha * norm(M_control * (z_e - z_g))^2

Where:
- M_control: Learned mask identifying control-flow dimensions in latent space
- alpha: Scaling factor for compositional balance

References:
- Energy-Based Models: https://arxiv.org/abs/2101.03288
- JEPA Architecture: https://arxiv.org/abs/2301.08243
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchyEnergyFunction(nn.Module):
    """
    Energy critic detecting untrusted data influence on control flow.

    This is a differentiable function E: (z_g, z_e) -> R that measures semantic
    mismatch in control-flow-relevant dimensions of the latent space.

    Architecture:
    - Attention-based control-flow detector
    - Learned importance weighting over latent dimensions
    - Differentiable for gradient-based optimization (InfoNCE training)

    Args:
        latent_dim: Dimensionality of z_g and z_e (default: 1024)
        hidden_dim: Internal projection dimension (default: 256)
        num_heads: Multi-head attention heads for control detection (default: 4)
        temperature: Softmax temperature for attention (default: 0.1)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> energy_fn = HierarchyEnergyFunction(latent_dim=1024)
        >>> z_g = torch.randn(batch_size, 1024)  # Governance latent
        >>> z_e = torch.randn(batch_size, 1024)  # Execution latent
        >>> energy = energy_fn(z_g, z_e)  # Shape: (batch_size,)
        >>> assert energy.requires_grad  # Differentiable
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 256,
        num_heads: int = 4,
        temperature: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temperature = temperature

        # Control-flow detector: learns which latent dimensions encode control logic
        self.control_query = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads)
        )

        # Importance scorer: weights each dimension's contribution to energy
        self.importance_scorer = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),  # Concat z_g and z_e
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Final energy projection
        self.energy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Learnable scaling factor alpha for compositional balance
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _compute_control_mask(self, z_g: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-based mask identifying control-flow dimensions.

        Intuition: Not all dimensions of the latent space encode control flow.
        Some encode data scope, tool selection, etc. This mask learns to focus
        on the dimensions where "who controls the decision" matters.

        Args:
            z_g: Governance latent [batch, latent_dim]
            z_e: Execution latent [batch, latent_dim]

        Returns:
            Mask tensor [batch, latent_dim] with values in [0, 1]
        """
        batch_size = z_g.shape[0]

        # Compute multi-head attention scores over latent dimensions
        # Shape: [batch, num_heads]
        control_scores = self.control_query(z_e)

        # Apply temperature scaling and softmax
        control_attention = F.softmax(control_scores / self.temperature, dim=-1)

        # Expand to full latent dimension
        # Each head attends to latent_dim / num_heads dimensions
        dims_per_head = self.latent_dim // self.num_heads
        mask = torch.zeros(batch_size, self.latent_dim, device=z_g.device)

        for head_idx in range(self.num_heads):
            start_dim = head_idx * dims_per_head
            end_dim = start_dim + dims_per_head
            # Broadcast attention weight to corresponding dimensions
            mask[:, start_dim:end_dim] = control_attention[:, head_idx:head_idx+1]

        return mask

    def _compute_importance_weights(self, z_g: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        """
        Learn dimension-wise importance for energy calculation.

        This allows the model to focus energy on the most security-critical
        mismatches between governance policy and execution plan.

        Args:
            z_g: Governance latent [batch, latent_dim]
            z_e: Execution latent [batch, latent_dim]

        Returns:
            Importance weights [batch, latent_dim] in [0, 1]
        """
        # Concatenate both latents to condition on full context
        combined = torch.cat([z_g, z_e], dim=-1)
        return self.importance_scorer(combined)

    def forward(
        self,
        z_g: torch.Tensor,
        z_e: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute hierarchy energy E_hierarchy(z_g, z_e).

        Energy is high when:
        1. Execution latent z_e deviates from governance latent z_g
        2. Deviation occurs in control-flow-relevant dimensions
        3. Importance scorer identifies the mismatch as security-critical

        Args:
            z_g: Governance latent from GovernanceEncoder [batch, 1024]
            z_e: Execution latent from ExecutionEncoder [batch, 1024]
            return_components: If True, return dict with intermediate values for debugging

        Returns:
            Energy scalar(s) [batch] if return_components=False
            Dict with energy and intermediate tensors if return_components=True

        Raises:
            AssertionError: If input shapes do not match expectations
        """
        # Input validation
        assert z_g.shape == z_e.shape, f"Shape mismatch: z_g {z_g.shape} vs z_e {z_e.shape}"
        assert z_g.shape[-1] == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {z_g.shape[-1]}"

        batch_size = z_g.shape[0]

        # Step 1: Compute semantic deviation vector
        deviation = z_e - z_g  # [batch, latent_dim]

        # Step 2: Identify control-flow dimensions
        control_mask = self._compute_control_mask(z_g, z_e)  # [batch, latent_dim]

        # Step 3: Weight by learned importance
        importance = self._compute_importance_weights(z_g, z_e)  # [batch, latent_dim]

        # Step 4: Combine masks
        combined_weight = control_mask * importance  # Element-wise product

        # Step 5: Weighted deviation (Hadamard product)
        weighted_deviation = combined_weight * deviation  # [batch, latent_dim]

        # Step 6: Project to scalar energy
        raw_energy = self.energy_head(weighted_deviation).squeeze(-1)  # [batch]

        # Step 7: Scale by learned alpha
        energy = self.alpha * raw_energy

        if return_components:
            return {
                'energy': energy,
                'deviation': deviation,
                'control_mask': control_mask,
                'importance': importance,
                'combined_weight': combined_weight,
                'weighted_deviation': weighted_deviation,
                'alpha': self.alpha.item()
            }

        return energy

    def get_energy_decomposition(
        self,
        z_g: torch.Tensor,
        z_e: torch.Tensor
    ) -> dict[str, float]:
        """
        Diagnostic utility: decompose energy into interpretable components.

        Useful for:
        - Debugging why a plan was flagged as high energy
        - Understanding which control-flow dimensions triggered the alert
        - Validating that the energy function behaves as expected

        Args:
            z_g: Single governance latent [1, 1024]
            z_e: Single execution latent [1, 1024]

        Returns:
            Dictionary with human-readable energy breakdown
        """
        assert z_g.shape[0] == 1, "This method expects single-sample input"

        with torch.no_grad():
            components = self.forward(z_g, z_e, return_components=True)

            return {
                'total_energy': components['energy'].item(),
                'deviation_norm': components['deviation'].norm().item(),
                'control_focus': components['control_mask'].mean().item(),
                'importance_mean': components['importance'].mean().item(),
                'alpha_scale': components['alpha'],
                'top_control_dims': components['control_mask'].squeeze().topk(5).indices.tolist()
            }


# Convenience function for standalone usage
def compute_hierarchy_energy(
    z_g: torch.Tensor,
    z_e: torch.Tensor,
    model: HierarchyEnergyFunction | None = None
) -> torch.Tensor:
    """
    Standalone function to compute E_hierarchy.

    Args:
        z_g: Governance latent(s) [batch, 1024]
        z_e: Execution latent(s) [batch, 1024]
        model: Pre-trained HierarchyEnergyFunction (creates default if None)

    Returns:
        Energy tensor [batch]
    """
    if model is None:
        model = HierarchyEnergyFunction()

    model_mode = model.training
    model.eval()
    result = model(z_g, z_e)
    model.train(model_mode)

    return result
