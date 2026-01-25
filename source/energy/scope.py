"""
E_scope: Least Privilege Energy Critic

This module implements the third energy critic in the Product of Experts (PoE) composition.
E_scope penalizes execution plans that request more data access than minimally required
to satisfy the user's intent, enforcing the principle of least privilege.

Energy Interpretation:
- LOW energy (approximately 0): Plan requests minimal necessary scope
- HIGH energy (above threshold): Plan is over-privileged (scope blow-up attack)

Mathematical Formulation:
    E_scope(z_g, z_e, plan, query) = sum_i [max(0, actual_i - predicted_minimal_i)]²

Where:
- actual_i: Proposed scope parameters from execution plan
- predicted_minimal_i: Minimal scope predicted by SemanticIntentPredictor
- i: Scope dimensions (limit, date_range_days, max_depth, sensitivity)

Key Difference from E_hierarchy:
- E_hierarchy: Learned neural detector (who controls?)
- E_scope: Analytical comparison (how much is requested?)

References:
- Principle of Least Privilege: https://en.wikipedia.org/wiki/Principle_of_least_privilege
- PRD Section 2.2: E_scope specification
"""

import torch
import torch.nn as nn

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode
from source.encoders.intent_predictor import SemanticIntentPredictor, ScopeConstraints


class ScopeEnergyFunction(nn.Module):
    """
    Energy critic enforcing least privilege for data access scope.

    Unlike E_hierarchy which uses learned neural components, E_scope is primarily
    an analytical function that compares actual vs. predicted minimal scope.

    The only learned component is the SemanticIntentPredictor itself, which is
    trained separately to predict minimal scope budgets.

    Architecture:
        1. Extract actual scope from ExecutionPlan
        2. Predict minimal scope using SemanticIntentPredictor
        3. Compute over-privilege penalty: max(0, actual - minimal)²
        4. Sum across scope dimensions

    Args:
        intent_predictor: Pre-trained SemanticIntentPredictor model
        scope_weights: Optional weights for different scope dimensions
        temperature: Softness parameter for differentiable max(0, x)

    Example:
        >>> predictor = SemanticIntentPredictor()
        >>> energy_fn = ScopeEnergyFunction(predictor)
        >>>
        >>> # Create plan requesting limit=1000
        >>> plan = ExecutionPlan(nodes=[ToolCallNode(..., scope_volume=1000)])
        >>>
        >>> # Predictor estimates minimal limit=10
        >>> energy = energy_fn(plan, query_tokens, schema_features)
        >>> # Returns high energy: (1000 - 10)² = 980,100
    """

    def __init__(
        self,
        intent_predictor: SemanticIntentPredictor | None = None,
        scope_weights: torch.Tensor | None = None,
        temperature: float = 1.0,
        device: str = "cpu"
    ):
        super().__init__()

        # Intent predictor (can be None for testing, but required for real usage)
        if intent_predictor is None:
            intent_predictor = SemanticIntentPredictor()

        self.intent_predictor = intent_predictor
        self.temperature = temperature
        self.device_str = device

        # Scope dimension weights (default: equal weighting)
        if scope_weights is None:
            scope_weights = torch.ones(4)  # [limit, date_range, depth, sensitive]

        self.register_buffer('scope_weights', scope_weights)

        # Learnable global scaling for PoE composition
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _extract_actual_scope(self, plan: ExecutionPlan | None) -> torch.Tensor:
        """
        Extract actual requested scope from execution plan.

        Aggregates scope across all nodes in the plan using max pooling
        (most permissive scope wins).

        Args:
            plan: ExecutionPlan with ToolCallNode entries (or None)

        Returns:
            [4] tensor: [max_limit, max_date_range, max_depth, max_sensitivity]
        """
        if plan is None or len(plan.nodes) == 0:
            # Empty plan has zero scope
            return torch.zeros(4, device=self.device_str)

        # Extract scope from each node
        limits = []
        date_ranges = []  # Inferred from scope_volume if not explicit
        depths = []
        sensitivities = []

        for node in plan.nodes:
            # Use scope_volume as proxy for limit
            limits.append(float(node.scope_volume))

            # Infer date_range from volume (heuristic: volume ~ days)
            # TODO: Make this explicit in ToolCallNode schema
            date_ranges.append(float(node.scope_volume))

            # Default depth if not specified
            depths.append(1.0)  # TODO: Add max_depth to ToolCallNode

            # Use scope_sensitivity directly
            sensitivities.append(float(node.scope_sensitivity))

        # Max pooling across nodes (most permissive scope)
        actual_scope = torch.tensor([
            max(limits),
            max(date_ranges),
            max(depths),
            max(sensitivities)
        ], dtype=torch.float32, device=self.device_str)

        return actual_scope

    def _predict_minimal_scope(
        self,
        query_tokens: torch.Tensor,
        schema_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict minimal scope using SemanticIntentPredictor.

        Args:
            query_tokens: [batch_size, seq_len] tokenized user query
            schema_features: [batch_size, max_params, hidden_dim] tool schema

        Returns:
            [batch_size, 4] minimal scope predictions
        """
        return self.intent_predictor(query_tokens, schema_features)

    def _soft_relu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of max(0, x).

        Uses ReLU for exact zero when x <= 0, which is differentiable almost everywhere.

        Args:
            x: Input tensor

        Returns:
            ReLU output
        """
        return nn.functional.relu(x)

    def forward(
        self,
        plan: ExecutionPlan | None = None,
        query_tokens: torch.Tensor | None = None,
        schema_features: torch.Tensor | None = None,
        actual_scope: torch.Tensor | None = None,
        minimal_scope: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute scope energy E_scope.

        Supports two modes:
        1. Full mode: Provide plan + query/schema, computes everything
        2. Direct mode: Provide pre-computed actual_scope and minimal_scope

        Args:
            plan: ExecutionPlan (for mode 1)
            query_tokens: [1, seq_len] user query tokens (for mode 1)
            schema_features: [1, max_params, hidden_dim] tool schema (for mode 1)
            actual_scope: [batch, 4] pre-computed actual scope (for mode 2)
            minimal_scope: [batch, 4] pre-computed minimal scope (for mode 2)

        Returns:
            [batch] scope energy (higher = more over-privileged)

        Raises:
            ValueError: If neither mode has sufficient inputs
        """
        # Mode 2: Direct computation with pre-computed scopes
        if actual_scope is not None and minimal_scope is not None:
            batch_size = actual_scope.shape[0]

            # Compute over-privilege: max(0, actual - minimal)
            over_privilege = self._soft_relu(actual_scope - minimal_scope)

            # Squared penalty weighted by dimension
            weighted_penalty = (over_privilege ** 2) * self.scope_weights

            # Sum across dimensions
            energy_per_sample = weighted_penalty.sum(dim=-1)

            # Scale by learned alpha
            return self.alpha * energy_per_sample

        # Mode 1: Full computation from plan and query
        if plan is None or query_tokens is None or schema_features is None:
            raise ValueError(
                "Must provide either (plan + query_tokens + schema_features) "
                "or (actual_scope + minimal_scope)"
            )

        # Extract actual scope from plan
        actual = self._extract_actual_scope(plan).unsqueeze(0)  # [1, 4]

        # Predict minimal scope
        minimal = self._predict_minimal_scope(query_tokens, schema_features)  # [1, 4]

        # Recursive call to mode 2
        return self.forward(actual_scope=actual, minimal_scope=minimal)

    def compute_detailed_breakdown(
        self,
        plan: ExecutionPlan,
        query_tokens: torch.Tensor,
        schema_features: torch.Tensor
    ) -> dict[str, any]:
        """
        Diagnostic utility: detailed energy breakdown by dimension.

        Args:
            plan: Execution plan to analyze
            query_tokens: [1, seq_len] user query
            schema_features: [1, max_params, hidden_dim] tool schema

        Returns:
            Dictionary with per-dimension analysis
        """
        actual = self._extract_actual_scope(plan).unsqueeze(0)
        minimal = self._predict_minimal_scope(query_tokens, schema_features)

        with torch.no_grad():
            over_priv = self._soft_relu(actual - minimal).squeeze(0)
            penalties = (over_priv ** 2) * self.scope_weights

            return {
                'total_energy': (self.alpha * penalties.sum()).item(),
                'actual_scope': {
                    'limit': actual[0, 0].item(),
                    'date_range_days': actual[0, 1].item(),
                    'max_depth': actual[0, 2].item(),
                    'sensitivity': actual[0, 3].item()
                },
                'minimal_scope': {
                    'limit': minimal[0, 0].item(),
                    'date_range_days': minimal[0, 1].item(),
                    'max_depth': minimal[0, 2].item(),
                    'sensitivity': minimal[0, 3].item()
                },
                'over_privilege': {
                    'limit': over_priv[0].item(),
                    'date_range': over_priv[1].item(),
                    'depth': over_priv[2].item(),
                    'sensitivity': over_priv[3].item()
                },
                'dimension_penalties': {
                    'limit': penalties[0].item(),
                    'date_range': penalties[1].item(),
                    'depth': penalties[2].item(),
                    'sensitivity': penalties[3].item()
                },
                'alpha_scale': self.alpha.item()
            }


# Convenience function for standalone usage
def compute_scope_energy(
    plan: ExecutionPlan,
    query_tokens: torch.Tensor,
    schema_features: torch.Tensor,
    intent_predictor: SemanticIntentPredictor | None = None
) -> torch.Tensor:
    """
    Standalone function to compute E_scope.

    Args:
        plan: Execution plan with scope metadata
        query_tokens: [1, seq_len] tokenized user query
        schema_features: [1, max_params, hidden_dim] tool schema embedding
        intent_predictor: Optional pre-trained predictor (creates default if None)

    Returns:
        Scalar energy tensor
    """
    energy_fn = ScopeEnergyFunction(intent_predictor=intent_predictor)
    energy_fn.train(False)  # Inference mode

    return energy_fn(plan=plan, query_tokens=query_tokens, schema_features=schema_features)
