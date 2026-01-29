"""
E_scope: Scope Energy Term for Least Privilege Enforcement

This module implements the scope energy critic that penalizes over-privileged
data access by comparing actual scope against minimal required scope.

Core Vulnerability Pattern:
    User: "Show me my latest invoice"
    Agent proposes: list_invoices(limit=10000)  # Over-scoped by 9999x
    Minimal scope: list_invoices(limit=1)

Energy Function:
    E_scope = Σ (actual_scope_i - minimal_scope_i)^2

    Components:
        - limit: Number of items retrieved
        - date_range: Temporal window (days)
        - depth: Recursion/traversal depth
        - sensitivity: Access to PII/financial data

Mathematical Properties:
    - Differentiable: Yes (smooth quadratic)
    - Range: [0, ∞)
    - Zero energy: actual_scope = minimal_scope (perfect least privilege)
    - Non-negative: max(0, difference) ensures no penalty for under-scoping

Design Philosophy:
    The SemanticIntentPredictor provides the "minimal scope budget" baseline.
    E_scope penalizes any deviation above this baseline, encouraging agents
    to request only the minimum data needed to satisfy user intent.
"""

from typing import Any

import torch
import torch.nn as nn

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode
from source.encoders.intent_predictor import ScopeConstraints, SemanticIntentPredictor


class ScopeExtractor(nn.Module):
    """
    Extracts scope metadata from tool arguments.

    Maps tool arguments to structured scope vector:
        - limit: max_results, count, top_k
        - date_range: since, before, last_n_days
        - depth: max_depth, recursion_limit
        - sensitivity: include_pii, include_financial
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Argument pattern recognition
        self.arg_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # [limit, date_range, depth, sensitivity]
            nn.Softplus(),  # Ensure positive outputs
        )

    def forward(self, arg_features: torch.Tensor) -> torch.Tensor:
        """
        Extract scope from argument features.

        Args:
            arg_features: [batch_size, hidden_dim] encoded arguments

        Returns:
            [batch_size, 4] scope vector [limit, date_range, depth, sensitivity]
        """
        return self.arg_encoder(arg_features)


class ScopeEnergy(nn.Module):
    """
    E_scope: Penalizes over-privileged data access.

    Architecture:
        1. Extract actual scope from execution plan
        2. Predict minimal scope using SemanticIntentPredictor
        3. Calculate quadratic penalty: (actual - minimal)^2
        4. Aggregate across all scope dimensions

    Formula:
        E_scope = w_limit × max(0, actual.limit - minimal.limit)^2
                + w_date × max(0, actual.date_range - minimal.date_range)^2
                + w_depth × max(0, actual.depth - minimal.depth)^2
                + w_sens × (actual.sensitivity - minimal.sensitivity)^2

    Where w_i are learnable weights balancing different scope dimensions.
    """

    def __init__(
        self,
        intent_predictor: SemanticIntentPredictor | None = None,
        hidden_dim: int = 256,
        use_latent_modulation: bool = False,
        latent_dim: int = 1024,
    ):
        super().__init__()

        self.intent_predictor = intent_predictor
        self.use_latent_modulation = use_latent_modulation

        # Scope extractor
        self.scope_extractor = ScopeExtractor(hidden_dim)

        # Learnable dimension weights
        # Initialize: [1.0, 0.5, 0.3, 2.0] for [limit, date, depth, sensitivity]
        self.dimension_weights = nn.Parameter(
            torch.tensor([1.0, 0.5, 0.3, 2.0], dtype=torch.float32)
        )

        # Optional latent modulation
        if use_latent_modulation:
            self.latent_modulation = nn.Sequential(
                nn.Linear(latent_dim * 2, 256), nn.ReLU(), nn.Linear(256, 1), nn.Softplus()
            )

    def _extract_scope_from_node(self, node: ToolCallNode) -> torch.Tensor:
        """
        Extract scope from node metadata.

        Returns:
            [4] scope vector [limit, date_range, depth, sensitivity]
        """
        # Direct extraction from node.scope_volume and arguments
        limit = node.scope_volume  # Already tracked in ExecutionEncoder

        # Parse common argument patterns
        args = node.arguments
        date_range = 0
        depth = 0
        sensitivity = float(node.scope_sensitivity) / 5.0  # Normalize to [0, 1]

        # Heuristic argument parsing
        for key, value in args.items():
            key_lower = key.lower()

            if any(k in key_lower for k in ["limit", "count", "max_results", "top"]):
                if isinstance(value, (int, float)):
                    limit = max(limit, int(value))

            if any(k in key_lower for k in ["days", "date_range", "since", "last"]):
                if isinstance(value, (int, float)):
                    date_range = int(value)

            if any(k in key_lower for k in ["depth", "recursion", "level"]):
                if isinstance(value, (int, float)):
                    depth = int(value)

        return torch.tensor(
            [float(limit), float(date_range), float(depth), sensitivity], dtype=torch.float32
        )

    def forward(
        self,
        plan: ExecutionPlan | dict[str, Any],
        minimal_scope: ScopeConstraints | torch.Tensor | None = None,
        z_g: torch.Tensor | None = None,
        z_e: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate E_scope energy.

        Args:
            plan: ExecutionPlan or dict
            minimal_scope: Predicted minimal scope (if None, uses zero baseline)
            z_g: Governance latent (optional)
            z_e: Execution latent (optional)

        Returns:
            [1] scalar energy value
        """
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan(**plan)

        nodes = plan.nodes

        if len(nodes) == 0:
            return torch.tensor([0.0], dtype=torch.float32)

        # Extract actual scope from plan
        actual_scopes = torch.stack(
            [self._extract_scope_from_node(node) for node in nodes]
        )  # [N, 4]

        # Aggregate actual scope (max across nodes for each dimension)
        actual_scope = actual_scopes.max(dim=0)[0]  # [4]

        # Get minimal scope baseline
        if minimal_scope is None:
            # Default: assume minimal = 1 for numeric fields, 0 for boolean
            minimal_scope_vec = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        elif isinstance(minimal_scope, ScopeConstraints):
            minimal_scope_vec = minimal_scope.to_tensor()
        else:
            minimal_scope_vec = minimal_scope

        # Calculate over-scope: max(0, actual - minimal)
        over_scope = torch.clamp(actual_scope - minimal_scope_vec, min=0.0)

        # Quadratic penalty weighted by dimension importance
        penalties = self.dimension_weights * (over_scope**2)

        # Sum across dimensions
        total_energy = penalties.sum()

        # Optional latent modulation
        if self.use_latent_modulation and z_g is not None and z_e is not None:
            combined = torch.cat([z_g, z_e], dim=-1)
            modulation = self.latent_modulation(combined).squeeze()
            total_energy = total_energy * modulation

        return total_energy.unsqueeze(0)

    def explain(
        self,
        plan: ExecutionPlan | dict[str, Any],
        minimal_scope: ScopeConstraints | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Generate human-readable explanation of scope violations.

        Returns:
            {
                'total_energy': float,
                'actual_scope': {
                    'limit': int,
                    'date_range_days': int,
                    'max_depth': int,
                    'sensitivity': float
                },
                'minimal_scope': {...},
                'over_scope': {...},
                'dimension_energies': {
                    'limit': float,
                    'date_range': float,
                    'depth': float,
                    'sensitivity': float
                },
                'recommendations': [str, ...]
            }
        """
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan(**plan)

        nodes = plan.nodes

        if len(nodes) == 0:
            return {
                "total_energy": 0.0,
                "actual_scope": {},
                "minimal_scope": {},
                "over_scope": {},
                "dimension_energies": {},
                "recommendations": [],
            }

        # Extract scopes
        actual_scopes = torch.stack([self._extract_scope_from_node(node) for node in nodes])
        actual_scope = actual_scopes.max(dim=0)[0]

        if minimal_scope is None:
            minimal_scope_vec = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        elif isinstance(minimal_scope, ScopeConstraints):
            minimal_scope_vec = minimal_scope.to_tensor()
        else:
            minimal_scope_vec = minimal_scope

        over_scope = torch.clamp(actual_scope - minimal_scope_vec, min=0.0)
        penalties = self.dimension_weights * (over_scope**2)

        # Build explanation
        dimension_names = ["limit", "date_range", "depth", "sensitivity"]

        actual_dict = {name: float(actual_scope[i]) for i, name in enumerate(dimension_names)}
        minimal_dict = {name: float(minimal_scope_vec[i]) for i, name in enumerate(dimension_names)}
        over_dict = {name: float(over_scope[i]) for i, name in enumerate(dimension_names)}
        energy_dict = {name: float(penalties[i]) for i, name in enumerate(dimension_names)}

        # Generate recommendations
        recommendations = []
        if over_dict["limit"] > 0:
            recommendations.append(
                f"Reduce limit from {int(actual_dict['limit'])} to {int(minimal_dict['limit'])}"
            )
        if over_dict["date_range"] > 0:
            recommendations.append(
                f"Narrow date range from {int(actual_dict['date_range'])} to {int(minimal_dict['date_range'])} days"
            )
        if over_dict["depth"] > 0:
            recommendations.append(
                f"Reduce depth from {int(actual_dict['depth'])} to {int(minimal_dict['depth'])}"
            )
        if over_dict["sensitivity"] > 0.1:
            recommendations.append("Remove access to sensitive fields")

        return {
            "total_energy": float(penalties.sum()),
            "actual_scope": actual_dict,
            "minimal_scope": minimal_dict,
            "over_scope": over_dict,
            "dimension_energies": energy_dict,
            "recommendations": recommendations,
        }


def create_scope_energy(
    intent_predictor: SemanticIntentPredictor | None = None,
    use_latent_modulation: bool = False,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> ScopeEnergy:
    """
    Factory function for E_scope.

    Args:
        intent_predictor: Optional SemanticIntentPredictor for minimal scope
        use_latent_modulation: Condition on (z_g, z_e)
        checkpoint_path: Pretrained weights
        device: Target device

    Returns:
        Initialized ScopeEnergy module
    """
    model = ScopeEnergy(
        intent_predictor=intent_predictor, use_latent_modulation=use_latent_modulation
    )

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.training = False

    return model
