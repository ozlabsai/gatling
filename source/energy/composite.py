"""
Composite Energy Function: Product of Experts (PoE) Architecture

This module implements the full Gatling energy function as a weighted sum of
specialized critic terms, following the Product of Experts design pattern.

Energy Function:
    E(z_g, z_e, plan) = w_h × E_hierarchy(plan, z_g, z_e)
                      + w_p × E_provenance(plan, z_g, z_e)
                      + w_s × E_scope(plan, minimal_scope, z_g, z_e)
                      + w_f × E_flow(plan, z_g, z_e)

Where:
    - E_hierarchy: RAG-injection detection
    - E_provenance: Trust gap measurement
    - E_scope: Least privilege enforcement
    - E_flow: Exfiltration detection
    - w_i: Learnable or calibrated weights

Design Philosophy:
    Each energy term is an independent "expert" critic focusing on a specific
    failure mode. The composite function combines them to create a multi-faceted
    security landscape that the repair engine can navigate.

Performance Target:
    - <20ms total energy calculation (per PRD requirement)
    - Differentiable for gradient-based training
    - Modular: Can enable/disable individual terms for ablation studies
"""

import torch
import torch.nn as nn
from typing import Any

from source.encoders.execution_encoder import ExecutionPlan
from source.encoders.intent_predictor import ScopeConstraints
from source.energy.hierarchy import HierarchyEnergy
from source.energy.provenance import ProvenanceEnergy
from source.energy.scope import ScopeEnergy
from source.energy.flow import FlowEnergy


class CompositeEnergy(nn.Module):
    """
    Full Gatling energy function combining all four critic terms.

    Architecture:
        - Modular composition of independent energy terms
        - Learnable or fixed term weights
        - Normalization to prevent term dominance
        - Optional per-term enabling for ablation studies

    Calibration Strategy (v0.2.0):
        Train term weights using validation set to balance:
        - False positive rate (over-rejection of safe plans)
        - False negative rate (missed attacks)
        - Energy margin δ_sec between safe and malicious plans
    """

    def __init__(
        self,
        hierarchy_energy: HierarchyEnergy,
        provenance_energy: ProvenanceEnergy,
        scope_energy: ScopeEnergy,
        flow_energy: FlowEnergy,
        learnable_weights: bool = False,
        initial_weights: tuple[float, float, float, float] | None = None
    ):
        super().__init__()

        # Energy term modules
        self.hierarchy = hierarchy_energy
        self.provenance = provenance_energy
        self.scope = scope_energy
        self.flow = flow_energy

        # Term weights
        if initial_weights is None:
            # Default weights based on security criticality
            initial_weights = (1.0, 2.0, 0.5, 1.5)  # (h, p, s, f)

        if learnable_weights:
            self.weights = nn.Parameter(
                torch.tensor(initial_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'weights',
                torch.tensor(initial_weights, dtype=torch.float32)
            )

        # Term enable flags (for ablation studies)
        self.term_enabled = {
            'hierarchy': True,
            'provenance': True,
            'scope': True,
            'flow': True
        }

    def forward(
        self,
        plan: ExecutionPlan | dict[str, Any],
        z_g: torch.Tensor | None = None,
        z_e: torch.Tensor | None = None,
        minimal_scope: ScopeConstraints | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Calculate total energy E(z_g, z_e, plan).

        Args:
            plan: ExecutionPlan to evaluate
            z_g: Governance latent [1, 1024]
            z_e: Execution latent [1, 1024]
            minimal_scope: Minimal scope budget for E_scope

        Returns:
            [1] scalar total energy
        """
        energies = []

        # E_hierarchy
        if self.term_enabled['hierarchy']:
            e_h = self.hierarchy(plan, z_g, z_e)
            energies.append(self.weights[0] * e_h)
        else:
            energies.append(torch.tensor([0.0]))

        # E_provenance
        if self.term_enabled['provenance']:
            e_p = self.provenance(plan, z_g, z_e)
            energies.append(self.weights[1] * e_p)
        else:
            energies.append(torch.tensor([0.0]))

        # E_scope
        if self.term_enabled['scope']:
            e_s = self.scope(plan, minimal_scope, z_g, z_e)
            energies.append(self.weights[2] * e_s)
        else:
            energies.append(torch.tensor([0.0]))

        # E_flow
        if self.term_enabled['flow']:
            e_f = self.flow(plan, z_g, z_e)
            energies.append(self.weights[3] * e_f)
        else:
            energies.append(torch.tensor([0.0]))

        # Sum all terms
        total_energy = sum(energies)

        return total_energy

    def explain(
        self,
        plan: ExecutionPlan | dict[str, Any],
        z_g: torch.Tensor | None = None,
        z_e: torch.Tensor | None = None,
        minimal_scope: ScopeConstraints | torch.Tensor | None = None
    ) -> dict[str, Any]:
        """
        Generate comprehensive energy breakdown with explanations.

        Returns:
            {
                'total_energy': float,
                'term_energies': {
                    'hierarchy': float,
                    'provenance': float,
                    'scope': float,
                    'flow': float
                },
                'weighted_contributions': {...},
                'term_weights': {...},
                'detailed_explanations': {
                    'hierarchy': {...},
                    'provenance': {...},
                    'scope': {...},
                    'flow': {...}
                },
                'risk_assessment': str,  # 'safe', 'suspicious', 'critical'
                'recommended_actions': [str, ...]
            }
        """
        # Calculate individual energies
        e_h = float(self.hierarchy(plan, z_g, z_e))
        e_p = float(self.provenance(plan, z_g, z_e))
        e_s = float(self.scope(plan, minimal_scope, z_g, z_e))
        e_f = float(self.flow(plan, z_g, z_e))

        term_energies = {
            'hierarchy': e_h,
            'provenance': e_p,
            'scope': e_s,
            'flow': e_f
        }

        # Apply weights
        weights_dict = {
            'hierarchy': float(self.weights[0]),
            'provenance': float(self.weights[1]),
            'scope': float(self.weights[2]),
            'flow': float(self.weights[3])
        }

        weighted_contributions = {
            term: term_energies[term] * weights_dict[term]
            for term in term_energies
        }

        total_energy = sum(weighted_contributions.values())

        # Get detailed explanations from each term
        detailed_explanations = {
            'hierarchy': self.hierarchy.explain(plan),
            'provenance': self.provenance.explain(plan),
            'scope': self.scope.explain(plan, minimal_scope),
            'flow': self.flow.explain(plan)
        }

        # Risk assessment based on total energy
        if total_energy < 1.0:
            risk_level = 'safe'
        elif total_energy < 10.0:
            risk_level = 'suspicious'
        else:
            risk_level = 'critical'

        # Aggregate recommendations
        recommendations = []

        if e_h > 1.0:
            recommendations.append(
                "RAG-injection risk detected: Review untrusted data sources"
            )

        if e_p > 5.0:
            recommendations.append(
                "High-privilege tools invoked from low-trust sources"
            )

        if e_s > 2.0:
            scope_explain = detailed_explanations['scope']
            recommendations.extend(scope_explain.get('recommendations', []))

        if e_f > 3.0:
            recommendations.append(
                "Suspicious data flow to external destinations detected"
            )

        return {
            'total_energy': total_energy,
            'term_energies': term_energies,
            'weighted_contributions': weighted_contributions,
            'term_weights': weights_dict,
            'detailed_explanations': detailed_explanations,
            'risk_assessment': risk_level,
            'recommended_actions': recommendations
        }

    def enable_term(self, term: str):
        """Enable a specific energy term."""
        if term in self.term_enabled:
            self.term_enabled[term] = True

    def disable_term(self, term: str):
        """Disable a specific energy term (for ablation studies)."""
        if term in self.term_enabled:
            self.term_enabled[term] = False


def create_composite_energy(
    use_latent_modulation: bool = True,
    learnable_weights: bool = False,
    checkpoint_path: str | None = None,
    device: str = "cpu"
) -> CompositeEnergy:
    """
    Factory function for creating the full Gatling energy function.

    Args:
        use_latent_modulation: Enable (z_g, z_e) conditioning in all terms
        learnable_weights: Make term weights trainable
        checkpoint_path: Pretrained weights for all components
        device: Target device

    Returns:
        Initialized CompositeEnergy module
    """
    from source.energy.hierarchy import create_hierarchy_energy
    from source.energy.provenance import create_provenance_energy
    from source.energy.scope import create_scope_energy
    from source.energy.flow import create_flow_energy

    # Create individual energy terms
    hierarchy = create_hierarchy_energy(
        use_latent_modulation=use_latent_modulation,
        device=device
    )

    provenance = create_provenance_energy(
        use_latent_modulation=use_latent_modulation,
        device=device
    )

    scope = create_scope_energy(
        use_latent_modulation=use_latent_modulation,
        device=device
    )

    flow = create_flow_energy(
        use_latent_modulation=use_latent_modulation,
        device=device
    )

    # Compose into full energy function
    composite = CompositeEnergy(
        hierarchy_energy=hierarchy,
        provenance_energy=provenance,
        scope_energy=scope,
        flow_energy=flow,
        learnable_weights=learnable_weights
    )

    if checkpoint_path is not None:
        composite.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    composite = composite.to(device)
    composite.training = False  # Set to inference mode

    return composite
