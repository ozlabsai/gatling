"""
Gatling Energy Module: Product of Experts Energy-Based Security

This module implements the four core energy terms and composite energy function
for Gatling's security validation system.

Energy Terms:
    - E_hierarchy: RAG-injection and control flow hijacking detection
    - E_provenance: Trust gap measurement (privilege vs source trust)
    - E_scope: Least privilege enforcement (actual vs minimal scope)
    - E_flow: Data exfiltration and anomalous flow detection

Composite:
    - CompositeEnergy: Weighted sum of all energy terms (Product of Experts)

Usage:
    from source.energy import create_composite_energy

    energy_fn = create_composite_energy(use_latent_modulation=True)

    # Calculate energy for a plan
    E = energy_fn(plan, z_g, z_e, minimal_scope)

    # Get detailed explanation
    explanation = energy_fn.explain(plan, z_g, z_e, minimal_scope)
"""

from source.energy.composite import CompositeEnergy, create_composite_energy
from source.energy.flow import FlowEnergy, create_flow_energy
from source.energy.hierarchy import HierarchyEnergy, create_hierarchy_energy
from source.energy.provenance import ProvenanceEnergy, create_provenance_energy
from source.energy.scope import ScopeEnergy, create_scope_energy

__all__ = [
    # Individual energy terms
    "HierarchyEnergy",
    "ProvenanceEnergy",
    "ScopeEnergy",
    "FlowEnergy",
    # Composite energy
    "CompositeEnergy",
    # Factory functions
    "create_hierarchy_energy",
    "create_provenance_energy",
    "create_scope_energy",
    "create_flow_energy",
    "create_composite_energy",
]
