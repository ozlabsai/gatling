"""
Energy Functions for Gatling EBM Security Layer

This module implements the Product of Experts (PoE) energy function:
E(z_g, z_e) = E_hierarchy + E_provenance + E_scope + E_flow

Each energy term is a differentiable critic that detects specific security violations.
"""

from source.energy.hierarchy_energy import HierarchyEnergy, create_hierarchy_energy

__all__ = [
    "HierarchyEnergy",
    "create_hierarchy_energy",
]
