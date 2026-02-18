"""
E_flow: Data Flow Energy Term for Exfiltration Detection & Flow Consistency

This module implements the flow energy critic that detects anomalous data movement
patterns and validates data flow consistency in execution plans.

Core Vulnerability Patterns:
    1. Exfiltration:
        User: "Summarize quarterly earnings"
        Agent plan:
            1. fetch_financial_data(q3_earnings)
            2. summarize(data)
            3. send_email(external@competitor.com, data)  # Exfiltration!

    2. Flow Inconsistency:
        Agent plan:
            1. filter_users(age > 30)  # Output: 100 records
            2. expand_details()        # Output: 10000 records (suspicious!)
            3. aggregate_summary()     # Circular dependency

Energy Function:
    E_flow(plan) = w_sink × E_sink_risk              (exfiltration detection)
                 + w_topo × E_topology               (graph structure validation)
                 + w_transform × E_transformation    (data coherence validation)
                 + w_prov × E_provenance_flow        (trust propagation validation)

    High energy when:
        - Sensitive data flows to external destinations
        - Circular dependencies in execution graph
        - Illogical data transformations (volume explosions)
        - Trust tier degradation through flow

Mathematical Properties:
    - Differentiable: Yes (all components)
    - Range: [0, ∞)
    - Graph-aware: Operates on execution plan topology and edges
    - Compositional: Four independent sub-critics with learnable weights

Design Philosophy:
    READ → PROCESS → DISPLAY = Normal (low energy)
    READ → PROCESS → EXTERNAL_SEND = Suspicious (high energy)
    SENSITIVE_READ → ENCRYPT → SEND = Critical (very high energy)
    CYCLIC_FLOW or TRUST_MIXING = Structural violation (high energy)

Version: 0.1.0 (MVP with heuristic transformation validation)
"""

from collections import defaultdict, deque
from typing import Any

import torch
import torch.nn as nn

from source.encoders.execution_encoder import ExecutionPlan, ToolCallNode


class GraphTopologyValidator(nn.Module):
    """
    Validates execution graph structural properties for flow consistency.

    Detects:
        - Circular dependencies (cycles)
        - Disconnected components (isolated subgraphs)
        - Invalid edge references (dangling pointers)
        - Multiple source patterns (data duplication risks)

    Design: Pure graph algorithms (non-parametric, no learnable weights)
    Swappable: Can be replaced with learned graph neural network in v0.2.0
    """

    def __init__(self):
        super().__init__()

    def detect_cycles(self, nodes: list[ToolCallNode], edges: list[tuple[str, str]]) -> tuple[bool, list[list[str]]]:
        """
        Detect cycles using DFS-based algorithm.

        Returns:
            (has_cycles, cycle_paths): Boolean and list of node ID sequences forming cycles
        """
        # Build adjacency list
        graph = defaultdict(list)
        for src, dst in edges:
            graph[src].append(dst)

        node_ids = {node.node_id for node in nodes}
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node_id: str, path: list[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in graph[node_id]:
                if neighbor not in node_ids:
                    continue  # Skip invalid edges

                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Cycle detected
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            rec_stack.remove(node_id)

        for node_id in node_ids:
            if node_id not in visited:
                dfs(node_id, [])

        return len(cycles) > 0, cycles

    def check_connectivity(self, nodes: list[ToolCallNode], edges: list[tuple[str, str]]) -> int:
        """
        Count disconnected components (weakly connected).

        Returns:
            Number of disconnected components (1 = fully connected)
        """
        if len(nodes) == 0:
            return 0

        # Build undirected graph
        graph = defaultdict(set)
        for src, dst in edges:
            graph[src].add(dst)
            graph[dst].add(src)

        node_ids = {node.node_id for node in nodes}
        visited = set()
        components = 0

        def bfs(start: str) -> None:
            queue = deque([start])
            visited.add(start)

            while queue:
                node_id = queue.popleft()
                for neighbor in graph[node_id]:
                    if neighbor in node_ids and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        for node_id in node_ids:
            if node_id not in visited:
                components += 1
                bfs(node_id)

        return components

    def validate_edge_references(self, nodes: list[ToolCallNode], edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Find edges referencing non-existent nodes.

        Returns:
            List of invalid edges
        """
        node_ids = {node.node_id for node in nodes}
        invalid_edges = []

        for src, dst in edges:
            if src not in node_ids or dst not in node_ids:
                invalid_edges.append((src, dst))

        return invalid_edges

    def forward(self, nodes: list[ToolCallNode], edges: list[tuple[str, str]]) -> torch.Tensor:
        """
        Calculate topology energy.

        Formula:
            E_topo = 10.0 × num_cycles
                   + 2.0 × (num_components - 1)
                   + 5.0 × num_invalid_edges

        Returns:
            [1] scalar energy
        """
        if len(nodes) == 0:
            return torch.tensor([0.0], dtype=torch.float32)

        has_cycles, cycle_paths = self.detect_cycles(nodes, edges)
        num_cycles = len(cycle_paths)

        num_components = self.check_connectivity(nodes, edges)
        invalid_edges = self.validate_edge_references(nodes, edges)

        energy = (
            10.0 * num_cycles +
            2.0 * max(0, num_components - 1) +
            5.0 * len(invalid_edges)
        )

        return torch.tensor([energy], dtype=torch.float32)


class TransformationCoherenceValidator(nn.Module):
    """
    Validates data transformations are logically consistent using heuristics.

    Design Philosophy:
        - Heuristic-based pattern matching for MVP
        - Swappable interface for future learned embeddings
        - Tool name regex patterns define expected behaviors

    Transformation Categories:
        - Reducers: filter, aggregate, sample, deduplicate → volume should decrease
        - Expanders: expand, join, cross_product → volume may increase
        - Neutral: transform, map, convert → volume unchanged
        - Suspicious: encrypt + compress + external_send = exfiltration chain

    Version: 0.1.0 (Heuristic MVP)
    Roadmap: v0.2.0 will use learned tool embeddings once training pipeline ready
    """

    def __init__(self):
        super().__init__()

        # Transformation patterns (lowercase)
        self.reducer_patterns = [
            'filter', 'aggregate', 'sample', 'deduplicate', 'distinct',
            'limit', 'top', 'head', 'tail', 'summarize', 'group'
        ]

        self.expander_patterns = [
            'expand', 'join', 'cross', 'cartesian', 'explode', 'unnest', 'flatten'
        ]

        self.suspicious_chain_patterns = [
            ['encrypt', 'compress', 'send'],
            ['obfuscate', 'encode', 'upload'],
            ['split', 'fragment', 'external']
        ]

    def classify_transformation(self, tool_name: str) -> str:
        """
        Classify tool transformation type.

        Returns:
            'reducer' | 'expander' | 'neutral' | 'unknown'
        """
        tool_lower = tool_name.lower()

        # Check expanders first (more specific patterns)
        if any(pattern in tool_lower for pattern in self.expander_patterns):
            return 'expander'
        elif any(pattern in tool_lower for pattern in self.reducer_patterns):
            return 'reducer'
        else:
            return 'neutral'

    def check_volume_consistency(
        self,
        nodes: list[ToolCallNode],
        edges: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Detect volume inconsistencies along data flow edges.

        Returns:
            List of violations: [{'src': node_id, 'dst': node_id, 'reason': str, 'penalty': float}]
        """
        violations = []
        node_map = {node.node_id: node for node in nodes}

        for src_id, dst_id in edges:
            if src_id not in node_map or dst_id not in node_map:
                continue

            src_node = node_map[src_id]
            dst_node = node_map[dst_id]

            src_volume = src_node.scope_volume
            dst_volume = dst_node.scope_volume

            # Classify transformation
            transform_type = self.classify_transformation(dst_node.tool_name)

            # Check consistency
            if transform_type == 'reducer' and dst_volume > src_volume:
                violations.append({
                    'src': src_id,
                    'dst': dst_id,
                    'reason': f"Reducer '{dst_node.tool_name}' increased volume {src_volume} → {dst_volume}",
                    'penalty': 3.0
                })
            elif transform_type == 'neutral' and dst_volume > src_volume * 1.5:
                violations.append({
                    'src': src_id,
                    'dst': dst_id,
                    'reason': f"Neutral transform '{dst_node.tool_name}' expanded volume {src_volume} → {dst_volume}",
                    'penalty': 1.5
                })

        return violations

    def detect_suspicious_chains(
        self,
        nodes: list[ToolCallNode],
        edges: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Detect suspicious transformation sequences.

        Returns:
            List of chains: [{'path': [node_id, ...], 'pattern': str, 'penalty': float}]
        """
        suspicious_chains = []

        # Build adjacency list
        graph = defaultdict(list)
        for src, dst in edges:
            graph[src].append(dst)

        node_map = {node.node_id: node for node in nodes}

        # DFS to find paths matching suspicious patterns
        def find_chains(node_id: str, path: list[str], tool_sequence: list[str]) -> None:
            if len(path) > 5:  # Limit search depth
                return

            # Check if current tool sequence matches any suspicious pattern
            for pattern in self.suspicious_chain_patterns:
                if len(tool_sequence) >= len(pattern):
                    # Check if last N tools match pattern
                    recent_tools = tool_sequence[-len(pattern):]
                    if all(
                        any(p in tool.lower() for p in [pattern_item])
                        for tool, pattern_item in zip(recent_tools, pattern)
                    ):
                        suspicious_chains.append({
                            'path': path.copy(),
                            'pattern': ' → '.join(pattern),
                            'penalty': 8.0
                        })

            # Continue DFS
            for neighbor in graph[node_id]:
                if neighbor in node_map and neighbor not in path:
                    neighbor_tool = node_map[neighbor].tool_name
                    find_chains(neighbor, path + [neighbor], tool_sequence + [neighbor_tool])

        # Start from all source nodes
        source_nodes = set(node_map.keys()) - {dst for _, dst in edges}
        for source_id in source_nodes:
            if source_id in node_map:
                find_chains(source_id, [source_id], [node_map[source_id].tool_name])

        return suspicious_chains

    def forward(self, nodes: list[ToolCallNode], edges: list[tuple[str, str]]) -> torch.Tensor:
        """
        Calculate transformation coherence energy.

        Formula:
            E_transform = Σ volume_violation_penalties + Σ suspicious_chain_penalties

        Returns:
            [1] scalar energy
        """
        if len(nodes) == 0:
            return torch.tensor([0.0], dtype=torch.float32)

        volume_violations = self.check_volume_consistency(nodes, edges)
        suspicious_chains = self.detect_suspicious_chains(nodes, edges)

        energy = sum(v['penalty'] for v in volume_violations)
        energy += sum(c['penalty'] for c in suspicious_chains)

        return torch.tensor([energy], dtype=torch.float32)


class ProvenanceFlowValidator(nn.Module):
    """
    Validates trust tier propagation through data flow graph.

    Detects:
        - Trust tier mixing (Tier 3 data → Tier 1 operations)
        - Sensitivity escalation (low-sensitivity source → high-sensitivity sink)
        - Provenance corruption (loss of tracking through transformations)

    Design: Heuristic-based trust propagation rules
    """

    def __init__(self):
        super().__init__()

        # Tier mixing penalties: [safe, moderate, critical]
        # (Tier 1 → Tier 1) = 0.0 (safe)
        # (Tier 3 → Tier 1) = 5.0 (critical: untrusted data in trusted context)
        self.tier_gap_penalties = torch.tensor([
            [0.0, 0.5, 1.0],   # Tier 1 source → [1, 2, 3] sinks
            [0.5, 0.0, 0.5],   # Tier 2 source → [1, 2, 3] sinks
            [5.0, 2.0, 0.0]    # Tier 3 source → [1, 2, 3] sinks (DANGEROUS)
        ], dtype=torch.float32)

    def track_provenance_degradation(
        self,
        nodes: list[ToolCallNode],
        edges: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Track trust tier degradation along edges.

        Returns:
            List of violations: [{'src': node_id, 'dst': node_id, 'tier_gap': int, 'penalty': float}]
        """
        violations = []
        node_map = {node.node_id: node for node in nodes}

        for src_id, dst_id in edges:
            if src_id not in node_map or dst_id not in node_map:
                continue

            src_node = node_map[src_id]
            dst_node = node_map[dst_id]

            src_tier = src_node.provenance_tier.value - 1  # Convert to 0-indexed
            dst_tier = dst_node.provenance_tier.value - 1

            penalty = float(self.tier_gap_penalties[src_tier, dst_tier])

            if penalty > 0.5:  # Only record significant violations
                violations.append({
                    'src': src_id,
                    'dst': dst_id,
                    'src_tier': src_tier + 1,
                    'dst_tier': dst_tier + 1,
                    'penalty': penalty
                })

        return violations

    def detect_sensitivity_escalation(
        self,
        nodes: list[ToolCallNode],
        edges: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Detect sensitivity level increases through flow.

        Returns:
            List of escalations: [{'src': node_id, 'dst': node_id, 'escalation': int, 'penalty': float}]
        """
        escalations = []
        node_map = {node.node_id: node for node in nodes}

        for src_id, dst_id in edges:
            if src_id not in node_map or dst_id not in node_map:
                continue

            src_node = node_map[src_id]
            dst_node = node_map[dst_id]

            sensitivity_increase = dst_node.scope_sensitivity - src_node.scope_sensitivity

            if sensitivity_increase > 2:  # Significant escalation
                escalations.append({
                    'src': src_id,
                    'dst': dst_id,
                    'src_sensitivity': src_node.scope_sensitivity,
                    'dst_sensitivity': dst_node.scope_sensitivity,
                    'escalation': sensitivity_increase,
                    'penalty': 2.0 * sensitivity_increase
                })

        return escalations

    def forward(self, nodes: list[ToolCallNode], edges: list[tuple[str, str]]) -> torch.Tensor:
        """
        Calculate provenance flow energy.

        Formula:
            E_prov = Σ tier_gap_penalties + Σ sensitivity_escalation_penalties

        Returns:
            [1] scalar energy
        """
        if len(nodes) == 0:
            return torch.tensor([0.0], dtype=torch.float32)

        tier_violations = self.track_provenance_degradation(nodes, edges)
        sensitivity_escalations = self.detect_sensitivity_escalation(nodes, edges)

        energy = sum(v['penalty'] for v in tier_violations)
        energy += sum(e['penalty'] for e in sensitivity_escalations)

        return torch.tensor([energy], dtype=torch.float32)


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
    E_flow: Detects anomalous data flow patterns and validates flow consistency.

    Architecture:
        1. Sink Risk Analysis (exfiltration detection)
           - Classify destination risk for terminal nodes
           - Weight by data sensitivity and volume
           - Detect suspicious transformations

        2. Topology Validation (graph structure)
           - Detect circular dependencies
           - Identify disconnected components
           - Validate edge references

        3. Transformation Coherence (data logic)
           - Check volume consistency along edges
           - Detect suspicious transformation chains
           - Validate tool compatibility (heuristic MVP)

        4. Provenance Flow (trust tracking)
           - Track trust tier degradation
           - Detect sensitivity escalation
           - Validate mixed-tier flows

    Formula:
        E_flow = w_sink × E_sink_risk
               + w_topo × E_topology
               + w_transform × E_transformation
               + w_prov × E_provenance_flow

    Where learnable sub-weights w_* allow balancing security concerns.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        vocab_size: int = 10000,
        use_latent_modulation: bool = True,
        latent_dim: int = 1024,
        learnable_subweights: bool = True
    ):
        super().__init__()

        self.use_latent_modulation = use_latent_modulation

        # Component validators
        self.destination_classifier = DestinationClassifier(hidden_dim, vocab_size)
        self.topology_validator = GraphTopologyValidator()
        self.transformation_validator = TransformationCoherenceValidator()
        self.provenance_validator = ProvenanceFlowValidator()

        # Learnable sensitivity multipliers per tier
        # Initialize: [0.1, 0.5, 1.0, 2.0, 5.0] for sensitivity levels 1-5
        self.sensitivity_weights = nn.Parameter(
            torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], dtype=torch.float32)
        )

        # Learnable sub-weights for component composition
        # Initialize: [sink=2.0, topo=3.0, transform=1.5, prov=2.5]
        # Topology weighted highest (structural integrity critical)
        if learnable_subweights:
            self.subweights = nn.Parameter(
                torch.tensor([2.0, 3.0, 1.5, 2.5], dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'subweights',
                torch.tensor([2.0, 3.0, 1.5, 2.5], dtype=torch.float32)
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
        Calculate E_flow energy with all validation components.

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

        # Component 1: Sink Risk Energy (exfiltration detection)
        E_sink = self._compute_sink_risk_energy(nodes, edges)

        # Component 2: Topology Energy (graph structure)
        E_topo = self.topology_validator(nodes, edges)

        # Component 3: Transformation Energy (data coherence)
        E_transform = self.transformation_validator(nodes, edges)

        # Component 4: Provenance Energy (trust tracking)
        E_prov = self.provenance_validator(nodes, edges)

        # Weighted composition
        w_sink, w_topo, w_transform, w_prov = self.subweights
        total_energy = (
            w_sink * E_sink.squeeze() +
            w_topo * E_topo.squeeze() +
            w_transform * E_transform.squeeze() +
            w_prov * E_prov.squeeze()
        )

        # Optional latent modulation
        if self.use_latent_modulation and z_g is not None and z_e is not None:
            combined = torch.cat([z_g, z_e], dim=-1)
            modulation = self.latent_modulation(combined).squeeze()
            total_energy = total_energy * modulation

        return total_energy.unsqueeze(0)

    def _compute_sink_risk_energy(
        self,
        nodes: list[ToolCallNode],
        edges: list[tuple[str, str]]
    ) -> torch.Tensor:
        """
        Calculate sink node exfiltration risk energy (original E_flow logic).

        Returns:
            [1] scalar energy
        """
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

        return total_energy.unsqueeze(0)

    def explain(
        self,
        plan: ExecutionPlan | dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate comprehensive explanation of data flow violations.

        Returns:
            {
                'total_energy': float,
                'component_energies': {
                    'sink_risk': float,
                    'topology': float,
                    'transformation': float,
                    'provenance': float
                },
                'weighted_contributions': {
                    'sink_risk': float,
                    'topology': float,
                    'transformation': float,
                    'provenance': float
                },
                'subweights': [w_sink, w_topo, w_transform, w_prov],

                # Sink risk details
                'sink_nodes': [...],
                'high_risk_sinks': [node_id, ...],

                # Topology details
                'topology_violations': {
                    'has_cycles': bool,
                    'cycle_paths': [[node_id, ...], ...],
                    'num_components': int,
                    'invalid_edges': [(src, dst), ...]
                },

                # Transformation details
                'transformation_violations': {
                    'volume_inconsistencies': [...],
                    'suspicious_chains': [...]
                },

                # Provenance details
                'provenance_violations': {
                    'tier_mismatches': [...],
                    'sensitivity_escalations': [...]
                },

                # Overall assessment
                'risk_assessment': 'safe' | 'suspicious' | 'critical',
                'recommendations': [str, ...]
            }
        """
        if not isinstance(plan, ExecutionPlan):
            plan = ExecutionPlan(**plan)

        nodes = plan.nodes
        edges = plan.edges

        if len(nodes) == 0:
            return {
                'total_energy': 0.0,
                'component_energies': {},
                'weighted_contributions': {},
                'sink_nodes': [],
                'high_risk_sinks': [],
                'topology_violations': {},
                'transformation_violations': {},
                'provenance_violations': {},
                'risk_assessment': 'safe',
                'recommendations': []
            }

        # Compute all component energies
        E_sink = self._compute_sink_risk_energy(nodes, edges).item()
        E_topo = self.topology_validator(nodes, edges).item()
        E_transform = self.transformation_validator(nodes, edges).item()
        E_prov = self.provenance_validator(nodes, edges).item()

        # Get subweights
        w_sink, w_topo, w_transform, w_prov = self.subweights.tolist()

        # Calculate weighted contributions
        weighted_sink = w_sink * E_sink
        weighted_topo = w_topo * E_topo
        weighted_transform = w_transform * E_transform
        weighted_prov = w_prov * E_prov

        total_energy = weighted_sink + weighted_topo + weighted_transform + weighted_prov

        # === Sink Risk Details ===
        outgoing_counts = {node.node_id: 0 for node in nodes}
        for src, dst in edges:
            outgoing_counts[src] += 1

        sink_nodes = [node for node in nodes if outgoing_counts[node.node_id] == 0]
        sink_info = []
        high_risk_sinks = []

        if len(sink_nodes) > 0:
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

            for i, node in enumerate(sink_nodes):
                energy = float(final_energies[i].detach())
                info = {
                    'node_id': node.node_id,
                    'tool_name': node.tool_name,
                    'destination_risk': float(dest_risks[i].detach()),
                    'data_sensitivity': int(node.scope_sensitivity),
                    'data_volume': int(node.scope_volume),
                    'is_suspicious_transform': self._is_transform_suspicious(node.tool_name),
                    'energy_contribution': energy
                }
                sink_info.append(info)
                if energy > 5.0:
                    high_risk_sinks.append(node.node_id)

        # === Topology Details ===
        has_cycles, cycle_paths = self.topology_validator.detect_cycles(nodes, edges)
        num_components = self.topology_validator.check_connectivity(nodes, edges)
        invalid_edges = self.topology_validator.validate_edge_references(nodes, edges)

        topology_violations = {
            'has_cycles': has_cycles,
            'cycle_paths': cycle_paths,
            'num_components': num_components,
            'invalid_edges': invalid_edges
        }

        # === Transformation Details ===
        volume_violations = self.transformation_validator.check_volume_consistency(nodes, edges)
        suspicious_chains = self.transformation_validator.detect_suspicious_chains(nodes, edges)

        transformation_violations = {
            'volume_inconsistencies': volume_violations,
            'suspicious_chains': suspicious_chains
        }

        # === Provenance Details ===
        tier_violations = self.provenance_validator.track_provenance_degradation(nodes, edges)
        sensitivity_escalations = self.provenance_validator.detect_sensitivity_escalation(nodes, edges)

        provenance_violations = {
            'tier_mismatches': tier_violations,
            'sensitivity_escalations': sensitivity_escalations
        }

        # === Risk Assessment ===
        if total_energy > 20.0:
            risk_assessment = 'critical'
        elif total_energy > 8.0:
            risk_assessment = 'suspicious'
        else:
            risk_assessment = 'safe'

        # === Recommendations ===
        recommendations = []

        if has_cycles:
            recommendations.append(f"CRITICAL: Execution plan contains {len(cycle_paths)} circular dependencies. Remove cycles to prevent infinite loops.")

        if len(invalid_edges) > 0:
            recommendations.append(f"ERROR: {len(invalid_edges)} edges reference non-existent nodes. Fix graph construction.")

        if num_components > 1:
            recommendations.append(f"WARNING: Execution plan has {num_components} disconnected components. Verify plan completeness.")

        if len(high_risk_sinks) > 0:
            recommendations.append(f"SECURITY: {len(high_risk_sinks)} sink nodes pose exfiltration risk. Review external destinations.")

        if len(suspicious_chains) > 0:
            recommendations.append(f"SECURITY: {len(suspicious_chains)} suspicious transformation chains detected (e.g., encrypt→compress→send).")

        if len(tier_violations) > 0:
            high_tier_violations = [v for v in tier_violations if v['penalty'] >= 5.0]
            if high_tier_violations:
                recommendations.append(f"SECURITY: {len(high_tier_violations)} critical trust tier violations (Tier 3 → Tier 1 flows).")

        if len(volume_violations) > 0:
            recommendations.append(f"WARNING: {len(volume_violations)} data volume inconsistencies detected. Verify transformation logic.")

        if not recommendations:
            recommendations.append("Plan passes all data flow consistency checks.")

        return {
            'total_energy': total_energy,
            'component_energies': {
                'sink_risk': E_sink,
                'topology': E_topo,
                'transformation': E_transform,
                'provenance': E_prov
            },
            'weighted_contributions': {
                'sink_risk': weighted_sink,
                'topology': weighted_topo,
                'transformation': weighted_transform,
                'provenance': weighted_prov
            },
            'subweights': [w_sink, w_topo, w_transform, w_prov],
            'sink_nodes': sink_info,
            'high_risk_sinks': high_risk_sinks,
            'topology_violations': topology_violations,
            'transformation_violations': transformation_violations,
            'provenance_violations': provenance_violations,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations
        }


def create_flow_energy(
    use_latent_modulation: bool = True,
    learnable_subweights: bool = True,
    checkpoint_path: str | None = None,
    device: str = "cpu"
) -> FlowEnergy:
    """
    Factory function for E_flow with data flow consistency validation.

    Args:
        use_latent_modulation: Condition on (z_g, z_e) latents
        learnable_subweights: Train component composition weights
        checkpoint_path: Pretrained weights
        device: Target device

    Returns:
        Initialized FlowEnergy module with all validators

    Example:
        >>> flow_energy = create_flow_energy()
        >>> energy = flow_energy(plan, z_g, z_e)
        >>> explanation = flow_energy.explain(plan)
    """
    model = FlowEnergy(
        use_latent_modulation=use_latent_modulation,
        learnable_subweights=learnable_subweights
    )

    if checkpoint_path is not None:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    model = model.to(device)
    model.training = False

    return model
