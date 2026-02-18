"""
Research Planner Agent for Project Gatling

This is the orchestrator agent that:
- Decomposes research workstreams into concrete implementation tasks
- Manages the task dependency graph
- Routes tasks to appropriate implementer agents
- Tracks progress against the 12-week timeline
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import networkx as nx
from datetime import datetime, timedelta
from agents.base_agent import GatlingAgent, TaskStatus


class ResearchPlannerAgent(GatlingAgent):
    """
    The Research Planner Agent orchestrates the entire multi-agent system.
    
    This agent doesn't implement code - instead, it creates the task breakdown
    and coordinates the execution of other agents.
    """
    
    def __init__(self):
        super().__init__("research_planner", "coordination")
        
        self.timeline = self._load_timeline()
        self.task_graph = nx.DiGraph()
        
    def _load_timeline(self) -> Dict[str, Any]:
        """Load the 12-week project timeline"""
        return {
            "weeks_1_3": {
                "phase": "foundation",
                "workstreams": ["latent_substrate"],
                "goal": "Train JEPA encoders on 100M benign execution logs"
            },
            "weeks_4_6": {
                "phase": "composition",
                "workstreams": ["energy_geometry", "provenance"],
                "goal": "Build and weight the four core energy terms"
            },
            "weeks_7_9": {
                "phase": "refinement",
                "workstreams": ["red_team", "dataset"],
                "goal": "Generate 10k hard negatives and optimize energy landscape"
            },
            "weeks_10_12": {
                "phase": "integration",
                "workstreams": ["integration"],
                "goal": "Deploy Repair Engine and benchmark against Kona suite"
            }
        }
    
    def decompose_workstream(self, workstream: str, phase: str) -> List[Dict[str, Any]]:
        """
        Break a workstream into concrete implementation tasks.
        
        Args:
            workstream: Name of workstream (latent_substrate, energy_geometry, etc.)
            phase: Current project phase (foundation, composition, etc.)
        
        Returns:
            List of task specifications
        """
        self.logger.info(f"Decomposing {workstream} workstream for phase {phase}")
        
        if workstream == "latent_substrate" and phase == "foundation":
            return self._decompose_latent_substrate()
        elif workstream == "energy_geometry" and phase == "composition":
            return self._decompose_energy_geometry()
        elif workstream == "provenance" and phase == "composition":
            return self._decompose_provenance()
        elif workstream == "red_team" and phase == "refinement":
            return self._decompose_red_team()
        elif workstream == "dataset" and phase == "refinement":
            return self._decompose_dataset()
        elif workstream == "integration" and phase == "integration":
            return self._decompose_integration()
        else:
            raise ValueError(f"Unknown workstream/phase: {workstream}/{phase}")
    
    def _decompose_latent_substrate(self) -> List[Dict[str, Any]]:
        """Decompose Latent Substrate workstream (Weeks 1-3)"""
        return [
            {
                "id": "LSA-001",
                "agent": "latent_substrate",
                "title": "Implement GovernanceEncoder",
                "description": """
                Build the transformer-based encoder that maps (policy, user_role, session_context)
                into a 1024-dimensional Governance Latent vector.
                
                Requirements:
                - Input: Policy schema (JSON/YAML), user role, session metadata
                - Output: z_g ∈ R^1024 (governance latent)
                - Architecture: Transformer encoder with policy-aware attention
                - Latency: <50ms inference on CPU
                - Differentiable: Must support backpropagation
                """,
                "dependencies": [],
                "acceptance_criteria": {
                    "code_quality": ["Type hints", "Docstrings", "PEP8 compliant"],
                    "functionality": [
                        "Encodes policy into 1024-dim vector",
                        "Passes gradient flow test",
                        "Handles variable-length policy schemas"
                    ],
                    "performance": ["<50ms inference latency", "<500MB memory"],
                    "tests": ["Unit tests >90% coverage", "Integration test with mock policy"]
                },
                "estimated_tokens": 50000,
                "priority": "high",
                "downstream_consumers": ["energy_geometry", "integration"],
                "outputs_required": [
                    "source/encoders/governance_encoder.py",
                    "test/test_governance_encoder.py",
                    "docs/encoders/governance_encoder.md"
                ]
            },
            {
                "id": "LSA-002",
                "agent": "latent_substrate",
                "title": "Implement ExecutionEncoder",
                "description": """
                Build the encoder that maps (proposed_plan, provenance_metadata)
                into a 1024-dimensional Execution Latent vector.
                
                Requirements:
                - Input: Tool-call graph with provenance tags
                - Output: z_e ∈ R^1024 (execution latent)
                - Architecture: Graph Neural Network or Transformer over plan DAG
                - Must preserve ordering and data flow semantics
                """,
                "dependencies": ["LSA-001"],  # Can learn from governance encoder patterns
                "acceptance_criteria": {
                    "code_quality": ["Type hints", "Docstrings", "PEP8 compliant"],
                    "functionality": [
                        "Encodes plan graph into 1024-dim vector",
                        "Preserves data flow dependencies",
                        "Handles variable plan lengths (1-100 nodes)"
                    ],
                    "performance": ["<50ms inference latency"],
                    "tests": ["Unit tests >90% coverage", "Graph encoding tests"]
                },
                "estimated_tokens": 45000,
                "priority": "high",
                "downstream_consumers": ["energy_geometry", "integration"]
            },
            {
                "id": "LSA-003",
                "agent": "latent_substrate",
                "title": "Implement Semantic Intent Predictor",
                "description": """
                Build a model that predicts the 'minimal scope budget' from user intent.
                This serves as the reference point for E_scope (least privilege energy term).
                
                Requirements:
                - Input: User message + tool schema
                - Output: Predicted minimal scope (e.g., limit=5 for "find invoice")
                - Used as ground truth for E_scope calculation
                """,
                "dependencies": ["LSA-001"],
                "acceptance_criteria": {
                    "functionality": [
                        "Predicts minimal data scope from user query",
                        "Outputs structured scope constraints"
                    ],
                    "performance": ["<100ms inference"],
                    "tests": ["Tested on 1000 diverse queries"]
                },
                "estimated_tokens": 35000,
                "priority": "medium",
                "downstream_consumers": ["energy_geometry"]
            },
            {
                "id": "LSA-004",
                "agent": "latent_substrate",
                "title": "Train JEPA Encoders on Benign Traces",
                "description": """
                Train the dual encoders using contrastive learning on benign execution logs.
                
                Requirements:
                - Dataset: 100M benign tool-use traces (to be provided by dataset team)
                - Training: InfoNCE loss, align (policy, execution) pairs
                - Validation: Embedding space visualization, nearest neighbor accuracy
                """,
                "dependencies": ["LSA-001", "LSA-002"],
                "acceptance_criteria": {
                    "functionality": [
                        "Training pipeline runs end-to-end",
                        "Embeddings cluster by policy type",
                        "Nearest neighbor accuracy >85%"
                    ],
                    "performance": ["Training completes in <24 hours on single GPU"]
                },
                "estimated_tokens": 40000,
                "priority": "high",
                "downstream_consumers": ["energy_geometry", "red_team"]
            }
        ]
    
    def _decompose_energy_geometry(self) -> List[Dict[str, Any]]:
        """Decompose Energy Geometry workstream (Weeks 4-6)"""
        return [
            {
                "id": "EGA-001",
                "agent": "energy_geometry",
                "title": "Implement E_hierarchy Energy Term",
                "description": """
                Build the energy critic that penalizes when untrusted retrieved data
                influences control flow.
                
                E_hierarchy(z_g, z_e) = ||trust_tier_embedding(z_e) - allowed_tier(z_g)||^2
                
                Requirements:
                - Differentiable function of (z_g, z_e)
                - Spikes when RAG snippets (Tier 3) control high-privilege tools
                - Returns scalar energy value
                """,
                "dependencies": ["LSA-001", "LSA-002"],
                "acceptance_criteria": {
                    "functionality": [
                        "Differentiable w.r.t. both inputs",
                        "Higher energy for hierarchy violations",
                        "Validated on synthetic violation examples"
                    ],
                    "performance": ["<5ms evaluation time"]
                },
                "estimated_tokens": 30000,
                "priority": "high",
                "downstream_consumers": ["integration"]
            },
            {
                "id": "EGA-002",
                "agent": "energy_geometry",
                "title": "Implement E_provenance Energy Term",
                "description": """
                Build the Trust Gap detector: measures Tool Privilege × Source Untrustworthiness.
                
                E_provenance = privilege(tool) × (1 - trust_score(source))
                
                Requirements:
                - Extract tool privilege from z_e metadata
                - Extract source trust tier from provenance tags
                - Return high energy when high-privilege tool uses untrusted source
                """,
                "dependencies": ["LSA-002"],
                "acceptance_criteria": {
                    "functionality": [
                        "Correctly extracts privilege and trust scores",
                        "Spikes on provenance rug-pull attacks",
                        "Differentiable"
                    ]
                },
                "estimated_tokens": 25000,
                "priority": "high",
                "downstream_consumers": ["integration"]
            },
            {
                "id": "EGA-003",
                "agent": "energy_geometry",
                "title": "Implement E_scope Energy Term",
                "description": """
                Build the Least Privilege detector: penalizes over-scoped data access.
                
                E_scope = max(0, actual_scope - minimal_scope)^2
                
                Requirements:
                - Uses Semantic Intent Predictor (LSA-003) for minimal_scope
                - Extracts actual_scope from execution plan
                - Returns energy proportional to over-scoping
                """,
                "dependencies": ["LSA-002", "LSA-003"],
                "acceptance_criteria": {
                    "functionality": [
                        "Correctly identifies scope blow-up attacks",
                        "Zero energy when scope equals minimal",
                        "Differentiable"
                    ]
                },
                "estimated_tokens": 25000,
                "priority": "high",
                "downstream_consumers": ["integration"]
            },
            {
                "id": "EGA-004",
                "agent": "energy_geometry",
                "title": "Implement E_flow Energy Term",
                "description": """
                Build the Exfiltration Detector: detects data flow to unauthorized endpoints.
                
                E_flow = semantic_distance(inferred_intent, actual_data_flow)
                
                Requirements:
                - Analyzes data flow graph in execution plan
                - Compares against expected flow from governance policy
                - Returns high energy for unexpected exfiltration steps
                """,
                "dependencies": ["LSA-001", "LSA-002"],
                "acceptance_criteria": {
                    "functionality": [
                        "Detects exfiltration pivot attacks",
                        "Low energy for legitimate flows",
                        "Differentiable"
                    ]
                },
                "estimated_tokens": 35000,
                "priority": "high",
                "downstream_consumers": ["integration"]
            },
            {
                "id": "EGA-005",
                "agent": "energy_geometry",
                "title": "Implement Composite Energy Function",
                "description": """
                Combine the four energy critics into a Product of Experts:
                
                E(z_g, z_e) = α₁·E_hierarchy + α₂·E_provenance + α₃·E_scope + α₄·E_flow
                
                Requirements:
                - Learnable or calibrated weights α_i
                - Scaling normalization to prevent dominance
                - Spectral regularization for landscape smoothness
                """,
                "dependencies": ["EGA-001", "EGA-002", "EGA-003", "EGA-004"],
                "acceptance_criteria": {
                    "functionality": [
                        "All four terms contribute meaningfully",
                        "Weights are calibrated or learned",
                        "Differentiable end-to-end"
                    ],
                    "performance": ["<20ms total energy computation"]
                },
                "estimated_tokens": 30000,
                "priority": "critical",
                "downstream_consumers": ["integration", "red_team"]
            }
        ]
    
    def _decompose_provenance(self) -> List[Dict[str, Any]]:
        """Decompose Provenance & Trust workstream (Weeks 4-6)"""
        return [
            {
                "id": "PA-001",
                "agent": "provenance",
                "title": "Implement Trust Tier Tagging System",
                "description": """
                Build the cryptographic tagging system for data provenance.
                
                Trust Tiers:
                - Tier 1: Internal verified sources
                - Tier 2: Signed partner APIs
                - Tier 3: Public web / RAG snippets
                
                Requirements:
                - Tag every data retrieval with trust tier
                - Cryptographic verification of tier assignment
                - Propagate tiers through data flow
                """,
                "dependencies": [],
                "acceptance_criteria": {
                    "functionality": [
                        "Tags all data sources with tier",
                        "Prevents tier spoofing",
                        "Efficient tier propagation"
                    ],
                    "security": ["Cryptographically secure tier verification"]
                },
                "estimated_tokens": 35000,
                "priority": "high",
                "downstream_consumers": ["energy_geometry", "integration"]
            },
            {
                "id": "PA-002",
                "agent": "provenance",
                "title": "Implement Discrete Repair Engine",
                "description": """
                Build the Greedy + Beam Search algorithm for plan repair.
                
                Algorithm:
                1. If E(z_g, z_e) > θ_safe:
                2. Generate candidate edits (narrowing, de-escalation, pruning)
                3. Select edit with max energy reduction
                4. Repeat until E < θ_safe or max_steps reached
                
                Requirements:
                - Atomic edit operations (swap tool, narrow scope, prune node)
                - Energy-guided search
                - Functional Intent Similarity (FIS) preservation >0.9
                """,
                "dependencies": ["EGA-005", "PA-001"],
                "acceptance_criteria": {
                    "functionality": [
                        "Reduces energy below threshold",
                        "Preserves FIS >90%",
                        "Terminates in <200ms"
                    ],
                    "correctness": ["Tested on 1000 malicious plans"]
                },
                "estimated_tokens": 50000,
                "priority": "critical",
                "downstream_consumers": ["integration"]
            },
            {
                "id": "PA-003",
                "agent": "provenance",
                "title": "Implement Fast-Path Policy Distillation",
                "description": """
                Distill common successful repairs into a fast-path policy.
                
                Strategy:
                - Cache (malicious_plan, repaired_plan) pairs
                - Train lightweight classifier: should_narrow_scope(plan) -> bool
                - Use for <10ms fast path, fall back to full EBM for novel cases
                
                Requirements:
                - Fast-path covers 80% of common violations
                - Full EBM validates all fast-path decisions
                - Prevents policy drift
                """,
                "dependencies": ["PA-002"],
                "acceptance_criteria": {
                    "performance": [
                        "<10ms for cached repairs",
                        "80% cache hit rate on validation set"
                    ],
                    "safety": ["Zero false positives from fast-path"]
                },
                "estimated_tokens": 35000,
                "priority": "medium",
                "downstream_consumers": ["integration"]
            }
        ]
    
    def _decompose_red_team(self) -> List[Dict[str, Any]]:
        """Decompose Red Team workstream (Weeks 7-9)"""
        return [
            {
                "id": "RTA-001",
                "agent": "red_team",
                "title": "Implement Plan-Injection Corrupter Agent",
                "description": """
                Build the automated adversarial mutation engine.
                
                Implements mutations from catalog:
                1. Scope Blow-up: limit=5 -> limit=10000
                2. Instruction Shadowing: Inject RAG content into control flow
                3. Provenance Rug-Pull: Swap trusted source for untrusted
                4. Exfiltration Pivot: Append data export step
                
                Requirements:
                - Applies mutations to gold traces
                - Generates "hard negatives" for training
                - Mutation difficulty calibration
                """,
                "dependencies": [],
                "acceptance_criteria": {
                    "functionality": [
                        "Generates all 4 mutation types",
                        "Produces valid (syntactically correct) plans",
                        "10k hard negatives in <1 hour"
                    ]
                },
                "estimated_tokens": 45000,
                "priority": "high",
                "downstream_consumers": ["dataset", "integration"]
            },
            {
                "id": "RTA-002",
                "agent": "red_team",
                "title": "Implement InfoNCE Training Loop",
                "description": """
                Build the contrastive training loop with hard negative mining.
                
                Training procedure:
                - Positive: (z_g, z_e_safe) pairs from gold traces
                - Negative: (z_g, z_e_malicious) from Corrupter Agent
                - Loss: InfoNCE with temperature τ
                - Hard negative mining: Prioritize samples where current EBM fails
                
                Requirements:
                - Integrates with existing encoders (LSA-001, LSA-002)
                - Integrates with energy function (EGA-005)
                - Maintains security margin δ_sec
                """,
                "dependencies": ["RTA-001", "LSA-004", "EGA-005"],
                "acceptance_criteria": {
                    "functionality": [
                        "Training loop converges",
                        "Energy gap δ_sec remains stable",
                        "AUC >0.999 on validation set"
                    ],
                    "performance": ["Training completes in <48 hours"]
                },
                "estimated_tokens": 40000,
                "priority": "critical",
                "downstream_consumers": ["integration"]
            }
        ]
    
    def _decompose_dataset(self) -> List[Dict[str, Any]]:
        """Decompose Dataset workstream (Weeks 7-9)"""
        return [
            {
                "id": "DA-001",
                "agent": "dataset",
                "title": "Generate Standard Utility Gold Traces (4M)",
                "description": """
                Use Oracle Agents (GPT-4, Claude Opus) to generate safe execution traces.
                
                Requirements:
                - 50+ tool domains (Finance, HR, DevOps, etc.)
                - Diverse user requests
                - 100% policy-compliant
                - Human verification sample (1000 traces)
                
                Output: 4M safe (policy, plan) pairs
                """,
                "dependencies": [],
                "acceptance_criteria": {
                    "quality": ["100% policy compliant", "Diverse coverage"],
                    "volume": ["4M traces generated"],
                    "validation": ["1000 human-verified samples"]
                },
                "estimated_tokens": 30000,
                "priority": "high",
                "downstream_consumers": ["red_team", "integration"]
            },
            {
                "id": "DA-002",
                "agent": "dataset",
                "title": "Generate Policy Boundary Traces (2M)",
                "description": """
                Generate "almost safe" plans that violate subtle boundaries.
                
                These are critical for calibrating the security margin δ_sec.
                
                Requirements:
                - Plans that are syntactically valid but violate policy
                - Subtle violations (not obvious jailbreaks)
                - Labeled with specific violation type
                """,
                "dependencies": ["DA-001"],
                "acceptance_criteria": {
                    "quality": ["Subtle, realistic violations"],
                    "volume": ["2M traces"],
                    "labeling": ["Each trace labeled with violation type"]
                },
                "estimated_tokens": 25000,
                "priority": "high",
                "downstream_consumers": ["red_team", "integration"]
            },
            {
                "id": "DA-003",
                "agent": "dataset",
                "title": "Label Minimal Scope Budgets",
                "description": """
                For each trace, label the minimal data scope required.
                
                Used as ground truth for E_scope energy term.
                
                Requirements:
                - Automated labeling using Semantic Intent Predictor (LSA-003)
                - Human verification of edge cases
                - Structured scope constraints (limit, date_range, fields)
                """,
                "dependencies": ["DA-001", "LSA-003"],
                "acceptance_criteria": {
                    "coverage": ["All 10M traces labeled"],
                    "accuracy": [">95% agreement with human labels (on sample)"]
                },
                "estimated_tokens": 20000,
                "priority": "medium",
                "downstream_consumers": ["integration"]
            }
        ]
    
    def _decompose_integration(self) -> List[Dict[str, Any]]:
        """Decompose Integration workstream (Weeks 10-12)"""
        return [
            {
                "id": "IA-001",
                "agent": "integration",
                "title": "Build End-to-End Gatling Pipeline",
                "description": """
                Integrate all components into unified system:
                1. Policy + Plan input
                2. Dual encoder forward pass
                3. Energy computation
                4. Repair if E > θ_safe
                5. Output repaired plan
                
                Requirements:
                - <200ms latency budget
                - All components working together
                - Comprehensive error handling
                """,
                "dependencies": ["LSA-004", "EGA-005", "PA-002"],
                "acceptance_criteria": {
                    "functionality": ["E2E pipeline executes successfully"],
                    "performance": ["<200ms p99 latency"],
                    "reliability": ["Handles errors gracefully"]
                },
                "estimated_tokens": 35000,
                "priority": "critical",
                "downstream_consumers": []
            },
            {
                "id": "IA-002",
                "agent": "integration",
                "title": "Run Kona Safety Benchmark Suite",
                "description": """
                Evaluate Gatling on comprehensive safety benchmarks.
                
                Metrics:
                1. Energy Mismatch Recall: AUC for safe vs. malicious separation
                2. Functional Intent Similarity: Semantic overlap of repairs
                3. Privilege Reduction: % reduction in sensitive access
                
                Requirements:
                - Run on 10k test set
                - Generate detailed performance report
                - Identify failure modes
                """,
                "dependencies": ["IA-001"],
                "acceptance_criteria": {
                    "safety": ["AUC >0.999", "Zero false positives on admin tasks"],
                    "utility": ["FIS >0.90", "Privilege reduction >70%"]
                },
                "estimated_tokens": 30000,
                "priority": "critical",
                "downstream_consumers": []
            }
        ]
    
    def populate_task_queue(self, tasks: List[Dict[str, Any]]):
        """
        Add tasks to the global task queue.
        
        Args:
            tasks: List of task specifications
        """
        # Load existing queue or create new
        if self.task_queue_path.exists():
            with open(self.task_queue_path) as f:
                queue = json.load(f)
        else:
            queue = {
                "pending": [],
                "ready": [],
                "in_progress": [],
                "review": [],
                "completed": [],
                "failed": []
            }
        
        # Add tasks to appropriate status
        for task in tasks:
            # Tasks with no dependencies are immediately ready
            if not task.get("dependencies", []):
                status = "ready"
            else:
                status = "pending"
            
            # Add metadata
            task["status"] = status
            task["created_at"] = datetime.now().isoformat()
            task["created_by"] = self.name
            
            queue[status].append(task)
            
            # Add to dependency graph
            self.task_graph.add_node(task["id"], **task)
            for dep in task.get("dependencies", []):
                self.task_graph.add_edge(dep, task["id"])
        
        # Write queue
        self.task_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.task_queue_path, 'w') as f:
            json.dump(queue, f, indent=2)
        
        self.logger.info(f"Added {len(tasks)} tasks to queue")
    
    def get_ready_tasks(self) -> List[str]:
        """
        Find tasks that are ready to execute (all dependencies complete).
        
        Returns:
            List of task IDs
        """
        with open(self.task_queue_path) as f:
            queue = json.load(f)
        
        # Return tasks already in ready status
        ready_task_ids = [t["id"] for t in queue.get("ready", [])]
        
        # Also check pending tasks whose dependencies are now complete
        completed = {t["id"] for t in queue.get("completed", [])}
        pending = queue.get("pending", [])
        
        for task in pending:
            deps = set(task.get("dependencies", []))
            if deps.issubset(completed):
                ready_task_ids.append(task["id"])
        
        return ready_task_ids
    
    def create_dependency_graph_visualization(self, output_path: Path):
        """
        Create a visualization of the task dependency graph.
        
        Args:
            output_path: Where to save the graph image
        """
        import matplotlib.pyplot as plt
        
        # Create layout
        pos = nx.spring_layout(self.task_graph, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(20, 12))
        
        # Color nodes by workstream
        workstream_colors = {
            "latent_substrate": "#FF6B6B",
            "energy_geometry": "#4ECDC4",
            "provenance": "#45B7D1",
            "red_team": "#FFA07A",
            "dataset": "#98D8C8",
            "integration": "#FFD93D"
        }
        
        node_colors = [
            workstream_colors.get(self.task_graph.nodes[node].get("agent", ""), "#CCCCCC")
            for node in self.task_graph.nodes()
        ]
        
        nx.draw(
            self.task_graph,
            pos,
            node_color=node_colors,
            node_size=3000,
            font_size=8,
            font_weight="bold",
            with_labels=True,
            arrows=True,
            edge_color="#888888",
            arrowsize=20
        )
        
        plt.title("Project Gatling: Task Dependency Graph", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Saved dependency graph to {output_path}")
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """
        Generate current project progress report.
        
        Returns:
            Progress statistics and status
        """
        with open(self.task_queue_path) as f:
            queue = json.load(f)
        
        total = sum(len(queue[status]) for status in queue.keys())
        completed = len(queue["completed"])
        in_progress = len(queue["in_progress"])
        ready = len(queue["ready"])
        blocked = len(queue["pending"])
        
        # Calculate per-workstream progress
        workstream_progress = {}
        for workstream in ["latent_substrate", "energy_geometry", "provenance", 
                          "red_team", "dataset", "integration"]:
            ws_tasks = [
                t for status_tasks in queue.values()
                for t in status_tasks
                if t.get("agent") == workstream
            ]
            ws_completed = [t for t in queue["completed"] if t.get("agent") == workstream]
            
            if ws_tasks:
                workstream_progress[workstream] = {
                    "total": len(ws_tasks),
                    "completed": len(ws_completed),
                    "percentage": len(ws_completed) / len(ws_tasks) * 100
                }
        
        report = {
            "overall_progress": completed / total * 100 if total > 0 else 0,
            "tasks": {
                "total": total,
                "completed": completed,
                "in_progress": in_progress,
                "ready": ready,
                "blocked": blocked
            },
            "workstream_progress": workstream_progress,
            "ready_tasks": self.get_ready_tasks(),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Research Planner doesn't execute implementation tasks.
        Instead it coordinates other agents.
        """
        raise NotImplementedError(
            "ResearchPlannerAgent doesn't execute tasks. "
            "Use decompose_workstream() and populate_task_queue() instead."
        )


if __name__ == "__main__":
    # Example usage
    planner = ResearchPlannerAgent()
    
    # Decompose Week 1-3 work
    tasks = planner.decompose_workstream("latent_substrate", "foundation")
    planner.populate_task_queue(tasks)
    
    # Generate progress report
    report = planner.generate_progress_report()
    print(json.dumps(report, indent=2))
    
    # Create dependency visualization
    planner.create_dependency_graph_visualization(Path("outputs/task_graph.png"))