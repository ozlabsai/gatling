#!/usr/bin/env python3
"""
Project Gatling Multi-Agent System - Quickstart Script

This script initializes the multi-agent infrastructure and kicks off
the first wave of implementation tasks.

Usage:
    uv run python quickstart.py --phase foundation
    uv run python quickstart.py --phase composition --parallel
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.research_planner import ResearchPlannerAgent


def create_directory_structure():
    """Create the complete directory structure for the project"""
    directories = [
        "agents",
        "tasks",
        "outputs/latent_substrate",
        "outputs/energy_geometry",
        "outputs/provenance",
        "outputs/red_team",
        "outputs/dataset",
        "outputs/integration",
        "handoffs/reviews",
        "acceptance_criteria",
        "source/encoders",
        "source/energy",
        "source/repair",
        "source/provenance",
        "test/test_agents",
        "test/test_encoders",
        "test/test_energy",
        "test/test_integration",
        "docs/encoders",
        "docs/energy",
        "docs/provenance",
        "logs",
        "config"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure")


def create_acceptance_criteria():
    """Create acceptance criteria files for each component type"""
    
    # Encoders acceptance criteria
    encoder_criteria = {
        "code_quality": {
            "required": [
                "Type hints for all functions",
                "Comprehensive docstrings",
                "PEP8 compliant",
                "No hardcoded values"
            ]
        },
        "functionality": {
            "required": [
                "Produces correct output shape",
                "Handles edge cases",
                "Validates inputs",
                "Differentiable (gradient flow test)"
            ]
        },
        "performance": {
            "required": [
                "Inference latency < 50ms on CPU",
                "Memory usage < 500MB",
                "Batch processing supported"
            ]
        },
        "tests": {
            "required": [
                "Unit test coverage > 90%",
                "Integration test with mock data",
                "Gradient flow test",
                "Edge case tests"
            ]
        }
    }
    
    # Energy functions acceptance criteria
    energy_criteria = {
        "code_quality": {
            "required": [
                "Type hints for all functions",
                "Mathematical notation in docstrings",
                "PEP8 compliant"
            ]
        },
        "functionality": {
            "required": [
                "Differentiable w.r.t. both inputs",
                "Returns scalar energy value",
                "Correct violation detection on test cases",
                "Handles variable input sizes"
            ]
        },
        "performance": {
            "required": [
                "Evaluation time < 5ms per plan",
                "Vectorized operations where possible"
            ]
        },
        "tests": {
            "required": [
                "Differentiability test",
                "Violation detection tests (TP/TN)",
                "Numerical stability tests",
                "Integration with encoders"
            ]
        }
    }
    
    # Save criteria files
    criteria_dir = Path("acceptance_criteria")
    
    with open(criteria_dir / "encoders.json", 'w') as f:
        json.dump(encoder_criteria, f, indent=2)
    
    with open(criteria_dir / "energy_functions.json", 'w') as f:
        json.dump(energy_criteria, f, indent=2)
    
    print("✓ Created acceptance criteria files")


def initialize_research_planner():
    """Initialize the Research Planner and create initial task breakdown"""
    
    planner = ResearchPlannerAgent()
    
    print("\n=== Initializing Research Planner ===")
    print(f"Created planner: {planner.name}")
    
    return planner


def decompose_phase(planner: ResearchPlannerAgent, phase: str):
    """Decompose tasks for a specific phase"""
    
    print(f"\n=== Decomposing Phase: {phase} ===")
    
    if phase == "foundation":
        workstreams = ["latent_substrate"]
    elif phase == "composition":
        workstreams = ["energy_geometry", "provenance"]
    elif phase == "refinement":
        workstreams = ["red_team", "dataset"]
    elif phase == "integration":
        workstreams = ["integration"]
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    all_tasks = []
    for workstream in workstreams:
        tasks = planner.decompose_workstream(workstream, phase)
        all_tasks.extend(tasks)
        print(f"  - {workstream}: {len(tasks)} tasks")
    
    planner.populate_task_queue(all_tasks)
    
    print(f"\n✓ Added {len(all_tasks)} tasks to queue")
    
    return all_tasks


def show_ready_tasks(planner: ResearchPlannerAgent):
    """Display tasks that are ready to execute"""
    
    ready_tasks = planner.get_ready_tasks()
    
    print("\n=== Ready Tasks ===")
    
    if not ready_tasks:
        print("  No tasks ready (waiting for dependencies)")
        return []
    
    # Load task details
    with open(planner.task_queue_path) as f:
        queue = json.load(f)
    
    ready_task_details = [
        t for t in queue["ready"]
        if t["id"] in ready_tasks
    ]
    
    for task in ready_task_details:
        print(f"\n  [{task['id']}] {task['title']}")
        print(f"    Agent: {task['agent']}")
        print(f"    Priority: {task['priority']}")
        print(f"    Est. tokens: {task['estimated_tokens']:,}")
    
    return ready_task_details


def generate_progress_report(planner: ResearchPlannerAgent):
    """Generate and display progress report"""
    
    report = planner.generate_progress_report()
    
    print("\n=== Project Progress ===")
    print(f"Overall: {report['overall_progress']:.1f}%")
    print(f"\nTasks:")
    print(f"  Total: {report['tasks']['total']}")
    print(f"  Completed: {report['tasks']['completed']}")
    print(f"  In Progress: {report['tasks']['in_progress']}")
    print(f"  Ready: {report['tasks']['ready']}")
    print(f"  Blocked: {report['tasks']['blocked']}")
    
    print(f"\nWorkstream Progress:")
    for ws, progress in report['workstream_progress'].items():
        print(f"  {ws}: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)")
    
    return report


def create_agent_launch_script():
    """Create a helper script for launching individual agents"""
    
    script = """#!/usr/bin/env python3
\"\"\"
Agent Launcher Script

Launch a specific agent to work on a task.

Usage:
    uv run python launch_agent.py latent_substrate LSA-001
    uv run python launch_agent.py energy_geometry EGA-001 --mode exploration
\"\"\"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Launch a Gatling agent")
    parser.add_argument("agent", help="Agent name (latent_substrate, energy_geometry, etc.)")
    parser.add_argument("task_id", help="Task ID to execute")
    parser.add_argument("--mode", default="focused", choices=["focused", "exploration"])
    
    args = parser.parse_args()
    
    print(f"Launching {args.agent} agent for task {args.task_id}")
    print(f"Mode: {args.mode}")
    print()
    print("=" * 60)
    print("AGENT EXECUTION INSTRUCTIONS")
    print("=" * 60)
    print()
    print("This is where you would spawn a new Claude Code session.")
    print("Since this is a coordination script, you'll need to:")
    print()
    print("1. Open a new Claude Code session (terminal or API)")
    print("2. Provide the agent-specific system prompt")
    print("3. Load the task specification")
    print("4. Monitor progress and collect outputs")
    print()
    print(f"Task file: tasks/task_queue.json (find {args.task_id})")
    print(f"Output dir: outputs/{args.agent}/")
    print()
    
    # In a real implementation, this would spawn a Claude Code session
    # using the Anthropic API with the agent-specific system prompt


if __name__ == "__main__":
    main()
"""
    
    with open("launch_agent.py", 'w') as f:
        f.write(script)
    
    Path("launch_agent.py").chmod(0o755)
    
    print("✓ Created launch_agent.py script")


def create_example_agent():
    """Create an example implementer agent to show the pattern"""
    
    example = '''"""
Example Latent Substrate Agent Implementation

This demonstrates how to create a specialized implementer agent.
"""

import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import GatlingAgent


class LatentSubstrateAgent(GatlingAgent):
    """
    Implementer agent for the Latent Substrate workstream.
    
    Responsibilities:
    - Implement GovernanceEncoder
    - Implement ExecutionEncoder
    - Train JEPA dual encoders
    """
    
    def __init__(self):
        super().__init__("latent_substrate_agent", "latent_substrate")
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a latent substrate task.
        
        This is where the actual implementation happens. In a real deployment,
        this would be a Claude Code session that:
        1. Reads the task specification
        2. Implements the required code
        3. Writes tests
        4. Generates documentation
        5. Creates the artifact manifest
        """
        
        task_id = task["id"]
        self.logger.info(f"Executing task {task_id}: {task['title']}")
        
        # Example: LSA-001 - Implement GovernanceEncoder
        if task_id == "LSA-001":
            return self._implement_governance_encoder(task)
        elif task_id == "LSA-002":
            return self._implement_execution_encoder(task)
        elif task_id == "LSA-003":
            return self._implement_intent_predictor(task)
        elif task_id == "LSA-004":
            return self._train_encoders(task)
        else:
            raise ValueError(f"Unknown task: {task_id}")
    
    def _implement_governance_encoder(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement GovernanceEncoder component"""
        
        # This is a placeholder - in reality, this would involve:
        # 1. Designing the transformer architecture
        # 2. Implementing the encoder class
        # 3. Writing comprehensive tests
        # 4. Generating documentation
        
        self.logger.info("Implementing GovernanceEncoder...")
        
        # Create output paths
        code_path = "source/encoders/governance_encoder.py"
        test_path = "test/test_governance_encoder.py"
        docs_path = "docs/encoders/governance_encoder.md"
        
        # Create artifact
        artifact = self.create_artifact(
            task_id=task["id"],
            component_name="GovernanceEncoder",
            outputs={
                "code": code_path,
                "tests": test_path,
                "docs": docs_path
            },
            interface={
                "input_shape": "(batch_size, policy_tokens)",
                "output_shape": "(batch_size, 1024)",
                "latency_p99": "45ms"
            }
        )
        
        self.logger.info("GovernanceEncoder implementation complete")
        
        return artifact
    
    def _implement_execution_encoder(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement ExecutionEncoder component"""
        # Similar to governance encoder
        pass
    
    def _implement_intent_predictor(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement Semantic Intent Predictor"""
        pass
    
    def _train_encoders(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Train JEPA dual encoders"""
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task ID to execute")
    args = parser.parse_args()
    
    agent = LatentSubstrateAgent()
    agent.run(args.task)
'''
    
    with open("agents/latent_substrate_agent.py", 'w') as f:
        f.write(example)
    
    print("✓ Created example agent: agents/latent_substrate_agent.py")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Project Gatling Multi-Agent System"
    )
    parser.add_argument(
        "--phase",
        choices=["foundation", "composition", "refinement", "integration"],
        default="foundation",
        help="Which phase to initialize"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Show parallel execution plan"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PROJECT GATLING - MULTI-AGENT SYSTEM INITIALIZATION")
    print("=" * 70)
    
    # Step 1: Create directory structure
    print("\nStep 1: Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Create acceptance criteria
    print("\nStep 2: Creating acceptance criteria...")
    create_acceptance_criteria()
    
    # Step 3: Initialize Research Planner
    print("\nStep 3: Initializing Research Planner...")
    planner = initialize_research_planner()
    
    # Step 4: Decompose tasks for the specified phase
    print("\nStep 4: Decomposing tasks...")
    tasks = decompose_phase(planner, args.phase)
    
    # Step 5: Show ready tasks
    print("\nStep 5: Identifying ready tasks...")
    ready_tasks = show_ready_tasks(planner)
    
    # Step 6: Generate progress report
    print("\nStep 6: Generating progress report...")
    report = generate_progress_report(planner)
    
    # Step 7: Create helper scripts
    print("\nStep 7: Creating helper scripts...")
    create_agent_launch_script()
    create_example_agent()
    
    # Step 8: Create dependency graph visualization
    print("\nStep 8: Creating dependency graph...")
    planner.create_dependency_graph_visualization(Path("outputs/task_graph.png"))
    
    # Final instructions
    print("\n" + "=" * 70)
    print("INITIALIZATION COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print()
    
    if ready_tasks:
        print("Ready to execute tasks:")
        for task in ready_tasks[:3]:  # Show first 3
            print(f"  • uv run python launch_agent.py {task['agent']} {task['id']}")
        
        if len(ready_tasks) > 3:
            print(f"  ... and {len(ready_tasks) - 3} more")
    else:
        print("No tasks ready yet (check dependencies)")
    
    print()
    print("View progress:")
    print("  • Task queue: tasks/task_queue.json")
    print("  • Dependency graph: outputs/task_graph.png")
    print(f"  • Logs: logs/research_planner.log")
    print()
    
    if args.parallel:
        print("\nParallel Execution Plan:")
        print("  You can run multiple agents simultaneously for independent tasks:")
        
        # Group ready tasks by agent
        from collections import defaultdict
        agents = defaultdict(list)
        for task in ready_tasks:
            agents[task['agent']].append(task['id'])
        
        for agent, task_ids in agents.items():
            print(f"\n  {agent}:")
            for tid in task_ids:
                print(f"    - {tid}")
    
    print()
    print("Documentation:")
    print("  • Architecture: MULTI_AGENT_ARCHITECTURE.md")
    print("  • Project overview: PRD.md")
    print("  • Work distribution: WORK-DISTRIBUTION.md")
    print()


if __name__ == "__main__":
    main()
