#!/usr/bin/env python3
"""
Manual Multi-Agent Runner for Project Gatling

This version works with your Claude.ai subscription instead of requiring API access.
You manually start Claude Code sessions for each task.

Usage:
    # See what tasks are ready
    uv run python agents/manual_runner.py --status

    # Get instructions for next task
    uv run python agents/manual_runner.py --next

    # Mark a task as complete
    uv run python agents/manual_runner.py --complete LSA-001

    # View progress report
    uv run python agents/manual_runner.py --report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.research_planner import ResearchPlannerAgent
from agents.base_agent import TaskStatus


class ManualAgentRunner:
    """
    Manual runner for use with Claude.ai subscription.
    
    You start Claude Code sessions manually and this script helps you:
    - See what tasks are ready
    - Get task instructions to paste into Claude
    - Mark tasks as complete
    - Track progress
    """
    
    def __init__(self):
        self.planner = ResearchPlannerAgent()
    
    def show_status(self):
        """Show current task queue status"""
        print("\n" + "="*70)
        print("TASK QUEUE STATUS")
        print("="*70 + "\n")
        
        with open(self.planner.task_queue_path) as f:
            queue = json.load(f)
        
        # Show counts
        print(f"Total tasks: {sum(len(queue[s]) for s in queue.keys())}")
        print(f"  ✓ Completed: {len(queue['completed'])}")
        print(f"  ⚡ In Progress: {len(queue['in_progress'])}")
        print(f"  ✅ Ready: {len(queue['ready'])}")
        print(f"  ⏸️  Blocked: {len(queue['pending'])}")
        
        # Show ready tasks
        if queue['ready']:
            print(f"\n📋 Ready Tasks (you can start these now):")
            for task in queue['ready']:
                print(f"\n  [{task['id']}] {task['title']}")
                print(f"    Agent: {task['agent']}")
                print(f"    Priority: {task.get('priority', 'medium')}")
                print(f"    Estimated: {task.get('estimated_tokens', 'unknown')} tokens")
        
        # Show in progress
        if queue['in_progress']:
            print(f"\n🔄 In Progress:")
            for task in queue['in_progress']:
                print(f"  [{task['id']}] {task['title']}")
        
        print()
    
    def show_next_task(self):
        """Show detailed instructions for the next task to work on"""
        ready_tasks = self.planner.get_ready_tasks()
        
        if not ready_tasks:
            print("\n❌ No tasks ready!")
            print("   Check dependencies or wait for in-progress tasks to complete.")
            return
        
        # Load task details
        with open(self.planner.task_queue_path) as f:
            queue = json.load(f)
        
        # Get highest priority ready task
        ready_task_details = [
            t for t in queue["ready"]
            if t["id"] in ready_tasks
        ]
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        ready_task_details.sort(
            key=lambda t: priority_order.get(t.get("priority", "medium"), 2)
        )
        
        task = ready_task_details[0]
        
        print("\n" + "="*70)
        print("NEXT TASK TO WORK ON")
        print("="*70 + "\n")
        
        print(f"Task ID: {task['id']}")
        print(f"Title: {task['title']}")
        print(f"Agent: {task['agent']}")
        print(f"Priority: {task.get('priority', 'medium')}")
        print(f"\nDescription:")
        print(task['description'])
        
        # Show dependencies if any
        if task.get('dependencies'):
            print(f"\nDependencies (already complete):")
            for dep_id in task['dependencies']:
                print(f"  ✓ {dep_id}")
        
        # Show required outputs
        print(f"\nRequired Outputs:")
        for output in task.get('outputs_required', []):
            print(f"  • {output}")
        
        # Generate Claude Code prompt
        print("\n" + "="*70)
        print("COPY THIS INTO YOUR CLAUDE CODE SESSION:")
        print("="*70 + "\n")
        
        prompt = f"""Task: {task['id']} - {task['title']}

{task['description']}

Required Outputs:
{chr(10).join('- ' + o for o in task.get('outputs_required', []))}

Acceptance Criteria:
{json.dumps(task.get('acceptance_criteria', {}), indent=2)}

When complete, create an artifact manifest at:
outputs/{task['agent']}/{task['id']}_artifact.json

Artifact format:
{{
    "task_id": "{task['id']}",
    "component": "<ComponentName>",
    "version": "0.1.0",
    "outputs": {{
        "code": "path/to/your/code.py",
        "tests": "path/to/your/tests.py",
        "docs": "path/to/your/docs.md"
    }},
    "interface": {{
        "input_shape": "...",
        "output_shape": "..."
    }},
    "validation_status": "passed"
}}

Begin implementation using UV for all operations.
"""
        
        print(prompt)
        
        # Mark as in progress
        print("\n" + "="*70)
        print("Ready to start? I'll mark this as 'in progress'")
        response = input("Mark as in progress? (y/N): ")
        
        if response.lower() == 'y':
            self.planner.update_task_status(task['id'], TaskStatus.IN_PROGRESS)
            print(f"✓ Marked {task['id']} as IN_PROGRESS")
            print("\nNow paste the prompt above into your Claude Code session!")
        
        print()
    
    def complete_task(self, task_id: str):
        """Mark a task as complete"""
        
        # Check if artifact exists
        task = self.planner.load_task(task_id)
        agent = task['agent']
        
        artifact_path = Path(f"outputs/{agent}/{task_id}_artifact.json")
        
        if not artifact_path.exists():
            print(f"\n❌ ERROR: Artifact not found at {artifact_path}")
            print("\nYou need to create the artifact manifest first!")
            print("Make sure your Claude Code session created:")
            print(f"  outputs/{agent}/{task_id}_artifact.json")
            return
        
        # Load and validate artifact
        with open(artifact_path) as f:
            artifact = json.load(f)
        
        print(f"\n✓ Found artifact: {artifact.get('component', 'Unknown')}")
        
        # Check outputs exist
        missing_outputs = []
        for output_type, path in artifact.get('outputs', {}).items():
            if not Path(path).exists():
                missing_outputs.append(path)
        
        if missing_outputs:
            print(f"\n⚠️  WARNING: Some output files not found:")
            for path in missing_outputs:
                print(f"  ✗ {path}")
            
            response = input("\nMark complete anyway? (y/N): ")
            if response.lower() != 'y':
                return
        
        # Mark as review
        self.planner.update_task_status(task_id, TaskStatus.REVIEW)
        
        print(f"\n✓ Marked {task_id} as REVIEW")
        print("\nTask complete! Running validation...")
        
        # Auto-approve for now (you can add manual review later)
        response = input("Approve and mark COMPLETED? (y/N): ")
        
        if response.lower() == 'y':
            self.planner.update_task_status(task_id, TaskStatus.COMPLETED)
            print(f"✓ Marked {task_id} as COMPLETED")
            
            # Create handoffs for downstream tasks
            from agents.base_agent import GatlingAgent
            
            class TempAgent(GatlingAgent):
                def execute_task(self, task): pass
            
            temp = TempAgent('manual', agent)
            temp._create_handoffs(task, artifact)
            
            print("\n✓ Created handoff documents for downstream tasks")
            
            # Check if new tasks are ready
            ready = self.planner.get_ready_tasks()
            if ready:
                print(f"\n🎉 New tasks are now ready: {', '.join(ready)}")
        
        print()
    
    def show_report(self):
        """Show detailed progress report"""
        report = self.planner.generate_progress_report()
        
        print("\n" + "="*70)
        print("PROJECT PROGRESS REPORT")
        print("="*70 + "\n")
        
        print(f"Overall Progress: {report['overall_progress']:.1f}%")
        
        print(f"\nTasks:")
        print(f"  Total: {report['tasks']['total']}")
        print(f"  ✓ Completed: {report['tasks']['completed']}")
        print(f"  🔄 In Progress: {report['tasks']['in_progress']}")
        print(f"  ✅ Ready: {report['tasks']['ready']}")
        print(f"  ⏸️  Blocked: {report['tasks']['blocked']}")
        
        print(f"\nWorkstream Progress:")
        for ws, prog in sorted(report['workstream_progress'].items()):
            bar_length = 30
            filled = int(bar_length * prog['percentage'] / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  {ws:20s} {bar} {prog['percentage']:5.1f}% ({prog['completed']}/{prog['total']})")
        
        if report.get('ready_tasks'):
            print(f"\n✅ Ready to Work On:")
            for task_id in report['ready_tasks']:
                print(f"  • {task_id}")
        
        print()
    
    def list_tasks(self, status: str = None):
        """List tasks by status"""
        with open(self.planner.task_queue_path) as f:
            queue = json.load(f)
        
        statuses = [status] if status else queue.keys()
        
        for s in statuses:
            tasks = queue.get(s, [])
            if tasks:
                print(f"\n{s.upper()} ({len(tasks)}):")
                for task in tasks:
                    print(f"  [{task['id']}] {task['title']}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual Multi-Agent Runner (works with Claude.ai subscription)"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Show task queue status")
    group.add_argument("--next", action="store_true", help="Show next task to work on")
    group.add_argument("--complete", metavar="TASK_ID", help="Mark task as complete")
    group.add_argument("--report", action="store_true", help="Show progress report")
    group.add_argument("--list", metavar="STATUS", help="List tasks by status (ready/in_progress/completed)")
    
    args = parser.parse_args()
    
    runner = ManualAgentRunner()
    
    if args.status:
        runner.show_status()
    elif args.next:
        runner.show_next_task()
    elif args.complete:
        runner.complete_task(args.complete)
    elif args.report:
        runner.show_report()
    elif args.list:
        runner.list_tasks(args.list)


if __name__ == "__main__":
    main()
