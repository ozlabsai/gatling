#!/usr/bin/env python3
"""
Sub-Agent Manager for Project Gatling

Allows main agents to decompose tasks into sub-tasks and track them.

Usage:
    # Decompose a task into sub-tasks
    uv run python agents/subagent_manager.py --decompose LSA-001

    # Show sub-tasks for a task
    uv run python agents/subagent_manager.py --status LSA-001

    # Get next sub-task to work on
    uv run python agents/subagent_manager.py --next LSA-001

    # Mark sub-task complete
    uv run python agents/subagent_manager.py --complete LSA-001-A

    # Mark entire task complete (when all sub-tasks done)
    uv run python agents/subagent_manager.py --finalize LSA-001
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class SubAgentManager:
    """
    Manages sub-task decomposition and tracking within main tasks.
    """
    
    def __init__(self):
        self.subtasks_dir = Path("tasks/subtasks")
        self.subtasks_dir.mkdir(parents=True, exist_ok=True)
    
    def get_subtask_file(self, task_id: str) -> Path:
        """Get path to subtasks file for a task"""
        return self.subtasks_dir / f"{task_id}_subtasks.json"
    
    def decompose_task(self, task_id: str):
        """
        Generate a decomposition prompt for the user to paste into Claude.
        """
        # Load main task
        from agents.research_planner import ResearchPlannerAgent
        planner = ResearchPlannerAgent()
        task = planner.load_task(task_id)
        
        print("\n" + "="*70)
        print(f"TASK DECOMPOSITION PROMPT FOR {task_id}")
        print("="*70 + "\n")
        
        print("Paste this into a Claude Code session to decompose the task:\n")
        print("-" * 70)
        
        decomposition_prompt = f"""I need to decompose this task into sub-tasks:

Task ID: {task_id}
Title: {task['title']}
Description:
{task['description']}

Required Outputs:
{chr(10).join('- ' + o for o in task.get('outputs_required', []))}

Please decompose this into 4-6 manageable sub-tasks that can be worked on 
sequentially or in parallel by different Claude Code sessions.

For each sub-task, provide:
1. Sub-task ID (format: {task_id}-A, {task_id}-B, etc.)
2. Title (concise, actionable)
3. Description (what needs to be done)
4. Dependencies (list of other sub-task IDs that must complete first)
5. Outputs (specific files this sub-task creates)
6. Complexity (low/medium/high)

Output ONLY a JSON object in this exact format:

{{
  "main_task_id": "{task_id}",
  "sub_tasks": [
    {{
      "id": "{task_id}-A",
      "title": "Design Architecture",
      "description": "Design the transformer architecture including layer counts, dimensions, attention heads, etc.",
      "dependencies": [],
      "outputs": ["docs/architecture/{task_id}_design.md"],
      "complexity": "medium"
    }},
    {{
      "id": "{task_id}-B",
      "title": "Implement Core Class",
      "description": "Implement the main encoder class with forward pass",
      "dependencies": ["{task_id}-A"],
      "outputs": ["source/encoders/governance_encoder.py"],
      "complexity": "high"
    }}
  ]
}}

Provide the JSON only, no other text.
"""
        
        print(decomposition_prompt)
        print("-" * 70)
        print()
        print("After Claude generates the JSON:")
        print()
        print(f"1. Copy the JSON output")
        print(f"2. Save it to: {self.get_subtask_file(task_id)}")
        print(f"3. Or paste it here and I'll save it:")
        print()
        
        response = input("Paste JSON here (or press Enter to skip): ")
        
        if response.strip():
            try:
                subtasks = json.loads(response)
                
                # Add metadata
                subtasks["created_at"] = datetime.now().isoformat()
                subtasks["status"] = {
                    st["id"]: "pending" for st in subtasks["sub_tasks"]
                }
                
                # Save
                with open(self.get_subtask_file(task_id), 'w') as f:
                    json.dump(subtasks, f, indent=2)
                
                print(f"\n✓ Saved {len(subtasks['sub_tasks'])} sub-tasks!")
                self.show_subtask_status(task_id)
                
            except json.JSONDecodeError as e:
                print(f"\n✗ Invalid JSON: {e}")
                print("   Please try again or save manually.")
    
    def show_subtask_status(self, task_id: str):
        """Show status of all sub-tasks"""
        subtask_file = self.get_subtask_file(task_id)
        
        if not subtask_file.exists():
            print(f"\n❌ No sub-tasks found for {task_id}")
            print(f"   Run: uv run python agents/subagent_manager.py --decompose {task_id}")
            return
        
        with open(subtask_file) as f:
            data = json.load(f)
        
        print("\n" + "="*70)
        print(f"SUB-TASKS FOR {task_id}")
        print("="*70 + "\n")
        
        status_counts = {}
        for st_id, status in data["status"].items():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total = len(data["sub_tasks"])
        completed = status_counts.get("completed", 0)
        in_progress = status_counts.get("in_progress", 0)
        ready = status_counts.get("ready", 0)
        
        print(f"Progress: {completed}/{total} completed ({completed/total*100:.0f}%)")
        print(f"  ✓ Completed: {completed}")
        print(f"  🔄 In Progress: {in_progress}")
        print(f"  ✅ Ready: {ready}")
        print(f"  ⏸️  Pending: {status_counts.get('pending', 0)}")
        
        print("\nSub-Tasks:\n")
        
        for subtask in data["sub_tasks"]:
            st_id = subtask["id"]
            status = data["status"][st_id]
            
            icon = {
                "completed": "✓",
                "in_progress": "🔄",
                "ready": "✅",
                "pending": "⏸️"
            }.get(status, "?")
            
            print(f"{icon} [{st_id}] {subtask['title']}")
            print(f"   Status: {status}")
            print(f"   Complexity: {subtask['complexity']}")
            
            if subtask.get('dependencies'):
                deps_status = [
                    f"{d} ({data['status'][d]})" 
                    for d in subtask['dependencies']
                ]
                print(f"   Dependencies: {', '.join(deps_status)}")
            
            print()
    
    def get_next_subtask(self, task_id: str):
        """Get the next sub-task ready to work on"""
        subtask_file = self.get_subtask_file(task_id)
        
        if not subtask_file.exists():
            print(f"\n❌ No sub-tasks found for {task_id}")
            return
        
        with open(subtask_file) as f:
            data = json.load(f)
        
        # Find ready sub-tasks (all dependencies complete)
        completed = {
            st_id for st_id, status in data["status"].items()
            if status == "completed"
        }
        
        ready_subtasks = []
        for subtask in data["sub_tasks"]:
            st_id = subtask["id"]
            status = data["status"][st_id]
            
            # Already ready or in progress
            if status in ["ready", "in_progress"]:
                ready_subtasks.append(subtask)
            # Pending but dependencies complete
            elif status == "pending":
                deps = set(subtask.get("dependencies", []))
                if deps.issubset(completed):
                    ready_subtasks.append(subtask)
                    # Mark as ready
                    data["status"][st_id] = "ready"
        
        # Save updated status
        with open(subtask_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        if not ready_subtasks:
            print("\n❌ No sub-tasks ready!")
            print("   All sub-tasks either in progress or blocked by dependencies.")
            return
        
        # Get highest priority (lowest complexity first for quick wins)
        complexity_order = {"low": 0, "medium": 1, "high": 2}
        ready_subtasks.sort(
            key=lambda st: complexity_order.get(st['complexity'], 1)
        )
        
        subtask = ready_subtasks[0]
        
        print("\n" + "="*70)
        print(f"NEXT SUB-TASK: {subtask['id']}")
        print("="*70 + "\n")
        
        print(f"Title: {subtask['title']}")
        print(f"Complexity: {subtask['complexity']}")
        print(f"\nDescription:")
        print(subtask['description'])
        
        if subtask.get('dependencies'):
            print(f"\nDependencies (completed):")
            for dep in subtask['dependencies']:
                print(f"  ✓ {dep}")
        
        print(f"\nRequired Outputs:")
        for output in subtask.get('outputs', []):
            print(f"  • {output}")
        
        # Generate prompt
        print("\n" + "="*70)
        print("PASTE THIS INTO CLAUDE CODE:")
        print("="*70 + "\n")
        
        prompt = f"""Sub-Task: {subtask['id']} - {subtask['title']}

This is part of main task: {task_id}

{subtask['description']}

Required Outputs:
{chr(10).join('- ' + o for o in subtask.get('outputs', []))}

When complete, create outputs and let me know so I can mark this sub-task complete.

Begin implementation using UV for all operations.
"""
        
        print(prompt)
        
        # Offer to mark as in progress
        print("\n" + "="*70)
        response = input(f"Mark {subtask['id']} as in progress? (y/N): ")
        
        if response.lower() == 'y':
            data["status"][subtask['id']] = "in_progress"
            with open(subtask_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Marked {subtask['id']} as IN_PROGRESS")
        
        print()
    
    def complete_subtask(self, subtask_id: str):
        """Mark a sub-task as complete"""
        # Extract main task ID (e.g., LSA-001 from LSA-001-A)
        task_id = '-'.join(subtask_id.split('-')[:-1])
        
        subtask_file = self.get_subtask_file(task_id)
        
        if not subtask_file.exists():
            print(f"\n❌ No sub-tasks found for {task_id}")
            return
        
        with open(subtask_file) as f:
            data = json.load(f)
        
        if subtask_id not in data["status"]:
            print(f"\n❌ Sub-task {subtask_id} not found!")
            return
        
        # Mark complete
        data["status"][subtask_id] = "completed"
        data.setdefault("completed_at", {})[subtask_id] = datetime.now().isoformat()
        
        with open(subtask_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Marked {subtask_id} as COMPLETED")
        
        # Check if all sub-tasks complete
        all_complete = all(
            status == "completed" 
            for status in data["status"].values()
        )
        
        if all_complete:
            print(f"\n🎉 All sub-tasks for {task_id} are complete!")
            print(f"   Run: uv run python agents/subagent_manager.py --finalize {task_id}")
        else:
            # Show what's ready next
            completed_count = sum(
                1 for s in data["status"].values() if s == "completed"
            )
            total = len(data["sub_tasks"])
            print(f"\nProgress: {completed_count}/{total} sub-tasks complete")
            
            # Check if new tasks are ready
            completed_ids = {
                st_id for st_id, status in data["status"].items()
                if status == "completed"
            }
            
            newly_ready = []
            for subtask in data["sub_tasks"]:
                st_id = subtask["id"]
                if data["status"][st_id] == "pending":
                    deps = set(subtask.get("dependencies", []))
                    if deps.issubset(completed_ids):
                        newly_ready.append(st_id)
            
            if newly_ready:
                print(f"\n✅ New sub-tasks now ready: {', '.join(newly_ready)}")
        
        print()
    
    def finalize_task(self, task_id: str):
        """Combine all sub-task outputs and create final artifact"""
        subtask_file = self.get_subtask_file(task_id)
        
        if not subtask_file.exists():
            print(f"\n❌ No sub-tasks found for {task_id}")
            return
        
        with open(subtask_file) as f:
            data = json.load(f)
        
        # Verify all complete
        all_complete = all(
            status == "completed" 
            for status in data["status"].values()
        )
        
        if not all_complete:
            incomplete = [
                st_id for st_id, status in data["status"].items()
                if status != "completed"
            ]
            print(f"\n❌ Not all sub-tasks complete!")
            print(f"   Incomplete: {', '.join(incomplete)}")
            return
        
        print("\n" + "="*70)
        print(f"FINALIZING {task_id}")
        print("="*70 + "\n")
        
        print("All sub-tasks complete! ✓")
        print("\nNow you need to:")
        print("1. Verify all outputs were created")
        print("2. Run integration tests")
        print("3. Create the final artifact manifest")
        print()
        print("Paste this into Claude Code for finalization:\n")
        print("-" * 70)
        
        finalization_prompt = f"""All sub-tasks for {task_id} are complete!

Sub-tasks completed:
{chr(10).join('- ' + st['id'] + ': ' + st['title'] for st in data['sub_tasks'])}

All outputs created:
{chr(10).join('- ' + o for st in data['sub_tasks'] for o in st.get('outputs', []))}

Please:
1. Verify all files exist and are correct
2. Run complete test suite
3. Create final artifact manifest at outputs/{data['sub_tasks'][0].get('id', '').split('-')[0]}/{task_id}_artifact.json

Artifact format:
{{
    "task_id": "{task_id}",
    "component": "<MainComponentName>",
    "version": "0.1.0",
    "outputs": {{
        "code": "...",
        "tests": "...",
        "docs": "..."
    }},
    "interface": {{...}},
    "validation_status": "passed",
    "sub_tasks_completed": {len(data['sub_tasks'])}
}}

Begin finalization.
"""
        
        print(finalization_prompt)
        print("-" * 70)
        print()
        print("After finalization, run:")
        print(f"  uv run python agents/manual_runner.py --complete {task_id}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Sub-Agent Manager for task decomposition"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--decompose", metavar="TASK_ID", help="Decompose a task into sub-tasks")
    group.add_argument("--status", metavar="TASK_ID", help="Show sub-task status")
    group.add_argument("--next", metavar="TASK_ID", help="Get next sub-task to work on")
    group.add_argument("--complete", metavar="SUBTASK_ID", help="Mark sub-task complete (e.g., LSA-001-A)")
    group.add_argument("--finalize", metavar="TASK_ID", help="Finalize task (all sub-tasks complete)")
    
    args = parser.parse_args()
    
    manager = SubAgentManager()
    
    if args.decompose:
        manager.decompose_task(args.decompose)
    elif args.status:
        manager.show_subtask_status(args.status)
    elif args.next:
        manager.get_next_subtask(args.next)
    elif args.complete:
        manager.complete_subtask(args.complete)
    elif args.finalize:
        manager.finalize_task(args.finalize)


if __name__ == "__main__":
    main()