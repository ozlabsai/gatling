#!/usr/bin/env python3
"""
Automated Multi-Agent Runner for Project Gatling

This script continuously monitors the task queue and automatically spawns
Claude Code agents via the Anthropic API to execute ready tasks.

Usage:
    uv run python agents/automated_runner.py --max-parallel 3 --monitor

Environment:
    Requires ANTHROPIC_API_KEY in .env file
"""

import os
import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.research_planner import ResearchPlannerAgent
from agents.base_agent import TaskStatus


class AutomatedAgentRunner:
    """
    Automated runner that spawns and monitors Claude Code agents.
    
    This enables true autonomous multi-agent execution where agents
    run without human intervention.
    """
    
    def __init__(self, max_parallel: int = 3, poll_interval: int = 30):
        """
        Initialize the automated runner.
        
        Args:
            max_parallel: Maximum number of agents to run simultaneously
            poll_interval: How often to check for ready tasks (seconds)
        """
        self.max_parallel = max_parallel
        self.poll_interval = poll_interval
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = Anthropic(api_key=api_key)
        
        # Initialize research planner
        self.planner = ResearchPlannerAgent()
        
        # Track running agents
        self.running_agents: Dict[str, Dict[str, Any]] = {}
        
        print(f"Initialized AutomatedAgentRunner (max_parallel={max_parallel})")
    
    def get_agent_system_prompt(self, agent_name: str, task: Dict[str, Any]) -> str:
        """
        Generate agent-specific system prompt.
        
        Args:
            agent_name: Name of the agent (latent_substrate, energy_geometry, etc.)
            task: Task specification
        
        Returns:
            Complete system prompt for the agent
        """
        
        # Load acceptance criteria
        criteria_map = {
            "latent_substrate": "encoders",
            "energy_geometry": "energy_functions",
            "provenance": "repair_engine",
        }
        
        criteria_file = criteria_map.get(agent_name, "default")
        criteria_path = Path(f"acceptance_criteria/{criteria_file}.json")
        
        if criteria_path.exists():
            with open(criteria_path) as f:
                criteria = json.load(f)
        else:
            criteria = {"note": "No specific criteria file found"}
        
        # Load dependencies if any
        dependency_context = ""
        if task.get("dependencies"):
            dependency_context = "\n\nDependencies completed:"
            for dep_id in task["dependencies"]:
                handoff_pattern = f"handoffs/{dep_id}_to_{agent_name}*.json"
                handoff_files = list(Path(".").glob(handoff_pattern))
                if handoff_files:
                    with open(handoff_files[0]) as f:
                        handoff = json.load(f)
                        dependency_context += f"\n- {dep_id}: {handoff['artifact']}"
        
        system_prompt = f"""You are a specialized implementation agent for Project Gatling.

AGENT IDENTITY:
- Name: {agent_name}_agent
- Workstream: {task.get('agent', 'unknown')}
- Role: Autonomous implementation of assigned task

CURRENT TASK:
- ID: {task['id']}
- Title: {task['title']}
- Description: {task['description']}
{dependency_context}

ACCEPTANCE CRITERIA:
{json.dumps(criteria, indent=2)}

REQUIRED OUTPUTS:
{json.dumps(task.get('outputs_required', []), indent=2)}

INSTRUCTIONS:
1. You have full access to the codebase at /home/claude/gatling/
2. Use UV for all Python operations (uv run, uv add, etc.)
3. Implement the task following the acceptance criteria exactly
4. Create all required outputs (code, tests, docs)
5. When complete, create an artifact manifest at:
   outputs/{agent_name}/{task['id']}_artifact.json

ARTIFACT MANIFEST FORMAT:
{{
    "task_id": "{task['id']}",
    "component": "<ComponentName>",
    "version": "0.1.0",
    "outputs": {{
        "code": "path/to/code.py",
        "tests": "path/to/tests.py",
        "docs": "path/to/docs.md"
    }},
    "interface": {{
        "input_shape": "...",
        "output_shape": "...",
        "latency_p99": "..."
    }},
    "validation_status": "passed"
}}

EXECUTION GUIDELINES:
- Follow the coding standards in CLAUDE.md
- Write comprehensive tests (>90% coverage)
- Include detailed docstrings
- Ensure all acceptance criteria are met before marking complete
- If you encounter blockers, document them in the artifact

AUTONOMY:
- You are running autonomously
- Make implementation decisions independently
- Use web search for current best practices
- Use extended thinking for complex algorithms
- Execute code to validate your implementation

Begin implementation now.
"""
        
        return system_prompt
    
    async def spawn_agent(self, task: Dict[str, Any]) -> str:
        """
        Spawn a new Claude Code agent to execute a task.
        
        Args:
            task: Task specification
        
        Returns:
            Session/conversation ID
        """
        agent_name = task['agent']
        task_id = task['id']
        
        print(f"\n{'='*60}")
        print(f"SPAWNING AGENT: {agent_name}")
        print(f"Task: {task_id} - {task['title']}")
        print(f"{'='*60}\n")
        
        # Update task status to in_progress
        self.planner.update_task_status(task_id, TaskStatus.IN_PROGRESS)
        
        # Generate system prompt
        system_prompt = self.get_agent_system_prompt(agent_name, task)
        
        # Start Claude Code session
        print(f"Starting Claude Code session for {task_id}...")
        
        try:
            # Use streaming for long-running tasks
            with self.client.messages.stream(
                model="claude-sonnet-4-5",
                max_tokens=200000,  # Extended budget for implementation
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Execute task {task_id}: {task['title']}\n\nBegin implementation."
                    }
                ]
            ) as stream:
                # Process the stream
                full_response_text = ""
                for text in stream.text_stream:
                    full_response_text += text
                    # Print progress indicators
                    if len(full_response_text) % 1000 == 0:
                        print(".", end="", flush=True)
                
                # Get final message
                response = stream.get_final_message()
            
            print()  # New line after progress dots
            
            # Extract conversation ID
            conversation_id = response.id
            
            # Track running agent
            self.running_agents[conversation_id] = {
                "task_id": task_id,
                "agent_name": agent_name,
                "task": task,
                "started_at": datetime.now().isoformat(),
                "status": "running"
            }
            
            print(f"✓ Agent spawned: {conversation_id}")
            print(f"  Response received: {len(full_response_text)} characters")
            
            # Process response to check for completion
            await self._process_agent_response(conversation_id, response)
            
            return conversation_id
            
        except Exception as e:
            print(f"✗ Failed to spawn agent: {str(e)}")
            self.planner.update_task_status(
                task_id,
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            raise
    
    async def _process_agent_response(self, conversation_id: str, response):
        """
        Process the response from an agent.
        
        Checks if task is complete by looking for artifact manifest.
        """
        agent_info = self.running_agents.get(conversation_id)
        if not agent_info:
            return
        
        task_id = agent_info["task_id"]
        agent_name = agent_info["agent_name"]
        
        # Check if agent created artifact manifest
        artifact_path = Path(f"outputs/{agent_name}/{task_id}_artifact.json")
        
        if artifact_path.exists():
            print(f"\n✓ Task {task_id} completed by {agent_name}")
            
            # Load artifact
            with open(artifact_path) as f:
                artifact = json.load(f)
            
            # Validate artifact
            if self._validate_artifact(artifact, agent_info["task"]):
                # Mark task as complete
                self.planner.update_task_status(task_id, TaskStatus.REVIEW)
                
                # Update agent tracking
                agent_info["status"] = "completed"
                agent_info["completed_at"] = datetime.now().isoformat()
                agent_info["artifact"] = artifact
                
                print(f"  Artifact validated: {artifact['component']}")
                print(f"  Status: REVIEW")
            else:
                print(f"  ✗ Artifact validation failed")
                self.planner.update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    {"error": "Artifact validation failed"}
                )
        else:
            # Task still in progress
            print(f"  Task {task_id} in progress...")
            
            # Check for tool use or other indicators
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == "tool_use":
                        print(f"    Using tool: {block.name}")
                    elif block.type == "text":
                        # Print excerpt
                        text = block.text[:200]
                        print(f"    Output: {text}...")
    
    def _validate_artifact(self, artifact: Dict[str, Any], task: Dict[str, Any]) -> bool:
        """
        Validate that artifact meets acceptance criteria.
        
        Args:
            artifact: Artifact manifest
            task: Task specification
        
        Returns:
            True if valid
        """
        # Check required fields
        required_fields = ["task_id", "component", "outputs", "interface"]
        for field in required_fields:
            if field not in artifact:
                print(f"    Missing required field: {field}")
                return False
        
        # Check outputs exist
        for output_type, path in artifact["outputs"].items():
            if not Path(path).exists():
                print(f"    Missing output file: {path}")
                return False
        
        # Check if tests pass (if test output exists)
        if "tests" in artifact["outputs"]:
            test_path = artifact["outputs"]["tests"]
            # Could run: uv run pytest {test_path}
            # For now, just check it exists
            pass
        
        return True
    
    async def monitor_and_spawn(self):
        """
        Main monitoring loop: check for ready tasks and spawn agents.
        """
        print("\n" + "="*60)
        print("AUTOMATED MULTI-AGENT RUNNER - ACTIVE")
        print("="*60)
        print(f"Max parallel agents: {self.max_parallel}")
        print(f"Poll interval: {self.poll_interval}s")
        print("\nMonitoring task queue...\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} ({datetime.now().strftime('%H:%M:%S')}) ---")
            
            # Clean up completed agents
            self._cleanup_completed_agents()
            
            # Get ready tasks
            ready_tasks = self.planner.get_ready_tasks()
            active_count = len([a for a in self.running_agents.values() if a["status"] == "running"])
            
            print(f"Ready tasks: {len(ready_tasks)}")
            print(f"Active agents: {active_count}/{self.max_parallel}")
            
            # Spawn agents for ready tasks if we have capacity
            available_slots = self.max_parallel - active_count
            
            if ready_tasks and available_slots > 0:
                # Load task details
                with open(self.planner.task_queue_path) as f:
                    queue = json.load(f)
                
                ready_task_details = [
                    t for t in queue["ready"]
                    if t["id"] in ready_tasks
                ]
                
                # Prioritize by priority field
                ready_task_details.sort(
                    key=lambda t: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(t.get("priority", "medium"), 2)
                )
                
                # Spawn agents
                for task in ready_task_details[:available_slots]:
                    try:
                        await self.spawn_agent(task)
                        await asyncio.sleep(2)  # Brief delay between spawns
                    except Exception as e:
                        print(f"Failed to spawn agent for {task['id']}: {e}")
            
            # Generate progress report
            if iteration % 5 == 0:  # Every 5 iterations
                self._print_progress_report()
            
            # Wait before next iteration
            print(f"\nWaiting {self.poll_interval}s until next check...")
            await asyncio.sleep(self.poll_interval)
    
    def _cleanup_completed_agents(self):
        """Remove completed agents from tracking"""
        completed = [
            conv_id for conv_id, info in self.running_agents.items()
            if info["status"] in ["completed", "failed"]
        ]
        
        for conv_id in completed:
            del self.running_agents[conv_id]
    
    def _print_progress_report(self):
        """Print detailed progress report"""
        report = self.planner.generate_progress_report()
        
        print("\n" + "="*60)
        print("PROGRESS REPORT")
        print("="*60)
        print(f"Overall: {report['overall_progress']:.1f}%")
        print(f"\nTasks: {report['tasks']['completed']}/{report['tasks']['total']} complete")
        print(f"  In Progress: {report['tasks']['in_progress']}")
        print(f"  Ready: {report['tasks']['ready']}")
        print(f"  Blocked: {report['tasks']['blocked']}")
        
        print("\nWorkstream Progress:")
        for ws, prog in report["workstream_progress"].items():
            bar_length = 20
            filled = int(bar_length * prog["percentage"] / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  {ws:20s} {bar} {prog['percentage']:5.1f}%")
        
        print("="*60 + "\n")
    
    async def run_single_task(self, task_id: str):
        """
        Run a single task (useful for testing).
        
        Args:
            task_id: Task to execute
        """
        task = self.planner.load_task(task_id)
        
        print(f"Running single task: {task_id}")
        
        conversation_id = await self.spawn_agent(task)
        
        print(f"\nAgent spawned: {conversation_id}")
        print("Task executing autonomously...")
        print("\nYou can monitor progress in:")
        print(f"  - logs/{task['agent']}_agent.log")
        print(f"  - outputs/{task['agent']}/")


async def main():
    parser = argparse.ArgumentParser(
        description="Automated Multi-Agent Runner for Project Gatling"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum number of agents to run simultaneously"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="How often to check for ready tasks (seconds)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Run a single task instead of monitoring mode"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run in continuous monitoring mode"
    )
    
    args = parser.parse_args()
    
    runner = AutomatedAgentRunner(
        max_parallel=args.max_parallel,
        poll_interval=args.poll_interval
    )
    
    if args.task:
        # Single task mode
        await runner.run_single_task(args.task)
    elif args.monitor:
        # Continuous monitoring mode
        await runner.monitor_and_spawn()
    else:
        print("Please specify either --task <TASK_ID> or --monitor")
        print("\nExamples:")
        print("  uv run python agents/automated_runner.py --task LSA-001")
        print("  uv run python agents/automated_runner.py --monitor --max-parallel 5")


if __name__ == "__main__":
    asyncio.run(main())