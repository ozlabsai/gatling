#!/usr/bin/env python3
"""
Agent Launcher Script

Launch a specific agent to work on a task.

Usage:
    uv run python launch_agent.py latent_substrate LSA-001
    uv run python launch_agent.py energy_geometry EGA-001 --mode exploration
"""

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
