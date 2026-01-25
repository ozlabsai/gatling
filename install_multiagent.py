#!/usr/bin/env python3
"""
Multi-Agent System Installer for Gatling Project

This script installs the multi-agent infrastructure into your existing
Gatling project directory.

Usage:
    python install_multiagent.py
"""

import os
import shutil
from pathlib import Path


def create_directory_structure(base_path: Path):
    """Create all required directories"""
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
        "logs",
    ]
    
    print("Creating directory structure...")
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}/")
    
    print()


def create_agents_init(base_path: Path):
    """Create agents/__init__.py"""
    init_content = '''"""
Multi-Agent System for Project Gatling

This package contains the autonomous agent infrastructure for parallel
development across research workstreams.
"""

from .base_agent import GatlingAgent, TaskStatus
from .research_planner import ResearchPlannerAgent

__all__ = ["GatlingAgent", "TaskStatus", "ResearchPlannerAgent"]
'''
    
    init_path = base_path / "agents" / "__init__.py"
    with open(init_path, 'w') as f:
        f.write(init_content)
    
    print("✓ Created agents/__init__.py")


def update_pyproject_toml(base_path: Path):
    """Add dependencies to pyproject.toml"""
    pyproject_path = base_path / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("⚠ Warning: pyproject.toml not found")
        return
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Check if dependencies already added
    if "networkx" in content and "matplotlib" in content:
        print("✓ Dependencies already in pyproject.toml")
        return
    
    # Add dependencies
    new_deps = '''
    # Multi-agent system dependencies
    "networkx>=3.0",        # Dependency graph management
    "anthropic>=0.18.0",    # API-based autonomous agents
    "matplotlib>=3.7.0",    # Graph visualization
'''
    
    # Find dependencies section and add
    if 'dependencies = [' in content:
        # Add before the closing bracket
        content = content.replace(
            'dependencies = [',
            f'dependencies = [{new_deps}'
        )
        
        with open(pyproject_path, 'w') as f:
            f.write(content)
        
        print("✓ Added dependencies to pyproject.toml")
        print("  Run: uv sync")
    else:
        print("⚠ Could not auto-update pyproject.toml")
        print("  Please manually add:")
        print(new_deps)


def create_acceptance_criteria(base_path: Path):
    """Create acceptance criteria files"""
    
    # Encoders criteria
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
    
    # Energy functions criteria
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
    
    # Repair engine criteria
    repair_criteria = {
        "code_quality": {
            "required": [
                "Type hints for all functions",
                "Algorithm complexity documented",
                "PEP8 compliant"
            ]
        },
        "functionality": {
            "required": [
                "Reduces energy below threshold",
                "Preserves Functional Intent Similarity > 0.9",
                "Handles all mutation types",
                "Terminates in bounded time"
            ]
        },
        "performance": {
            "required": [
                "Total latency < 200ms",
                "Greedy search < 50ms",
                "Beam search < 150ms"
            ]
        },
        "tests": {
            "required": [
                "Tested on 1000+ malicious plans",
                "FIS validation",
                "Convergence tests",
                "Edge case handling"
            ]
        }
    }
    
    import json
    
    criteria_dir = base_path / "acceptance_criteria"
    
    with open(criteria_dir / "encoders.json", 'w') as f:
        json.dump(encoder_criteria, f, indent=2)
    
    with open(criteria_dir / "energy_functions.json", 'w') as f:
        json.dump(energy_criteria, f, indent=2)
    
    with open(criteria_dir / "repair_engine.json", 'w') as f:
        json.dump(repair_criteria, f, indent=2)
    
    print("✓ Created acceptance criteria files")


def create_readme_update(base_path: Path):
    """Add multi-agent section to README"""
    readme_path = base_path / "README.md"
    
    addition = """

## Multi-Agent Development System

This project uses an autonomous multi-agent system for parallel development across research workstreams.

### Quick Start

```bash
# Initialize the multi-agent system
uv run python quickstart.py --phase foundation

# Run autonomous agents
uv run python agents/automated_runner.py --monitor --max-parallel 3
```

### Documentation

- [Multi-Agent Architecture](docs/MULTI_AGENT_ARCHITECTURE.md) - Complete system design
- [Usage Guide](docs/USAGE_GUIDE.md) - How to use the system
- [How It Works](docs/HOW_IT_WORKS_VISUAL.md) - Visual explanation

### Agents

The system includes specialized agents for each workstream:
- **LatentSubstrateAgent**: JEPA encoders (Weeks 1-3)
- **EnergyGeometryAgent**: Energy functions (Weeks 4-6)
- **ProvenanceAgent**: Trust architecture & repair (Weeks 4-6)
- **RedTeamAgent**: Adversarial mutations (Weeks 7-9)
- **DatasetAgent**: Gatling-10M dataset (Weeks 7-9)
- **IntegrationAgent**: E2E testing (Weeks 10-12)

Each agent runs autonomously with full Claude Code capabilities.
"""
    
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
        
        if "Multi-Agent Development System" not in content:
            with open(readme_path, 'a') as f:
                f.write(addition)
            print("✓ Updated README.md")
        else:
            print("✓ README.md already has multi-agent section")
    else:
        print("⚠ README.md not found")


def main():
    """Main installation routine"""
    print("=" * 70)
    print("PROJECT GATLING - MULTI-AGENT SYSTEM INSTALLER")
    print("=" * 70)
    print()
    
    # Get base path (current directory)
    base_path = Path.cwd()
    
    # Verify we're in the gatling project
    if not (base_path / "pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found!")
        print("   Please run this script from your gatling project root directory")
        return 1
    
    if not (base_path / "CLAUDE.md").exists():
        print("⚠ Warning: CLAUDE.md not found - are you in the gatling directory?")
    
    print(f"Installing into: {base_path}")
    print()
    
    # Step 1: Create directory structure
    print("Step 1: Creating directory structure")
    create_directory_structure(base_path)
    
    # Step 2: Create agents/__init__.py
    print("Step 2: Creating agents/__init__.py")
    create_agents_init(base_path)
    print()
    
    # Step 3: Update pyproject.toml
    print("Step 3: Updating pyproject.toml")
    update_pyproject_toml(base_path)
    print()
    
    # Step 4: Create acceptance criteria
    print("Step 4: Creating acceptance criteria")
    create_acceptance_criteria(base_path)
    print()
    
    # Step 5: Update README
    print("Step 5: Updating README.md")
    create_readme_update(base_path)
    print()
    
    print("=" * 70)
    print("INSTALLATION COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print()
    print("1. Copy these files from the chat outputs to your project:")
    print("   - agents/base_agent.py")
    print("   - agents/research_planner.py")
    print("   - agents/automated_runner.py")
    print("   - agents/latent_substrate_agent.py (example)")
    print("   - quickstart.py")
    print("   - docs/MULTI_AGENT_ARCHITECTURE.md")
    print("   - docs/USAGE_GUIDE.md")
    print("   - docs/HOW_IT_WORKS_VISUAL.md")
    print()
    print("2. Install dependencies:")
    print("   uv sync")
    print()
    print("3. Initialize the system:")
    print("   uv run python quickstart.py --phase foundation")
    print()
    print("4. Run autonomous agents:")
    print("   uv run python agents/automated_runner.py --monitor")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
