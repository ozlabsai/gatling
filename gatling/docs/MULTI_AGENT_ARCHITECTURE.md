# Multi-Agent Claude Code Architecture for Project Gatling

## Overview

This document outlines how to implement Project Gatling using a multi-agent Claude Code system inspired by [Anthropic's multi-agent research approach](https://www.anthropic.com/engineering/multi-agent-research-system).

## Core Architecture Pattern

The implementation uses **Specialized Expert Agents** coordinated by a **Research Planner Agent**, mirroring the theoretical workstream distribution while leveraging Claude's extended thinking and tool use capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                   Research Planner Agent                     │
│  (Coordinator, Task Decomposition, Integration Oversight)   │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌───────────────┐
│  Implementer  │    │   Reviewer    │
│    Agents     │◄──►│    Agents     │
└───────┬───────┘    └───────┬───────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Integration Agent   │
        │  (Testing, Merging)  │
        └─────────────────────┘
```

## Agent Role Definitions

### 1. Research Planner Agent (RPAgent)
**Role**: Orchestrator and strategic decomposer
**Responsibilities**:
- Break down workstreams into concrete implementation tasks
- Maintain the project state and dependency graph
- Route tasks to appropriate implementer agents
- Verify integration points between workstreams
- Track progress against the 12-week timeline

**Claude Code Configuration**:
```python
# RPAgent operates with extended context and planning mode
{
    "name": "research_planner",
    "model": "claude-sonnet-4-5",
    "extended_thinking": True,
    "system_prompt": "You are the Research Planner for Project Gatling...",
    "tools": [
        "bash_tool",
        "view",
        "create_file",
        "str_replace"
    ],
    "context_files": [
        "PRD.md",
        "WORK-DISTRIBUTION.md",
        "IMPLEMENTATION_LOG.md"
    ]
}
```

### 2. Implementer Agents (5 Specialized Agents)

#### 2.1 LatentSubstrateAgent (LeCun Workstream)
**Focus**: JEPA encoders and dual-encoder architecture
**Key Tasks**:
- Implement `source/encoders/governance_encoder.py`
- Implement `source/encoders/execution_encoder.py`
- Build transformer-based feature extraction
- Create semantic intent vector space (1024-dim)

**Specialized Skills**:
- PyTorch/JAX deep learning
- Transformer architectures
- Representation learning
- HuggingFace integration

**Implementation Pattern**:
```python
# Each agent gets a focused task with clear inputs/outputs
task = {
    "objective": "Implement GovernanceEncoder class",
    "inputs": ["policy_schema.json", "encoder_spec.md"],
    "outputs": ["source/encoders/governance_encoder.py", "test/test_governance_encoder.py"],
    "dependencies": ["torch", "transformers"],
    "acceptance_criteria": [
        "Encodes policy into 1024-dim vector",
        "Passes gradient flow test",
        "< 50ms inference latency"
    ]
}
```

#### 2.2 EnergyGeometryAgent (Du + Freedman Workstream)
**Focus**: Energy function composition and landscape smoothness
**Key Tasks**:
- Implement four energy critics (E_hierarchy, E_provenance, E_scope, E_flow)
- Build Product-of-Experts composition
- Implement spectral regularization
- Create energy landscape visualization tools

**Specialized Skills**:
- Energy-based models
- Differentiable optimization
- Topological analysis
- Numerical stability

#### 2.3 RedTeamAgent (Kolter Workstream)
**Focus**: Adversarial mutation catalog and hard negative generation
**Key Tasks**:
- Implement Corrupter Agent
- Build mutation catalog (4 types)
- Generate InfoNCE training pairs
- Create adversarial evaluation suite

**Specialized Skills**:
- Adversarial ML
- Data augmentation
- Contrastive learning
- Security testing

#### 2.4 ProvenanceAgent (Song Workstream)
**Focus**: Trust architecture and repair engine
**Key Tasks**:
- Implement cryptographic provenance tagging
- Build discrete repair engine (Greedy + Beam Search)
- Create fast-path policy distillation
- Implement repair acceptance criteria (FIS)

**Specialized Skills**:
- Cryptographic protocols
- Discrete optimization
- Search algorithms
- Policy distillation

#### 2.5 DatasetAgent (Librarians + Verifiers)
**Focus**: Gatling-10M dataset generation and curation
**Key Tasks**:
- Generate gold traces using Oracle Agents
- Apply mutation pipeline
- Label minimal scope budgets
- Validate energy annotations

**Specialized Skills**:
- Synthetic data generation
- Data validation
- LLM prompting for oracle behavior
- Dataset balancing

### 3. Reviewer Agents

Each implementer agent has a paired **Reviewer Agent** that:
- Validates code against acceptance criteria
- Checks integration points with other workstreams
- Runs tests and benchmarks
- Suggests improvements or flags issues

**Key Pattern**: Reviewer agents use a different Claude instance with fresh context to avoid confirmation bias.

### 4. Integration Agent
**Role**: Continuous integration and system-level testing
**Responsibilities**:
- Merge verified components
- Run end-to-end tests
- Profile performance (latency, memory)
- Update IMPLEMENTATION_LOG.md and CHANGELOG.md

## Multi-Agent Communication Protocol

### Task Queue System

```python
# tasks/task_queue.json
{
    "pending": [
        {
            "id": "LSA-001",
            "agent": "latent_substrate",
            "priority": "high",
            "title": "Implement GovernanceEncoder",
            "dependencies": [],
            "status": "ready"
        }
    ],
    "in_progress": [...],
    "review": [...],
    "completed": [...]
}
```

### Handoff Protocol

When Agent A completes a task that Agent B depends on:

1. **Agent A** writes completion artifact to `outputs/<workstream>/`
2. **Agent A** updates task queue with completion status
3. **Agent A** writes handoff document: `handoffs/<taskID>_to_<agentB>.md`
4. **Research Planner** triggers Agent B with dependency artifacts

### Artifact Schema

```python
# outputs/latent_substrate/governance_encoder/artifact.json
{
    "component": "GovernanceEncoder",
    "version": "0.1.0",
    "outputs": {
        "code": "source/encoders/governance_encoder.py",
        "tests": "test/test_governance_encoder.py",
        "docs": "docs/encoders/governance.md"
    },
    "interface": {
        "input_shape": "(batch_size, policy_tokens)",
        "output_shape": "(batch_size, 1024)",
        "latency_p99": "45ms"
    },
    "dependencies": ["torch>=2.0", "transformers>=4.35"],
    "downstream_consumers": ["energy_geometry", "integration"],
    "validation_status": "passed",
    "reviewer": "reviewer_latent_substrate"
}
```

## Implementation Workflow

### Week 1-3: Foundation (Latent Substrate)

```bash
# Research Planner decomposes Workstream 1
uv run python agents/research_planner.py \
    --phase foundation \
    --workstream latent_substrate \
    --output tasks/week1-3.json

# Spawn LatentSubstrateAgent
uv run python agents/latent_substrate_agent.py \
    --task tasks/LSA-001.json \
    --context docs/PRD.md \
    --output outputs/latent_substrate/

# Reviewer validates
uv run python agents/reviewer_latent_substrate.py \
    --artifact outputs/latent_substrate/governance_encoder/artifact.json \
    --criteria acceptance_criteria/encoders.json

# Integration merges if approved
uv run python agents/integration_agent.py \
    --merge outputs/latent_substrate/governance_encoder/
```

### Week 4-6: Composition (Energy + Provenance)

Tasks depend on completed encoders. Research Planner triggers parallel execution:

```python
# Parallel task execution
tasks = [
    ("energy_geometry", "EGA-001", "Implement E_hierarchy"),
    ("energy_geometry", "EGA-002", "Implement E_provenance"),
    ("provenance", "PA-001", "Build Trust Tier system")
]

# Each runs in isolated Claude Code session
for agent, task_id, description in tasks:
    spawn_agent(agent, task_id, wait=False)

# Integration waits for all completions
wait_for_dependencies(["EGA-001", "EGA-002", "PA-001"])
```

### Week 7-9: Adversarial Refinement

RedTeamAgent runs continuous mutation cycles:

```python
# Mutation loop
while not convergence():
    # Generate hard negatives
    hard_negs = red_team_agent.generate_mutations(gold_traces)
    
    # Test current EBM
    false_negatives = energy_agent.audit(hard_negs)
    
    # Feedback to dataset agent
    dataset_agent.prioritize_samples(false_negatives)
    
    # Retrain energy function
    energy_agent.train(new_batch)
```

### Week 10-12: Integration & Deployment

Integration Agent runs full E2E pipeline:

```python
# End-to-end integration test
def test_gatling_pipeline():
    # 1. Load policy
    policy = load_policy("config/example_policy.yaml")
    
    # 2. Generate test plan (potentially malicious)
    plan = load_plan("test/plans/scope_blowup.json")
    
    # 3. Audit
    z_g = governance_encoder(policy)
    z_e = execution_encoder(plan)
    energy = energy_function(z_g, z_e)
    
    # 4. Repair if needed
    if energy > theta_safe:
        repaired_plan = repair_engine.search(plan, policy, max_steps=10)
    
    # 5. Validate
    assert energy_function(z_g, execution_encoder(repaired_plan)) < theta_safe
    assert functional_intent_similarity(plan, repaired_plan) > 0.9
```

## Agent Execution Patterns

### Pattern 1: Focused Implementation

```bash
# Single agent, single task, deep focus
uv run python agents/latent_substrate_agent.py \
    --mode focused \
    --task LSA-001 \
    --max_thinking_budget 10000 \
    --checkpoint_every 100_lines
```

**Best for**: Complex algorithm implementation, novel research translation

### Pattern 2: Parallel Sweep

```bash
# Multiple agents, independent tasks
parallel --jobs 3 \
    uv run python agents/{}_agent.py --task {} ::: \
    energy_geometry EGA-001 ::: \
    energy_geometry EGA-002 ::: \
    provenance PA-001
```

**Best for**: Modular components with clear interfaces

### Pattern 3: Iterative Refinement

```bash
# Agent-reviewer loop until acceptance
while ! reviewer_accepts; do
    uv run python agents/energy_geometry_agent.py --task EGA-001 --iteration $i
    uv run python agents/reviewer_energy.py --artifact outputs/energy/E_hierarchy.json
    i=$((i+1))
done
```

**Best for**: Components requiring high precision (energy critics)

### Pattern 4: Research Exploration

```bash
# Agent explores multiple approaches, reports findings
uv run python agents/energy_geometry_agent.py \
    --mode exploration \
    --task "Compare spectral vs. adversarial regularization" \
    --output_format research_report
```

**Best for**: Novel research questions, architecture decisions

## Practical Setup

### Directory Structure

```
gatling/
├── agents/                      # Agent implementations
│   ├── research_planner.py
│   ├── latent_substrate_agent.py
│   ├── energy_geometry_agent.py
│   ├── red_team_agent.py
│   ├── provenance_agent.py
│   ├── dataset_agent.py
│   ├── reviewer_*.py
│   └── integration_agent.py
├── tasks/                       # Task queue and definitions
│   ├── task_queue.json
│   ├── week1-3/
│   ├── week4-6/
│   └── ...
├── outputs/                     # Agent outputs organized by workstream
│   ├── latent_substrate/
│   ├── energy_geometry/
│   ├── red_team/
│   ├── provenance/
│   └── dataset/
├── handoffs/                    # Inter-agent communication
│   └── LSA-001_to_EGA.md
├── acceptance_criteria/         # Validation specifications
│   ├── encoders.json
│   ├── energy_functions.json
│   └── repair_engine.json
├── source/                      # Final integrated code
├── test/                        # Test suite
└── docs/                        # Documentation
```

### Agent Base Class

```python
# agents/base_agent.py
from abc import ABC, abstractmethod
import json
from pathlib import Path

class GatlingAgent(ABC):
    def __init__(self, agent_name: str, workstream: str):
        self.name = agent_name
        self.workstream = workstream
        self.task_queue = Path("tasks/task_queue.json")
        self.output_dir = Path(f"outputs/{workstream}")
        
    @abstractmethod
    def execute_task(self, task: dict) -> dict:
        """Main task execution logic"""
        pass
    
    def load_task(self, task_id: str) -> dict:
        """Load task from queue"""
        with open(self.task_queue) as f:
            queue = json.load(f)
        return next(t for t in queue["pending"] if t["id"] == task_id)
    
    def complete_task(self, task_id: str, artifact: dict):
        """Mark task complete and create artifact"""
        # Update task queue
        # Write artifact.json
        # Create handoff documents
        pass
    
    def request_review(self, artifact_path: Path) -> bool:
        """Trigger reviewer agent"""
        pass
```

### Research Planner Implementation

```python
# agents/research_planner.py
class ResearchPlannerAgent(GatlingAgent):
    def __init__(self):
        super().__init__("research_planner", "coordination")
        self.timeline = self.load_timeline()
        
    def decompose_workstream(self, workstream: str, phase: str) -> list[dict]:
        """Break workstream into concrete tasks"""
        if workstream == "latent_substrate" and phase == "foundation":
            return [
                {
                    "id": "LSA-001",
                    "title": "Implement GovernanceEncoder",
                    "dependencies": [],
                    "acceptance_criteria": self.load_criteria("encoders"),
                    "estimated_tokens": 50000
                },
                {
                    "id": "LSA-002", 
                    "title": "Implement ExecutionEncoder",
                    "dependencies": ["LSA-001"],  # Can reuse patterns
                    "acceptance_criteria": self.load_criteria("encoders"),
                    "estimated_tokens": 40000
                },
                # ... more tasks
            ]
    
    def create_dependency_graph(self) -> nx.DiGraph:
        """Build complete project dependency graph"""
        G = nx.DiGraph()
        # Add all tasks as nodes
        # Add dependency edges
        # Validate no cycles
        return G
    
    def schedule_next_tasks(self) -> list[str]:
        """Find ready tasks (all dependencies complete)"""
        G = self.create_dependency_graph()
        completed = self.get_completed_tasks()
        return [
            task for task in G.nodes()
            if all(dep in completed for dep in G.predecessors(task))
            and task not in completed
        ]
```

## Claude Code Session Management

### Isolated Agent Sessions

Each agent runs in an isolated Claude Code session to prevent context pollution:

```python
# agents/runner.py
from anthropic import Anthropic
import os

def spawn_agent_session(agent_name: str, task: dict) -> str:
    """Create isolated Claude Code session for agent"""
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create agent-specific system prompt
    system_prompt = f"""You are {agent_name} for Project Gatling.
    
Your specific responsibility: {task['description']}

You have access to:
- The project codebase at /home/claude/gatling/
- Task specification: {json.dumps(task)}
- Acceptance criteria: {task['acceptance_criteria']}

Your output must include:
1. Implementation code in source/
2. Tests in test/
3. Documentation in docs/
4. Artifact manifest in outputs/{agent_name}/

Use UV for all Python operations. Follow the coding guidelines in CLAUDE.md.
"""
    
    # Start extended thinking session
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100000,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Begin implementation of task {task['id']}: {task['title']}"
        }]
    )
    
    return response.id  # Session ID for tracking
```

### Parallel Agent Execution

```python
# agents/parallel_runner.py
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def run_agents_parallel(tasks: list[dict]):
    """Run multiple agents in parallel"""
    
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(spawn_agent_session, task['agent'], task)
            for task in tasks
        ]
        
        results = await asyncio.gather(*[
            asyncio.wrap_future(f) for f in futures
        ])
    
    return results
```

## Testing Strategy for Multi-Agent System

### Agent Unit Tests

```python
# test/test_agents/test_latent_substrate_agent.py
def test_governance_encoder_implementation():
    """Test that LatentSubstrateAgent produces valid encoder"""
    
    agent = LatentSubstrateAgent()
    task = load_task("LSA-001")
    
    artifact = agent.execute_task(task)
    
    # Verify outputs exist
    assert Path(artifact['outputs']['code']).exists()
    assert Path(artifact['outputs']['tests']).exists()
    
    # Verify interface contract
    assert artifact['interface']['output_shape'] == "(batch_size, 1024)"
    
    # Verify tests pass
    result = subprocess.run(['uv', 'run', 'pytest', artifact['outputs']['tests']])
    assert result.returncode == 0
```

### Integration Tests

```python
# test/test_integration/test_encoder_energy_integration.py
def test_encoder_to_energy_handoff():
    """Test that encoders integrate correctly with energy functions"""
    
    # Assume encoders are complete
    governance_encoder = load_component("outputs/latent_substrate/governance_encoder")
    execution_encoder = load_component("outputs/latent_substrate/execution_encoder")
    
    # Assume energy function is complete
    E_hierarchy = load_component("outputs/energy_geometry/E_hierarchy")
    
    # Test integration
    policy = load_test_policy()
    plan = load_test_plan()
    
    z_g = governance_encoder(policy)
    z_e = execution_encoder(plan)
    energy = E_hierarchy(z_g, z_e)
    
    assert energy.shape == ()  # Scalar
    assert energy.requires_grad  # Differentiable
```

### System-Level Tests

```python
# test/test_system/test_end_to_end.py
def test_gatling_full_pipeline():
    """Test complete audit + repair pipeline"""
    
    # This test requires all workstreams complete
    if not all_workstreams_complete():
        pytest.skip("Waiting for integration")
    
    gatling = GatlingSystem()
    
    # Test 1: Safe plan passes audit
    safe_plan = load_plan("test/fixtures/safe_email_query.json")
    assert gatling.audit(safe_plan) < gatling.theta_safe
    
    # Test 2: Malicious plan detected
    malicious_plan = load_plan("test/fixtures/scope_blowup.json")
    assert gatling.audit(malicious_plan) > gatling.theta_safe
    
    # Test 3: Repair succeeds
    repaired = gatling.repair(malicious_plan)
    assert gatling.audit(repaired) < gatling.theta_safe
    assert functional_intent_similarity(malicious_plan, repaired) > 0.9
```

## Monitoring and Observability

### Agent Activity Dashboard

```python
# monitoring/dashboard.py
def generate_agent_status():
    """Create real-time status of all agents"""
    
    status = {
        "latent_substrate": {
            "current_task": "LSA-002",
            "progress": "60%",
            "eta": "2h 15m",
            "tokens_used": 35000,
            "status": "in_progress"
        },
        "energy_geometry": {
            "current_task": "EGA-001",
            "progress": "waiting",
            "blocked_by": ["LSA-001"],
            "status": "blocked"
        },
        # ... other agents
    }
    
    return status
```

### Progress Tracking

```python
# monitoring/progress.py
def calculate_project_progress():
    """Calculate overall project completion"""
    
    completed = get_completed_tasks()
    total = get_all_tasks()
    
    return {
        "overall": len(completed) / len(total),
        "by_workstream": {
            "latent_substrate": workstream_progress("latent_substrate"),
            "energy_geometry": workstream_progress("energy_geometry"),
            # ...
        },
        "by_week": week_by_week_progress(),
        "blockers": identify_critical_path_blockers()
    }
```

## Advantages of This Multi-Agent Approach

### 1. **Parallel Development**
- 5 workstreams can progress simultaneously
- Reduces 12-week timeline to actual wall-clock time
- Each agent has focused context (no overwhelming monolithic context)

### 2. **Specialization**
- Each agent becomes expert in its domain
- Accumulated learning within workstream
- Consistent code style per module

### 3. **Quality Control**
- Dedicated reviewer agents provide independent validation
- Separation of implementation and review prevents confirmation bias
- Acceptance criteria enforcement

### 4. **Reproducibility**
- Task queue provides complete execution trace
- Artifacts document every decision
- Handoffs create knowledge transfer record

### 5. **Resilience**
- Agent failure doesn't crash entire project
- Can retry failed tasks with different approaches
- Research exploration mode allows trying multiple solutions

## Getting Started: Practical First Steps

### Step 1: Initialize Project Structure

```bash
# Create directory structure
mkdir -p gatling/{agents,tasks,outputs,handoffs,acceptance_criteria,source,test,docs}

# Initialize UV project
cd gatling
uv init

# Add base dependencies
uv add torch transformers pytest numpy networkx
```

### Step 2: Create Research Planner

```bash
# Implement research planner first
uv run python -c "
from agents.research_planner import ResearchPlannerAgent

rp = ResearchPlannerAgent()
tasks = rp.decompose_workstream('latent_substrate', 'foundation')
rp.populate_task_queue(tasks)
"
```

### Step 3: Launch First Agent

```bash
# Start with LatentSubstrateAgent on LSA-001
uv run python agents/latent_substrate_agent.py --task LSA-001
```

### Step 4: Iterate

As each task completes:
1. Reviewer validates
2. Integration agent merges
3. Research Planner identifies next ready tasks
4. Spawn new agents

## Key Success Factors

1. **Clear Task Decomposition**: Research Planner must create well-defined, atomic tasks
2. **Strong Acceptance Criteria**: Each task needs measurable success conditions
3. **Artifact Discipline**: Every agent output must be documented and versioned
4. **Integration Frequency**: Merge often to catch incompatibilities early
5. **Communication Protocol**: Standardize handoff documents and artifact schemas

## Recommended Timeline

- **Week 1**: Setup infrastructure, implement Research Planner, start LSA
- **Weeks 2-3**: Complete latent substrate (encoders)
- **Weeks 4-6**: Parallel execution of energy + provenance workstreams
- **Weeks 7-9**: Red team + dataset generation + first training runs
- **Weeks 10-12**: Integration, optimization, E2E testing

## Conclusion

This multi-agent architecture maps naturally to Project Gatling's consortium structure while leveraging Claude Code's strengths:

- **Extended thinking** for complex algorithm implementation
- **Tool use** for code generation, testing, file management
- **Parallel sessions** for true concurrent development
- **Specialization** through focused system prompts and context

The result is a development process that mirrors a research team, with each "agent" acting as a specialized researcher working toward the common goal of building the Gatling integrity layer.
