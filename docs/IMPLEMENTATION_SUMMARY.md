# Project Gatling: Multi-Agent Implementation Guide

## Quick Summary

I've designed and implemented a complete multi-agent Claude Code architecture for Project Gatling, inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system). This system enables parallel development across 5 specialized research workstreams, coordinated by a Research Planner agent.

## What Was Created

### 1. **Core Architecture Documents**

#### `MULTI_AGENT_ARCHITECTURE.md` (13,000+ words)
Comprehensive architecture guide covering:
- Specialized Expert Agents pattern with Research Planner coordination
- Agent role definitions for all 5 workstreams + reviewers + integration
- Multi-agent communication protocol (task queue, handoffs, artifacts)
- Implementation workflow for the 12-week timeline
- Testing strategy and monitoring system
- Practical execution patterns (focused, parallel, iterative, exploratory)

### 2. **Base Infrastructure**

#### `agents/base_agent.py`
Abstract base class that all specialized agents inherit from:
- Task loading from global queue
- Task status management (pending → ready → in_progress → review → completed)
- Artifact creation with standardized schema
- Review request triggering
- Handoff document generation for downstream consumers
- Dependency loading from completed tasks
- Comprehensive logging system

**Key Features:**
- Type-safe with enum-based status tracking
- Automatic metadata tagging (timestamps, agent attribution)
- Standardized artifact format for integration
- Built-in error handling and state recovery

#### `agents/research_planner.py`
The orchestrator that coordinates the entire multi-agent system:
- Complete task decomposition for all 5 workstreams (20+ tasks total)
- Dependency graph management using NetworkX
- Task readiness detection (when all dependencies complete)
- Progress tracking and reporting
- Timeline management (12-week sprint)
- Dependency graph visualization

**Implemented Workstream Decompositions:**
1. **Latent Substrate** (4 tasks): GovernanceEncoder, ExecutionEncoder, Intent Predictor, JEPA Training
2. **Energy Geometry** (5 tasks): E_hierarchy, E_provenance, E_scope, E_flow, Composite Energy
3. **Provenance** (3 tasks): Trust Tagging, Repair Engine, Fast-Path Distillation
4. **Red Team** (2 tasks): Corrupter Agent, InfoNCE Training Loop
5. **Dataset** (3 tasks): Gold Traces, Policy Boundary, Minimal Scope Labeling
6. **Integration** (2 tasks): E2E Pipeline, Kona Benchmarks

### 3. **Quickstart System**

#### `quickstart.py`
One-command initialization of the entire multi-agent infrastructure:
- Creates complete directory structure (agents, tasks, outputs, handoffs, etc.)
- Generates acceptance criteria for each component type
- Initializes Research Planner with task breakdowns
- Identifies ready-to-execute tasks
- Creates dependency graph visualization
- Generates helper scripts

**Usage:**
```bash
# Initialize foundation phase (Weeks 1-3)
uv run python quickstart.py --phase foundation

# Initialize with parallel execution plan
uv run python quickstart.py --phase composition --parallel
```

#### `launch_agent.py` (auto-generated)
Helper script for spawning individual agent sessions:
```bash
uv run python launch_agent.py latent_substrate LSA-001
uv run python launch_agent.py energy_geometry EGA-001 --mode exploration
```

#### `agents/latent_substrate_agent.py` (example)
Complete example implementer agent showing the pattern for specialization.

## Key Architectural Patterns

### 1. **Separation of Concerns**
Each agent has a single, focused responsibility:
- **Research Planner**: Task decomposition and coordination only
- **Implementer Agents**: Code implementation for their workstream only
- **Reviewer Agents**: Independent validation with fresh context
- **Integration Agent**: System-level testing and merging

### 2. **Communication via Artifacts**
Agents don't directly communicate - they produce and consume standardized artifacts:
```json
{
  "component": "GovernanceEncoder",
  "outputs": {"code": "...", "tests": "...", "docs": "..."},
  "interface": {"input_shape": "...", "output_shape": "..."},
  "validation_status": "passed",
  "downstream_consumers": ["energy_geometry", "integration"]
}
```

### 3. **Dependency-Driven Execution**
Tasks automatically become "ready" when dependencies complete:
```
LSA-001 (GovernanceEncoder) → complete
  ↓
LSA-002 (ExecutionEncoder) → automatically ready
  ↓
EGA-001 (E_hierarchy) → automatically ready
```

### 4. **Acceptance Criteria Enforcement**
Every task has measurable success conditions:
- Code quality (type hints, docstrings, PEP8)
- Functionality (tests, validation)
- Performance (latency, memory)
- Integration (interface contracts)

### 5. **Parallel Execution**
Independent tasks can run simultaneously:
```bash
# These can all run in parallel (no dependencies between them)
uv run python agents/energy_geometry_agent.py --task EGA-001 &
uv run python agents/energy_geometry_agent.py --task EGA-002 &
uv run python agents/provenance_agent.py --task PA-001 &
```

## Mapping to Anthropic's Multi-Agent Research Pattern

This implementation follows the patterns from Anthropic's blog post:

1. **Specialized Expert Agents** ✓
   - Each workstream has a dedicated agent (LeCun → LatentSubstrateAgent, etc.)
   - Agents have specialized system prompts and focused context

2. **Coordinator Agent** ✓
   - Research Planner acts as orchestrator
   - Manages task decomposition and dependency tracking

3. **Artifact-Based Communication** ✓
   - Agents produce standardized artifact manifests
   - Handoff documents enable knowledge transfer

4. **Independent Review** ✓
   - Reviewer agents use fresh context to avoid confirmation bias
   - Separate validation prevents implementation blind spots

5. **Extended Thinking for Complex Tasks** ✓
   - Each agent can use extended thinking budget for deep implementation
   - Mode flags (focused/exploration) allow different cognitive strategies

## Practical Implementation Flow

### Week 1-3: Foundation (Latent Substrate)

```bash
# 1. Initialize system
uv run python quickstart.py --phase foundation

# 2. Launch first agent
uv run python agents/latent_substrate_agent.py --task LSA-001

# Agent implements GovernanceEncoder:
#   - Reads task spec and acceptance criteria
#   - Implements source/encoders/governance_encoder.py
#   - Writes comprehensive tests
#   - Generates documentation
#   - Creates artifact manifest

# 3. Reviewer validates
uv run python agents/reviewer_latent_substrate.py --artifact outputs/latent_substrate/GovernanceEncoder/artifact.json

# 4. Integration agent merges if approved
uv run python agents/integration_agent.py --merge outputs/latent_substrate/GovernanceEncoder/

# 5. Research Planner identifies next ready task (LSA-002)
# Repeat process
```

### Week 4-6: Composition (Energy + Provenance)

Multiple agents work in parallel:
```bash
# All these tasks are now ready (dependencies from Week 1-3 complete)
parallel --jobs 3 uv run python agents/{1}_agent.py --task {2} ::: \
  energy_geometry EGA-001 ::: \
  energy_geometry EGA-002 ::: \
  provenance PA-001
```

### Week 7-9: Refinement (Red Team + Dataset)

Adversarial loop with feedback:
```bash
# Red team generates hard negatives
uv run python agents/red_team_agent.py --task RTA-001

# Dataset agent labels and validates
uv run python agents/dataset_agent.py --task DA-003

# Energy agent retrains on hard negatives
uv run python agents/energy_geometry_agent.py --task EGA-005 --mode retrain
```

### Week 10-12: Integration & Deployment

System-level validation:
```bash
# Build complete pipeline
uv run python agents/integration_agent.py --task IA-001

# Run comprehensive benchmarks
uv run python agents/integration_agent.py --task IA-002

# Generate deployment artifacts
```

## Advantages of This Approach

### 1. **True Parallelism**
- 5 workstreams can progress simultaneously
- Reduces 12-week sequential timeline to actual wall-clock time
- Each agent has focused, manageable context

### 2. **Specialization & Expertise**
- Agents accumulate domain knowledge within their workstream
- Consistent code style per module
- Deep focus leads to higher quality implementations

### 3. **Built-in Quality Control**
- Dedicated reviewer agents with independent perspective
- Acceptance criteria enforcement prevents drift
- Integration tests catch incompatibilities early

### 4. **Reproducibility & Traceability**
- Task queue provides complete execution history
- Artifacts document every decision
- Handoffs create knowledge transfer record
- Easy to reproduce or restart from any point

### 5. **Resilience**
- Agent failure doesn't crash entire project
- Can retry failed tasks with different approaches
- Research exploration mode allows trying multiple solutions
- Modular design enables swapping components

### 6. **Natural Research Workflow**
Maps directly to how research teams actually work:
- Principal investigator (Research Planner) coordinates
- Researchers (Implementer Agents) execute focused tasks
- Peer review (Reviewer Agents) validates quality
- Integration meetings (Integration Agent) ensure coherence

## How to Get Started

### Immediate Next Steps (5 minutes)

1. **Initialize the system:**
```bash
cd /path/to/gatling
uv run python quickstart.py --phase foundation
```

2. **Inspect what was created:**
```bash
# View task queue
cat tasks/task_queue.json

# View dependency graph
open outputs/task_graph.png

# View ready tasks
uv run python -c "
from agents.research_planner import ResearchPlannerAgent
planner = ResearchPlannerAgent()
print(planner.get_ready_tasks())
"
```

3. **Launch your first agent:**

Since you're already using Claude Code, you can manually execute tasks by:

**Option A: Direct Implementation**
Just start implementing the first ready task (LSA-001) following the spec in `tasks/task_queue.json`. When complete, create the artifact manifest manually.

**Option B: Use API to Spawn Agents**
Create separate Claude Code sessions via the Anthropic API for each agent, passing agent-specific system prompts.

**Option C: Sequential Multi-Agent**
Work through tasks sequentially but maintain the artifact/handoff structure for reproducibility.

### Implementation Strategy Recommendation

Given your constraints, I recommend this hybrid approach:

1. **Use Research Planner for coordination** (keep using it!)
   - It already provides task breakdown
   - Tracks dependencies and progress
   - Identifies what's ready to work on

2. **Implement tasks yourself but follow agent patterns**
   - Work on one task at a time
   - Create proper artifact manifests
   - Write handoff documents
   - Follow acceptance criteria strictly

3. **Use Claude Code for each task**
   - Start fresh session per task for clean context
   - Load task spec + acceptance criteria
   - Implement until criteria met
   - Create artifact before moving to next

4. **Periodically check coordination**
   ```bash
   # Update task status
   uv run python -c "
   from agents.research_planner import ResearchPlannerAgent
   planner = ResearchPlannerAgent()
   report = planner.generate_progress_report()
   print(report)
   "
   ```

## Advanced: True Multi-Agent with API

For full automation, you'd spawn agents via Anthropic API:

```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load task spec
task = load_task("LSA-001")

# Create agent-specific system prompt
system_prompt = f"""You are LatentSubstrateAgent for Project Gatling.

Your task: {task['title']}
{task['description']}

Acceptance criteria:
{json.dumps(task['acceptance_criteria'], indent=2)}

You must:
1. Implement code in source/
2. Write tests in test/
3. Create docs in docs/
4. Generate artifact manifest

Use UV for all operations. Follow guidelines in CLAUDE.md.
"""

# Start session
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=100000,
    system=system_prompt,
    messages=[{
        "role": "user",
        "content": f"Begin implementation of {task['id']}"
    }]
)

# Monitor and collect outputs...
```

## File Manifest

Here's everything that was created:

```
/home/claude/
├── MULTI_AGENT_ARCHITECTURE.md    # Complete architecture guide (13K+ words)
├── agents/
│   ├── base_agent.py               # Abstract base class for all agents
│   ├── research_planner.py         # Orchestrator agent with full task decomposition
│   └── latent_substrate_agent.py   # Example implementer agent
├── quickstart.py                   # One-command initialization script
└── launch_agent.py                 # Helper script for agent spawning (auto-generated)

Directory structure (created by quickstart.py):
├── tasks/                          # Task queue and specs
├── outputs/                        # Agent outputs by workstream
├── handoffs/                       # Inter-agent communication
├── acceptance_criteria/            # Validation specs
├── source/                         # Final integrated code
├── test/                           # Test suite
├── docs/                           # Documentation
└── logs/                           # Agent execution logs
```

## Success Metrics

You'll know the multi-agent system is working when:

1. **Task queue shows progress**
   - Tasks move from pending → ready → in_progress → review → completed
   - Dependencies automatically unblock downstream tasks

2. **Artifacts accumulate**
   - Each completed task has artifact.json in outputs/
   - Artifact interfaces match downstream consumer expectations

3. **Handoffs enable integration**
   - Handoff documents provide all context for next agent
   - No need to re-read entire project history

4. **Progress is measurable**
   - Research Planner reports show % completion
   - Dependency graph visualizes critical path
   - Blockers are identifiable

5. **Quality is enforced**
   - Acceptance criteria prevent premature completion
   - Review step catches issues before merge
   - Integration tests validate component interactions

## Troubleshooting

**Q: Task shows as "blocked" but dependencies are complete?**
Update task queue manually or re-run Research Planner to refresh ready tasks.

**Q: How do I add a new task mid-project?**
Use Research Planner's `populate_task_queue()` with new task spec. It will automatically determine if ready based on dependencies.

**Q: Can I change acceptance criteria after task starts?**
Yes, update `acceptance_criteria/*.json` but be aware this might invalidate in-progress work.

**Q: How do I handle failed tasks?**
Set status to "failed" in queue, analyze logs, fix issue, reset to "ready" and re-run.

## Conclusion

You now have a complete multi-agent architecture for Project Gatling that:

- Maps naturally to your 5-workstream research structure
- Enables parallel development across components
- Enforces quality through acceptance criteria
- Maintains reproducibility through artifact tracking
- Scales from single-developer to full team coordination

The system is designed to work equally well whether you're:
- Working solo and using it for structure/tracking
- Coordinating a small team with manual agent execution
- Fully automating with API-spawned Claude Code sessions

Start with `quickstart.py` and grow into whichever execution model fits your workflow!
