# How to Actually Use the Multi-Agent System

## Three Usage Modes (Pick What Works for You)

### Mode 1: **Solo Developer with Structure** (Recommended for Starting)
You use the task breakdown for organization, but you do the implementation yourself.

**Step-by-step:**

```bash
# 1. Initialize the project
cd /path/to/gatling
uv run python quickstart.py --phase foundation

# 2. See what tasks are ready
uv run python -c "
from agents.research_planner import ResearchPlannerAgent
planner = ResearchPlannerAgent()
ready = planner.get_ready_tasks()
print('Ready tasks:', ready)
"

# Output: Ready tasks: ['LSA-001', 'PA-001', 'DA-001']

# 3. Pick a task to work on
# Read the task spec:
cat tasks/task_queue.json | jq '.ready[] | select(.id=="LSA-001")'

# 4. Implement the task (you do this manually or with Claude Code)
# For example, for LSA-001 (GovernanceEncoder):
# - Create source/encoders/governance_encoder.py
# - Create test/test_governance_encoder.py
# - Create docs/encoders/governance_encoder.md

# 5. Mark task complete
uv run python -c "
from agents.base_agent import GatlingAgent, TaskStatus
import json

class MyAgent(GatlingAgent):
    def execute_task(self, task): pass

agent = MyAgent('manual', 'latent_substrate')

# Create artifact
artifact = {
    'component': 'GovernanceEncoder',
    'outputs': {
        'code': 'source/encoders/governance_encoder.py',
        'tests': 'test/test_governance_encoder.py',
        'docs': 'docs/encoders/governance_encoder.md'
    },
    'interface': {
        'input_shape': '(batch_size, policy_tokens)',
        'output_shape': '(batch_size, 1024)'
    }
}

agent.complete_task('LSA-001', artifact)
"

# 6. Check progress
uv run python -c "
from agents.research_planner import ResearchPlannerAgent
planner = ResearchPlannerAgent()
report = planner.generate_progress_report()
print(json.dumps(report, indent=2))
"

# 7. See what's ready next (LSA-002 should now be ready since LSA-001 is its dependency)
```

**Pro:** Maximum control, works with existing workflow
**Con:** Manual tracking of completions

---

### Mode 2: **Semi-Automated with Claude Code Sessions**
You spawn new Claude Code sessions for each task, passing in the task context.

**Step-by-step:**

```bash
# 1. Initialize
uv run python quickstart.py --phase foundation

# 2. For each ready task, start a fresh Claude Code session
# In your terminal or via API:

# Session for LSA-001
claude-code --context "
Task: LSA-001 - Implement GovernanceEncoder
$(cat tasks/task_queue.json | jq '.ready[] | select(.id==\"LSA-001\")')

Requirements:
- Implement source/encoders/governance_encoder.py
- Create comprehensive tests
- Follow acceptance criteria in acceptance_criteria/encoders.json
- Create artifact manifest when done

Begin implementation.
"

# Claude Code then:
# - Reads the requirements
# - Implements the code
# - Writes tests
# - Creates documentation
# - Outputs: 'I've implemented GovernanceEncoder at source/encoders/...'

# 3. When Claude finishes, you manually mark task complete (same as Mode 1)

# 4. Repeat for next ready task
```

**Pro:** Fresh context per task, Claude does implementation
**Con:** Still requires you to spawn each session

---

### Mode 3: **Fully Automated Multi-Agent** (Advanced)
Agents spawn themselves via the Anthropic API and run autonomously.

**Step-by-step:**

```bash
# 1. Set up automated agent runner
# Create: agents/automated_runner.py
```

```python
# agents/automated_runner.py (already created for you!)
```

**Full automated workflow:**

```bash
# 1. Initialize project
uv run python quickstart.py --phase foundation

# 2. Start automated runner in monitoring mode
uv run python agents/automated_runner.py --monitor --max-parallel 3

# The runner will:
# - Check task queue every 30 seconds
# - Spawn Claude Code agents for up to 3 ready tasks
# - Monitor agent progress
# - Mark tasks complete when artifacts are created
# - Automatically start next ready tasks
# - Run until all tasks complete
```

**What happens autonomously:**

```
[10:00:00] --- Iteration 1 ---
Ready tasks: 3 (LSA-001, PA-001, DA-001)
Active agents: 0/3

SPAWNING AGENT: latent_substrate
Task: LSA-001 - Implement GovernanceEncoder
✓ Agent spawned: msg_abc123

SPAWNING AGENT: provenance
Task: PA-001 - Implement Trust Tier Tagging
✓ Agent spawned: msg_def456

SPAWNING AGENT: dataset
Task: DA-001 - Generate Gold Traces
✓ Agent spawned: msg_ghi789

[10:00:30] --- Iteration 2 ---
Ready tasks: 0
Active agents: 3/3
Waiting 30s...

[10:15:00] --- Iteration 30 ---
✓ Task LSA-001 completed by latent_substrate
  Artifact validated: GovernanceEncoder
  Status: REVIEW

Ready tasks: 1 (LSA-002)  # Now ready because LSA-001 is done!
Active agents: 2/3

SPAWNING AGENT: latent_substrate
Task: LSA-002 - Implement ExecutionEncoder
✓ Agent spawned: msg_jkl012

... continues until all tasks done ...
```

**Pro:** Zero intervention required, true parallel execution
**Con:** Requires API access, costs API credits, need monitoring

---

## 2. **YES - Each Agent Has Full Claude Code Capabilities!**

Each spawned agent is a **full Claude Code session** with:

### ✓ All Computer Tools
- `bash_tool` - Run any command
- `str_replace` - Edit files
- `create_file` - Create new files  
- `view` - Read files and directories
- `web_search` - Search the web
- `web_fetch` - Fetch web pages

### ✓ All Skills
- docx creation/editing
- pptx creation/editing
- xlsx creation/editing
- PDF manipulation
- Frontend design
- Any custom skills you've added

### ✓ Extended Thinking
- Can use extended thinking for complex algorithms
- Makes independent decisions
- Searches web for best practices
- Self-corrects when tests fail

### Example of Agent Autonomy

Here's what happens when you spawn an agent for LSA-001 (GovernanceEncoder):

```
Agent receives system prompt with task spec...

Agent thinks: "I need to implement a transformer-based encoder. Let me search for 
current best practices..."

[Agent uses web_search: "transformer encoder implementation pytorch 2024"]

Agent: "Found HuggingFace transformers library is standard. Let me check the 
specific requirements..."

[Agent uses view: acceptance_criteria/encoders.json]

Agent: "I need to output 1024-dim vectors with <50ms latency. Let me design 
the architecture..."

[Agent uses extended thinking to design the model]

Agent: "Now implementing..."

[Agent uses create_file: source/encoders/governance_encoder.py]

Agent writes the code...

[Agent uses create_file: test/test_governance_encoder.py]

Agent writes tests...

[Agent uses bash_tool: "uv run pytest test/test_governance_encoder.py"]

Tests pass! ✓

[Agent uses create_file: docs/encoders/governance_encoder.md]

Agent writes documentation...

[Agent uses create_file: outputs/latent_substrate/LSA-001_artifact.json]

Agent creates artifact manifest...

Done! Task LSA-001 complete.
```

**The agent did all of this autonomously!** No intervention from you.

---

## How Agents Run Without Constant Intervention

### The Key: Artifact-Based Completion Detection

The `automated_runner.py` checks if an agent is done by looking for the artifact manifest:

```python
# Agent creates this when complete:
outputs/latent_substrate/LSA-001_artifact.json

# Runner checks for this file:
if artifact_path.exists():
    # Task complete!
    mark_task_as_done()
    spawn_next_ready_task()
```

### Agent's Internal Loop

Each agent follows this autonomous loop:

1. **Read task spec** (from system prompt)
2. **Search for information** if needed (web_search)
3. **Design solution** (extended thinking)
4. **Implement code** (create_file, str_replace)
5. **Write tests** (create_file)
6. **Run tests** (bash_tool: pytest)
7. **Fix failures** (if tests fail, agent debugs and fixes)
8. **Write docs** (create_file)
9. **Create artifact** (signals completion)
10. **Exit**

The runner's job is just to:
- Spawn agents with task specs
- Wait for artifact files
- Mark tasks complete
- Spawn next ready agents

### How Multiple Agents Work in Parallel

```
Timeline:

10:00 - Spawn Agent A (LSA-001), Agent B (PA-001), Agent C (DA-001)

10:05 - Agent A: Implementing GovernanceEncoder...
        Agent B: Implementing Trust Tagging...
        Agent C: Searching for dataset generation best practices...

10:10 - Agent A: Running tests...
        Agent B: Running tests...
        Agent C: Setting up dataset pipeline...

10:15 - Agent A: ✓ Complete! Artifact created.
        Agent B: Still testing...
        Agent C: Generating samples...

10:15 - Runner detects Agent A done
        Spawns Agent D (LSA-002) - can start because LSA-001 done!

10:20 - Agent B: ✓ Complete!
        Agent C: Still generating...
        Agent D: Implementing ExecutionEncoder...

... and so on ...
```

All agents run **completely independently** in parallel!

---

## Practical Example: Running Your First Autonomous Agent

### Option A: Single Task Test

```bash
# 1. Initialize
uv run python quickstart.py --phase foundation

# 2. Run ONE task autonomously
uv run python agents/automated_runner.py --task LSA-001

# This will:
# - Spawn a Claude Code agent
# - Agent implements GovernanceEncoder autonomously
# - Creates artifact when done
# - You can watch logs in real-time

# 3. Check results
cat outputs/latent_substrate/LSA-001_artifact.json
cat source/encoders/governance_encoder.py
uv run pytest test/test_governance_encoder.py
```

### Option B: Full Autonomous Mode

```bash
# 1. Initialize
uv run python quickstart.py --phase foundation

# 2. Start autonomous runner (runs forever until all tasks done)
uv run python agents/automated_runner.py --monitor --max-parallel 5

# Now walk away! The system will:
# - Spawn 5 agents immediately for ready tasks
# - When any agent finishes, spawn next ready task
# - Continue until all foundation phase tasks are done
# - You can check progress anytime in logs/

# 3. Monitor progress (in another terminal)
watch -n 10 'python -c "
from agents.research_planner import ResearchPlannerAgent
planner = ResearchPlannerAgent()
report = planner.generate_progress_report()
print(f\"Progress: {report[\"overall_progress\"]:.1f}%\")
print(f\"Completed: {report[\"tasks\"][\"completed\"]}\")
print(f\"Active: {report[\"tasks\"][\"in_progress\"]}\")
"'
```

---

## What You Need to Do (Minimal!)

### For Fully Automated Mode:

1. **Setup (once):**
   ```bash
   # Add API key to .env
   echo "ANTHROPIC_API_KEY=your_key_here" >> .env
   
   # Initialize project
   uv run python quickstart.py --phase foundation
   ```

2. **Run (once per phase):**
   ```bash
   uv run python agents/automated_runner.py --monitor
   ```

3. **Monitor (optional):**
   ```bash
   # Check progress
   tail -f logs/research_planner.log
   
   # Or check specific agent
   tail -f logs/latent_substrate_agent.log
   ```

4. **Review (when phase completes):**
   ```bash
   # Check all artifacts were created
   find outputs/ -name "*_artifact.json"
   
   # Run integration tests
   uv run pytest test/
   ```

That's it! The agents handle everything else.

---

## Cost Estimates

### Tokens per Task (Estimated)

- **Simple task** (LSA-003): ~30,000 tokens = ~$0.90
- **Medium task** (LSA-001): ~50,000 tokens = ~$1.50  
- **Complex task** (PA-002): ~70,000 tokens = ~$2.10

### Total Project Cost (Approximate)

- 20 tasks × average 50k tokens = 1M tokens
- **Total: ~$30-50** for entire project

If using extended thinking (higher quality):
- 20 tasks × 100k tokens = 2M tokens
- **Total: ~$60-100**

Much cheaper than your time doing it manually!

---

## Troubleshooting

### Q: Agent gets stuck?
```bash
# Check what it's doing
tail -f logs/latent_substrate_agent.log

# If truly stuck (rare), kill and restart:
# (automated_runner will mark task as failed and you can retry)
```

### Q: Agent completes but artifact invalid?
```bash
# Manually check what went wrong
cat outputs/latent_substrate/LSA-001_artifact.json

# Check the code it created
cat source/encoders/governance_encoder.py

# Re-run with more specific instructions by editing task spec
```

### Q: Want to give agent more guidance mid-task?
Currently, agents run fully autonomously once spawned. For more control, use **Mode 2** (semi-automated) where you interact with each Claude Code session.

---

## Best Practice Recommendations

### Start Small
```bash
# Test with one task first
uv run python agents/automated_runner.py --task LSA-001

# If that works, run 2-3 in parallel
uv run python agents/automated_runner.py --monitor --max-parallel 2

# Once confident, scale up
uv run python agents/automated_runner.py --monitor --max-parallel 5
```

### Monitor First Run
Keep logs open to see what agents are doing:
```bash
# Terminal 1: Run agents
uv run python agents/automated_runner.py --monitor

# Terminal 2: Watch logs
tail -f logs/*.log

# Terminal 3: Check progress
watch -n 30 'find outputs/ -name "*.json" | wc -l'
```

### Hybrid Approach
- Use **automated mode** for straightforward tasks (encoders, energy terms)
- Use **manual mode** for novel research tasks (architecture decisions)
- Use **semi-automated** for tasks needing occasional guidance

---

## Summary

**Question 1: How do I use it?**

Three modes, pick what fits your style:
- **Solo**: Use task structure, implement yourself
- **Semi-Auto**: Spawn Claude sessions per task, some automation
- **Full-Auto**: `automated_runner.py --monitor`, walk away

**Question 2: Do agents have full capabilities?**

**YES!** Each agent is a complete Claude Code session with:
- All computer tools (bash, file ops, web search)
- All skills (docx, pptx, coding)
- Extended thinking for hard problems
- Full autonomy to implement, test, debug, document

They run **completely independently** and signal completion via artifact files. The runner just spawns them and collects results.

**You can literally start the runner and come back hours later to completed implementations!**

