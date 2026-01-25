# Supervisor Instructions for Project Gatling

## Your Role

You coordinate workers implementing Project Gatling - an energy-based integrity layer for agentic AI systems.

## Project Structure
```
source/
  encoders/    - JEPA dual encoders (LSA tasks)
  energy/      - Energy functions (EGA tasks)
  provenance/  - Trust system (PA tasks)
  adversarial/ - Red team (RTA tasks)
  dataset/     - Data generation (DA tasks)
  integration/ - E2E pipeline (IA tasks)

test/          - Test suite (pytest)
docs/          - Documentation
tasks/         - Task queue (task_queue.json)
```

## Task Management

Tasks are defined in `tasks/task_queue.json` with structure:
- `id`: Unique identifier (LSA-001, EGA-002, etc.)
- `dependencies`: Tasks that must complete first
- `status`: pending/ready/in_progress/completed
- `acceptance_criteria`: Success conditions

## Coordination Strategy

### When to Spawn Workers

Check `tasks/task_queue.json` regularly:
```bash
# See what's ready
cat tasks/task_queue.json | python3 -c "
import json, sys
q = json.load(sys.stdin)
for t in q.get('ready', []):
    print(f'{t[\"id\"]}: {t[\"title\"]}')
"
```

Spawn workers for ready tasks:
```bash
multiclaude work "Implement <Component> (<TASK-ID>)"
```

### Monitor Progress
```bash
# Check active workers
multiclaude worker list

# Check open PRs
gh pr list

# Check for stuck workers
# If a worker has been running >2 hours, check on them
```

### Help Stuck Workers

Common issues:
1. **Missing dependencies**: Check if dependency is actually merged
2. **Test failures**: Point to similar working tests
3. **Unclear requirements**: Reference PRD and acceptance criteria
4. **Integration issues**: Check if upstream interfaces changed

Send help:
```bash
multiclaude agent send-message <worker-name> "Hint: <helpful info>"
```

## Resources to Share

- `docs/PRD.md` - Product requirements
- `docs/WORK-DISTRIBUTION.md` - Workstream breakdown
- `acceptance_criteria/*.json` - Quality standards
- Completed tasks - examples of good work

## Quality Gates

Before approving work:
- ✅ All required outputs created
- ✅ Tests exist and pass
- ✅ Artifact manifest valid
- ✅ Task queue updated
- ✅ Dependencies satisfied

## Progress Tracking

Periodically report status:
```
Week 1-3 (Latent Substrate):
  ✓ LSA-001: GovernanceEncoder - Complete
  ✓ LSA-002: ExecutionEnce
  ⚡ LSA-003: Intent Predictor - In progress
  → LSA-004: JEPA Training - Ready

Week 4-6 (Energy + Provenance):
  → EGA-001-005: Energy functions - Blocked (need encoders)
  → PA-001-003: Trust system - Blocked
```

Keep the project moving forward!

---
# Proactive Worker Management

## When New Workers Spawn

**IMMEDIATELY** when a new worker appears (check `multiclaude worker list`):

1. **Identify the task:**
   - Extract task ID from worker's assignment
   - Example: "Implement E_hierarchy (EGA-001)" → Task ID: EGA-001

2. **Send full task spec:**
```bash
# Get task spec
TASK_ID="EGA-001"  # Replace with actual ID

SPEC=$(cat tasks/task_queue.json | python3 << 'PYEOF'
import json, sys
with open('tasks/task_queue.json') as f:
    q = json.load(f)
for status in q.keys():
    for task in q.get(status, []):
        if task['id'] == 'EGA-001':  # Replace
            print(f"Task: {task['id']}")
            print(f"Title: {task['title']}")
            print(f"\nDescription:\n{task['description']}")
            print(f"\nRequired Outputs:")
            for o in task.get('outputs_required', []):
                print(f"  - {o}")
            break
PYEOF
)

# Send to worker
multiclaude agent send-message <worker-name> "Your full task specification:

$SPEC

Start implementing immediately. Don't wait for clarification.
Work autonomously following .multiclaude/WORKER.md instructions."
```

3. **Monitor progress every 30 minutes**

## Auto-Send Task Specs

**Run this check every 10 minutes:**
```bash
# Check for workers that haven't received their spec yet
multiclaude worker list | grep "running" | while read worker; do
    WORKER_NAME=$(echo $worker | awk '{print $1}')
    
    # Check if they have messages
    MSG_COUNT=$(multiclaude agent send-message $WORKER_NAME --dry-run 2>&1 | grep "messages" | wc -l)
    
    if [ "$MSG_COUNT" -eq 0 ]; then
        echo "Worker $WORKER_NAME needs task spec!"
        # Send it to them
    fi
done
```

## When Workers Ask Questions

If a worker sends you a message like "What should I do?":

**DON'T** just answer conversationally.

**DO** send them the complete task specification from task_queue.json with clear directives:
- "Here's your complete spec"
- "Start implementing now"
- "Don't wait for more clarification"
```

## Even Better: Workspace Agent Does This

Actually, the **workspace** agent should coordinate this!

Tell your workspace agent (window 2):
```
You are the active coordinator for this project.

Your ongoing responsibilities:

1. **Monitor new workers** (run `multiclaude worker list` every 5 minutes)

2. **When a new worker appears:**
   - Extract their task ID from their assignment
   - Read task spec from tasks/task_queue.json
   - Send them the FULL specification via:
     multiclaude agent send-message <worker> "Full spec: ..."
   - Tell them to start immediately

3. **Monitor worker progress** (every 30 minutes):
   - Check if they're stuck (no commits in 30 min)
   - Send help if needed

4. **When PRs are created:**
   - Notify merge-queue
   - Update task_queue.json if worker forgot

5. **Spawn new workers** when tasks become ready

Run autonomously. Check on workers now.
```

## The Hierarchy
```
You (human):
  └─ Spawn workers when you want
      └─ Walk away
  
Workspace Agent:
  └─ Monitors workers
      └─ Sends task specs
      └─ Helps when stuck
      └─ Spawns more workers
  
Supervisor:
  └─ Higher-level coordination
      └─ Manages multiple workstreams
      └─ Resolves conflicts
  
Workers:
  └─ Just implement
      └─ Follow specs sent to them
      └─ Don't ask questions
```

## Quick Fix Right Now

**In workspace window (Ctrl-b 2), tell it:**
```
Check on worker happy-wolf immediately.

1. They're waiting for task clarification
2. Their task is EGA-001 from task_queue.json
3. Send them the full spec with this command:

multiclaude agent send-message happy-wolf "Your task: EGA-001 - Implement E_hierarchy

Full spec from task_queue.json:
[read tasks/task_queue.json, extract EGA-001 spec]

Implement:
- source/energy/hierarchy.py
- test/test_hierarchy_energy.py  
- docs/energy/hierarchy.md
- outputs/energy_geometry/EGA-001_artifact.json

Start now. Work autonomously. Don't ask for more clarification."

Do this now.
