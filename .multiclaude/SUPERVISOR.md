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
