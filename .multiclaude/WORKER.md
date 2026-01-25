# Worker Instructions - PRD-Driven Approach

## Your Role

You implement components for Project Gatling by reading requirements from the PRD.

## Workflow

### 1. Read Your Task Assignment

When assigned a task like: `"Implement E_hierarchy (EGA-001)"`

Extract:
- Task ID: `EGA-001`
- Component: `E_hierarchy`

### 2. Get PRD Section from Task Queue

```bash
TASK_ID="EGA-001"  # Your task ID

# Get PRD section
PRD_SECTION=$(cat tasks/task_queue.json | python3 << PYEOF
import json
with open('tasks/task_queue.json') as f:
    q = json.load(f)
for status in q.values():
    if isinstance(status, list):
        for task in status:
            if task.get('id') == '$TASK_ID':
                print(task.get('prd_section', 'docs/PRD.md'))
                break
PYEOF
)

echo "Reading requirements from: $PRD_SECTION"
```

### 3. Read Requirements from PRD

```bash
# Read the PRD section
cat $PRD_SECTION | grep -A 50 "E_hierarchy"

# Or read entire section
cat docs/WORK-DISTRIBUTION.md
```

The PRD contains:
- **What** to build (component description)
- **Why** we need it (motivation)
- **How** it should work (architecture)
- **Acceptance criteria** (quality standards)

**Use the PRD as your spec, not task_queue.json!**

### 4. Implement Per PRD

Follow the architecture described in the PRD:

```python
# Example from reading PRD:
# "E_hierarchy: MLP classifier on [z_g, z_e] → binary violation score"

class HierarchyEnergy(nn.Module):
    """
    Energy function penalizing untrusted data in control flow.
    
    Per PRD Section: Energy Geometry Workstream
    Architecture: MLP on concatenated latents
    """
    def __init__(self, latent_dim=1024):
        # Implementation per PRD specs
        pass
```

### 5. Standard Outputs

Every task requires:
- **Code**: `source/<workstream>/<component>.py`
- **Tests**: `test/test_<component>.py` (>90% coverage)
- **Docs**: `docs/<workstream>/<component>.md`
- **Artifact**: `outputs/<workstream>/<TASK-ID>_artifact.json`

### 6. Update Task Queue on Completion

```bash
python3 << 'PYEOF'
import json
from datetime import datetime

TASK_ID = "EGA-001"  # Your task

with open('tasks/task_queue.json') as f:
    queue = json.load(f)

# Move from ready/in_progress to completed
for status in ['ready', 'in_progress']:
    for i, task in enumerate(queue.get(status, [])):
        if task.get('id') == TASK_ID:
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            queue['completed'].append(task)
            queue[status].pop(i)
            break

# Check for newly ready tasks
completed_ids = {t['id'] for t in queue['completed']}
newly_ready = []

for task in list(queue.get('pending', [])):
    deps = set(task.get('depends_on', []))
    if deps.issubset(completed_ids):
        task['status'] = 'ready'
        task.pop('blocked_by', None)
        newly_ready.append(task)
        queue['ready'].append(task)

queue['pending'] = [t for t in queue['pending'] if t not in newly_ready]

with open('tasks/task_queue.json', 'w') as f:
    json.dump(queue, f, indent=2)

print(f"✓ Marked {TASK_ID} complete")
if newly_ready:
    print(f"\nNewly ready tasks:")
    for t in newly_ready:
        print(f"  {t['id']}: {t['component']}")
PYEOF
```

### 7. Create PR

```bash
git add .
git commit -m "Implement <Component> (<TASK-ID>)

Implements <TASK-ID> per PRD section: <PRD-SECTION>

Components:
- <Component implementation>
- Comprehensive tests (>90% coverage)
- Documentation

All acceptance criteria met."

git push -u origin HEAD

gh pr create \
  --title "Implement <Component> (<TASK-ID>)" \
  --body "Per PRD: <PRD-SECTION>

**Implementation:**
- Component as specified in PRD
- Tests verify all requirements
- Documentation complete

**Validation:**
✅ Tests pass
✅ PRD requirements met
✅ Task queue updated"
```

## Key Principle

**PRD = Requirements**
**Task Queue = State**

- ✅ Read PRD for "what to build"
- ✅ Read task queue for "what's done"
- ❌ Don't expect task queue to have detailed specs

## Example Workflow

```bash
# 1. Assigned: EGA-001
# 2. Read task queue → prd_section: "docs/WORK-DISTRIBUTION.md#energy-geometry"
# 3. Read PRD section → learn about E_hierarchy
# 4. Implement per PRD architecture
# 5. Create tests, docs, artifact
# 6. Update task queue
# 7. Create PR
```

## Resources

- **Requirements**: `docs/PRD.md`, `docs/WORK-DISTRIBUTION.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **State**: `tasks/task_queue.json`
- **Standards**: `CLAUDE.md`

Work autonomously. The PRD has everything you need.