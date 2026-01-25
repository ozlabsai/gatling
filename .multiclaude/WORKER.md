# Worker Instructions for Project Gatling

## Your Role

You are a worker agent implementing a specific task from the Gatling project task queue.

## Workflow

### 1. Extract Task Information

When assigned a task like:
"Implement GovernanceEncoder (LSA-001)"

Extract:
- Task ID: `LSA-001`
- Component: `GovernanceEncoder`

### 2. Read Full Task Specification
```bash
# Read the complete task spec
cat tasks/task_queue.json | python3 -c "
import json, sys
q = json.load(sys.stdin)
for status in q.keys():
    for task in q.get(status, []):
        if task['id'] == 'LSA-001':  # Replace with your task ID
            import pprint
            pprint.pprint(task)
"
```

### 3. Check Dependencies

BEFORE implementing, verify dependencies are merged:
```bash
# Check if dependency exists in main
dep_id="LSA-001"  # Replace with actual dependency
if ! git log origin/main --oneline | grep -q "$dep_id"; then
    echo "ERROR: Dependency $dep_id not merged yet!"
    multiclaude agent send-message supervisor "Cannot start - dependency $dep_id not merged"
    exit 1
fi
```

### 4. Implementation Standards

**Use UV for all operations:**
```bash
uv add torch transformers    # Add dependencies
uv run pytest test/          # Run tests
uv run ruff check source/    # Lint (if available)
```

**Code location by workstream:**
- LSA (Latent Substrate): `source/encoders/`
- EGA (Energy Geometry): `source/energy/`
- PA (Provenance): `source/provenance/`
- RTA (Red Team): `source/adversarial/`
- DA (Dataset): `source/dataset/`
- IA (Integration): `source/integration/`

**Test location:**
- `test/test_<component>.py`

**Documentation:**
- `docs/<workstream>/<component>.md`

### 5. Development Process
```bash
# 1. Research best practices
# Use web search for current approaches in your domain

# 2. Design solution
# Plan architecture, choose libraries

# 3. Implement incrementally
# - Start with interfaces/types
# - Implement core logic
# - Add validation
# - Write tests alongside code

# 4. Run tests frequently
uv run pytest test/test_your_component.py -v

# 5. Fix failures
# Debug, refine, re-test

# 6. Document
# Write clear docstrings and usage examples
```

### 6. Required Outputs

Every task MUST create:

1. **Code**: Implementation in `source/`
2. **Tests**: Comprehensive tests in `test/` (>90% coverage target)
3. **Docs**: Documentation in `docs/`
4. **Artifact**: Manifest in `outputs/<workstream>/<task-id>_artifact.json`

### 7. Artifact Format
```json
{
    "task_id": "LSA-001",
    "component": "GovernanceEncoder",
    "version": "0.1.0",
    "outputs": {
        "code": "source/encoders/governance_encoder.py",
        "tests": "test/test_governance_encoder.py",
        "docs": "docs/encoders/governance_encoder.md"
    },
    "interface": {
        "input_shape": "(batch_size, policy_tokens)",
        "output_shape": "(batch_size, 1024)",
        "latency_p99": "45ms"
    },
    "validation_status": "passed",
    "test_coverage": "94%",
    "dependencies_used": ["torch", "transformers"]
}
```

### 8. Update Task Queue (CRITICAL!)

When complete, mark the task as done:
```bash
python3 << 'PYEOF'
import json
from datetime import datetime

# Replace with your task ID
TASK_ID = "LSA-001"

with open('tasks/task_queue.json') as f:
    queue = json.load(f)

# Find and update task
for status in ['ready', 'in_progress', 'pending']:
    for i, task in enumerate(queue.get(status, [])):
        if task['id'] == TASK_ID:
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            queue['completed'].append(task)
            queue[status].pop(i)
            print(f"✓ Marked {TASK_ID} as completed")
            break

# Check for newly ready tasks
completed_set = {t['id'] for t in queue['completed']}
newly_ready = []

for task in list(queue.get('pending', [])):
    deps = set(task.get('dependencies', []))
    if deps.issubset(completed_set):
        task['status'] = 'ready'
        newly_ready.append(task)
      print(f"→ {task['id']} is now ready")

# Update queue
queue['pending'] = [t for t in queue.get('pending', []) if t not in newly_ready]
queue['ready'].extend(newly_ready)

with open('tasks/task_queue.json', 'w') as f:
    json.dump(queue, f, indent=2)

print(f"\n✓ Updated task_queue.json")
PYEOF
```

### 9. Create Pull Request
```bash
# Commit your work
git add .
git commit -m "Implement <Component> (<TASK-ID>)

Implements task <TASK-ID> from task queue.

Changes:
- Implemented <Component> in source/
- Added comprehensive tests (>90% coverage)
- Created documentation
- Generated artifact manifest

All tepass. Ready for review."

# Push and create PR
git push -u origin HEAD

gh pr create \
  --title "Implement <Component> (<TASK-ID>)" \
  --body "Implements <TASK-ID>.

**Outputs:**
- Implementation: source/path/to/file.py
- Tests: test/test_file.py
- Documentation: docs/path/to/doc.md
- Artifact: outputs/workstream/TASK-ID_artifact.json

**Validation:**
- ✅ All tests pass
- ✅ Coverage >90%
- ✅ Task queue updated
- ✅ Dependencies satisfied"
```

### 10. Pre-PR Checklist

Before creating PR:
- [ ] All required files created
- [ ] Tests pass: `uv run pytest test/test_*.py -v`
- [ ] Code follows Python standards (type hints, docstrings)
- [ ] Artifact manifest created and valid
- [ ] Task dated (task marked completed)
- [ ] Dependencies properly declared in pyproject.toml
- [ ] Documentation is clear and complete

## Resources

- **Task definitions**: `tasks/task_queue.json`
- **Project requirements**: `docs/PRD.md`
- **Workstream details**: `docs/WORK-DISTRIBUTION.md`
- **Coding standards**: `CLAUDE.md` (if exists)
- **Acceptance criteria**: `acceptance_criteria/<type>.json`

## When Stuck

1. Check if dependencies are actually merged
2. Review similar completed tasks for patterns
3. Ask supervisor: `multiclaude agent send-message supervisor "Need help with <issue>"`
4. Search for current best practices online
5. Review the PRD and requirements again

## Quality Standards

This is production code for a research project. High standards apply:
- Comprehensive testing (not just smoke tests)
- Clear documentation (others will use this)
- Proper error handling
- Type safety where possible
- Performance considerations (latency requirements)

Work autonomously but thoughtfully. Take time to do it right.
