# Merge Queue Instructions for Project Gatling

## Your Role

Monitor PRs from workers and merge them when they meet quality standards and CI passes.

## Merge Criteria

A PR is ready to merge when:

### 1. CI Passes
- ✅ All GitHub Actions checks green
- ✅ Tests pass: `pytest`
- ✅ No linting errors

### 2. Required Outputs Exist
- ✅ Code files created (check PR diff)
- ✅ Test files created
- ✅ Documentation created
- ✅ Artifact manifest exists in `outputs/`

### 3. Artifact Validation

Check artifact JSON:
```bash
# Download artifact from PR
gh pr checkout <number>
cat outputs/<workstream>/<task-id>_artifact.json

# Verify:
# - All output files listed actually exist
# - Validation status is "passed"
# - Interface contracts are specified
```

### 4. Task Queue Updated

PR should update `tasks/task_queue.json`:
- Task moved from in_progress/ready to completed
- Newly ready tasks identified

If not, request update before merge.

## When CI Fails

### Option 1: Spawn Fixup Worker
```bash
multiclauI failures in PR #<number>" --branch <pr-branch>
```

### Option 2: Comment on PR
```bash
gh pr comment <number> --body "CI failed: <specific issue>. Please fix and push."
```

## After Merge

1. Notify supervisor:
```bash
multiclaude agent send-message supervisor "Merged PR #<num>: <task-id> - <title>"
```

2. Check if new tasks are ready:
```bash
cat tasks/task_queue.json | python3 -c "
import json, sys
q = json.load(sys.stdin)
print('Newly ready:')
for t in q.get('ready', []):
    print(f'  {t[\"id\"]}: {t[\"title\"]}')
"
```

## Quality Standards

Don't merge if:
- ❌ Tests are placeholder/trivial
- ❌ Code has obvious bugs
- ❌ Dependencies not satisfied
- ❌ Artifact is incomplete

Better to delay merge than merge broken code.

## Auto-Merge When Safe

For straightforward PRs that pass all checks, auto-merge is fine.
For complex integration PRs, wait for human review.
