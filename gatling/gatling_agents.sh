#!/usr/bin/env bash
#
# Gatling Autonomous Agent System
#
# Simple approach: Spawn Claude Code instances to work on tasks autonomously.
# No complex teams, no supervisor, no merge queue - just workers doing tasks.
#
# Usage:
#   ./gatling_agents.sh init                    # Setup
#   ./gatling_agents.sh work                    # Spawn agent for next ready task
#   ./gatling_agents.sh work LSA-002            # Spawn agent for specific task
#   ./gatling_agents.sh watch                   # Attach to see agents working
#   ./gatling_agents.sh status                  # Show what's happening
#   ./gatling_agents.sh complete LSA-002        # Mark task complete (after agent finishes)

set -euo pipefail

PROJECT_ROOT="$(pwd)"
TMUX_SESSION="gatling"
WORKTREE_DIR="$PROJECT_ROOT/.workers"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[gatling]${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

# Check prerequisites
check_prereqs() {
    local missing=0
    
    if ! command -v tmux &> /dev/null; then
        error "tmux not installed (brew install tmux)"
        missing=1
    fi
    
    if ! command -v claude &> /dev/null; then
        error "claude CLI not found (install from claude.ai/code)"
        missing=1
    fi
    
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a git repository"
        missing=1
    fi
    
    if [[ ! -f "tasks/task_queue.json" ]]; then
        error "tasks/task_queue.json not found (run: uv run python quickstart.py --phase foundation)"
        missing=1
    fi
    
    if [[ $missing -eq 1 ]]; then
        exit 1
    fi
}

# Initialize
init() {
    log "Initializing Gatling autonomous agent system..."
    
    check_prereqs
    
    # Create directories
    mkdir -p "$WORKTREE_DIR"
    mkdir -p outputs/{latent_substrate,energy_geometry,provenance,red_team,dataset,integration}
    
    # Create tmux session if needed
    if ! tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        tmux new-session -d -s $TMUX_SESSION -n main -c "$PROJECT_ROOT"
        success "Created tmux session: $TMUX_SESSION"
    else
        success "Tmux session already exists: $TMUX_SESSION"
    fi
    
    success "Initialization complete"
    echo ""
    echo "Next steps:"
    echo "  ./gatling_agents.sh work      # Start working on next task"
    echo "  ./gatling_agents.sh watch     # Watch agents work"
}

# Get next ready task from queue
get_next_task() {
    python3 << 'EOF'
import json
import sys

with open('tasks/task_queue.json') as f:
    queue = json.load(f)

ready = queue.get('ready', [])
if not ready:
    sys.exit(1)

# Get highest priority task
priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
ready.sort(key=lambda t: priority_order.get(t.get('priority', 'medium'), 2))

print(json.dumps(ready[0]))
EOF
}

# Get specific task
get_task() {
    local task_id="$1"
    
    python3 << EOF
import json
import sys

with open('tasks/task_queue.json') as f:
    queue = json.load(f)

for status in ['ready', 'pending', 'in_progress']:
    for task in queue.get(status, []):
        if task['id'] == '$task_id':
            print(json.dumps(task))
            sys.exit(0)

sys.exit(1)
EOF
}

# Spawn agent for task
spawn_agent() {
    local task_id="${1:-}"
    
    check_prereqs
    
    # Get task
    local task_json
    if [[ -z "$task_id" ]]; then
        log "Getting next ready task..."
        task_json=$(get_next_task) || {
            warn "No ready tasks found"
            echo ""
            echo "Check status with: ./gatling_agents.sh status"
            echo "Or initialize with: uv run python quickstart.py --phase foundation"
            exit 1
        }
        task_id=$(echo "$task_json" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
    else
        task_json=$(get_task "$task_id") || {
            error "Task $task_id not found"
            exit 1
        }
    fi
    
    # Extract task details
    local agent_name=$(echo "$task_json" | python3 -c "import sys, json; print(json.load(sys.stdin)['agent'])")
    local title=$(echo "$task_json" | python3 -c "import sys, json; print(json.load(sys.stdin)['title'])")
    local description=$(echo "$task_json" | python3 -c "import sys, json; print(json.load(sys.stdin)['description'])")
    local outputs=$(echo "$task_json" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin).get('outputs_required', [])))")
    
    log "Spawning agent for: $task_id - $title"
    
    # Create git worktree
    local worktree="$WORKTREE_DIR/$task_id"
    local branch="work/$task_id"
    
    if [[ -d "$worktree" ]]; then
        warn "Worktree already exists, reusing: $worktree"
    else
        git worktree add "$worktree" -b "$branch" 2>/dev/null || {
            # Branch might exist, try without -b
            git worktree add "$worktree" "$branch" 2>/dev/null || {
                git worktree add "$worktree" -B "$branch"
            }
        }
        success "Created worktree: $worktree"
    fi
    
    # Create tmux window
    local window="$task_id"
    
    if tmux list-windows -t $TMUX_SESSION 2>/dev/null | grep -q "^[0-9]*: $window"; then
        warn "Agent already running for $task_id"
        echo "   Watch: tmux select-window -t $TMUX_SESSION:$window && tmux attach -t $TMUX_SESSION"
        return
    fi
    
    tmux new-window -t $TMUX_SESSION -n "$window" -c "$worktree"
    
    # Start Claude Code
    tmux send-keys -t "$TMUX_SESSION:$window" "claude" Enter
    sleep 2
    
    # Create the system prompt
    read -r -d '' PROMPT << EOM || true
I am an autonomous worker agent for Project Gatling.

MY TASK: $task_id - $title

DESCRIPTION:
$description

MY JOB:
1. Implement everything required for this task
2. Write comprehensive tests (aim for >90% coverage)
3. Run tests and fix any failures
4. Create documentation
5. When everything works, create an artifact manifest

WORKING DIRECTORY:
I'm in an isolated git worktree at: $worktree
This is a branch: $branch
I can work without affecting the main codebase.

TOOLS AVAILABLE:
- bash tool: I can run any command (uv, pytest, etc.)
- create_file: I can create any file
- str_replace: I can edit files
- view: I can read files

REQUIRED OUTPUTS:
$outputs

COMPLETION CRITERIA:
When I'm done, I must create: $PROJECT_ROOT/outputs/$agent_name/${task_id}_artifact.json

Artifact format:
{
    "task_id": "$task_id",
    "component": "ComponentName",
    "version": "0.1.0",
    "outputs": {
        "code": "path/to/code.py",
        "tests": "path/to/test.py",
        "docs": "path/to/docs.md"
    },
    "interface": {
        "input_shape": "...",
        "output_shape": "...",
        "latency_p99": "..."
    },
    "validation_status": "passed"
}

WORKFLOW:
1. First, I'll search the web for current best practices
2. Design the solution
3. Implement in small, testable pieces
4. Write tests as I go
5. Run tests frequently: uv run pytest -v
6. Fix any failures
7. Create docs
8. Create artifact when all tests pass

IMPORTANT:
- Use UV for all Python operations (uv run, uv add)
- Follow coding standards in $PROJECT_ROOT/CLAUDE.md
- Tests must pass before creating artifact
- Be thorough - this is production code

I will now begin working autonomously.
EOM

    # Send the prompt
    tmux send-keys -t "$TMUX_SESSION:$window" "$PROMPT" Enter
    
    # Mark as in progress
    python3 << EOF
import json
with open('tasks/task_queue.json') as f:
    queue = json.load(f)

# Move from ready to in_progress
for i, task in enumerate(queue.get('ready', [])):
    if task['id'] == '$task_id':
        task['status'] = 'in_progress'
        queue['in_progress'].append(task)
        queue['ready'].pop(i)
        break

with open('tasks/task_queue.json', 'w') as f:
    json.dump(queue, f, indent=2)
EOF
    
    success "Agent spawned and working on $task_id"
    echo ""
    echo "The agent is now working autonomously!"
    echo ""
    echo "To watch progress:"
    echo "  tmux attach -t $TMUX_SESSION"
    echo "  (then Ctrl-b n/p to switch between agents)"
    echo ""
    echo "To detach and let it work:"
    echo "  Ctrl-b d"
    echo ""
    echo "When complete, mark it:"
    echo "  ./gatling_agents.sh complete $task_id"
}

# Mark task complete
complete_task() {
    local task_id="$1"
    
    check_prereqs
    
    # Check if artifact exists
    local artifact
    artifact=$(find outputs -name "${task_id}_artifact.json" 2>/dev/null | head -1)
    
    if [[ -z "$artifact" ]]; then
        error "No artifact found for $task_id"
        echo ""
        echo "The agent should create: outputs/*/{{task_id}}_artifact.json"
        echo "Check if the agent is still working or encountered errors."
        echo ""
        echo "Watch the agent: tmux select-window -t $TMUX_SESSION:$task_id && tmux attach"
        exit 1
    fi
    
    log "Found artifact: $artifact"
    
    # Validate artifact
    local component=$(cat "$artifact" | python3 -c "import sys, json; print(json.load(sys.stdin).get('component', 'Unknown'))")
    
    echo ""
    echo "Artifact: $component"
    echo "Location: $artifact"
    echo ""
    
    # Check outputs exist
    local missing=0
    while IFS= read -r output_path; do
        if [[ ! -f "$output_path" ]]; then
            error "Missing: $output_path"
            missing=1
        else
            success "Found: $output_path"
        fi
    done < <(cat "$artifact" | python3 -c "import sys, json; outputs = json.load(sys.stdin).get('outputs', {}); [print(v) for v in outputs.values()]")
    
    if [[ $missing -eq 1 ]]; then
        echo ""
        warn "Some outputs are missing - mark complete anyway? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Mark as completed in queue
    python3 << EOF
import json
from datetime import datetime

with open('tasks/task_queue.json') as f:
    queue = json.load(f)

# Move from in_progress to completed
for i, task in enumerate(queue.get('in_progress', [])):
    if task['id'] == '$task_id':
        task['status'] = 'completed'
        task['completed_at'] = datetime.now().isoformat()
        queue['completed'].append(task)
        queue['in_progress'].pop(i)
        break

# Check if new tasks are ready
completed_ids = {t['id'] for t in queue.get('completed', [])}
new_ready = []

for i, task in enumerate(queue.get('pending', [])[:]):  # Copy to avoid modification issues
    deps = set(task.get('dependencies', []))
    if deps.issubset(completed_ids):
        task['status'] = 'ready'
        new_ready.append(task)

# Remove newly ready tasks from pending
queue['pending'] = [t for t in queue.get('pending', []) if t['id'] not in {t['id'] for t in new_ready}]
queue['ready'].extend(new_ready)

with open('tasks/task_queue.json', 'w') as f:
    json.dump(queue, f, indent=2)

# Print newly ready tasks
if new_ready:
    print("\\nNewly ready tasks:")
    for task in new_ready:
        print(f"  - {task['id']}: {task['title']}")
EOF
    
    success "Marked $task_id as completed"
    
    # Kill the tmux window
    if tmux list-windows -t $TMUX_SESSION 2>/dev/null | grep -q "^[0-9]*: $task_id"; then
        tmux kill-window -t "$TMUX_SESSION:$task_id"
        log "Closed agent window"
    fi
}

# Show status
show_status() {
    check_prereqs
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  GATLING AUTONOMOUS AGENT STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Task queue status
    python3 << 'EOF'
import json

with open('tasks/task_queue.json') as f:
    queue = json.load(f)

total = sum(len(queue.get(s, [])) for s in queue.keys())
completed = len(queue.get('completed', []))
in_progress = len(queue.get('in_progress', []))
ready = len(queue.get('ready', []))
blocked = len(queue.get('pending', []))

print(f"Tasks: {completed}/{total} complete")
print(f"  ✓ Completed: {completed}")
print(f"  ⚡ In Progress: {in_progress}")
print(f"  ✅ Ready: {ready}")
print(f"  ⏸️  Blocked: {blocked}")
print()

if in_progress > 0:
    print("Working on:")
    for task in queue.get('in_progress', []):
        print(f"  • {task['id']}: {task['title'][:60]}")
    print()

if ready > 0:
    print("Ready to start:")
    for task in queue.get('ready', []):
        print(f"  • {task['id']}: {task['title'][:60]}")
    print()
EOF
    
    # Active agents
    if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        echo "Active agent windows:"
        tmux list-windows -t $TMUX_SESSION -F "  #{window_index}: #{window_name}" | grep -v "^  0: main"
        echo ""
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Watch agents work
watch_agents() {
    if ! tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        error "No agents running"
        echo "  Start one with: ./gatling_agents.sh work"
        exit 1
    fi
    
    echo ""
    echo "Attaching to agent session..."
    echo ""
    echo "Commands:"
    echo "  Ctrl-b n/p    - Next/previous agent"
    echo "  Ctrl-b 0-9    - Jump to specific agent"
    echo "  Ctrl-b d      - Detach (agents keep working!)"
    echo "  Ctrl-b ?      - Show all tmux commands"
    echo ""
    sleep 2
    
    tmux attach -t $TMUX_SESSION
}

# Cleanup
cleanup() {
    log "Cleaning up..."
    
    # Kill tmux session
    if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        tmux kill-session -t $TMUX_SESSION
        success "Killed tmux session"
    fi
    
    # Remove worktrees
    if [[ -d "$WORKTREE_DIR" ]]; then
        for wt in "$WORKTREE_DIR"/*; do
            if [[ -d "$wt" ]]; then
                git worktree remove "$wt" --force 2>/dev/null || true
            fi
        done
        rm -rf "$WORKTREE_DIR"
        success "Removed worktrees"
    fi
    
    success "Cleanup complete"
}

# Help
show_help() {
    cat << 'EOF'
Gatling Autonomous Agent System

Simple autonomous agents using Claude Code in tmux.

COMMANDS:
  init                  Setup the system
  work [task-id]        Spawn agent for next (or specific) task
  status                Show what's happening
  watch                 Attach to see agents work
  complete <task-id>    Mark task as done
  cleanup               Remove all agents and worktrees

WORKFLOW:
  1. ./gatling_agents.sh init
  2. ./gatling_agents.sh work          # Spawns agent for LSA-001
  3. ./gatling_agents.sh work          # Spawns agent for LSA-002
  4. ./gatling_agents.sh watch         # Watch them work
  5. [Detach with Ctrl-b d - agents keep working]
  6. [Come back later...]
  7. ./gatling_agents.sh status        # Check progress
  8. ./gatling_agents.sh complete LSA-001
  9. ./gatling_agents.sh work          # Start next ready task

TIPS:
  - Agents work autonomously - just spawn and walk away
  - Use 'watch' to see them in action
  - Detach (Ctrl-b d) and they keep working
  - Each agent is isolated in its own git worktree
  - Uses your Claude.ai subscription (no API key needed!)

EXAMPLES:
  ./gatling_agents.sh work             # Work on next ready task
  ./gatling_agents.sh work LSA-003     # Work on specific task
  ./gatling_agents.sh status           # See what's happening
  ./gatling_agents.sh watch            # Watch agents work

EOF
}

# Main
case "${1:-}" in
    init)
        init
        ;;
    work)
        spawn_agent "${2:-}"
        ;;
    complete)
        if [[ -z "${2:-}" ]]; then
            error "Usage: $0 complete <task-id>"
            exit 1
        fi
        complete_task "$2"
        ;;
    status)
        show_status
        ;;
    watch)
        watch_agents
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        exit 1
        ;;
esac