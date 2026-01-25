#!/usr/bin/env python3
"""
Add all tasks with PRD references (hybrid approach)

Task queue tracks state + points to PRD for requirements.
"""

import json
from datetime import datetime

# Read current queue
with open('tasks/task_queue.json') as f:
    queue = json.load(f)

# Minimal task definitions - just IDs, dependencies, and PRD pointers
all_tasks = [
    # Energy Geometry (Weeks 4-6)
    {
        "id": "EGA-001",
        "component": "E_hierarchy",
        "prd_section": "docs/WORK-DISTRIBUTION.md#energy-geometry-workstream",
        "depends_on": ["LSA-001", "LSA-002"],
        "priority": "high"
    },
    {
        "id": "EGA-002",
        "component": "E_provenance", 
        "prd_section": "docs/WORK-DISTRIBUTION.md#energy-geometry-workstream",
        "depends_on": ["LSA-001", "LSA-002"],
        "priority": "high"
    },
    {
        "id": "EGA-003",
        "component": "E_scope",
        "prd_section": "docs/WORK-DISTRIBUTION.md#energy-geometry-workstream",
        "depends_on": ["LSA-003"],
        "priority": "high"
    },
    {
        "id": "EGA-004",
        "component": "E_flow",
        "prd_section": "docs/WORK-DISTRIBUTION.md#energy-geometry-workstream",
        "depends_on": ["LSA-002"],
        "priority": "medium"
    },
    {
        "id": "EGA-005",
        "component": "CompositeEnergy",
        "prd_section": "docs/WORK-DISTRIBUTION.md#energy-geometry-workstream",
        "depends_on": ["EGA-001", "EGA-002", "EGA-003", "EGA-004"],
        "priority": "critical"
    },
    
    # Provenance (Weeks 4-6)
    {
        "id": "PA-001",
        "component": "TrustTierTagging",
        "prd_section": "docs/WORK-DISTRIBUTION.md#provenance-workstream",
        "depends_on": ["LSA-001"],
        "priority": "high"
    },
    {
        "id": "PA-002",
        "component": "RepairEngine",
        "prd_section": "docs/WORK-DISTRIBUTION.md#provenance-workstream",
        "depends_on": ["EGA-005", "PA-001"],
        "priority": "critical"
    },
    {
        "id": "PA-003",
        "component": "FastPathDistillation",
        "prd_section": "docs/WORK-DISTRIBUTION.md#provenance-workstream",
        "depends_on": ["PA-002"],
        "priority": "medium"
    },
    
    # Red Team (Weeks 7-9)
    {
        "id": "RTA-001",
        "component": "CorrupterAgent",
        "prd_section": "docs/WORK-DISTRIBUTION.md#red-team-workstream",
        "depends_on": ["EGA-005"],
        "priority": "high"
    },
    {
        "id": "RTA-002",
        "component": "ContrastiveLearning",
        "prd_section": "docs/WORK-DISTRIBUTION.md#red-team-workstream",
        "depends_on": ["RTA-001", "DA-001"],
        "priority": "high"
    },
    
    # Dataset (Weeks 7-9)
    {
        "id": "DA-001",
        "component": "GoldTraceGeneration",
        "prd_section": "docs/WORK-DISTRIBUTION.md#dataset-workstream",
        "depends_on": ["LSA-004"],
        "priority": "critical"
    },
    {
        "id": "DA-002",
        "component": "PolicyBoundaryCases",
        "prd_section": "docs/WORK-DISTRIBUTION.md#dataset-workstream",
        "depends_on": ["DA-001"],
        "priority": "high"
    },
    {
        "id": "DA-003",
        "component": "MinimalScopeLabels",
        "prd_section": "docs/WORK-DISTRIBUTION.md#dataset-workstream",
        "depends_on": ["DA-001"],
        "priority": "medium"
    },
    
    # Integration (Weeks 10-12)
    {
        "id": "IA-001",
        "component": "E2EPipeline",
        "prd_section": "docs/WORK-DISTRIBUTION.md#integration-workstream",
        "depends_on": ["EGA-005", "PA-002", "RTA-002"],
        "priority": "critical"
    },
    {
        "id": "IA-002",
        "component": "KonaBenchmarks",
        "prd_section": "docs/WORK-DISTRIBUTION.md#integration-workstream",
        "depends_on": ["IA-001"],
        "priority": "critical"
    }
]

# Check what's completed
completed_ids = {t['id'] for t in queue.get('completed', [])}

# Determine ready vs pending
ready = []
pending = []

for task in all_tasks:
    # Skip if already completed
    if task['id'] in completed_ids:
        continue
    
    # Check dependencies
    deps = set(task.get('depends_on', []))
    if deps.issubset(completed_ids):
        task['status'] = 'ready'
        ready.append(task)
    else:
        task['status'] = 'pending'
        task['blocked_by'] = list(deps - completed_ids)
        pending.append(task)

# Update queue
queue['ready'].extend(ready)
queue['pending'].extend(pending)

# Save
with open('tasks/task_queue.json', 'w') as f:
    json.dump(queue, f, indent=2)

print(f"✓ Added {len(all_tasks)} tasks")
print(f"\nCompleted: {len(completed_ids)}")
print(f"Ready: {len(ready)}")
print(f"Pending: {len(pending)}")
print("\nReady tasks:")
for t in ready:
    print(f"  {t['id']}: {t['component']}")