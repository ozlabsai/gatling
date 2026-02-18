#!/usr/bin/env python3
"""
Create Step 1 beads and spawn polecats to work on them.
"""

import json
import subprocess
import sys
from pathlib import Path

def create_bead_via_api(bead_id, title, description, labels, priority="P1"):
    """Create a bead using direct file manipulation."""

    # Create bead file
    bead_dir = Path("/Users/guynachshon/gt/gatling/.beads/beads")
    bead_dir.mkdir(parents=True, exist_ok=True)

    bead_file = bead_dir / f"{bead_id}.md"

    content = f"""---
id: {bead_id}
title: {title}
priority: {priority}
type: task
owner: mayor
status: OPEN
labels: {', '.join(labels)}
created: 2026-01-29
---

{description}
"""

    with open(bead_file, 'w') as f:
        f.write(content)

    print(f"✅ Created bead: {bead_id}")
    return bead_id

def spawn_polecat(name, bead_id):
    """Spawn a polecat and assign it work."""

    gatling_root = Path("/Users/guynachshon/gt/gatling")

    # Spawn polecat
    result = subprocess.run(
        ["gt", "polecat", "spawn", "gatling", name],
        cwd=gatling_root,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"✅ Spawned polecat: {name}")

        # Send work assignment via mail
        subprocess.run(
            [
                "gt", "mail", "send", f"gatling/{name}",
                "--subject", f"Work Assignment: {bead_id}",
                "--message", f"Please work on bead {bead_id}. Check bead details with: gt show {bead_id}",
                "--type", "task",
                "--priority", "0"
            ],
            cwd=gatling_root
        )
        print(f"📬 Sent work assignment to {name}")
        return True
    else:
        print(f"⚠️  Failed to spawn {name}: {result.stderr}")
        return False

def main():
    print("🚀 Creating Step 1 Beads and Spawning Polecats\n")
    print("=" * 60)

    # Define beads and their assigned polecats
    beads_and_polecats = [
        {
            "bead_id": "ga-loader-jasper",
            "title": "Fix Jasper Loader Imports (8 loaders)",
            "polecat": "sapphire",
            "labels": ["dataset", "loaders", "import-fix", "jasper"],
            "description": """Fix import and module path issues for jasper's 8 loaders.

**Current Issue**: Cannot import GSM8KLoader - namespace collision with quartz

**Tasks**:
1. Verify loader classes in jasper/gatling/source/dataset/loaders.py
2. Update import strategy using importlib for module isolation
3. Test isolated import in jasper's UV environment
4. Add to aggregation script with proper namespace handling
5. Validate 100-sample generation from each loader

**Target**: 8 loaders (GSM8K, MATH, InstructionDataset variants)

**Output**: Updated generate_tier1_dataset.py with jasper working
"""
        },
        {
            "bead_id": "ga-loader-opal",
            "title": "Fix Opal Loader Imports (7 specialized loaders)",
            "polecat": "emerald",
            "labels": ["dataset", "loaders", "uv-environment", "opal"],
            "description": """Fix UV environment isolation for opal's 7 specialized loaders.

**Current Issue**: UV venv isolation prevents cross-polecat imports

**Tasks**:
1. Test direct import in opal's environment
2. Choose isolation strategy (subprocess/copy/unified-requirements)
3. Implement in generate_tier1_dataset.py
4. Test 100-sample generation
5. Add to metadata statistics

**Target**: 7 loaders (AppleMMau, NvidiaToolScale, etc.)

**Output**: Cross-polecat import solution + working integration
"""
        },
        {
            "bead_id": "ga-loader-topaz",
            "title": "Fix Topaz Conversation Loaders (LMSYS + WildChat)",
            "polecat": "pearl",
            "labels": ["dataset", "loaders", "conversations", "topaz"],
            "description": """Fix module path for topaz conversation loaders.

**Current Issue**: Import path needs loaders.conversations (directory)

**Tasks**:
1. Verify directory structure
2. Update import (already done, validate)
3. Test imports in topaz environment
4. Define sampling strategy for 1M+ datasets
5. Generate test samples

**Target**: 2 loaders (LMSYS, WildChat) with sampling

**Output**: Working conversation loaders with 1M sample strategy
"""
        },
        {
            "bead_id": "ga-tier1-full",
            "title": "Run Full Tier I Generation (4M samples)",
            "polecat": "diamond",
            "labels": ["dataset", "generation", "tier-1", "production"],
            "description": """Execute full Tier I benign dataset generation.

**Phase 1a**: Generate 1M with current 5 loaders (immediate)
**Phase 1b**: Generate 4M with all 23 loaders (after fixes)

**Tasks**:
1. Run Phase 1a baseline (1M samples)
2. Monitor progress and validate quality
3. Wait for loader fixes to complete
4. Run Phase 1b full generation (4M)
5. Apply deduplication (intent-based hashing)
6. Final quality validation

**Outputs**:
- data/tier1_baseline_1m.jsonl (Phase 1a)
- data/tier1_benign_4m.jsonl (Phase 1b)
- Metadata with per-loader statistics
"""
        }
    ]

    # Create beads and spawn polecats
    for item in beads_and_polecats:
        print(f"\n📋 {item['bead_id']}: {item['title']}")
        print(f"   Polecat: {item['polecat']}")

        # Create bead
        create_bead_via_api(
            item['bead_id'],
            item['title'],
            item['description'],
            item['labels']
        )

        # Spawn polecat
        spawn_polecat(item['polecat'], item['bead_id'])

        print()

    print("=" * 60)
    print("\n✅ Step 1 Deployment Complete!")
    print("\n4 polecats spawned and assigned:")
    print("  - sapphire  → ga-loader-jasper")
    print("  - emerald   → ga-loader-opal")
    print("  - pearl     → ga-loader-topaz")
    print("  - diamond   → ga-tier1-full")
    print("\nMonitor progress: gt polecat list --all")

if __name__ == "__main__":
    main()
