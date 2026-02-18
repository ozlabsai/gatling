#!/usr/bin/env python3
"""Test importing Topaz loaders - EXACT simulation of main script."""

import sys
from pathlib import Path

print("=== Simulating EXACT import sequence from generate_tier1_dataset.py ===\n")

# Setup paths
polecats_root = Path(__file__).parent / "polecats"

# Add ROOT gatling to path (like the main script does via sys.path.insert(0, str(Path(__file__).parent.parent)))
root_gatling = Path(__file__).parent
sys.path.insert(0, str(root_gatling))
print(f"Added root gatling to sys.path: {root_gatling}\n")

# QUARTZ
print("1. QUARTZ:")
quartz_path = str(polecats_root / "quartz" / "gatling")
sys.path.insert(0, quartz_path)
try:
    from source.dataset.function_calling_loaders import ToolBankLoader
    print(f"  ✓ Imported ToolBankLoader")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# OPAL
print("\n2. OPAL:")
opal_path = str(polecats_root / "opal" / "gatling")
if opal_path not in sys.path:
    sys.path.insert(0, opal_path)
try:
    if 'source.dataset.specialized_loaders' in sys.modules:
        del sys.modules['source.dataset.specialized_loaders']
    from source.dataset.specialized_loaders import AppleMMauLoader
    print(f"  ✓ Imported AppleMMauLoader")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# JASPER
print("\n3. JASPER:")
jasper_path = str(polecats_root / "jasper" / "gatling")
if jasper_path not in sys.path:
    sys.path.insert(0, jasper_path)
# Clear modules
for mod in list(sys.modules.keys()):
    if mod.startswith('source'):
        del sys.modules[mod]
print(f"  Cleared source modules")
try:
    from source.dataset.loaders import GSM8KLoader
    print(f"  ✓ Imported GSM8KLoader")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# TOPAZ
print("\n4. TOPAZ:")
topaz_path = str(polecats_root / "topaz" / "gatling")
if topaz_path not in sys.path:
    sys.path.insert(0, topaz_path)
print(f"  topaz_path: {topaz_path}")
print(f"  topaz first in sys.path? {sys.path[0] == topaz_path}")
# Clear modules
for mod in list(sys.modules.keys()):
    if mod.startswith('source'):
        del sys.modules[mod]
print(f"  Cleared source modules")
try:
    from source.dataset.loaders.conversations import LMSYSLoader, WildChatLoader
    print(f"  ✓ SUCCESS! Imported: {LMSYSLoader}, {WildChatLoader}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# RUBY
print("\n5. RUBY:")
try:
    from source.dataset.loaders import AgentHarmLoader as RubyLoader
    print(f"  ✓ Imported AgentHarmLoader from Ruby")
except Exception as e:
    print(f"  ✗ Failed: {e}")
