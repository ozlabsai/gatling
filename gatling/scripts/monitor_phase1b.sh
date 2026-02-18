#!/bin/bash
# Monitor Phase 1b Tier I Dataset Generation

echo "📊 Phase 1b Generation Monitor"
echo "========================================"

# Check if process is still running
PID=9466
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Generation process running (PID: $PID)"
else
    echo "⚠️  Generation process completed or stopped"
fi

# Check output file
if [ -f "data/tier1_phase1b.jsonl" ]; then
    LINES=$(wc -l < data/tier1_phase1b.jsonl)
    SIZE=$(ls -lh data/tier1_phase1b.jsonl | awk '{print $5}')
    echo "📝 Output file: $LINES samples, $SIZE"
else
    echo "📝 Output file: Not created yet (loading in progress)"
fi

# Show last 10 lines of log
echo ""
echo "📋 Latest progress:"
echo "------------------------------------------------------------------------"
tail -10 data/phase1b_generation.log
echo "------------------------------------------------------------------------"

# Show loader completion status
echo ""
echo "🔧 Loader Status:"
grep -E "(✓ Loaded|✗ Error)" data/phase1b_generation.log | tail -15
