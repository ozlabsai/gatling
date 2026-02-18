#!/bin/bash
# Monitor Full Tier I Dataset Generation (All Loaders)

echo "📊 Full Tier I Generation Monitor (ALL Loaders)"
echo "========================================"

# Check if process is still running
PID=40194
if ps -p $PID > /dev/null 2>&1; then
    CPU_TIME=$(ps -p $PID -o time= | tr -d ' ')
    echo "✅ Generation process running (PID: $PID, CPU time: $CPU_TIME)"
else
    echo "⚠️  Generation process completed or stopped"
fi

# Check output file
if [ -f "data/tier1_full_all_loaders.jsonl" ]; then
    LINES=$(wc -l < data/tier1_full_all_loaders.jsonl)
    SIZE=$(ls -lh data/tier1_full_all_loaders.jsonl | awk '{print $5}')
    echo "📝 Output file: $LINES samples, $SIZE"
else
    echo "📝 Output file: Not created yet (loading in progress)"
fi

# Check log file
if [ -f "data/tier1_full_generation.log" ]; then
    LOG_SIZE=$(ls -lh data/tier1_full_generation.log | awk '{print $5}')
    LOG_LINES=$(wc -l < data/tier1_full_generation.log)
    echo "📋 Log file: $LOG_LINES lines, $LOG_SIZE"
    echo ""
    echo "📋 Latest progress (last 15 lines):"
    echo "------------------------------------------------------------------------"
    tail -15 data/tier1_full_generation.log
    echo "------------------------------------------------------------------------"
else
    echo "📋 Log file: Not created yet (buffer not flushed)"
fi

echo ""
echo "🔧 Loader completion count:"
grep -c "✓ Loaded" data/tier1_full_generation.log 2>/dev/null || echo "0 (log not ready)"

echo ""
echo "💡 Tip: Run 'tail -f data/tier1_full_generation.log' for live updates"
