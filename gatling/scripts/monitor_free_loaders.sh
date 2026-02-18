#!/bin/bash
# Monitor Free Loaders Generation (Excluding Topaz)

echo "📊 Free Loaders Generation Monitor (No Conversation Datasets)"
echo "========================================================"

# Check if process is still running
if [ -f "/tmp/tier1_generation.pid" ]; then
    PID=$(cat /tmp/tier1_generation.pid)
    if ps -p $PID > /dev/null 2>&1; then
        CPU_TIME=$(ps -p $PID -o time= | tr -d ' ')
        echo "✅ Generation process running (PID: $PID, CPU time: $CPU_TIME)"
    else
        echo "⚠️  Generation process completed or stopped"
    fi
else
    echo "⚠️  PID file not found"
fi

# Check output file
if [ -f "data/tier1_free_loaders.jsonl" ]; then
    LINES=$(wc -l < data/tier1_free_loaders.jsonl)
    SIZE=$(ls -lh data/tier1_free_loaders.jsonl | awk '{print $5}')
    echo "📝 Output file: $LINES samples, $SIZE"
else
    echo "📝 Output file: Not created yet (loading in progress)"
fi

# Check log file
if [ -f "data/tier1_free.log" ]; then
    LOG_SIZE=$(ls -lh data/tier1_free.log | awk '{print $5}')
    LOG_LINES=$(wc -l < data/tier1_free.log)
    echo "📋 Log file: $LOG_LINES lines, $LOG_SIZE"

    if [ $LOG_LINES -gt 0 ]; then
        echo ""
        echo "📋 Latest progress (last 20 lines):"
        echo "------------------------------------------------------------------------"
        tail -20 data/tier1_free.log
        echo "------------------------------------------------------------------------"
    fi
else
    echo "📋 Log file: Not created yet (buffer not flushed)"
fi

echo ""
echo "💡 Tip: Run 'tail -f data/tier1_free.log' for live updates"
