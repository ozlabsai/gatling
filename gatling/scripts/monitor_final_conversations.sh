#!/bin/bash
# Monitor Final Conversation Generation (Verified Fix)

echo "📊 Final Conversation Generation Monitor (Bugs Fixed!)"
echo "========================================================"

# Check if process is still running
if [ -f "/tmp/tier1_conversations.pid" ]; then
    PID=$(cat /tmp/tier1_conversations.pid)
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
if [ -f "data/tier1_conversations_final.jsonl" ]; then
    LINES=$(wc -l < data/tier1_conversations_final.jsonl)
    SIZE=$(ls -lh data/tier1_conversations_final.jsonl | awk '{print $5}')

    # Estimate cost based on samples (~$0.063 per conversation)
    COST=$(echo "scale=2; $LINES * 0.063" | bc)

    echo "📝 Output file: $LINES samples, $SIZE"
    echo "💰 Estimated cost so far: \$$COST"

    # Progress percentage
    PROGRESS=$(echo "scale=1; ($LINES / 2400) * 100" | bc)
    echo "📈 Progress: $PROGRESS%"
else
    echo "📝 Output file: Not created yet (processing in progress)"
fi

# Check log file
if [ -f "data/tier1_conversations_final.log" ]; then
    LOG_SIZE=$(ls -lh data/tier1_conversations_final.log | awk '{print $5}')
    LOG_LINES=$(wc -l < data/tier1_conversations_final.log)
    echo "📋 Log file: $LOG_LINES lines, $LOG_SIZE"

    if [ $LOG_LINES -gt 0 ]; then
        echo ""
        echo "📋 Latest progress (last 20 lines):"
        echo "------------------------------------------------------------------------"
        tail -20 data/tier1_conversations_final.log
        echo "------------------------------------------------------------------------"
    fi
else
    echo "📋 Log file: Not created yet"
fi

echo ""
echo "🎯 Target: 2,400 conversation samples"
echo "💰 Estimated total cost: ~\$150"
echo "💡 Tip: Run 'tail -f data/tier1_conversations_final.log' for live updates"
