#!/bin/bash
# Monitor Budget-Capped Conversation Generation

echo "📊 Conversation Dataset Generation Monitor (Budget: ~\$151)"
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
if [ -f "data/tier1_conversations_sampled.jsonl" ]; then
    LINES=$(wc -l < data/tier1_conversations_sampled.jsonl)
    SIZE=$(ls -lh data/tier1_conversations_sampled.jsonl | awk '{print $5}')

    # Estimate cost based on samples (assuming ~$0.063 per conversation)
    COST=$(echo "scale=2; $LINES * 0.063" | bc)

    echo "📝 Output file: $LINES samples, $SIZE"
    echo "💰 Estimated cost so far: \$$COST"
else
    echo "📝 Output file: Not created yet (loading in progress)"
fi

# Check log file
if [ -f "data/tier1_conversations.log" ]; then
    LOG_SIZE=$(ls -lh data/tier1_conversations.log | awk '{print $5}')
    LOG_LINES=$(wc -l < data/tier1_conversations.log)
    echo "📋 Log file: $LOG_LINES lines, $LOG_SIZE"

    if [ $LOG_LINES -gt 0 ]; then
        echo ""
        echo "📋 Latest progress (last 25 lines):"
        echo "------------------------------------------------------------------------"
        tail -25 data/tier1_conversations.log
        echo "------------------------------------------------------------------------"
    fi
else
    echo "📋 Log file: Not created yet (buffer not flushed)"
fi

echo ""
echo "🎯 Target: 2,400 conversations (1,200 LMSYS + 1,200 WildChat)"
echo "💡 Tip: Run 'tail -f data/tier1_conversations.log' for live updates"
