#!/bin/bash

# 用法：./scripts/run_training.sh [dqn|ppo|compare]
ALGO=${1:-dqn}

if [ "$ALGO" == "dqn" ]; then
    echo "🚀 Training DQN..."
    python train_dqn.py

elif [ "$ALGO" == "ppo" ]; then
    echo "🚀 Training PPO..."
    python train_ppo.py

elif [ "$ALGO" == "compare" ]; then
    echo "📊 Comparing strategies..."
    python scripts/compare_strategies.py

else
    echo "❌ Unknown mode: $ALGO"
    echo "Usage: ./scripts/run_training.sh [dqn|ppo|compare]"
    exit 1
fi

echo "✅ Done."




