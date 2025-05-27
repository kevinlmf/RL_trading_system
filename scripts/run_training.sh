#!/bin/bash

# 用法：./scripts/run_training.sh [dqn|ppo|compare]
ALGO=${1:-dqn}

# 🚧 编译 cpp_trading.so 模块（自动跳过已存在的情况）
CPP_BUILD_DIR="./cpp_core/build"
CPP_SO_FILE="$CPP_BUILD_DIR/cpp_trading.so"

if [ ! -f "$CPP_SO_FILE" ]; then
    echo "🔧 C++ module not found. Building cpp_trading.so..."
    mkdir -p "$CPP_BUILD_DIR"
    cd "$CPP_BUILD_DIR"
    cmake ..
    make -j$(nproc)
    cd - > /dev/null
    echo "✅ C++ module built."
else
    echo "🧠 Using existing C++ module: $CPP_SO_FILE"
fi

# 🚀 训练逻辑分发
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
