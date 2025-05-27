#!/bin/bash

# 自动构建 cpp_trading.so（确保使用当前 Python 版本和 arm64 架构）

echo "🔧 Cleaning build directory..."
cd "$(dirname "$0")/build"
rm -rf *

echo "🔍 Detecting Python executable..."
PYTHON_EXEC=$(which python3)
echo "✅ Using Python: $PYTHON_EXEC"

echo "⚙️ Running CMake with arm64 and correct Python..."
arch -arm64 cmake .. -DPYTHON_EXECUTABLE=$PYTHON_EXEC

echo "⚒️ Building C++ module..."
arch -arm64 make -j4

echo "✅ Build complete. Output: build/cpp_trading.so"

