#!/bin/bash

set -e
echo "📦 Setting up Quant Trading System..."

# 1. 创建 Python 虚拟环境（仅当 venv 不存在时）
if [ ! -d "venv" ]; then
    echo "🧪 Creating virtual environment..."
    python3.10 -m venv venv
fi

# 2. 激活环境
echo "🔁 Activating environment..."
source venv/bin/activate

# 3. 安装 Python 依赖
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. 构建 C++ 模块
echo "🔧 Building C++ PyBind11 module..."
mkdir -p cpp_core/build
cd cpp_core/build
cmake ..
make
cd ../../

# 5. 测试 C++ 模块能否被正确加载
echo "🧪 Testing cpp_trading module..."
python scripts/test_cpp_module.py

echo "✅ Setup complete. You can now run train_dqn.py or train_ppo.py!"
