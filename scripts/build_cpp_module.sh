#!/bin/bash
set -e

echo "🔧 Building C++ PyBind11 module..."

cd "$(dirname "$0")/../cpp_core"
mkdir -p build
cd build
cmake ..
make

echo "✅ C++ module built: $(find . -name 'cpp_trading*.so')"

