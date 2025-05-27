#!/bin/bash

echo "🛠️  [Step 1] Cleaning old build..."
rm -rf cpp_core/build

echo "🐍 [Step 2] Detecting current Python path..."
PYTHON_PATH=$(which python3)
echo "   → Using: $PYTHON_PATH"

echo "🔧 [Step 3] Configuring CMake with PyBind11..."
cmake -DPYTHON_EXECUTABLE=$PYTHON_PATH \
      -DCMAKE_PREFIX_PATH=$(python3 -m pybind11 --cmakedir) \
      -B cpp_core/build -S cpp_core

echo "🔨 [Step 4] Compiling shared library..."
cmake --build cpp_core/build -- -j4

echo "✅ [Done] Built .so file:"
find cpp_core/build -name "*.so"


