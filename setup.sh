#!/bin/bash

echo "🔁 Rebuilding cpp_trading.so with current Python"

rm -rf cpp_core/build

cmake -DPYTHON_EXECUTABLE=$(which python) -B cpp_core/build -S cpp_core
cmake --build cpp_core/build -- -j4

echo "✅ Done: cpp_core/build/cpp_trading.so"
