cmake_minimum_required(VERSION 3.14)
project(cpp_trading)

set(CMAKE_CXX_STANDARD 17)

# 📌 加入 pybind11 子模块（你已有 pybind11/ 目录）
add_subdirectory(pybind11)

# 📌 包含头文件目录
include_directories(include)

# 📦 构建 pybind11 模块（绑定 + 实现）
pybind11_add_module(cpp_trading MODULE
    bindings/main_bindings.cpp     # ⬅️ PyBind11 主入口
    bindings/data_bindings.cpp     # ⬅️ DataFeed Python 绑定
    bindings/order_bindings.cpp    # ⬅️ OrderExecutor Python 绑定
    src/data_feed.cpp              # ✅ DataFeed C++ 实现
    src/order_executor.cpp         # ✅ OrderExecutor C++ 实现
)

# 🧩 生成 cpp_trading.so，而不是默认 Python 风格命名
set_target_properties(cpp_trading PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)





