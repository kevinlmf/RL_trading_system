cmake_minimum_required(VERSION 3.14)
project(cpp_trading)

set(CMAKE_CXX_STANDARD 17)

# 📌 加入 pybind11 子模块（你已有 pybind11/ 目录）
add_subdirectory(pybind11)

# 📌 包含头文件目录
include_directories(include)

# 📦 构建 pybind11 模块（合并所有绑定与 C++ 源文件）
pybind11_add_module(cpp_trading MODULE
    bindings/main_bindings.cpp     # ⬅️ PyBind11 主入口
    bindings/data_bindings.cpp     # ⬅️ DataFeed
    bindings/order_bindings.cpp    # ⬅️ OrderExecutor
    src/order_executor.cpp         # ⬅️ OrderExecutor 实现
    # src/data_feed.cpp            # ⬅️ 若有 data_feed 实现，也加上
)

# 🧩 禁用默认 Python 后缀命名（让最终输出为 cpp_trading.so）
set_target_properties(cpp_trading PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)




