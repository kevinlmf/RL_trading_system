cmake_minimum_required(VERSION 3.14)
project(cpp_trading)

# 使用 pybind11 包（确保用户用 pip 安装 pybind11）
find_package(pybind11 REQUIRED)

# 添加你的源文件和绑定文件
pybind11_add_module(cpp_trading
    bindings/main_bindings.cpp
    bindings/data_bindings.cpp
    bindings/order_bindings.cpp
    src/data_feed.cpp
    src/order_executor.cpp
)

target_include_directories(cpp_trading PRIVATE include)





