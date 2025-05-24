cmake_minimum_required(VERSION 3.14)
project(cpp_trading)

set(CMAKE_CXX_STANDARD 17)

# 引入 pybind11
add_subdirectory(pybind11)
include_directories(include)

# 👇 手动创建 shared object（不要 Python 后缀）
pybind11_add_module(cpp_trading MODULE
    bindings/data_bindings.cpp
    src/data_feed.cpp
)

# 👇 禁用默认 Python 后缀命名
set_target_properties(cpp_trading PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)




