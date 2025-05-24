from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "cpp_trading",
        ["bindings/data_bindings.cpp", "src/data_feed.cpp"],
        include_dirs=["include"],
    ),
]

setup(
    name="cpp_trading",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

