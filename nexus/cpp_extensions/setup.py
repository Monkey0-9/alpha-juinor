from setuptools import setup, Extension
import pybind11

# Add standard library paths for C++ features on Windows MinGW/MSVC
common_args = ["-O3", "-std=c++11", "-ffast-math"]

ext_modules = [
    Extension(
        "nexus_cpp",
        sources=[
            "fast_math.cpp",
            "kalman.cpp",
            "fractal.cpp",
            "stats.cpp",
            "timeseries.cpp",
            "signals.cpp",
            "matrix.cpp",
            "optimization.cpp",
            "feed.cpp",
            "backtest.cpp",
            "onnx_inference.cpp",
        ],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=common_args,
        extra_link_args=["-static-libgcc", "-static-libstdc++"],
    ),
    Extension(
        "nexus_onnx",
        sources=["onnx_inference.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=common_args,
        extra_link_args=["-static-libgcc", "-static-libstdc++"],
    ),
    Extension(
        "nexus_backtester",
        sources=["backtester.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=common_args,
        extra_link_args=["-static-libgcc", "-static-libstdc++"],
    ),
    Extension(
        "nexus_feed",
        sources=["market_feed.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=common_args,
        extra_link_args=["-static-libgcc", "-static-libstdc++"],
    ),
]

setup(
    name="nexus_cpp_extensions",
    version="0.2.0",
    description="Nexus High-Performance C++ Trading Engine",
    ext_modules=ext_modules,
)
