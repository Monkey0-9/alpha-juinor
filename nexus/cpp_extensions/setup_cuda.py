from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="nexus_cuda",
    ext_modules=[
        CUDAExtension(
            "nexus_cuda",
            [
                "monte_carlo.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
