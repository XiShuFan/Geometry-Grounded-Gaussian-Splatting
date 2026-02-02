import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name="tetra-nerf",
    packages=find_packages(),
    author="Jonas Kulhanek",
    author_email="jonas.kulhanek@live.com",
    description="Official implementation of Tetra-NeRF paper",
    ext_modules=[
        CUDAExtension(
            "tetranerf.utils.extension.tetranerf_cpp_extension",
            ["src/triangulation.cpp", "src/py_binding.cpp"],
            extra_compile_args={
                "nvcc": ["-O3", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/utils")],
                "cxx": ["-O3"],
            },
            extra_link_args=["-lgmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
