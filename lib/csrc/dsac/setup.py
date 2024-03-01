from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ransac_voting',
    ext_modules=[
        CUDAExtension('dsac', [
            './src/dsac.cpp'
        ],
        extra_compile_args={"cxx":['-fopenmp'], "nvcc":[]},
        extra_link_args=['-fopenmp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
