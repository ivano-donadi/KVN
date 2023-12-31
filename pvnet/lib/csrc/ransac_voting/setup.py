from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ransac_voting',
    ext_modules=[
        CUDAExtension('ransac_voting', [
            './src/ransac_voting.cpp',
            './src/ransac_voting_kernel.cu'
        ],
        extra_compile_args={"cxx":['-fopenmp'], "nvcc":[]},
        extra_link_args=['-fopenmp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
