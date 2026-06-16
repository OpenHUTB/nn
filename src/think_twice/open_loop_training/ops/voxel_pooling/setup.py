from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='voxel_pooling_ext',
    ext_modules=[
        CUDAExtension(
            name='voxel_pooling_ext',
            sources=[
                'src/voxel_pooling_forward.cpp',
                'src/voxel_pooling_forward_cuda.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
