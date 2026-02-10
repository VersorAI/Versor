from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='versor_cpp',
    ext_modules=[
        CppExtension(
            name='versor_cpp',
            sources=['versor_core.cpp'],
            extra_compile_args=['-O3'],
            runtime_library_dirs=['/Users/mac/Library/Python/3.14/lib/python/site-packages/torch/lib']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
