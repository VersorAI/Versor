from setuptools import setup, find_packages
import os

# Get version from gacore/_version.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'gacore', '_version.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split("'")[1]
    return '0.0.1'

setup(
    name="gacore",
    version=get_version(),
    author="Hugo Hadfield, Eric Wieser, Alex Arsenovic, Robert Kern, Antigravity AI",
    author_email="contact@versor.ai",
    description="High-Performance Geometric Algebra Core for Deep Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VersorAI/Versor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "sparse>=0.13.0",
    ],
    extras_require={
        "cuda": ["triton>=2.1.0"],
        "apple": ["mlx>=0.10.0"],
    }
)
