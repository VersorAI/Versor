#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'gacore', '_version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gacore',
    version=__version__,
    license='bsd',
    description='Numerical Geometric Algebra Module',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Robert Kern',
    maintainer='Alex Arsenovic',
    maintainer_email='alexarsenovic@gmail.com',
    url='http://clifford.readthedocs.io',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.17',
        'scipy',
        'numba > 0.46',
        'h5py',
        'sparse',
    ],
    extras_require={
        'cuda': ['torch', 'triton'],
        'apple': ['mlx'],
    },
    # package_dir={'gacore':'gacore'},

    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        "Source Code": "https://github.com/pygae/clifford", # Legacy reference
    },

    python_requires='>=3.5',
)
