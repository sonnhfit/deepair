#!/usr/bin/env python3

# import os
from setuptools import setup, find_packages
from deepair.version import __version__

# load the README file and use it as the long_description for PyPI
with open('README.md', 'r', encoding="utf8") as f:
    readme = f.read()

install_requires =[
    'gym',
    'numpy',
    'torch>=1.11',
    'cloudpickle',
    'pandas',
    'tqdm',
    'ipython',
    'pylint',
    'sphinx-material',
    'numpydoc',
    'nbsphinx',
    'recommonmark',
    'sphinx-markdown-tables',
    'sphinx_copybutton',
    'readthedocs-sphinx-search'
]

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(
    name="deepair",
    description="PyTorch implementations of Deep reinforcement learning algorithms.",
    long_description=readme,
    long_description_content_type='text/markdown',
    version=__version__,
    author="Son Nguyen Huu",
    author_email="sonnhfit@gmail.com",
    url="https://github.com/sonnhfit/deepair",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=install_requires,
    license="MIT",
    zip_safe=False,
    entry_points={
        'console_scripts': ['py-package-template=py_pkg.entry_points:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords='reinforcement-learning-algorithms reinforcement-learning machine-learning deep rl'
)
