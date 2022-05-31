#!/usr/bin/env python3

# import os
from setuptools import setup

# load the README file and use it as the long_description for PyPI
with open('README.md', 'r') as f:
    readme = f.read()

with open("./requirements.txt", "r") as f:
    required = f.read().splitlines()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(
    name="Deepair",
    description="PyTorch implementations of Deep reinforcement learning algorithms.",
    long_description=readme,
    long_description_content_type='text/markdown',
    version=0.1,
    author="Son Nguyen Huu",
    author_email="sonnhfit@gmail.com",
    url="https://github.com/sonnhfit/deepair",
    packages=['deepair'],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=required,
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
