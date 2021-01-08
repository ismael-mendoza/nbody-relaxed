#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = ["pip"]

author = "Ismael Mendoza"

setup(
    name="relaxed",
    author=author,
    author_email="imendoza@umich.edu",
    description="N-body Relaxed data analysis",
    long_description=readme,
    package_dir={"relaxed": "relaxed"},
    python_requires=">=3.7",
    packages=["relaxed"],
    setup_requires=setup_requirements,
    url="https://github.com/ismael2395/nbody-relaxed",
    version="0.1.0",
    ext_modules=cythonize("relaxed/subhaloes/cy_binning.pyx"),
)
