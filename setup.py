# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

from setuptools import find_packages, setup

setup(
    name='PyRID',
    packages=find_packages(),
    version='0.0.1',
    description='PyRID is a Python library for particle-based reaction diffusion simulations',
    author='Moritz F P Becker',
    license='MIT',
    install_requires = ["numba == 0.55.1", "h5py == 3.7.0", "scipy == 1.9.1", "seaborn == 0.13.1", "matplotlib == 3.7.4", "numpy == 1.20.3", "plotly == 5.7.0", "pyvis == 0.2.1", "kaleido == 0.2.1"],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest==4.4.1'],
    test_suite = 'Unit_Tests',
)