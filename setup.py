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
    install_requires = ["seaborn == 0.11.2", "matplotlib == 3.5.1", "numpy == 1.20.3", "plotly == 5.7.0", "pyvis == 0.2.1", "kaleido == 0.2.1"],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest==4.4.1'],
    test_suite = 'Unit_Tests',
)