import doctest
import os
import unittest

from setuptools import find_packages, setup

requirements = [
    "numpy==1.18.0",
    "matplotlib==3.4.3",
    "scipy==1.7.1",
    "networkx==2.4",
    "sympy==1.5.1",
    "ciw==2.1.0",
    "dask==2021.08.1",
    "nashpy==0.0.23",
]


setup(
    name="ambulance_game",
    install_requires=requirements,
    author="Michalis Panayides",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
