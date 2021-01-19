import doctest
import os
import unittest

from setuptools import find_packages, setup

requirements = ["numpy>=1.12.1"]


setup(
    name="ambulance_game",
    install_requires=requirements,
    author="Michalis Panayides",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
