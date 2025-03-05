#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="adaptsam",
    version="0.0.1",
    description="Adapting Segment Anything 2.1 to extract 3D objects from volumetric data",
    author="Romain Karpinski",
    author_email="romain.karpinski@loria.fr",
    url="https://github.com/KarpRom/adaptSAM-3D",
    packages=find_packages(),
)
