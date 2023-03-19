# -*- coding: utf8 -*-
from setuptools import setup,find_packages

# Required packages are included in this list
# required_packages = []

#if location of package is different than the name of package
#package_dir= {'helper': 'path where __init__.py is searched for'} 
#this applies recursively

setup(
    name="helper",
    version="0.1",
    description="Lib to facilitate the Information Operation project",
    author="Manita Pote",
    packages=find_packages(),
    # install_requires=required_packages,
)