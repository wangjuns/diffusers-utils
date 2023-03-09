import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="diffusers-utils",
    py_modules=["diffusers_utils"],
    version="0.1",
    description="Utils used with diffusion.",
    author="Simo Ryu",
    packages=find_packages(),
    
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)