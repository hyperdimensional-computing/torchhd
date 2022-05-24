"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages

# Read the version without importing any dependencies
version = {}
with open("torchhd/version.py") as f:
    exec(f.read(), version)

setup(
    name="torch-hd",  # use torch-hd on PyPi to install torchhd, torchhd is too similar according to PyPi
    version=version["__version__"],
    description="Torchhd is a Python library for Hyperdimensional Computing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hyperdimensional-computing/torchhd",
    license="MIT",
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "requests",
    ],
    packages=find_packages(exclude=["docs", "torchhd.tests", "examples"]),
    python_requires=">=3.6, <4",
    project_urls={
        "Source": "https://github.com/hyperdimensional-computing/torchhd",
        "Documentation": "https://torchhd.readthedocs.io",
    },
)
