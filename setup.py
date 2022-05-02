"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
from setuptools import setup

setup(
    name="hdc",
    version="0.5.3",
    description="Python library for Hyperdimensional Computing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mikeheddes/hdc-lib",
    license="MIT",
    install_requires=open("requirements.txt").readlines(),
    packages=["hdc"],
    python_requires=">=3.6, <4",
    project_urls={
        "Source": "https://github.com/mikeheddes/hdc-lib/",
        "Documentation": "https://mikeheddes.github.io/hdc-lib/",
    },
)
