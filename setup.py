from setuptools import setup, find_packages

setup(
    name="ssv2",
    version="0.1.0",
    url="https://github.com/matsengrp/sci-space-v2.git",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "pytest",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
)
