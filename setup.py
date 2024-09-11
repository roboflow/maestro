from typing import List, Union

import setuptools
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements(path: Union[str, List[str]]) -> List[str]:
    if not isinstance(path, list):
        path = [path]
    requirements = []
    for p in path:
        with open(p) as fh:
            requirements.extend([line.strip() for line in fh])
    return requirements


setuptools.setup(
    name="maestro",
    version="0.2.0rc2",
    author="Roboflow",
    author_email="help@roboflow.com",
    description="Visual Prompting for Large Multimodal Models (LMMs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow/multimodal-maestro",
    packages=find_packages(
        where=".",
        exclude=(
            "cookbooks",
            "docs",
            "tests",
            "tests.*",
            "requirements",
        ),
    ),
    install_requires=read_requirements("requirements/requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements/requirements.test.txt"),
        "docs": read_requirements("requirements/requirements.docs.txt"),
    },
    entry_points={
        "console_scripts": [
            "maestro=maestro.cli.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.9,<3.13",
)
