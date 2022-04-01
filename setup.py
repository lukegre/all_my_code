from os.path import exists

from setuptools import find_packages, setup

if exists("README.rst"):
    with open("README.rst") as f:
        long_description = f.read()
else:
    long_description = ""

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

test_requirements = ["pytest-cov"]
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

setup(
    name="all_my_code",
    author="Luke Gregor",
    author_email="lukegre@gmail.com",
    description=(
        "tools that I've developed over time that form a part of my daily "
        "workflow. Thought good to share."
    ),
    keywords="Oceanography;xarray",
    license="GNUv3",
    classifiers=CLASSIFIERS,
    url="https://github.com/lukegre/all_my_code",
    use_scm_version={
        "version_scheme": "python-simplified-semver",
        "local_scheme": "no-local-version",
    },
    long_description=long_description,
    packages=find_packages(),
    install_requires=install_requires,
    test_suite="all_my_code/tests",
    tests_require=test_requirements,
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
        "setuptools_scm_git_archive",
    ],
)
