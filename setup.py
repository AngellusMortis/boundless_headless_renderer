#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from os import path

from setuptools import find_packages, setup

BASE_DIR = path.abspath(path.dirname(__file__))


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lines = (line.strip() for line in open(filename))
    return [line.strip() for line in lines if line and not line.strip().startswith("#")]


with open(path.join(BASE_DIR, "README.md")) as readme_file:
    readme = readme_file.read()

# with open(path.join(BASE_DIR, "HISTORY.rst")) as history_file:
#     history = history_file.read()

req_files = {
    "requirements": "requirements.in",
}

requirements = {}
for req, req_file in req_files.items():
    requirements[req] = parse_requirements(req_file)

setup(
    name="boundless_headless_renderer",
    version="0.1.0",
    description=("Boundless Icon Renderer by @willcrutchley"),
    entry_points={"console_scripts": ["boundless-icon-render=boundless_headless_renderer.__main__:main"]},
    long_description=readme,
    author="Will Crutchley",
    author_email="",
    url="https://gitlab.com/willcrutchley/boundless_headless_renderer/",
    packages=find_packages(include=["boundless_headless_renderer"]),
    include_package_data=True,
    install_requires=requirements["requirements"],
    license="MIT license",
    zip_safe=False,
    keywords="boundless_headless_renderer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
)
