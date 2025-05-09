import ast
import os
import re

from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

ver_file = os.path.join(this_directory, "kaira", "version.py")
with open(ver_file, encoding="utf-8") as f:
    content = f.read()
    match = re.search(r"__version_info__\s*=\s*(\([^)]+\))", content)
    if not match:
        raise RuntimeError("Unable to find __version_info__ in version.py")
    version_info = ast.literal_eval(match.group(1))
    VERSION = ".".join(map(str, version_info))

setup(
    name="pykaira",
    version=VERSION,
    url="https://github.com/ipc-lab/kaira",
    license="MIT",
    author="Selim Firat Yilmaz",
    author_email="yilmazselimfirat@gmail.com",
    description="A PyTorch-based toolkit for simulating communication systems.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=["setuptools>=38.6.0"],
    keywords=[
        "wireless communication",
        "communications",
        "simulation",
        "toolbox",
        "channel modeling",
        "signal processing",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    project_urls={
        "Documentation": "https://kaira.readthedocs.io",
        "Source": "https://github.com/ipc-lab/kaira",
        "Tracker": "https://github.com/ipc-lab/kaira/issues",
    },
)
