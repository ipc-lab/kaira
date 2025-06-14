import ast
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent.resolve()

# Read README with error handling
try:
    with open(this_directory / "README.md", encoding="utf-8") as f:
        readme = f.read()
except FileNotFoundError:
    print("Warning: README.md not found", file=sys.stderr)
    readme = ""

# Read requirements with filtering
try:
    with open(this_directory / "requirements.txt", encoding="utf-8") as f:
        requirements = [line.strip() for line in f.read().splitlines() if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    print("Warning: requirements.txt not found", file=sys.stderr)
    requirements = []

# Get version with better error handling
ver_file = this_directory / "kaira" / "version.py"
try:
    with open(ver_file, encoding="utf-8") as f:
        content = f.read()
        match = re.search(r"__version_info__\s*=\s*(\([^)]+\))", content)
        if not match:
            raise RuntimeError("Unable to find __version_info__ in version.py")
        version_info = ast.literal_eval(match.group(1))
        VERSION = ".".join(map(str, version_info))
except FileNotFoundError:
    raise RuntimeError(f"Version file not found: {ver_file}")
except Exception as e:
    raise RuntimeError(f"Error parsing version from {ver_file}: {e}")

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
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    include_package_data=True,
    package_data={
        "kaira": ["py.typed"],
    },
    install_requires=requirements,
    setup_requires=["setuptools>=38.6.0"],
    entry_points={
        "console_scripts": [
            "kaira-benchmark=scripts.kaira_benchmark:main",
        ],
    },
    keywords=[
        "wireless communication",
        "communications",
        "simulation",
        "toolbox",
        "channel modeling",
        "signal processing",
        "pytorch",
        "deep learning",
        "machine learning",
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
    project_urls={"Documentation": "https://kaira.readthedocs.io", "Source": "https://github.com/ipc-lab/kaira", "Tracker": "https://github.com/ipc-lab/kaira/issues", "Changelog": "https://github.com/ipc-lab/kaira/blob/main/CHANGELOG.md"},
)
