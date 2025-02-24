#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Remove previous builds.
rm -rf build/ dist/ kaira.egg-info/

# Build source and wheel distributions.
python setup.py sdist bdist_wheel

# Upload the distributions to PyPI.
python3 -m twine upload dist/*
