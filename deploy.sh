#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not installed. Please install it and try again."
        exit 1
    fi
}

# Check for required tools
check_command python3
check_command pip
check_command twine

# Ensure dependencies are installed
pip install --quiet --upgrade setuptools wheel twine

echo "=== Starting deployment process for Kaira ==="

# Extract version from setup.py or another source
VERSION=$(python3 -c "import re; print(re.search(r'version=[\"\'](.*?)[\"\']', open('setup.py').read()).group(1))")
echo "Preparing to deploy version: $VERSION"

# Check if this version already exists on PyPI
if pip install --quiet kaira==$VERSION 2>/dev/null; then
    echo "Error: Version $VERSION already exists on PyPI!"
    echo "Please update the version number in setup.py before deploying."
    exit 1
fi

# Ask for confirmation before proceeding
read -p "Are you sure you want to deploy version $VERSION to PyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info/

echo "Building distribution packages..."
python3 setup.py sdist bdist_wheel

echo "Checking package with twine..."
twine check dist/*

echo "Ready to upload to PyPI..."
read -p "Proceed with upload? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled. Build artifacts are available in dist/ directory."
    exit 0
fi

echo "Uploading to PyPI..."
python3 -m twine upload dist/*

echo "=== Deployment completed successfully ==="
