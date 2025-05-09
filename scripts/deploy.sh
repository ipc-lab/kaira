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

# Parse command line arguments
SKIP_VERSION_CHECK=0
FORCE_DEPLOY=0

while getopts "sf" opt; do
  case $opt in
    s) SKIP_VERSION_CHECK=1 ;;
    f) FORCE_DEPLOY=1 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Extract version from kaira/version.py which is the source of truth for version number
VERSION=$(python3 -c "import sys; sys.path.append('..'); from kaira.version import __version__; print(__version__)")
echo "Preparing to deploy version: $VERSION"

# Check if this version already exists on PyPI, unless version check is skipped
if [ $SKIP_VERSION_CHECK -eq 0 ] && pip install --quiet pykaira=="$VERSION" 2>/dev/null; then
    echo "Error: Version $VERSION already exists on PyPI!"
    echo "Please update the version number in kaira/version.py before deploying."

    if [ $FORCE_DEPLOY -eq 0 ]; then
        echo "To override this check, run the script with -s option to skip version check"
        echo "or -f to force deployment with same version (not recommended)."
        exit 1
    else
        echo "Warning: Force deployment flag set. Proceeding with deployment of existing version."
        echo "This is not recommended and may cause issues with PyPI!"
    fi
fi

# Ask for confirmation before proceeding
read -p "Are you sure you want to deploy version $VERSION to PyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Change to parent directory before cleaning and building
cd ..

echo "Cleaning previous build artifacts..."
# More thorough cleaning of Python build artifacts and cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name ".eggs" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type f -name "coverage.xml" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".coverage*" -exec rm -rf {} +
find . -type d -name "htmlcov" -exec rm -rf {} +

# Clean documentation build artifacts
echo "Cleaning documentation build artifacts..."
if [ -d "docs/_build" ]; then
    echo "Removing docs/_build directory..."
    rm -rf docs/_build/
fi
if [ -d "docs/gen_modules" ]; then
    echo "Removing docs/gen_modules directory..."
    rm -rf docs/gen_modules/
fi
if [ -d "docs/generated" ]; then
    echo "Removing docs/generated directory..."
    rm -rf docs/generated/
fi
if [ -d "docs/auto_examples" ]; then
    echo "Removing docs/auto_examples directory..."
    rm -rf docs/auto_examples/
fi
# Also clean any sphinx-related temporary files
find docs -name ".DS_Store" -delete
find docs -name "sg_execution_times.rst" -delete

# Remove build and dist directories
rm -rf build/ dist/ ./*.egg-info/

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
