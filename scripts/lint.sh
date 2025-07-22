#!/bin/bash

# Kaira Linting Script
# This script runs all linting and code quality checks using pre-commit hooks

set -e  # Exit on any error

echo "Running linting and code quality checks..."

# Run all pre-commit hooks
pre-commit run --all-files

echo "✅ All linting checks completed successfully!"
