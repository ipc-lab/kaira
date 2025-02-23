#!/usr/bin/env bash
set -euo pipefail

echo "Running flake8 linting..."
flake8 --exclude build,dist,kaira.egg-info .
echo "Linting completed successfully."