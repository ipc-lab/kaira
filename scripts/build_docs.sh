#!/usr/bin/env bash

# Move to the docs directory, accounting for the new script location
cd "$(dirname "$0")/../docs" || exit

# Generate API documentation for the current Kaira library (adjust path)
sphinx-apidoc --no-headings --no-toc -f -o . ../kaira

# Clean existing builds and generate HTML documentation.
make clean
make html
