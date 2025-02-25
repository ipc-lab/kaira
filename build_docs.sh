#!/usr/bin/env bash

# Change to the docs directory (ensure the script is executed from any directory)
cd "$(dirname "$0")" || exit

# Generate API documentation for the current Kaira library.
sphinx-apidoc --no-headings --no-toc -f -o . ../kaira

# Clean existing builds and generate HTML documentation.
make clean
make html
