#!/usr/bin/env bash

# Move to the project root directory
cd "$(dirname "$0")/.." || exit

# Generate the API reference documentation automatically
echo "Generating API reference documentation..."
python scripts/generate_api_reference.py docs/api_reference.rst

# Generate the changelog documentation from CHANGELOG.md
echo "Generating changelog documentation..."
python scripts/generate_changelog.py

# Move to the docs directory
cd docs || exit

# Generate API documentation for the current Kaira library with no-index option
# to avoid duplicate object descriptions
# sphinx-apidoc --no-headings --no-toc --implicit-namespaces --no-index -f -o . ../kaira

# Clean existing builds and generate HTML documentation.
make clean
make html

echo "Documentation build complete!"
