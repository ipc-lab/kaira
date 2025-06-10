#!/usr/bin/env python3
"""Generate index.rst files for example galleries automatically.

This script scans the examples directories and generates corresponding index.rst files
in the docs/examples/ directories. It extracts metadata from example docstrings and
creates properly formatted Sphinx-Gallery compatible index files.

Usage:
    python scripts/generate_example_indices.py
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional


class ExampleIndexGenerator:
    """Generate index.rst files for example galleries."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.examples_dir = project_root / "examples"
        self.docs_examples_dir = project_root / "docs" / "examples"

        # Category descriptions
        self.category_descriptions = {
            "channels": "Channel models for wireless communications, including AWGN, fading channels, and composite channel effects.",
            "constraints": "Constraint handling and optimization techniques for communication systems design and signal processing.",
            "modulation": "Digital modulation schemes and their characteristics in Kaira. These examples show how to implement, analyze, and compare different digital modulation techniques commonly used in modern communications systems.",
            "metrics": "Performance metrics and evaluation tools for communication systems, including error rates, capacity measures, and signal quality metrics.",
            "data": "Data handling utilities, dataset management, and preprocessing tools for machine learning and communications applications.",
            "losses": "Loss functions and optimization objectives for neural networks in communications, including custom losses for specific tasks.",
            "models": "Neural network models and architectures for communications, including deep learning approaches to channel coding, modulation, and signal processing.",
            "models_fec": "Forward Error Correction (FEC) models and coding techniques, including modern deep learning approaches to error correction and classical coding schemes.",
            "benchmarks": "Benchmarking tools and performance comparisons for different algorithms, models, and system configurations.",
            "utils": "Utility functions and helper tools for signal processing, visualization, and system analysis.",
        }

    def extract_example_metadata(self, example_file: Path) -> Optional[Dict[str, str]]:
        """Extract metadata from an example file's docstring.

        Args:
            example_file: Path to the example Python file

        Returns:
            Dictionary containing title, description, and other metadata
        """
        try:
            with open(example_file, encoding="utf-8") as f:
                content = f.read()

            # Parse the AST to get the module docstring
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)

            if not docstring:
                print(f"Warning: No docstring found in {example_file}")
                return None

            # Parse the docstring to extract title and description
            lines = docstring.strip().split("\n")

            # Find the title (first non-empty line, often surrounded by =)
            title = None
            description = ""

            # Remove leading/trailing empty lines
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            if not lines:
                return None

            # Look for title in docstring format
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith("="):
                    title = line
                    break

            if not title:
                # Fallback: use filename as title
                title = example_file.stem.replace("plot_", "").replace("_", " ").title()

            # Extract description (lines after title and separator)
            desc_lines = []
            started_desc = False

            for line in lines:
                line = line.strip()
                if started_desc and line and not line.startswith("="):
                    desc_lines.append(line)
                elif line and not line.startswith("=") and line != title:
                    if title in line or started_desc:
                        continue
                    started_desc = True
                    desc_lines.append(line)

            # Join description lines and clean up
            description = " ".join(desc_lines).strip()
            if not description:
                # Fallback description
                description = f"Demonstrates {title.lower()} techniques and analysis."

            # Clean up title - remove extra spaces and improve formatting
            title = re.sub(r"\s+", " ", title).strip()

            return {"title": title, "description": description, "filename": example_file.stem, "alt_text": title}

        except Exception as e:
            print(f"Error processing {example_file}: {e}")
            return None

    def get_examples_in_category(self, category: str) -> List[Dict[str, str]]:
        """Get all examples in a category with their metadata.

        Args:
            category: Category name (e.g., 'modulation', 'channels')

        Returns:
            List of example metadata dictionaries
        """
        category_path = self.examples_dir / category

        if not category_path.exists():
            return []

        examples = []

        # Find all Python files that start with 'plot_'
        for example_file in sorted(category_path.glob("plot_*.py")):
            metadata = self.extract_example_metadata(example_file)
            if metadata:
                examples.append(metadata)

        return examples

    def generate_index_content(self, category: str, examples: List[Dict[str, str]]) -> str:
        """Generate the content for an index.rst file.

        Args:
            category: Category name
            examples: List of example metadata

        Returns:
            Generated RST content
        """
        # Category title and description
        title = category.replace("_", " ").title()
        description = self.category_descriptions.get(category, f"Examples for {title.lower()}.")

        content = [":orphan:", "", title, "=" * len(title), "", description, "", ".. raw:: html", "", '    <div class="sphx-glr-thumbnails">', ""]

        # Add each example
        for example in examples:
            content.extend(
                [
                    ".. raw:: html",
                    "",
                    f"    <div class=\"sphx-glr-thumbcontainer\" tooltip=\"{example['description']}\">",
                    "",
                    ".. only:: html",
                    "",
                    f"    .. image:: /auto_examples/{category}/images/thumb/sphx_glr_{example['filename']}_thumb.png",
                    f"      :alt: {example['alt_text']}",
                    "",
                    f"    :ref:`sphx_glr_auto_examples_{category}_{example['filename']}.py`",
                    "",
                    ".. raw:: html",
                    "",
                    f"      <div class=\"sphx-glr-thumbnail-title\">{example['title']}</div>",
                    "    </div>",
                    "",
                ]
            )

        # Close the gallery
        content.extend([".. raw:: html", "", "    </div>", "", "", ".. toctree:", "   :hidden:", ""])

        # Add toctree entries
        for example in examples:
            content.append(f"   /auto_examples/{category}/{example['filename']}")

        return "\n".join(content) + "\n"

    def generate_index_file(self, category: str) -> bool:
        """Generate index.rst file for a specific category.

        Args:
            category: Category name

        Returns:
            True if successful, False otherwise
        """
        examples = self.get_examples_in_category(category)

        if not examples:
            print(f"No examples found for category: {category}")
            return False

        # Generate content
        content = self.generate_index_content(category, examples)

        # Ensure the docs/examples directory exists
        docs_category_dir = self.docs_examples_dir / category
        docs_category_dir.mkdir(parents=True, exist_ok=True)

        # Write the index file
        index_file = docs_category_dir / "index.rst"

        try:
            with open(index_file, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Generated: {index_file}")
            print(f"  - Found {len(examples)} examples")

            return True

        except Exception as e:
            print(f"Error writing {index_file}: {e}")
            return False

    def generate_all_indices(self) -> None:
        """Generate index.rst files for all example categories."""
        # Get all categories from examples directory
        categories = []

        for item in self.examples_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Check if there are any plot_*.py files
                if list(item.glob("plot_*.py")):
                    categories.append(item.name)

        if not categories:
            print("No example categories found!")
            return

        print(f"Found {len(categories)} example categories: {', '.join(categories)}")
        print()

        successful = 0
        for category in sorted(categories):
            if self.generate_index_file(category):
                successful += 1
            print()

        print(f"Successfully generated {successful}/{len(categories)} index files.")

    def verify_examples_sync(self) -> None:
        """Verify that all examples are included in their index files."""
        print("Verifying example synchronization...")
        print("-" * 50)

        for category_dir in self.examples_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("."):
                continue

            category = category_dir.name
            examples = list(category_dir.glob("plot_*.py"))

            if not examples:
                continue

            index_file = self.docs_examples_dir / category / "index.rst"

            if not index_file.exists():
                print(f"❌ {category}: Missing index.rst file")
                continue

            # Read the index file and check for references
            try:
                with open(index_file, encoding="utf-8") as f:
                    index_content = f.read()

                missing_examples = []
                for example_file in examples:
                    example_name = example_file.stem
                    if example_name not in index_content:
                        missing_examples.append(example_name)

                if missing_examples:
                    print(f"❌ {category}: Missing examples in index: {', '.join(missing_examples)}")
                else:
                    print(f"✅ {category}: All {len(examples)} examples included")

            except Exception as e:
                print(f"❌ {category}: Error reading index file: {e}")


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(description="Generate example gallery index files")
    parser.add_argument("--verify", action="store_true", help="Verify synchronization instead of generating files")
    parser.add_argument("--category", type=str, help="Generate index for specific category only")

    args = parser.parse_args()

    # Find project root (directory containing this script's parent)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    generator = ExampleIndexGenerator(project_root)

    if args.verify:
        generator.verify_examples_sync()
    elif args.category:
        success = generator.generate_index_file(args.category)
        if not success:
            exit(1)
    else:
        generator.generate_all_indices()


if __name__ == "__main__":
    main()
