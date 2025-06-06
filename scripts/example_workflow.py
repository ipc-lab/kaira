#!/usr/bin/env python3
"""Development workflow script for managing example galleries.

This script provides convenient commands for developers working with examples:
- Add new examples and automatically update indices
- Test example synchronization
- Preview generated index files
- Validate example formatting

Usage:
    python scripts/example_workflow.py add --category modulation --example plot_new_modulation.py
    python scripts/example_workflow.py sync
    python scripts/example_workflow.py preview --category channels
"""

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import must be after path modification
from scripts.generate_example_indices import ExampleIndexGenerator  # noqa: E402


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec B603
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def add_example(category: str, example_file: str, project_root: Path) -> bool:
    """Add a new example and update the corresponding index."""
    generator = ExampleIndexGenerator(project_root)

    example_path = project_root / "examples" / category / example_file

    if not example_path.exists():
        print(f"Error: Example file {example_path} does not exist!")
        print("Please create the example file first.")
        return False

    print(f"Adding example {example_file} to category {category}")

    # Regenerate the index for this category
    if generator.generate_index_file(category):
        print(f"✅ Successfully updated index for {category}")

        # Verify the addition worked
        print("\nVerifying synchronization...")
        examples = generator.get_examples_in_category(category)
        example_names = [ex["filename"] for ex in examples]

        if example_file.replace(".py", "") in example_names:
            print(f"✅ Example {example_file} is now included in the gallery")
            return True
        else:
            print(f"❌ Example {example_file} was not properly added")
            return False
    else:
        return False


def sync_all(project_root: Path) -> bool:
    """Synchronize all example indices."""
    generator = ExampleIndexGenerator(project_root)

    print("Synchronizing all example galleries...")
    generator.generate_all_indices()
    print("\nVerifying synchronization...")
    generator.verify_examples_sync()
    return True


def preview_category(category: str, project_root: Path) -> bool:
    """Preview the generated index content for a category."""
    generator = ExampleIndexGenerator(project_root)

    examples = generator.get_examples_in_category(category)
    if not examples:
        print(f"No examples found in category: {category}")
        return False

    print(f"\n📋 Preview of {category} gallery index:")
    print("=" * 50)

    title = category.replace("_", " ").title()
    description = generator.category_descriptions.get(category, f"Examples for {title.lower()}.")

    print(f"Title: {title}")
    print(f"Description: {description}")
    print(f"Number of examples: {len(examples)}")
    print("\nExamples:")

    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example['title']}")
        print(f"     File: {example['filename']}.py")
        print(f"     Description: {example['description'][:100]}...")
        print()

    return True


def validate_example(example_file: Path) -> bool:
    """Validate an example file format."""
    if not example_file.exists():
        print(f"❌ File does not exist: {example_file}")
        return False

    if not example_file.name.startswith("plot_"):
        print(f"❌ Example filename should start with 'plot_': {example_file.name}")
        return False

    try:
        with open(example_file, encoding="utf-8") as f:
            content = f.read()

        # Check for docstring
        if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
            print(f"❌ Example should start with a docstring: {example_file.name}")
            return False

        # Check for basic structure markers
        if "# %%" not in content:
            print(f"⚠️  Consider adding section markers (# %%) for better documentation: {example_file.name}")

        print(f"✅ Example format looks good: {example_file.name}")
        return True

    except Exception as e:
        print(f"❌ Error reading file {example_file}: {e}")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Example gallery development workflow")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new example to a category")
    add_parser.add_argument("--category", required=True, help="Example category (e.g., modulation, channels)")
    add_parser.add_argument("--example", required=True, help="Example filename (e.g., plot_new_example.py)")

    # Sync command
    subparsers.add_parser("sync", help="Synchronize all example galleries")

    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview a category gallery")
    preview_parser.add_argument("--category", required=True, help="Category to preview")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate example file format")
    validate_parser.add_argument("--file", required=True, help="Path to example file")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test example execution")
    test_parser.add_argument("--category", help="Test all examples in category")
    test_parser.add_argument("--file", help="Test specific example file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    success = True

    if args.command == "add":
        success = add_example(args.category, args.example, project_root)

    elif args.command == "sync":
        success = sync_all(project_root)

    elif args.command == "preview":
        success = preview_category(args.category, project_root)

    elif args.command == "validate":
        example_file = Path(args.file)
        if not example_file.is_absolute():
            example_file = project_root / args.file
        success = validate_example(example_file)

    elif args.command == "test":
        if args.category:
            category_dir = project_root / "examples" / args.category
            if category_dir.exists():
                for example_file in category_dir.glob("plot_*.py"):
                    print(f"\n🧪 Testing {example_file.name}...")
                    success &= run_command(["python", str(example_file)], f"Execute {example_file.name}")
            else:
                print(f"Category directory not found: {category_dir}")
                success = False

        elif args.file:
            example_file = Path(args.file)
            if not example_file.is_absolute():
                example_file = project_root / args.file
            success = run_command(["python", str(example_file)], f"Execute {example_file.name}")
        else:
            print("Please specify either --category or --file for testing")
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
