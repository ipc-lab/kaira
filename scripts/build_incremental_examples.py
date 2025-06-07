#!/usr/bin/env python3
"""Incremental sphinx-gallery build script.

This script builds only the changed example files instead of rebuilding all examples from scratch,
which significantly speeds up the CI process.
"""

import argparse
import os
import subprocess  # nosec B404 # subprocess needed for sphinx-build integration
import sys
import tempfile
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command with proper error handling."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=capture_output, text=True)  # nosec B602 # shell=True needed for sphinx commands
    if capture_output:
        return result.stdout.strip()
    return result


def build_specific_examples(changed_files, docs_dir):
    """Build only the specific changed example files using sphinx-gallery.

    This bypasses sphinx-build's tendency to regenerate everything and directly uses sphinx-
    gallery's API to build only changed files.
    """
    print(f"🔧 Building {len(changed_files)} changed examples incrementally...")

    # Convert to absolute paths
    docs_dir = Path(docs_dir).absolute()
    os.chdir(docs_dir)

    # Import required modules
    sys.path.insert(0, str(docs_dir))

    try:
        import conf  # type: ignore[import]
        from sphinx.application import Sphinx
        from sphinx.util.docutils import patch_docutils, unpatch_docutils
        from sphinx_gallery.gen_gallery import generate_gallery, parse_config
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("Falling back to full sphinx-build...")
        return False

    success_count = 0
    error_count = 0

    # Get the base gallery configuration
    gallery_conf = getattr(conf, "sphinx_gallery_conf", {})

    for changed_file in changed_files:
        if not changed_file.strip() or not changed_file.endswith(".py"):
            continue

        # Parse the file path: examples/category/file.py
        parts = changed_file.strip().split("/")
        if len(parts) < 3 or parts[0] != "examples":
            print(f"⚠️ Skipping invalid path: {changed_file}")
            continue

        category = parts[1]
        example_name = parts[2]

        print(f"\n📝 Processing {changed_file}")

        # Check if source file exists
        example_src = docs_dir.parent / "examples" / category / example_name
        if not example_src.exists():
            print(f"⚠️ Source file not found: {example_src}")
            error_count += 1
            continue

        # Ensure output directory exists
        output_dir = docs_dir / "auto_examples" / category
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create a minimal configuration for this specific file
            file_conf = gallery_conf.copy()
            file_conf.update(
                {
                    "examples_dirs": [f"../examples/{category}"],
                    "gallery_dirs": [f"auto_examples/{category}"],
                    "filename_pattern": f"/{example_name}$",
                    "abort_on_example_error": False,
                    "only_warn_on_example_error": True,
                    "plot_gallery": True,
                    "download_all_examples": True,
                }
            )

            # Create a temporary Sphinx app context
            with tempfile.TemporaryDirectory():
                try:
                    # Set up minimal Sphinx environment
                    srcdir = str(docs_dir)
                    outdir = str(docs_dir / "_build" / "html")
                    doctreedir = str(docs_dir / "_build" / "doctrees")
                    confdir = str(docs_dir)

                    # Initialize Sphinx app
                    app = Sphinx(srcdir, confdir, outdir, doctreedir, "html")

                    # Parse and validate the configuration
                    parsed_conf = parse_config(app, file_conf)

                    # Apply docutils patches and generate
                    patch_docutils("0.17")
                    generate_gallery(app, parsed_conf)

                    print(f"✅ Successfully generated {changed_file}")
                    success_count += 1

                except Exception as e:
                    print(f"❌ Error generating {changed_file}: {e}")
                    error_count += 1

                finally:
                    try:
                        unpatch_docutils()
                    except Exception as e:
                        # Ignore unpatch errors as they're not critical for the build
                        print(f"Warning: Failed to unpatch docutils: {e}")

        except Exception as e:
            print(f"❌ Configuration error for {changed_file}: {e}")
            error_count += 1

    print("\n🎯 Incremental build completed:")
    print(f"  ✅ Success: {success_count} files")
    print(f"  ❌ Errors: {error_count} files")

    return success_count > 0


def update_gallery_index(docs_dir):
    """Update gallery indexes and cross-references without rebuilding examples.

    This runs a quick sphinx-build with example generation disabled to update the gallery structure
    and navigation.
    """
    print("🔗 Updating gallery indexes and cross-references...")

    cmd = "sphinx-build -b html " "-D sphinx_gallery_conf.plot_gallery=False " "-D sphinx_gallery_conf.run_stale_examples=False " ". _build/html -q -W --keep-going"

    try:
        run_command(cmd, cwd=docs_dir)
        print("✅ Gallery indexes updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Gallery index update failed: {e}")
        return False


def main():
    """Main entry point for the incremental build script.

    Parse command line arguments and orchestrate the incremental build process.
    """
    parser = argparse.ArgumentParser(description="Build changed examples incrementally")
    parser.add_argument("--changed-files", required=True, help="Space-separated list of changed example files")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory (default: docs)")
    parser.add_argument("--fallback-full-build", action="store_true", help="Fall back to full sphinx-build if incremental build fails")

    args = parser.parse_args()

    # Parse changed files
    changed_files = [f.strip() for f in args.changed_files.split() if f.strip()]

    if not changed_files:
        print("ℹ️ No changed files provided")
        return 0

    print(f"🚀 Starting incremental build for {len(changed_files)} files:")
    for f in changed_files:
        print(f"  - {f}")

    docs_dir = Path(args.docs_dir).absolute()
    if not docs_dir.exists():
        print(f"❌ Documentation directory not found: {docs_dir}")
        return 1

    # Attempt incremental build
    success = build_specific_examples(changed_files, docs_dir)

    if success:
        # Update gallery indexes
        update_gallery_index(docs_dir)
        print("🎉 Incremental build completed successfully!")
        return 0
    elif args.fallback_full_build:
        print("🔄 Incremental build failed, falling back to full build...")
        os.chdir(docs_dir)
        try:
            run_command("sphinx-build -b html " "-D sphinx_gallery_conf.plot_gallery=True " "-D sphinx_gallery_conf.download_all_examples=True " ". _build/html -v")
            print("✅ Full build completed successfully")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Full build also failed: {e}")
            return 1
    else:
        print("❌ Incremental build failed and fallback disabled")
        return 1


if __name__ == "__main__":
    sys.exit(main())
