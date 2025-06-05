#!/usr/bin/env python
"""Script to automatically generate API reference documentation for Kaira.

This script inspects the Kaira package structure and generates reStructuredText content for the API
reference documentation, ensuring all exposed modules, classes, functions, and other objects are
properly documented.
"""

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

# Add the parent directory to the path so we can import kaira
sys.path.insert(0, str(Path(__file__).parent.parent))

import kaira


def get_module_members(module: ModuleType) -> Dict[str, List[str]]:
    """Get all exposed members from a module and categorize them by type.

    Args:
        module: The module to inspect.

    Returns:
        A dictionary mapping member types to lists of member names.
    """
    if not hasattr(module, "__all__"):
        return {}

    members: Dict[str, List[str]] = {}
    for name in module.__all__:
        if not hasattr(module, name):
            continue

        obj = getattr(module, name)

        # Categorize by type
        if inspect.isclass(obj):
            members.setdefault("classes", []).append(name)
        elif inspect.isfunction(obj):
            members.setdefault("functions", []).append(name)
        elif inspect.ismodule(obj):
            members.setdefault("modules", []).append(name)
        else:
            members.setdefault("others", []).append(name)

    # Sort each category alphabetically
    for category in members:
        members[category].sort()

    return members


def generate_autosummary_block(module_path: str, members: List[str], template: str = "class.rst") -> str:
    """Generate an autosummary block for a list of members.

    Args:
        module_path: The full module path.
        members: List of member names to include.
        template: The template to use for the autosummary.

    Returns:
        A string containing the reStructuredText autosummary block.
    """
    if not members:
        return ""

    block = [
        f".. currentmodule:: {module_path}",
        "",
        ".. autosummary::",
        "   :toctree: generated",
        f"   :template: {template}",
        "   :nosignatures:",
        "",
    ]

    for member in members:
        block.append(f"   {member}")

    return "\n".join(block)


def process_module(module_path: str, module: ModuleType) -> Dict[str, str]:
    """Process a module to generate documentation blocks.

    Args:
        module_path: The full module path.
        module: The module object.

    Returns:
        A dictionary mapping content types to reStructuredText blocks.
    """
    members = get_module_members(module)
    blocks = {}

    if "classes" in members and members["classes"]:
        blocks["classes"] = generate_autosummary_block(module_path, members["classes"], "class.rst")

    if "functions" in members and members["functions"]:
        blocks["functions"] = generate_autosummary_block(module_path, members["functions"], "function.rst")

    return blocks


def scan_modules_recursively(base_module: ModuleType, base_path: str, visited: set) -> Dict[str, Dict[str, str]]:
    """Recursively scan modules and their submodules for documentation.

    Args:
        base_module: The base module to start from.
        base_path: The base module path.
        visited: Set of already visited module paths to avoid cycles.

    Returns:
        A nested dictionary of module paths and their documentation blocks.
    """
    if base_path in visited:
        return {}

    visited.add(base_path)
    result = {}

    # Process the base module itself
    base_blocks = process_module(base_path, base_module)
    if base_blocks:
        result[base_path] = base_blocks

    # Check if the module is a package (has __file__ and is a directory with __init__.py)
    is_package = False
    if hasattr(base_module, "__file__") and base_module.__file__ is not None:
        module_file = Path(base_module.__file__)
        module_dir = module_file.parent
        is_package = module_file.name == "__init__.py" and module_dir.is_dir()

        # If it's a package, scan its subdirectories for submodules
        if is_package:
            for item in module_dir.iterdir():
                # Check for Python files (modules) or directories with __init__.py (packages)
                if (item.is_file() and item.suffix == ".py" and item.name != "__init__.py") or (item.is_dir() and (item / "__init__.py").exists()):

                    # Determine the submodule name and path
                    submodule_name = item.stem if item.is_file() else item.name
                    submodule_path = f"{base_path}.{submodule_name}"

                    # Skip if this module path has already been processed
                    if submodule_path in visited:
                        continue

                    try:
                        # Try to import the submodule
                        submodule = importlib.import_module(submodule_path)

                        # Process the submodule recursively
                        submodule_results = scan_modules_recursively(submodule, submodule_path, visited)
                        result.update(submodule_results)
                    except (ImportError, AttributeError, ModuleNotFoundError) as e:
                        print(f"Warning: Could not import {submodule_path}: {e}")

    # Also check the __all__ attribute for any modules
    if hasattr(base_module, "__all__"):
        for name in base_module.__all__:
            # Skip if not a module or already processed
            if not hasattr(base_module, name) or not inspect.ismodule(getattr(base_module, name)) or f"{base_path}.{name}" in visited:
                continue

            submodule = getattr(base_module, name)
            submodule_path = f"{base_path}.{name}"

            # Process the submodule recursively if not already processed
            submodule_results = scan_modules_recursively(submodule, submodule_path, visited)
            result.update(submodule_results)

    return result


def scan_submodules(base_module: ModuleType, base_path: str) -> Dict[str, Dict[str, str]]:
    """Recursively scan submodules and generate documentation.

    This is a wrapper around scan_modules_recursively that initializes the visited set.

    Args:
        base_module: The base module to start from.
        base_path: The base module path.

    Returns:
        A nested dictionary of module paths and their documentation blocks.
    """
    return scan_modules_recursively(base_module, base_path, set())


def generate_api_reference() -> str:
    """Generate the full API reference content.

    Returns:
        A string containing the reStructuredText content for the API reference.
    """
    # Header of the API reference
    header = """Kaira API Reference
=====================================

.. note::
   Kaira version |version| documentation. For older versions, please refer to the version selector above.

This documentation provides a comprehensive reference of Kaira's components organized by functional category.
Each component is documented with its parameters, methods, and usage examples.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
--------------

Kaira is a modular toolkit for communication systems built on PyTorch. The library is organized into
several key modules that handle different aspects of communication systems:

- **Channels**: Model transmission mediums with various noise and distortion characteristics
- **Constraints**: Enforce practical limitations on transmitted signals
- **Metrics**: Evaluate quality and performance of communication systems
- **Models**: Implement neural network architectures for encoding/decoding and end-to-end communication systems
- **Modulations**: Implement digital modulation schemes for wireless transmission
- **Losses**: Provide objective functions for training neural networks
- **Utilities**: Helper functions and tools for common operations

Base Components
--------------------------

Base classes define the fundamental interfaces that all implementations must adhere to.
These abstract classes establish the contract that derived classes must fulfill.

.. currentmodule:: kaira

.. autosummary::
   :toctree: generated
   :template: class.rst
   :recursive:
   :nosignatures:

   channels.BaseChannel
   constraints.BaseConstraint
   metrics.BaseMetric
   models.BaseModel
   modulations.BaseModulator
   modulations.BaseDemodulator
   losses.BaseLoss
"""

    # Collect all documentation blocks
    all_blocks: Dict[str, Dict[str, str]] = {}

    # Process main modules
    for module_name in kaira.__all__:
        if module_name == "__version__":
            continue

        module = getattr(kaira, module_name)
        module_path = f"kaira.{module_name}"

        # Add the main module section
        section_title = module_name.capitalize()
        section_underline = "-" * len(section_title)

        description = module.__doc__.strip() if module.__doc__ else f"{section_title} module for Kaira."

        all_blocks[module_path] = {
            "title": f"{section_title}\n{section_underline}\n\n{description}\n",
        }

        # Process the module and its submodules
        submodule_blocks = scan_submodules(module, module_path)

        # Add the blocks to the result
        for submodule_path, blocks in submodule_blocks.items():
            # Skip the main module as we've already added it
            if submodule_path == module_path:
                all_blocks[submodule_path].update(blocks)
            else:
                # Extract the submodule name for the section title
                submodule_name = submodule_path.split(".")[-1]
                formatted_name = " ".join(word.capitalize() for word in submodule_name.split("_"))

                subsection_title = formatted_name
                subsection_underline = "^" * len(subsection_title)

                # Try to get the module docstring
                try:
                    submodule = importlib.import_module(submodule_path)
                    description = submodule.__doc__.strip() if submodule.__doc__ else f"{formatted_name} submodule."
                except (ImportError, AttributeError):
                    description = f"{formatted_name} submodule."

                all_blocks[submodule_path] = {
                    "title": f"{subsection_title}\n{subsection_underline}\n\n{description}\n",
                }
                all_blocks[submodule_path].update(blocks)

    # Combine all blocks into the final document
    output: List[str] = [header]

    # Add the main module sections in a specific order
    module_order = ["channels", "constraints", "metrics", "models", "modulations", "losses", "data", "utils"]

    for module_name in module_order:
        module_path = f"kaira.{module_name}"
        if module_path not in all_blocks:
            continue

        module_blocks = all_blocks.pop(module_path)
        output.append(module_blocks.pop("title"))

        # Add the main module's classes and functions
        if "classes" in module_blocks:
            output.append(module_blocks["classes"])
            output.append("\n")

        if "functions" in module_blocks:
            output.append(module_blocks["functions"])
            output.append("\n")

        # Add submodule sections for this module
        submodule_paths = [p for p in all_blocks.keys() if p.startswith(f"{module_path}.")]
        for submodule_path in sorted(submodule_paths):
            cur_submodule_blocks = all_blocks.pop(submodule_path)
            output.append(cur_submodule_blocks.pop("title"))

            if "classes" in cur_submodule_blocks:
                output.append(cur_submodule_blocks["classes"])
                output.append("\n")

            if "functions" in cur_submodule_blocks:
                output.append(cur_submodule_blocks["functions"])
                output.append("\n")

    # Add any remaining modules
    for module_path, module_blocks in sorted(all_blocks.items()):
        output.append(module_blocks.pop("title"))

        if "classes" in module_blocks:
            output.append(module_blocks["classes"])
            output.append("\n")

        if "functions" in module_blocks:
            output.append(module_blocks["functions"])
            output.append("\n")

    return "\n".join(output).rstrip("\n") + "\n"


if __name__ == "__main__":
    api_reference = generate_api_reference()

    # Get the output path from command line arguments or use a default
    output_path = sys.argv[1] if len(sys.argv) > 1 else "docs/api_reference.rst"

    # Ensure output_path is not None (for type checking)
    assert output_path is not None, "Output path cannot be None"

    with open(output_path, "w") as f:
        f.write(api_reference)

    print(f"API reference generated and saved to {output_path}")
