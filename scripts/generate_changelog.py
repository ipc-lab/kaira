#!/usr/bin/env python3
"""Script to automatically generate changelog.rst from CHANGELOG.md.

This script reads the CHANGELOG.md file and converts it to reStructuredText format for inclusion in
the Sphinx documentation.
"""

import re
import sys
from pathlib import Path


def parse_markdown_changelog(content: str) -> str:
    """Parse the Markdown changelog content and convert it to reStructuredText.

    Args:
        content: The CHANGELOG.md content as a string

    Returns:
        The converted reStructuredText content
    """
    lines = content.split("\n")
    rst_lines = []

    # Skip the first few lines (comment and title)
    in_content = False
    header_added = False

    for line in lines:
        # Skip HTML comments
        if line.strip().startswith("<!--"):
            continue

        # Main title
        if line.startswith("# Changelog"):
            if not header_added:
                rst_lines.extend(
                    [
                        "Changelog",
                        "=========",
                        "",
                        "All notable changes to the Kaira project are documented in this file.",
                        "",
                        "The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,",
                        "and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.",
                        "",
                    ]
                )
                header_added = True
            in_content = True
            continue

        if not in_content:
            continue

        # Skip the intro paragraph that repeats info
        if "The format is based on" in line or "and this project adheres" in line or "All notable changes to" in line:
            continue

        # Version headers (## [version] - date)
        version_match = re.match(r"^## \[([^\]]+)\](?:\s*-\s*(.+))?", line)
        if version_match:
            version = version_match.group(1)
            date = version_match.group(2) if version_match.group(2) else ""

            if version.lower() == "unreleased":
                title = "Unreleased Changes"
            else:
                title = f"Version {version}"
                if date:
                    title += f" ({date})"

            rst_lines.extend([title, "-" * len(title), ""])
            continue

        # Subsection headers (### Added, ### Changed, etc.)
        subsection_match = re.match(r"^### (.+)", line)
        if subsection_match:
            subsection = subsection_match.group(1)
            rst_lines.extend([subsection, "^" * len(subsection), ""])
            continue

        # Convert bullet points
        if line.strip().startswith("- "):
            # Handle nested bullet points with proper indentation
            indent_level = len(line) - len(line.lstrip()) - 2  # -2 for '- '
            bullet_content = line.strip()[2:]  # Remove '- '

            # Convert markdown links to rst format
            bullet_content = convert_markdown_links(bullet_content)

            # Convert bold text
            bullet_content = re.sub(r"\*\*(.+?)\*\*", r"**\1**", bullet_content)

            # Add proper indentation
            rst_line = "  " * (indent_level // 2) + "* " + bullet_content
            rst_lines.append(rst_line)
            continue

        # Convert markdown links in regular text
        if line.strip() and not line.startswith("["):
            converted_line = convert_markdown_links(line)
            converted_line = re.sub(r"\*\*(.+?)\*\*", r"**\1**", converted_line)
            rst_lines.append(converted_line)
            continue

        # Skip link reference definitions at the end
        if re.match(r"^\[.+\]:\s*https?://", line):
            continue

        # Add empty lines as they are
        if not line.strip():
            rst_lines.append("")

    return "\n".join(rst_lines).rstrip("\n") + "\n"


def convert_markdown_links(text: str) -> str:
    """Convert markdown links to reStructuredText format.

    Args:
        text: Text that may contain markdown links

    Returns:
        Text with converted links
    """
    # Convert [text](url) to `text <url>`_
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    return re.sub(link_pattern, r"`\1 <\2>`_", text)


def main():
    """Main function to generate changelog.rst from CHANGELOG.md."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    changelog_md_path = project_root / "CHANGELOG.md"
    changelog_rst_path = project_root / "docs" / "changelog.rst"

    # Check if CHANGELOG.md exists
    if not changelog_md_path.exists():
        print(f"Error: CHANGELOG.md not found at {changelog_md_path}")
        sys.exit(1)

    # Read the markdown content
    try:
        with open(changelog_md_path, encoding="utf-8") as f:
            md_content = f.read()
    except Exception as e:
        print(f"Error reading CHANGELOG.md: {e}")
        sys.exit(1)

    # Convert to reStructuredText
    rst_content = parse_markdown_changelog(md_content)

    # Ensure docs directory exists
    changelog_rst_path.parent.mkdir(exist_ok=True)

    # Write the reStructuredText content
    try:
        with open(changelog_rst_path, "w", encoding="utf-8") as f:
            f.write(rst_content)

        print(f"Successfully generated {changelog_rst_path}")
        print(f"Converted {changelog_md_path} to reStructuredText format")

    except Exception as e:
        print(f"Error writing changelog.rst: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
