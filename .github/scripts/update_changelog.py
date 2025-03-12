#!/usr/bin/env python3
"""This script updates CHANGELOG.md when a new release is published on GitHub.

It extracts information from the release and formats it according to Keep a Changelog format.
"""

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/ipc-lab/kaira"
CHANGELOG_PATH = "CHANGELOG.md"


def get_current_date():
    """Return the current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def get_release_date():
    """Get the release date from the environment variables if available, otherwise use the current
    date."""
    release_body = os.environ.get("RELEASE_BODY", "")
    # Try to find a date in the release body
    date_match = re.search(r"Released on: (\d{4}-\d{2}-\d{2})", release_body)
    if date_match:
        return date_match.group(1)

    # Fallback to current date
    return get_current_date()


def get_latest_version_info():
    """Extract version information from release tag and body."""
    tag = os.environ.get("RELEASE_TAG", "").lstrip("v")
    name = os.environ.get("RELEASE_NAME", tag if tag else "Unknown Version")
    body = os.environ.get("RELEASE_BODY", "")

    # Get release date
    date = get_release_date()

    # Process the release body to extract categories
    categories = {}
    current_category = None

    # Default categories if none found

    for line in body.splitlines():
        # Check for category headers like "### Added" or "## Added"
        category_match = re.match(r"^#+\s+(\w+)$", line.strip())
        if category_match:
            current_category = category_match.group(1)
            if current_category not in categories:
                categories[current_category] = []
            continue

        # If we're in a category and the line isn't empty, add it to that category
        if current_category and line.strip() and not line.startswith("#"):
            # Clean up bullet points for consistency
            line = re.sub(r"^\s*[-*]\s*", "- ", line.strip())
            if not line.startswith("-"):
                line = f"- {line}"
            categories[current_category].append(line)

    # If no categories were found, try to extract from unformatted text
    if not categories:
        categories["Added"] = []
        for line in body.splitlines():
            if line.strip() and not line.startswith("#"):
                # Clean up bullet points for consistency
                line = re.sub(r"^\s*[-*]\s*", "- ", line.strip())
                if not line.startswith("-"):
                    line = f"- {line}"
                categories["Added"].append(line)

    return {"version": tag, "name": name, "date": date, "categories": categories}


def update_changelog(version_info):
    """Update the CHANGELOG.md file with the new release information."""
    changelog_path = Path(CHANGELOG_PATH)
    if not changelog_path.exists():
        print(f"Error: {CHANGELOG_PATH} not found.")
        return False

    with open(changelog_path) as file:
        content = file.read()

    # Extract the unreleased section
    unreleased_pattern = r"## \[Unreleased\](.*?)##"
    unreleased_match = re.search(unreleased_pattern, content, re.DOTALL)

    if unreleased_match:
        unreleased_match.group(1).strip()
    else:
        pass

    # Format the new release section
    version = version_info["version"]
    date = version_info["date"]

    # Create new release section
    new_release_section = f"## [{version}] - {date}\n"

    # Add categories and their items
    for category, items in version_info["categories"].items():
        if items:  # Only add categories with content
            new_release_section += f"### {category}\n"
            for item in items:
                new_release_section += f"{item}\n"
            new_release_section += "\n"

    # Clean up any trailing newlines
    new_release_section = new_release_section.rstrip() + "\n"

    # Reset the Unreleased section
    new_unreleased_section = "## [Unreleased]\n### Added\n- Future features and improvements will be listed here\n\n"

    # Update the links at the bottom
    get_previous_version(version)

    links_section = f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD\n"
    links_section += f"[{version}]: {REPO_URL}/releases/tag/v{version}\n"

    # Replace the Unreleased section with new content
    if unreleased_match:
        content = content.replace(unreleased_match.group(0), new_unreleased_section + new_release_section + "## ")
    else:
        content = re.sub(r"# Changelog.*?##", "# Changelog\n\n" + new_unreleased_section + new_release_section + "##", content, flags=re.DOTALL)

    # Replace the links section
    links_pattern = r"\[Unreleased\]: .*"
    if re.search(links_pattern, content):
        # Replace the unreleased link
        content = re.sub(r"\[Unreleased\]: .*", f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD", content)

        # Add the new version link if it doesn't exist
        version_link_pattern = rf"\[{version}\]: .*"
        if not re.search(version_link_pattern, content):
            links_match = re.search(r"(\[Unreleased\]: .*)", content)
            if links_match:
                content = content.replace(links_match.group(0), f"{links_match.group(0)}\n[{version}]: {REPO_URL}/releases/tag/v{version}")
    else:
        # If no links section exists, add it at the end
        content += f"\n[Unreleased]: {REPO_URL}/compare/v{version}...HEAD\n"
        content += f"[{version}]: {REPO_URL}/releases/tag/v{version}\n"

    # Write the updated content back to the file
    with open(changelog_path, "w") as file:
        file.write(content)

    return True


def get_previous_version(current_version):
    """Get the previous version from git tags."""
    try:
        result = subprocess.run(["git", "tag", "--sort=-v:refname"], capture_output=True, text=True, check=True)
        tags = result.stdout.strip().split("\n")

        # Filter tags to only include version tags
        version_tags = [tag for tag in tags if tag.startswith("v")]

        current_tag = f"v{current_version}"

        # Find the previous version
        for tag in version_tags:
            if tag != current_tag:
                return tag.lstrip("v")

        # If no previous tag is found
        return "0.0.0"
    except Exception as e:
        print(f"Error getting previous version: {e}")
        return "0.0.0"


def main():
    """Main function to update the changelog."""
    print("Updating CHANGELOG.md with new release information...")

    version_info = get_latest_version_info()
    print(f"Processing release: {version_info['version']} - {version_info['date']}")

    if update_changelog(version_info):
        print("CHANGELOG.md updated successfully.")
    else:
        print("Failed to update CHANGELOG.md.")


if __name__ == "__main__":
    main()
