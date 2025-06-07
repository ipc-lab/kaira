#!/usr/bin/env python3
"""Download auto_examples folder from a remote source (such as GitHub releases or artifacts) before
documentation builds.

It's designed to work both locally and on ReadTheDocs.
"""

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    # Fallback for environments without requests
    requests = None  # type: ignore[assignment]

# Always import urllib for fallback scenarios
import urllib.error
import urllib.request


def log_message(message: str) -> None:
    """Log a message with timestamp."""
    print(f"[download_auto_examples] {message}")


def check_github_artifacts_for_examples(repo_owner: str, repo_name: str, token: Optional[str] = None) -> Optional[str]:
    """Check GitHub Actions artifacts for auto_examples.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        token: GitHub token for API access (optional)

    Returns:
        Download URL for the auto_examples artifact, or None if not found
    """
    try:
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/artifacts"  # Use requests if available, otherwise fallback to urllib
        if requests:
            headers = {}
            if token:
                headers["Authorization"] = f"token {token}"

            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            artifacts_data = response.json()
        else:
            # Fallback to urllib for compatibility
            req = urllib.request.Request(api_url)  # nosec B310 - Using HTTPS URL from trusted GitHub API
            if token:
                req.add_header("Authorization", f"token {token}")

            with urllib.request.urlopen(req) as urllib_response:  # nosec B310 - Using HTTPS URL from trusted GitHub API
                artifacts_data = json.loads(urllib_response.read().decode())

        # Look for recent auto_examples artifacts
        # Sort by creation date to get the most recent first
        artifacts = sorted(
            artifacts_data.get("artifacts", []),
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        for artifact in artifacts:
            name_lower = artifact["name"].lower()
            if ("auto_examples" in name_lower or "auto-examples" in name_lower) and not artifact["expired"]:
                log_message(f"Found auto_examples artifact: {artifact['name']} (created: {artifact.get('created_at', 'unknown')})")
                return artifact["archive_download_url"]

    except Exception as e:
        if requests and hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 404:
                log_message("No GitHub artifacts found or repository not accessible")
            elif e.response.status_code == 403:
                log_message("GitHub artifacts require authentication (GITHUB_TOKEN)")
            else:
                log_message(f"Could not check GitHub artifacts: HTTP {e.response.status_code}")
        elif hasattr(e, "code"):  # urllib error
            if e.code == 404:
                log_message("No GitHub artifacts found or repository not accessible")
            elif e.code == 403:
                log_message("GitHub artifacts require authentication (GITHUB_TOKEN)")
            else:
                log_message(f"Could not check GitHub artifacts: HTTP {e.code} - {e.reason}")  # type: ignore[attr-defined]
        else:
            log_message(f"Could not check GitHub artifacts: {e}")

    return None


def check_github_release_for_examples(repo_owner: str, repo_name: str) -> Optional[str]:
    """Check if there's a recent GitHub release with auto_examples.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name

    Returns:
        Download URL for the auto_examples archive, or None if not found
    """
    try:
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"

        # Use requests if available, otherwise fallback to urllib
        if requests:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            release_data = response.json()
        else:
            # Fallback to urllib for compatibility
            with urllib.request.urlopen(api_url) as urllib_response:  # nosec B310 - Using HTTPS URL from trusted GitHub API
                release_data = json.loads(urllib_response.read().decode())

        # Look for an asset named something like "auto_examples.zip"
        for asset in release_data.get("assets", []):
            if "auto_examples" in asset["name"].lower() and asset["name"].endswith(".zip"):
                log_message(f"Found auto_examples asset: {asset['name']}")
                return asset["browser_download_url"]

    except Exception as e:
        if requests and hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 404:
                log_message("No GitHub releases found or no auto_examples assets available")
            else:
                log_message(f"Could not check GitHub releases: HTTP {e.response.status_code}")
        elif hasattr(e, "code"):  # urllib error
            if e.code == 404:
                log_message("No GitHub releases found or no auto_examples assets available")
            else:
                log_message(f"Could not check GitHub releases: HTTP {e.code} - {e.reason}")  # type: ignore[attr-defined]
        else:
            log_message(f"Could not check GitHub releases: {e}")

    return None


def download_and_extract_examples(download_url: str, target_dir: Path) -> bool:
    """Download and extract auto_examples from a URL.

    Args:
        download_url: URL to download the examples archive
        target_dir: Target directory where auto_examples should be extracted

    Returns:
        True if successful, False otherwise
    """
    try:
        log_message(f"Downloading auto_examples from: {download_url}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Download the file
        if requests:
            response = requests.get(download_url, timeout=60, stream=True)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            # Fallback to urllib for compatibility
            urllib.request.urlretrieve(download_url, tmp_path)  # nosec B310 - URL is validated from trusted GitHub API
        log_message(f"Downloaded to temporary file: {tmp_path}")

        # Extract the archive
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            # List contents to understand structure
            file_list = zip_ref.namelist()
            log_message(f"Archive contains {len(file_list)} files")

            # Find the auto_examples directory in the archive
            auto_examples_prefix = None
            for file_path in file_list:
                if "auto_examples/" in file_path:
                    # Get the prefix before auto_examples
                    parts = file_path.split("auto_examples/")
                    if len(parts) >= 2:
                        auto_examples_prefix = parts[0] + "auto_examples/"
                        break

            if auto_examples_prefix:
                log_message(f"Found auto_examples with prefix: {auto_examples_prefix}")

                # Extract only auto_examples files
                for file_path in file_list:
                    if file_path.startswith(auto_examples_prefix):
                        # Calculate relative path within auto_examples
                        relative_path = file_path[len(auto_examples_prefix) :]
                        if relative_path:  # Skip the auto_examples/ directory itself
                            target_file_path = target_dir / "auto_examples" / relative_path
                            target_file_path.parent.mkdir(parents=True, exist_ok=True)

                            # Extract the file
                            with zip_ref.open(file_path) as source:
                                with open(target_file_path, "wb") as target:
                                    shutil.copyfileobj(source, target)

                log_message(f"Successfully extracted auto_examples to {target_dir / 'auto_examples'}")
            else:
                log_message("Could not find auto_examples directory in archive")
                return False

        # Clean up temporary file
        os.unlink(tmp_path)
        return True

    except Exception as e:
        log_message(f"Error downloading/extracting examples: {e}")
        return False


def generate_placeholder_examples(target_dir: Path) -> None:
    """Generate placeholder auto_examples when download fails.

    Args:
        target_dir: Target directory for the docs
    """
    log_message("Generating placeholder auto_examples...")

    auto_examples_dir = target_dir / "auto_examples"
    auto_examples_dir.mkdir(exist_ok=True)

    # Create placeholder directories based on the examples structure
    example_categories = [
        "channels",
        "constraints",
        "modulation",
        "metrics",
        "data",
        "losses",
        "models",
        "models_fec",
        "benchmarks",
        "utils",
    ]

    for category in example_categories:
        category_dir = auto_examples_dir / category
        category_dir.mkdir(exist_ok=True)

        # Create a placeholder index file
        index_file = category_dir / "index.rst"
        with open(index_file, "w") as f:
            f.write(
                f"""
{category.title()} Examples
{'=' * (len(category) + 9)}

.. note::
   Auto-generated examples are not available in this build.
   Please check the main documentation or repository for full examples.

"""
            )

    log_message("Placeholder auto_examples created")


def get_default_config() -> dict[str, bool | int | str]:
    """Get default configuration for auto_examples download."""
    return {
        "github_owner": "ipc-lab",
        "github_repo": "kaira",
        "use_github_releases": True,
        "use_github_artifacts": True,
        "use_local_examples": True,
        "create_placeholders": False,
        "min_files_threshold": 20,
        "skip_if_exists": True,
    }


def main():
    """Main function to download auto_examples."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs"

    log_message("Starting auto_examples download process...")
    log_message(f"Project root: {project_root}")
    log_message(f"Docs directory: {docs_dir}")

    # Load configuration
    config = get_default_config()

    # Check if auto_examples already exists and is not empty
    auto_examples_dir = docs_dir / "auto_examples"
    if config["skip_if_exists"] and auto_examples_dir.exists() and any(auto_examples_dir.iterdir()):
        log_message("auto_examples directory already exists and is not empty")
        # Check if it has substantial content
        total_files = sum(1 for _ in auto_examples_dir.rglob("*") if _.is_file())
        if total_files > config["min_files_threshold"]:
            log_message(f"Found {total_files} files in auto_examples, skipping download")
            return
        else:
            log_message(f"Found only {total_files} files in auto_examples, will attempt download")

    repo_owner = config["github_owner"]
    repo_name = config["github_repo"]

    # Try to download from GitHub releases first
    if config["use_github_releases"]:
        download_url = check_github_release_for_examples(repo_owner, repo_name)
        if download_url:
            success = download_and_extract_examples(download_url, docs_dir)
            if success:
                log_message("Successfully downloaded auto_examples from GitHub release")
                return

    # Try to download from GitHub Actions artifacts (if token is available)
    if config["use_github_artifacts"]:
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if github_token:
            artifact_url = check_github_artifacts_for_examples(repo_owner, repo_name, github_token)
            if artifact_url:
                success = download_and_extract_examples(artifact_url, docs_dir)
                if success:
                    log_message("Successfully downloaded auto_examples from GitHub artifact")
                    return

    # Check if we're in a local development environment with examples
    if config["use_local_examples"]:
        examples_dir = project_root / "examples"
        if examples_dir.exists() and any(examples_dir.iterdir()):
            log_message("Local examples directory found, will rely on sphinx-gallery to generate auto_examples")
            return

    # Final fallback: Generate placeholder examples only if configured
    if config["create_placeholders"]:
        log_message("Could not download auto_examples, generating placeholders")
        generate_placeholder_examples(docs_dir)
    else:
        log_message("Could not download auto_examples and placeholder generation is disabled")
        log_message("Sphinx-gallery will generate examples during build if local examples exist")


if __name__ == "__main__":
    main()
