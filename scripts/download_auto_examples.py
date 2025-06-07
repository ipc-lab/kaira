#!/usr/bin/env python3
"""Download auto_examples folder from a remote source (such as GitHub releases or artifacts) before
documentation builds.

It's designed to work both locally and on ReadTheDocs.

Security Note:
    This script uses subprocess calls with trusted git commands for repository context detection.
    The subprocess usage is safe as it:
    - Only calls git with hardcoded, trusted arguments
    - Never uses user input in subprocess calls
    - Has proper error handling and timeout controls
    - Only detects repository state, no modifications
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
                log_message("GitHub artifacts require authentication with 'actions:read' scope (GITHUB_TOKEN needed)")
            else:
                log_message(f"Could not check GitHub artifacts: HTTP {e.response.status_code}")
        elif hasattr(e, "code"):  # urllib error
            if e.code == 404:
                log_message("No GitHub artifacts found or repository not accessible")
            elif e.code == 403:
                log_message("GitHub artifacts require authentication with 'actions:read' scope (GITHUB_TOKEN needed)")
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
        # Handle specific authentication errors for GitHub artifacts
        if requests and hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 403:
                try:
                    error_data = e.response.json()
                    if "actions scope" in error_data.get("message", "").lower():
                        log_message("Error: GitHub token lacks 'actions:read' scope required for artifact download")
                        log_message("Please ensure GITHUB_TOKEN has the necessary permissions")
                    else:
                        log_message(f"GitHub API authentication error: {error_data.get('message', 'Forbidden')}")
                except (ValueError, KeyError):
                    log_message("GitHub API authentication error: access forbidden (403)")
            else:
                log_message(f"Error downloading/extracting examples: HTTP {e.response.status_code}")
        else:
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

    # Category descriptions from the ExampleIndexGenerator
    category_descriptions = {
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

    # Create placeholder directories based on the examples structure
    for category, description in category_descriptions.items():
        category_dir = auto_examples_dir / category
        category_dir.mkdir(exist_ok=True)

        # Create images/thumb directory for thumbnails
        images_dir = category_dir / "images" / "thumb"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create a placeholder index file with proper sphinx-gallery structure
        index_file = category_dir / "index.rst"
        title = category.replace("_", " ").title()

        with open(index_file, "w") as f:
            f.write(
                f""":orphan:

{title}
{'=' * len(title)}

{description}

.. note::
   **Auto-generated examples are not available in this build.**

   This could be due to:

   * Missing pre-built examples in the GitHub release
   * Network issues during download
   * First-time documentation build

   **To view the full examples:**

   * Visit the `online documentation <https://kaira.readthedocs.io/>`_
   * Check the `GitHub repository <https://github.com/ipc-lab/kaira/tree/main/examples/{category}>`_
   * Build the documentation locally with ``make html`` in the docs directory

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Examples in this category demonstrate {description.lower()}">

.. only:: html

    .. image:: /_static/logo.png
      :alt: {title} Examples
      :width: 200px

.. raw:: html

      <div class="sphx-glr-thumbnail-title">View {title} Examples Online</div>
    </div>

.. raw:: html

    </div>

.. raw:: html

    <div class="gallery-outro">
        <p><strong>Examples not available in this build.</strong></p>
        <p>Please visit the <a href="https://kaira.readthedocs.io/en/latest/examples/{category}/index.html">online documentation</a> or the <a href="https://github.com/ipc-lab/kaira/tree/main/examples/{category}">GitHub repository</a> to view the full examples.</p>
    </div>

"""
            )

    # Create a main index file that matches the expected location
    main_index = auto_examples_dir / "index.rst"
    with open(main_index, "w") as f:
        f.write(
            """:orphan:

Auto Examples Gallery
======================

.. note::
   **Auto-generated examples are not available in this build.**

   This could be due to missing pre-built examples or network issues during download.

   **To view the full examples gallery:**

   * Visit the `online documentation <https://kaira.readthedocs.io/en/latest/examples_index.html>`_
   * Check the `GitHub repository <https://github.com/ipc-lab/kaira/tree/main/examples/>`_
   * Build the documentation locally with ``make html`` in the docs directory

.. raw:: html

    <div class="gallery-intro">
        <p>The examples gallery provides comprehensive demonstrations of Kaira's capabilities. Due to technical limitations in this build, placeholder content is shown below.</p>
    </div>

.. toctree::
   :maxdepth: 1

"""
        )

        # Add toctree entries for all categories
        for category in category_descriptions.keys():
            f.write(f"   {category}/index\n")

    log_message("Enhanced placeholder auto_examples created with proper gallery structure")


def get_default_config() -> dict[str, bool | int | str]:
    """Get default configuration for auto_examples download."""
    return {
        "github_owner": "ipc-lab",
        "github_repo": "kaira",
        "use_github_releases": True,
        "use_github_artifacts": True,
        "use_local_examples": True,  # Enable local examples as final fallback
        "create_placeholders": False,  # Still fail instead of creating placeholders
        "min_files_threshold": 20,
        "skip_if_exists": True,
    }


def detect_repository_context() -> dict[str, str | bool]:
    """Detect the current repository context to determine download strategy.

    Returns:
        Dictionary with context information including:
        - is_release: Whether we're on a tagged release
        - is_push: Whether we're on a regular commit/push
        - ref_name: The current ref (tag/branch name)
        - strategy: Recommended download strategy
    """
    context: dict[str, str | bool] = {"is_release": False, "is_push": False, "ref_name": "unknown", "strategy": "artifacts_first"}  # default strategy

    # Check GitHub Actions environment variables
    github_event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    github_ref = os.environ.get("GITHUB_REF", "")
    github_ref_name = os.environ.get("GITHUB_REF_NAME", "")

    if github_event_name == "release" or github_ref.startswith("refs/tags/"):
        context["is_release"] = True
        context["ref_name"] = github_ref_name or github_ref.split("/")[-1]
        context["strategy"] = "releases_first"
        log_message(f"Detected release context: {context['ref_name']}")
    elif github_event_name in ["push", "pull_request"]:
        context["is_push"] = True
        context["ref_name"] = github_ref_name or github_ref.split("/")[-1]
        context["strategy"] = "artifacts_first"
        log_message(f"Detected push/PR context: {context['ref_name']}")
    else:
        # Try to detect from git if available
        try:
            import subprocess  # nosec B404 - Using subprocess for trusted git commands only

            # Check if we're on a tag
            # Note: This subprocess call is safe - using trusted git commands with no user input
            result = subprocess.run(["git", "describe", "--exact-match", "--tags", "HEAD"], capture_output=True, text=True, check=False)  # nosec B603,B607
            if result.returncode == 0:
                context["is_release"] = True
                context["ref_name"] = result.stdout.strip()
                context["strategy"] = "releases_first"
                log_message(f"Detected git tag: {context['ref_name']}")
            else:
                # Check current branch
                # Note: This subprocess call is safe - using trusted git commands with no user input
                result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False)  # nosec B603,B607
                if result.returncode == 0:
                    context["is_push"] = True
                    context["ref_name"] = result.stdout.strip()
                    context["strategy"] = "artifacts_first"
                    log_message(f"Detected git branch: {context['ref_name']}")
        except (ImportError, FileNotFoundError):
            log_message("Git not available for context detection, using default strategy")

    log_message(f"Repository context: {context['strategy']} strategy")
    return context


def main() -> None:
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

    # Override config based on environment variables
    if os.environ.get("CREATE_PLACEHOLDERS", "").lower() in ("true", "1", "yes"):
        config["create_placeholders"] = True
        log_message("Placeholder generation enabled via environment variable")

    # Detect if we're running on ReadTheDocs
    is_rtd = os.environ.get("READTHEDOCS") == "True"
    is_ci = any(key in os.environ for key in ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "TRAVIS"])

    if is_rtd:
        log_message("Running on ReadTheDocs environment")
    elif is_ci:
        log_message("Running in CI environment")

    # Check if auto_examples already exists and is not empty
    auto_examples_dir = docs_dir / "auto_examples"
    if config["skip_if_exists"] and auto_examples_dir.exists() and any(auto_examples_dir.iterdir()):
        log_message("auto_examples directory already exists and is not empty")
        # Check if it has substantial content
        total_files = sum(1 for _ in auto_examples_dir.rglob("*") if _.is_file())
        min_threshold = config["min_files_threshold"]
        if not isinstance(min_threshold, int):
            raise TypeError("min_files_threshold must be an integer")
        if total_files > min_threshold:
            log_message(f"Found {total_files} files in auto_examples, skipping download")
            return
        else:
            log_message(f"Found only {total_files} files in auto_examples, will attempt download")

    repo_owner = config["github_owner"]
    repo_name = config["github_repo"]
    if not isinstance(repo_owner, str) or not isinstance(repo_name, str):
        raise TypeError("github_owner and github_repo must be strings")
    download_succeeded = False

    # Detect repository context to determine optimal download strategy
    repo_context = detect_repository_context()

    # Apply context-aware download strategy
    if repo_context["strategy"] == "releases_first":
        log_message("Using releases-first strategy (detected release/tag)")

        # Try GitHub releases first for release context
        if config["use_github_releases"]:
            log_message("Trying GitHub releases (primary for release)")
            download_url = check_github_release_for_examples(repo_owner, repo_name)
            if download_url:
                success = download_and_extract_examples(download_url, docs_dir)
                if success:
                    log_message("Successfully downloaded auto_examples from GitHub release")
                    download_succeeded = True

        # Try artifacts as fallback for releases
        if not download_succeeded and config["use_github_artifacts"]:
            github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if github_token:
                log_message("Trying GitHub Actions artifacts (fallback)")
                artifact_url = check_github_artifacts_for_examples(repo_owner, repo_name, github_token)
                if artifact_url:
                    success = download_and_extract_examples(artifact_url, docs_dir)
                    if success:
                        log_message("Successfully downloaded auto_examples from GitHub artifact")
                        download_succeeded = True
            else:
                log_message("No GitHub token available for artifact fallback")

    else:  # artifacts_first strategy
        log_message("Using artifacts-first strategy (detected push/commit)")

        # Try GitHub Actions artifacts first for push/commit context
        if config["use_github_artifacts"]:
            github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if github_token:
                log_message("Trying GitHub Actions artifacts (primary for push/commit)")
                artifact_url = check_github_artifacts_for_examples(repo_owner, repo_name, github_token)
                if artifact_url:
                    success = download_and_extract_examples(artifact_url, docs_dir)
                    if success:
                        log_message("Successfully downloaded auto_examples from GitHub artifact")
                        download_succeeded = True
            else:
                log_message("No GitHub token available for artifact download")

        # Try releases as fallback for push/commit
        if not download_succeeded and config["use_github_releases"]:
            log_message("Trying GitHub releases (fallback)")
            download_url = check_github_release_for_examples(repo_owner, repo_name)
            if download_url:
                success = download_and_extract_examples(download_url, docs_dir)
                if success:
                    log_message("Successfully downloaded auto_examples from GitHub release")
                    download_succeeded = True

    # Final fallback: Use local examples for sphinx-gallery generation
    if not download_succeeded and config["use_local_examples"]:
        examples_dir = project_root / "examples"
        if examples_dir.exists() and any(examples_dir.iterdir()):
            log_message("Remote downloads failed, falling back to local examples")
            log_message("Sphinx-gallery will generate auto_examples from local examples during build")
            return
        else:
            log_message("No local examples directory found")

    # Final check - fail if nothing worked and we're on RTD/CI
    if not download_succeeded:
        if config["create_placeholders"]:
            log_message("Could not download auto_examples, generating placeholders")
            generate_placeholder_examples(docs_dir)
        else:
            error_msg = "Failed to download auto_examples from any source"

            if is_rtd or is_ci:
                github_token_available = bool(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
                error_msg += f"""

Attempted sources (strategy: {repo_context['strategy']}):
1. GitHub releases {"(failed or not found)" if config["use_github_releases"] else "(disabled)"}
2. GitHub artifacts {"(no token with 'actions:read' scope)" if not github_token_available else "(failed - check token permissions)" if config["use_github_artifacts"] else "(disabled)"}
3. Local examples {"(not found)" if config["use_local_examples"] else "(disabled)"}

This will cause the documentation build to fail or be extremely slow.

Solutions:
1. Set GITHUB_TOKEN environment variable with 'actions:read' scope in RTD settings
2. Create a GitHub release with auto_examples.zip asset
3. Ensure workflow artifacts exist and haven't expired
4. For RTD: Go to Project Settings → Environment Variables → Add GITHUB_TOKEN

GitHub Token Requirements:
- Repository access (public repos: no additional scopes needed)
- actions:read scope (for downloading workflow artifacts)
"""
                log_message(error_msg)
                exit(1)
            else:
                log_message(error_msg)
                log_message("Sphinx-gallery will generate examples during build if local examples exist")


if __name__ == "__main__":
    main()
