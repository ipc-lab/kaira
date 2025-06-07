#!/usr/bin/env python3
"""Download auto_examples folder from a remote source (GitHub releases) before documentation
builds.

This script downloads the auto_examples folder from the latest GitHub release to enable incremental
builds. It's designed to work both locally and in CI environments.

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


def check_github_releases_for_examples(repo_owner: str, repo_name: str, token: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Check GitHub Actions artifacts for auto_examples.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        token: GitHub token for API access (required for artifacts)

    Returns:
        Tuple of (artifact_download_url, token) for the auto_examples artifact, or None if not found

    Note:
        GitHub artifacts ALWAYS require authentication. If no token is provided,
        this function will return None and log a helpful message.
    """
    if not token:
        log_message("Skipping GitHub artifacts: No token provided (artifacts require authentication)")
        return None

    try:
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/artifacts"

        # Use proper GitHub API headers as per 2022-11-28 API version
        if requests:
            headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}

            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            artifacts_data = response.json()
        else:
            # Fallback to urllib for compatibility
            req = urllib.request.Request(api_url)  # nosec B310 - Using HTTPS URL from trusted GitHub API
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("X-GitHub-Api-Version", "2022-11-28")

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
                # Return the artifact ID and token for proper GitHub API download
                artifact_id = artifact["id"]
                download_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/artifacts/{artifact_id}/zip"
                return (download_url, token)

    except Exception as e:
        if requests and hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 404:
                log_message("No GitHub artifacts found or repository not accessible")
            elif e.response.status_code == 403:
                log_message("GitHub artifacts require authentication with 'actions:read' scope")
                log_message("Current GITHUB_TOKEN lacks required permissions - check token scopes")
            else:
                log_message(f"Could not check GitHub artifacts: HTTP {e.response.status_code}")
        elif hasattr(e, "code"):  # urllib error
            if e.code == 404:
                log_message("No GitHub artifacts found or repository not accessible")
            elif e.code == 403:
                log_message("GitHub artifacts require authentication with 'actions:read' scope")
                log_message("Current GITHUB_TOKEN lacks required permissions - check token scopes")
            else:
                log_message(f"Could not check GitHub artifacts: HTTP {e.code} - {e.reason}")  # type: ignore[attr-defined]
        else:
            log_message(f"Could not check GitHub artifacts: {e}")

    return None


def check_github_release_for_examples(repo_owner: str, repo_name: str) -> Optional[str]:
    """Check if there's a recent GitHub release with auto_examples.

    First tries to find commit-specific auto-examples releases for the current commit tree,
    then falls back to the latest release if no commit-specific release is found.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name

    Returns:
        Download URL for the auto_examples archive, or None if not found
    """
    # First try to find commit-specific auto-examples releases
    commit_specific_url = check_commit_specific_releases(repo_owner, repo_name)
    if commit_specific_url:
        return commit_specific_url

    # Fallback to latest release
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
                log_message(f"Found auto_examples asset in latest release: {asset['name']}")
                download_url: str = asset["browser_download_url"]
                return download_url

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


def check_commit_specific_releases(repo_owner: str, repo_name: str) -> Optional[str]:
    """Check for commit-specific auto-examples releases in the current commit tree.

    Searches for releases with tags like 'auto-examples-{commit_sha_prefix}' and finds
    the most recent one that corresponds to a commit in the current branch's history.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name

    Returns:
        Download URL for the auto_examples archive, or None if not found
    """
    try:
        # Get current commit SHA
        current_commit = get_current_commit_sha()
        if not current_commit:
            log_message("Could not determine current commit SHA")
            return None

        # Get list of commits in current branch (last 50 to avoid too many API calls)
        commit_history = get_commit_history(repo_owner, repo_name, current_commit)
        if not commit_history:
            log_message("Could not get commit history")
            return None

        log_message(f"Searching for auto-examples releases in commit tree (current: {current_commit[:8]})")

        # Get all releases
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"

        if requests:
            response = requests.get(api_url, timeout=30, params={"per_page": 100})
            response.raise_for_status()
            releases = response.json()
        else:
            # Fallback to urllib for compatibility
            with urllib.request.urlopen(f"{api_url}?per_page=100") as urllib_response:  # nosec B310 - Using HTTPS URL from trusted GitHub API
                releases = json.loads(urllib_response.read().decode())

        # Look for auto-examples releases and check if their commit is in our history
        for release in releases:
            tag_name = release.get("tag_name", "")
            if not tag_name.startswith("auto-examples-"):
                continue

            # Extract commit SHA prefix from tag name
            commit_prefix = tag_name.replace("auto-examples-", "")

            # Check if this commit is in our history
            matching_commit = None
            for commit_sha in commit_history:
                if commit_sha.startswith(commit_prefix):
                    matching_commit = commit_sha
                    break

            if matching_commit:
                # Look for auto_examples.zip asset in this release
                for asset in release.get("assets", []):
                    if "auto_examples" in asset["name"].lower() and asset["name"].endswith(".zip"):
                        log_message(f"Found commit-specific auto_examples: {tag_name} (commit: {matching_commit[:8]})")
                        download_url: str = asset["browser_download_url"]
                        return download_url

        log_message("No commit-specific auto-examples releases found in current commit tree")
        return None

    except Exception as e:
        if requests and hasattr(e, "response") and e.response is not None:
            log_message(f"Error checking commit-specific releases: HTTP {e.response.status_code}")
        else:
            log_message(f"Error checking commit-specific releases: {e}")
        return None


def get_current_commit_sha() -> Optional[str]:
    """Get the current commit SHA from environment or git."""
    # First try GitHub Actions environment
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha

    # Try git command
    try:
        import subprocess  # nosec B404 - Using subprocess for trusted git commands only

        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)  # nosec B603,B607 - trusted git command

        if result.returncode == 0:
            return result.stdout.strip()

    except (ImportError, FileNotFoundError):
        pass

    return None


def get_commit_history(repo_owner: str, repo_name: str, current_commit: str, limit: int = 50) -> list[str]:
    """Get commit history for the current branch.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        current_commit: Current commit SHA
        limit: Maximum number of commits to retrieve

    Returns:
        List of commit SHAs in reverse chronological order
    """
    try:
        # First try GitHub API to get commit history
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
        params: dict[str, str | int] = {"sha": current_commit, "per_page": limit}

        if requests:
            response = requests.get(api_url, timeout=30, params=params)
            response.raise_for_status()
            commits = response.json()
        else:
            # Fallback to urllib for compatibility
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            with urllib.request.urlopen(f"{api_url}?{query_string}") as urllib_response:  # nosec B310 - Using HTTPS URL from trusted GitHub API
                commits = json.loads(urllib_response.read().decode())

        return [commit["sha"] for commit in commits]

    except Exception as e:
        log_message(f"Could not get commit history from GitHub API: {e}")

        # Fallback to git command if available
        try:
            import subprocess  # nosec B404 - Using subprocess for trusted git commands only

            result = subprocess.run(["git", "log", "--format=%H", f"-{limit}", current_commit], capture_output=True, text=True, check=False)  # nosec B603,B607 - trusted git command

            if result.returncode == 0:
                return result.stdout.strip().split("\n")

        except (ImportError, FileNotFoundError):
            pass

        return []


def extract_auto_examples(zip_ref, file_list, target_dir):
    """Extract auto_examples from archive.

    Args:
        zip_ref: ZipFile object for the archive
        file_list: List of file paths in the archive
        target_dir: Target directory for extraction

    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    # Remove existing auto_examples directory completely
    auto_examples_target = target_dir / "auto_examples"
    if auto_examples_target.exists():
        log_message("Removing existing auto_examples directory")
        try:
            shutil.rmtree(auto_examples_target)
        except OSError as e:
            log_message(f"Warning: Could not fully remove existing directory: {e}")
            return False

    # Extract auto_examples files, but skip problematic files
    auto_examples_extracted = False
    successful_extractions = 0
    failed_extractions = 0
    skipped_files = 0

    # Files to skip - these are build artifacts that shouldn't be transferred
    skip_patterns = [
        "searchindex",  # sphinx-gallery search database (environment-specific)
        ".doctrees",  # sphinx doctree cache
        ".buildinfo",  # sphinx build info
        "__pycache__",  # python cache
        ".pyc",  # compiled python files
        ".tmp",  # temporary files
    ]

    for file_path in file_list:
        if file_path.startswith("auto_examples/"):
            # Check if file should be skipped
            should_skip = any(pattern in file_path for pattern in skip_patterns)

            if should_skip:
                skipped_files += 1
                continue

            try:
                zip_ref.extract(file_path, target_dir)
                auto_examples_extracted = True
                successful_extractions += 1
            except Exception as e:
                log_message(f"Warning: Could not extract file {file_path}: {e}")
                failed_extractions += 1
                continue

    log_message(f"Extraction summary: {successful_extractions} successful, {failed_extractions} failed, {skipped_files} skipped")

    if auto_examples_extracted:
        log_message(f"Successfully extracted auto_examples to {target_dir / 'auto_examples'}")
        return True
    else:
        log_message("ERROR: No auto_examples files were successfully extracted")
        return False


def download_and_extract_examples(download_url: str, target_dir: Path, github_token: Optional[str] = None) -> bool:
    """Download and extract auto_examples from a URL.

    Args:
        download_url: URL to download the examples archive
        target_dir: Target directory where auto_examples should be extracted
        github_token: GitHub token for authenticated downloads (required for artifacts)

    Returns:
        True if successful, False otherwise
    """
    try:
        log_message(f"Downloading auto_examples from: {download_url}")

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        # Download the file
        if requests:
            # Check if this is a GitHub API URL requiring authentication
            headers = {}
            if github_token and "api.github.com" in download_url:
                headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {github_token}", "X-GitHub-Api-Version": "2022-11-28"}

            response = requests.get(download_url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            # Fallback to urllib for compatibility
            if github_token and "api.github.com" in download_url:
                req = urllib.request.Request(download_url)  # nosec B310 - URL is validated from trusted GitHub API
                req.add_header("Accept", "application/vnd.github+json")
                req.add_header("Authorization", f"Bearer {github_token}")
                req.add_header("X-GitHub-Api-Version", "2022-11-28")
                with urllib.request.urlopen(req) as urllib_response:  # nosec B310 - URL is validated from trusted GitHub API
                    with open(tmp_path, "wb") as f:
                        # Read in chunks to avoid type issues
                        while True:
                            chunk = urllib_response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
            else:
                urllib.request.urlretrieve(download_url, tmp_path)  # nosec B310 - URL is validated from trusted GitHub API
        log_message(f"Downloaded to temporary file: {tmp_path}")

        # Extract the archive
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            # List contents to understand structure
            file_list = zip_ref.namelist()
            log_message(f"Archive contains {len(file_list)} files")

            # Debug: Log first few files to understand structure
            for i, file_path in enumerate(file_list[:10]):  # Show first 10 files
                log_message(f"File {i+1}: {file_path}")
            if len(file_list) > 10:
                log_message(f"... and {len(file_list) - 10} more files")

            # Check if this is a GitHub artifact containing auto_examples.zip
            if len(file_list) == 1 and file_list[0] == "auto_examples.zip":
                log_message("Found nested auto_examples.zip - extracting inner archive")

                # Extract the nested zip file to a temporary location
                with zip_ref.open("auto_examples.zip") as nested_zip_data:
                    nested_tmp_path = tmp_path.with_name(tmp_path.name + "_nested.zip")
                    with open(nested_tmp_path, "wb") as f:
                        shutil.copyfileobj(nested_zip_data, f)

                # Now extract from the nested zip
                with zipfile.ZipFile(nested_tmp_path, "r") as nested_zip_ref:
                    nested_file_list = nested_zip_ref.namelist()
                    log_message(f"Nested archive contains {len(nested_file_list)} files")

                    # Use simple extraction
                    success = extract_auto_examples(nested_zip_ref, nested_file_list, target_dir)
                    if not success:
                        log_message("ERROR: Failed to extract auto_examples from nested archive")
                        return False

                # Clean up temporary nested file
                nested_tmp_path.unlink()

            else:
                # Direct archive structure - use simple extraction
                log_message("Processing direct archive structure")
                success = extract_auto_examples(zip_ref, file_list, target_dir)
                if not success:
                    log_message("ERROR: Failed to extract auto_examples from direct archive")
                    return False

        # Clean up temporary file
        tmp_path.unlink()
        return True

    except Exception as e:
        # Clean up temporary files if they exist
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
            if "nested_tmp_path" in locals() and nested_tmp_path.exists():
                nested_tmp_path.unlink()
        except Exception as cleanup_error:
            log_message(f"Warning: Failed to clean up temporary files: {cleanup_error}")
            # Continue execution despite cleanup failure

        # Handle specific authentication errors for GitHub artifacts
        if requests and hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 403:
                try:
                    error_data = e.response.json()
                    if "actions scope" in error_data.get("message", "").lower():
                        log_message(error_data.get("message", ""))
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

    # Clean up any existing auto_examples directory to avoid conflicts
    if auto_examples_dir.exists():
        try:
            import shutil

            log_message(f"Removing existing auto_examples directory: {auto_examples_dir}")
            shutil.rmtree(auto_examples_dir)
        except OSError as e:
            log_message(f"Warning: Could not remove existing auto_examples directory: {e}")
            # Continue anyway, we'll handle individual directory conflicts below

    # Use try-except for more robust directory creation
    try:
        auto_examples_dir.mkdir(exist_ok=True)
    except (FileExistsError, OSError) as e:
        if auto_examples_dir.exists() and auto_examples_dir.is_dir():
            log_message(f"auto_examples directory already exists: {e}")
        else:
            log_message(f"Warning: Could not create auto_examples directory: {e}")
            return

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

        # More robust directory creation with better error handling
        try:
            category_dir.mkdir(exist_ok=True)
        except (FileExistsError, OSError) as e:
            if category_dir.exists() and category_dir.is_dir():
                log_message(f"Category directory {category} already exists: {e}")
            else:
                log_message(f"Warning: Could not create category directory {category}: {e}")
                continue

        # Create images/thumb directory for thumbnails
        images_dir = category_dir / "images" / "thumb"
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError) as e:
            if images_dir.exists() and images_dir.is_dir():
                log_message(f"Images directory for {category} already exists: {e}")
            else:
                log_message(f"Warning: Could not create images directory for {category}: {e}")

        # Create a placeholder index file with proper sphinx-gallery structure
        index_file = category_dir / "index.rst"
        title = category.replace("_", " ").title()

        try:
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
        except OSError as e:
            log_message(f"Warning: Could not create index file for {category}: {e}")
            continue

    # Create a main index file that matches the expected location
    main_index = auto_examples_dir / "index.rst"
    try:
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

    except OSError as e:
        log_message(f"Warning: Could not create main index file: {e}")
        return

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
        - is_github_actions: Whether running in GitHub Actions
    """
    context: dict[str, str | bool] = {"is_release": False, "is_push": False, "ref_name": "unknown", "strategy": "artifacts_first", "is_github_actions": False}  # default strategy

    # Check if we're running in GitHub Actions
    is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    context["is_github_actions"] = is_github_actions

    if is_github_actions:
        log_message("Running in GitHub Actions environment")

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
        # For ReadTheDocs, enable placeholders by default to ensure builds don't fail
        config["create_placeholders"] = True
        config["use_local_examples"] = True
        log_message("RTD: Enabled fallback strategies (placeholders + local examples)")
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

    # Get GitHub token (if available)
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if github_token:
        token_source = "automatic" if repo_context["is_github_actions"] else "configured"
        log_message(f"GitHub token available ({token_source}) - can access artifacts and releases")
        if is_rtd:
            log_message("RTD: GITHUB_TOKEN configured - optimal performance with artifact access")
    else:
        if is_rtd:
            log_message("RTD: No GitHub token configured - will only try public releases")
            log_message("      Artifacts require authentication but releases are public")
            log_message("      To access artifacts, add GITHUB_TOKEN in RTD project settings")
        elif repo_context["is_github_actions"]:
            log_message("Warning: No GITHUB_TOKEN available in GitHub Actions (this is unexpected)")
        else:
            log_message("Local environment: No GitHub token - will only try public releases")

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
            if github_token:
                # Try GitHub releases first (more accessible across workflows)
                log_message(f"Trying GitHub releases (primary) - using {token_source} token")
                release_url = check_github_release_for_examples(repo_owner, repo_name)
                if release_url:
                    success = download_and_extract_examples(release_url, docs_dir)
                    if success:
                        log_message("Successfully downloaded auto_examples from GitHub release")
                        download_succeeded = True

                # Fallback: Try GitHub artifacts (requires higher permissions)
                if not download_succeeded:
                    log_message(f"Trying GitHub Actions artifacts (fallback) - using {token_source} token")
                    artifact_result = check_github_releases_for_examples(repo_owner, repo_name, github_token)
                    if artifact_result:
                        artifact_url, token = artifact_result
                        success = download_and_extract_examples(artifact_url, docs_dir, token)
                        if success:
                            log_message("Successfully downloaded auto_examples from GitHub artifact")
                            download_succeeded = True

    else:  # artifacts_first strategy
        log_message("Using artifacts-first strategy (detected push/commit)")

        # Try GitHub Actions artifacts first for push/commit context
        if config["use_github_artifacts"]:
            if github_token:
                # Try GitHub releases first (more accessible across workflows)
                log_message(f"Trying GitHub releases (primary for push/commit) - using {token_source} token")
                release_url = check_github_release_for_examples(repo_owner, repo_name)
                if release_url:
                    success = download_and_extract_examples(release_url, docs_dir)
                    if success:
                        log_message("Successfully downloaded auto_examples from GitHub release")
                        download_succeeded = True

                # Fallback: Try GitHub artifacts (requires higher permissions)
                if not download_succeeded:
                    log_message(f"Trying GitHub Actions artifacts (fallback) - using {token_source} token")
                    artifact_result = check_github_releases_for_examples(repo_owner, repo_name, github_token)
                    if artifact_result:
                        artifact_url, token = artifact_result
                        success = download_and_extract_examples(artifact_url, docs_dir, token)
                        if success:
                            log_message("Successfully downloaded auto_examples from GitHub artifact")
                            download_succeeded = True
                else:
                    # Check if this is RTD with a manually configured token (not auto from GitHub Actions)
                    is_rtd_configured_token = is_rtd and not repo_context["is_github_actions"]  # nosec B105
                    if is_rtd_configured_token:
                        log_message("RTD: Artifact access failed - check if GITHUB_TOKEN has 'actions:read' scope")
                        log_message("      Tokens need: actions:read, contents:read permissions")

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
                github_token_available = bool(github_token)
                is_github_actions = repo_context["is_github_actions"]

                error_msg += f"""

Attempted sources (strategy: {repo_context['strategy']}):
1. GitHub releases {"(failed or not found)" if config["use_github_releases"] else "(disabled)"} - PUBLIC, no token needed
2. GitHub artifacts {"(no token - authentication required)" if not github_token_available else "(failed - check token permissions)" if config["use_github_artifacts"] else "(disabled)"} - PRIVATE, token required
3. Local examples {"(not found)" if config["use_local_examples"] else "(disabled)"} - LOCAL, no token needed

This will cause the documentation build to fail or be extremely slow.

Solutions:"""

                if is_github_actions:
                    error_msg += """
- In GitHub Actions: GITHUB_TOKEN should be automatically available
- Check workflow permissions in .github/workflows/*.yml:
  ```yaml
  permissions:
    actions: read  # Required for downloading artifacts
    contents: read
  ```
- Verify artifacts exist and haven't expired (30-day retention)
- Note: Cross-workflow artifact access may require explicit permissions"""
                elif is_rtd:
                    error_msg += """
For RTD (Read the Docs):
1. OPTIONAL: Set GITHUB_TOKEN environment variable for artifact access
   - Go to RTD Project Settings → Environment Variables → Add GITHUB_TOKEN
   - Create token at: GitHub Settings → Developer settings → Personal access tokens
   - Required scope: 'actions:read' (for downloading workflow artifacts)

2. RECOMMENDED: Create GitHub releases with auto_examples.zip assets
   - Releases are PUBLIC and don't require authentication
   - More reliable for external services like RTD
   - Automated via GitHub Actions on tag creation

3. FALLBACK: Build will use local examples (slower but works)"""
                else:
                    error_msg += """
For external CI/services:
1. Set GITHUB_TOKEN environment variable with 'actions:read' scope
2. Alternative: Use GitHub releases instead of artifacts (no token needed)
3. Ensure workflow artifacts exist and haven't expired (30-day retention)"""
                log_message(error_msg)
                # For ReadTheDocs, don't exit with error - let it fall back gracefully
                if is_rtd:
                    log_message("RTD: Continuing with build despite download failures - will use local examples or placeholders")
                    log_message("This is expected if no GITHUB_TOKEN is configured in RTD project settings")
                else:
                    exit(1)
            else:
                log_message(error_msg)
                log_message("Sphinx-gallery will generate examples during build if local examples exist")


if __name__ == "__main__":
    main()
