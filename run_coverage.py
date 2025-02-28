#!/usr/bin/env python3
"""Run pytest with coverage and generate reports."""
import subprocess  # nosec B404 - Usage is deliberate and controlled
import sys


def run_coverage():
    """Run coverage analysis on the codebase."""
    # Define the coverage command - using fixed commands with no user input
    coverage_commands = [
        ["coverage", "run", "-m", "pytest"],
        ["coverage", "report", "-m"],
        ["coverage", "html"],
    ]

    # Execute each command with proper security measures
    for cmd in coverage_commands:
        try:
            # Using a list of arguments prevents shell injection
            # Adding check=True to raise exception if command fails
            subprocess.run(cmd, check=True)  # nosec B603 - Using fixed command list
        except subprocess.SubprocessError as e:
            print(f"Error running coverage command {cmd}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_coverage())
