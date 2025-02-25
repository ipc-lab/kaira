#!/usr/bin/env python3
"""
Run pytest with coverage and generate reports.
"""
import os
import sys
import subprocess

def main():
    """Run tests with coverage."""
    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        "--cov=kaira",
        "--cov-report=term",
        "--cov-report=html:coverage_html",
        "--cov-report=xml:coverage.xml",
        "tests/"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
