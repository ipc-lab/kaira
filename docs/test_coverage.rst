.. _test_coverage:

Test Coverage
=============

This document explains how to measure and interpret test coverage in the Kaira project.

Running Coverage Analysis
-------------------------

To run tests with coverage analysis:

.. code-block:: bash

    # Install required development dependencies
    pip install -r requirements-dev.txt

    # Run the coverage script
    python scripts/run_coverage.py

Or manually with pytest:

.. code-block:: bash

    pytest --cov=kaira --cov-report=html tests/

Interpreting Coverage Reports
-----------------------------

After running the tests with coverage, you'll find:

1. A console summary showing coverage percentages
2. An HTML report in the ``coverage_html/`` directory
3. An XML report in ``coverage.xml`` (useful for CI integration)

Open the HTML report in your browser to see detailed line-by-line coverage:

.. code-block:: bash

    # On Linux
    xdg-open coverage_html/index.html

    # On macOS
    open coverage_html/index.html

    # On Windows
    start coverage_html/index.html

Coverage Goals
--------------

- Aim for at least 80% overall code coverage
- Focus on covering critical paths and edge cases
- Use ``# pragma: no cover`` sparingly for code that genuinely cannot be tested
