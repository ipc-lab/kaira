Development Instructions
========================

To set up the development environment, run:

.. code-block:: bash

    pip install -r requirements-dev.txt

After writing your code, perform the following checks:

.. code-block:: bash

    bash build_docs.sh    # Build the documentation
    bash lint.sh          # Verify PEP8 compliance
    pytest --cov=kaira --cov-config=.coveragerc   # Run tests with coverage
    pre-commit run -a     # Run pre-commit checks
