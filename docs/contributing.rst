.. _contributing:

Contributing
============

We appreciate your contributions to Kaira! Contributions come in many forms—including code, documentation, issue triage, tutorials, or community support.

We adhere to the principles of openness, respect, and collaboration as outlined in the
[Python Software Foundation Code of Conduct](https://www.python.org/psf/codeofconduct/).

If you encounter issues using Kaira, please submit a ticket via the
`GitHub issue tracker <https://github.com/ipc-lab/kaira/issues>`_. Additionally, you can
post feature requests or pull requests. For further inquiries, feel free to open an issue on GitHub or email at `yilmaselimfirat (at) gmail.com`.

Pull Request Checklist
----------------------
- **Alignment:** Does your contribution align with the project's goals?
- **Code Style:** Is your code PEP8 compliant? (You can check using `bash lint.sh`.)
- **Testing:** Do your changes pass the tests, including CI (e.g., Travis CI)?
- **Review Existing Work:** Have you reviewed active `pull requests <https://github.com/ipc-lab/kaira/pulls>`_ and `issues <https://github.com/ipc-lab/kaira/issues>`_ to avoid overlap?
- **For new features:**
  - Have you implemented tests ensuring more than 95% test coverage?
  - Have you added examples that demonstrate the new feature’s usage?

Development Instructions
------------------------
To set up the development environment, run:

.. code-block:: bash

    pip install -r requirements-dev.txt

After writing your code, perform the following checks:

.. code-block:: bash

    bash build_docs.sh    # Build the documentation
    bash lint.sh          # Verify PEP8 compliance
    pytest --cov=kaira --cov-config=.coveragerc   # Run tests with coverage

Running Documentation
---------------------
Follow these steps to build and view the project documentation using Sphinx:

1. **Open a Terminal:**  
   Launch the integrated terminal in Visual Studio Code.

2. **Navigate to the Docs Directory:**  
   Change to the documentation folder:

   .. code-block:: bash

       cd docs

3. **Build the Documentation:**  
   - On Linux/macOS:

     .. code-block:: bash

         make html

   - On Windows:

     .. code-block:: bat

         make.bat html

After building, open the generated HTML files in your browser to review the documentation.