.. _contributing:

Contributing
============

We welcome your contributions to Kaira! Your involvement is essential to our project's growth and improvement. Contributions can include code, documentation, issue triage, tutorials, and community support.

We adhere to the :ref:`code_of_conduct`. All contributors are required to uphold its principles of openness, respect, and collaboration.

If you encounter issues or have suggestions, please:

- Submit a ticket via the `GitHub Issue Tracker <https://github.com/ipc-lab/kaira/issues>`_
- Open a pull request for feature enhancements or bug fixes
- Email us at `yilmaselimfirat (at) gmail.com` for further inquiries

Types of Contributions
----------------------

You can contribute in many ways:

- **Code:** Fix bugs, add new features, or improve existing code.
- **Documentation:** Improve the documentation, add examples, or write tutorials.
- **Issue Triage:** Help manage issues by verifying them, suggesting labels, or closing resolved issues.
- **Community Support:** Answer questions, provide feedback, and help other users.

Getting Started with Development
--------------------------------

To start contributing to Kaira:

1. Set up your development environment following the :ref:`development-workflow` guide
2. Familiarize yourself with the :ref:`code_of_conduct`
3. Review the :ref:`makefile` for common development tasks
4. Build the documentation using the :ref:`build_documentation` guide
5. Submit a pull request with your changes
6. Engage with the community and help others
7. Share your experiences and feedback

Development Workflow
------------------------

.. _development-workflow:

1. **Setting up your environment:**

   - Clone the repository: `git clone https://github.com/ipc-lab/kaira.git`
   - Navigate to the project directory: `cd kaira`
   - Install development requirements: `pip install -r requirements-dev.txt`
   - Set up pre-commit hooks: `pre-commit install`

2. **Creating a feature branch:**

   - Create a new branch for your feature: `git checkout -b feature-branch-name`

3. **Making changes and adding tests:**

   - Implement your changes in the codebase
   - Add or update tests to cover your changes
   - Build the documentation: `bash build_docs.sh`
   - Ensure your code is PEP8 compliant: `bash lint.sh`
   - Run tests to ensure everything works: `pytest --cov=kaira --cov-config=.coveragerc`
   - Run pre-commit checks: `pre-commit run -a`

4. **Submitting a pull request:**

   - Commit your changes: `git commit -am "Description of changes"`
   - Push your branch to GitHub: `git push origin feature-branch-name`
   - Open a pull request on GitHub

Pull Request Checklist
----------------------
- **Alignment:** Verify that your contribution aligns with the project's goals.
- **Code Style:** Ensure your code is PEP8 compliant (check with `bash lint.sh`).
- **Testing:** Confirm that your changes pass all existing tests, including CI (e.g., Travis CI).
- **Review Existing Work:** Review active `pull requests <https://github.com/ipc-lab/kaira/pulls>`_ and `issues <https://github.com/ipc-lab/kaira/issues>`_ to avoid duplication.
- **For New Features:**
  - Have you written tests with at least 95% coverage?
  - Have you provided examples demonstrating the new feature's usage?
- **Documentation:** Update the documentation to reflect your changes, including any new features or modifications.
- **Pre-Commit Hooks:** Ensure pre-commit hooks are set up and run successfully (`pre-commit run --all-files`).

Next Steps
----------

* :ref:`build_documentation`
* :ref:`code_of_conduct`
* :ref:`makefile`
