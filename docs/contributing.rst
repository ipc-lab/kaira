.. _contributing:

Contributing
============

We welcome your contributions to Kaira! Your involvement is essential to our project's growth and improvement. Contributions can include code, documentation, issue triage, tutorials, and community support.

We adhere to the `Contributor Covenant Code of Conduct <../CODE_OF_CONDUCT.md>`. All contributors are required to uphold its principles of openness, respect, and collaboration.

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

1. Set up your development environment following the :ref:`development_setup` guide
2. Understand the :ref:`contributing` guidelines
3. Familiarize yourself with the :ref:`code_of_conduct`
4. Learn about our :ref:`test_coverage` requirements

Development Workflow
--------------------

The typical development workflow includes:

1. Setting up your environment
2. Creating a feature branch
3. Making changes and adding tests
4. Submitting a pull request

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

For detailed instructions on setting up your development environment and running tests, please see the :doc:`Development Instructions <development>`.
