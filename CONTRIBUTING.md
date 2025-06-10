# Contributing to Kaira

Thank you for considering contributing to Kaira! This document provides guidelines and instructions to help make the contribution process smooth and effective.

## Code of Conduct

We follow the [Python Software Foundation Code of Conduct](https://www.python.org/psf/codeofconduct/). All contributors are expected to adhere to its principles of openness, respect, and collaboration.

## Ways to Contribute

There are many ways to contribute to Kaira:

- **Code Contributions**: Bug fixes, feature implementations, optimizations
- **Documentation**: Improving existing documentation, adding examples, fixing typos
- **Issue Triage**: Helping to categorize, reproduce, or solve reported issues
- **Testing**: Writing test cases or finding edge cases
- **Examples and Tutorials**: Creating examples that demonstrate Kaira's capabilities
- **Community Support**: Answering questions in discussions and helping new users

## Development Workflow

### Setting Up Your Development Environment

1. **Fork the repository**:

   - Visit [https://github.com/ipc-lab/kaira](https://github.com/ipc-lab/kaira) and click the "Fork" button

2. **Clone your fork**:

   ```bash
   git clone https://github.com/YOUR-USERNAME/kaira.git
   cd kaira
   ```

3. **Set up the development environment**:

   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"
   # Or alternatively:
   pip install -e .
   pip install -r requirements-dev.txt
   ```

### Making Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature-name
   ```

2. **Make your changes**:

   - Write code that follows our style guidelines
   - Add or update tests as appropriate
   - Update documentation if necessary

3. **Run tests locally**:

   ```bash
   pytest
   ```

4. **Check code style**:

   ```bash
   bash scripts/lint.sh
   ```

### Submitting a Pull Request

1. **Push your changes**:

   ```bash
   git push origin feature-name
   ```

2. **Open a pull request**:

   - Go to your fork on GitHub and click "New pull request"
   - Follow the PR template provided

3. **Code review process**:

   - Maintainers will review your code
   - Address any feedback or requested changes
   - Once approved, your PR will be merged

## Pull Request Checklist

Before submitting a pull request, please ensure:

- [ ] **Alignment**: Your contribution aligns with the project's goals and roadmap
- [ ] **Code Style**: Your code follows our style guidelines and passes linting checks
- [ ] **Testing**: You've added tests that cover your changes with at least 95% coverage
- [ ] **Documentation**: You've updated relevant documentation and added examples if needed
- [ ] **Commits**: Your commits are logical, atomic, and have clear messages
- [ ] **Conflicts**: Your branch has no conflicts with the main branch
- [ ] **Review**: You've reviewed your own code for obvious issues

## Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding standards
- Use [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
- Use meaningful variable and function names
- Keep functions focused on a single responsibility
- Add comments for complex logic, but prioritize readable code

## Documentation

- Update documentation alongside code changes
- For new features, include:
  - API documentation
  - Usage examples
  - Implementation notes if helpful for users

## Creating Examples

Kaira uses an automated example gallery system that makes it easy to add new examples without manual maintenance of index files.

### Adding New Examples

1. **File Naming**: Use the `plot_` prefix for all examples (e.g., `plot_my_feature.py`)

2. **Location**: Place examples in the appropriate category directory under `examples/`

3. **Docstring Format**: Use the following format at the top of your example:

   ```python
   """
   ==========================================
   Clear and Descriptive Title
   ==========================================

   Comprehensive description explaining what the example demonstrates,
   key concepts covered, and expected outcomes.

   Additional paragraphs can provide more detail about implementation
   specifics, theoretical background, or usage notes.
   """
   ```

4. **Structure**: Follow the template in `examples/template_example.py`

### Automated Index Generation

The system automatically:

- Discovers all `plot_*.py` files in example directories
- Extracts titles and descriptions from docstrings
- Generates `index.rst` files for each category
- Maintains synchronization during documentation builds
- Updates example galleries automatically via pre-commit hooks when example files are modified

**No manual index.rst editing required!** The example gallery indices are automatically updated whenever you commit changes to example files through pre-commit hooks.

### Development Tools

Use the example workflow script for development:

```bash
# Preview examples in a category
python scripts/example_workflow.py preview --category modulation

# Sync all example galleries
python scripts/example_workflow.py sync

# Validate example format
python scripts/example_workflow.py validate --file examples/modulation/plot_new_example.py
```

### Example Categories

Examples are organized into categories:

- `channels` - Channel models for wireless communications
- `modulation` - Digital modulation schemes
- `metrics` - Performance metrics and evaluation tools
- `models` - Neural network models and architectures
- `models_fec` - Forward Error Correction models
- `benchmarks` - Benchmarking tools and comparisons
- And more...

For detailed information, see `docs/automated_example_gallery.md`.

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a PR
- Aim for at least 95% test coverage for new code
- Include edge cases and error conditions in tests

## Getting Help

If you need assistance with your contribution:

- Ask questions in [GitHub Discussions](https://github.com/ipc-lab/kaira/discussions)
- Join our [Gitter chat](https://gitter.im/kaira/community)
- Email us at `yilmaselimfirat (at) gmail.com`

## Issue Reporting

If you encounter issues or have suggestions:

1. Check existing [issues](https://github.com/ipc-lab/kaira/issues) to avoid duplicates
2. Submit a ticket via the [GitHub Issue Tracker](https://github.com/ipc-lab/kaira/issues)
3. Use the appropriate issue template
4. Provide as much detail as possible for effective troubleshooting

We appreciate your contributions to Kaira!
