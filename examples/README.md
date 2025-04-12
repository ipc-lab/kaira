# Kaira Examples

This directory contains examples demonstrating various features and use cases of the Kaira library. The examples are organized by category and are automatically rendered into documentation using [Sphinx-Gallery](https://sphinx-gallery.github.io/).

## Creating New Examples

When creating new examples, follow these guidelines to ensure they work properly with sphinx-gallery:

1. **Naming Convention**: Name your example files with a `plot_` prefix (e.g., `plot_my_example.py`), even if the example doesn't generate plots. This ensures sphinx-gallery processes them correctly.

2. **File Structure**: Follow the template in `template_example.py`. Each example should include:

   - A docstring title and description at the top
   - Code sections separated by `# %%` comments
   - Section titles and explanations using `# %% # Section Title` format
   - A conclusion section

3. **Documentation**: Write clear, informative text in the RST-formatted docstrings and comments. These will be rendered as documentation.

4. **Reproducibility**: Always set random seeds (e.g., `torch.manual_seed(42)`, `np.random.seed(42)`) to ensure reproducible results.

5. **Visual Output**: Include visualizations where appropriate. These will be automatically captured and included in the documentation.

## Example Template

See `template_example.py` in the main examples directory for a ready-to-use template.

## Building the Example Gallery

The example gallery is built automatically when building the documentation:

```bash
cd docs
make html
```

The generated gallery will be available at `docs/_build/html/auto_examples/index.html`.

## Testing Examples

Examples should be tested to ensure they run without errors:

```bash
# Run a specific example
python examples/channels/plot_example.py

# Run all examples in a category
python -m pytest docs/auto_examples/channels/ -v
```

## Additional Resources

- [Sphinx-Gallery Documentation](https://sphinx-gallery.github.io)
- [Example Gallery Best Practices](https://sphinx-gallery.github.io/stable/advanced.html)
