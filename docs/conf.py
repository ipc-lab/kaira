# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath("../kaira"))

print(os.path.abspath("../kaira"))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Kaira"
author = "Selim F. Yilmaz, Imperial IPC Lab"

copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Add source links
    "sphinx.ext.intersphinx",  # Link to other projects
    "sphinx.ext.coverage",  # Check documentation coverage
]

# Automatically generate autosummary pages
# autosummary_generate = True

bibtex_bibfiles = ["refs.bib"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Switch to ReadTheDocs theme
html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = []

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Options for source files ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-source-files

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Include README.md in the documentation
master_doc = "index"


def skip_member(app, what, name, obj, skip, options):
    """Determine whether to skip a member during documentation generation.

    This function is used to skip certain members (e.g., private attributes,
    inherited members) from appearing in the generated documentation.

    Args:
        app: The Sphinx application object.
        what (str): The type of the object being documented (e.g., "module", "class", "function").
        name (str): The name of the member.
        obj: The member object itself.
        skip (bool): A flag indicating whether the member should be skipped by default.
        options: Options passed to the directive.

    Returns:
        bool: True if the member should be skipped, False otherwise.
    """
    module = getattr(obj, "__module__", None)
    if module and (
        module.startswith("numpy.")
        or module.startswith("torch.")
        or module == "torch"
        or module == "numpy"
    ):
        return True

    return False


def setup(app):
    """Set up the Sphinx application.

    This function is called when the Sphinx application is initialized.
    It is used to connect event listeners and perform other setup tasks.

    Args:
        app: The Sphinx application object.
    """
    app.connect("autodoc-skip-member", skip_member)
