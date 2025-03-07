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
author = "Kaira Team"
version = "0.1.0"
release = "0.1.0"

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
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx_gallery.gen_gallery",  # Add sphinx-gallery for plot directive
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.todo",  # Add support for TODOs
    "sphinx.ext.ifconfig",  # Add support for conditional content
    "sphinx_design",  # Add sphinx-design for better UI components
    # "hoverxref.extension"
]

# Remove the duplicated and restrictive sphinx_gallery_conf
# and replace with a more comprehensive one
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": r"\.py$",  # Include all Python files
    "ignore_pattern": r"__init__\.py",  # Ignore __init__.py files
    "line_numbers": True,  # Show line numbers in code blocks
    "download_all_examples": True,  # Option to download all examples
    "plot_gallery": True,  # Generate plots from examples
    "thumbnail_size": (320, 224),  # Size for gallery thumbnails
    "remove_config_comments": True,  # Remove config comments in examples
    "min_reported_time": 1,  # Minimum time to report in examples
    "show_memory": False,  # Don't show memory usage
    "matplotlib_animations": True,  # Enable matplotlib animations
    "doc_module": ("kaira",),  # Document this module
    "default_thumb_file": "_static/logo.png",  # Default thumbnail image
    "show_signature": True,  # Show function signatures
    "reference_url": {
        "kaira": None,  # The module has no reference URL
    },
    "capture_repr": ("_repr_html_", "__repr__"),  # Capture representations for objects
}

# Configure autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__, __eq__, __format__, __ge__, __getattribute__, __gt__, __hash__, __lt__, __le__, __ne__, __reduce__, __reduce_ex__, __sizeof__, __str__",  # , __module__, __dict__, __dir__',
    "show-inheritance": True,
    "inherited-members": True,
    "private-members": False,
}

# Add settings to completely ignore inherited members from Python SDK
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False

# Automatically generate autosummary pages
autosummary_generate = True
autosummary_imported_members = False
autosummary_template_mapping = {
    "class": "class.rst",
    "function": "function.rst",
    "module": "module.rst",
    "attribute": "attribute.rst",
    "method": "method.rst",
}

# Configure sphinxcontrib.bibtex with improved settings
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"  # Use the built-in plain style which is more tolerant
bibtex_reference_style = "author_year"
bibtex_tooltips = False # disable default tooltips
# Remove other bibtex settings that might be incompatible
bibtex_cite_id = "cite-{key}"
# bibtex_bibliography_header = "<h2>References</h2>"
# bibtex_footbibliography_header = "<h3>Footnote Citations</h3>"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
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

# Add autosummary templates directory
templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Switch to ReadTheDocs theme
html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
    "plot_directive.css",  # Add the plot directive CSS file
]

# Add custom JavaScript file - ensure correct format
html_js_files = []

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_title = "Kaira Documentation"


# -- Options for source files ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-source-files

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    # 'display_version': True,
    "prev_next_buttons_location": "bottom",
    # "style_external_links": True,
    "style_nav_header_background": "#005f73",
}

html_context = {
    "display_github": True,
    "github_user": "ipc-lab",
    "github_repo": "kaira",
    "github_version": "main/docs/",
    "conf_py_path": "/docs/",
    "source_suffix": source_suffix,
}

# Include README.md in the documentation
master_doc = "index"

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "torchmetrics": ("https://torchmetrics.readthedocs.io/en/stable/", None),
}

todo_include_todos = True

hoverxref_auto_ref = True
hoverxref_role_types = {
    "hoverxref": "tooltip",
    "ref": "tooltip",
    "confval": "tooltip",
    "mod": "tooltip",
    "class": "tooltip",
    "term": "tooltip",
}


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
        bool: True if the member should be skipped, False otherwise.or name == "training" or name == "dump_patches" :
    """
    # Exclude common dunder methods that aren't helpful in documentation
    if name not in ["__init__", "__call__", "__enter__", "__exit__"]:
        if name.startswith("__") and name.endswith("__"):
            return True

        if name.startswith("_"):
            return True

    if name in ["training", "dump_patches", "call_super_init"]:
        return True

    module = getattr(obj, "__module__", None)
    if module and (module.startswith("numpy.") or module.startswith("torch.") or module == "torch" or module == "numpy" or module == "torchmetrics"):
        return True

    # Skip specific problematic methods
    if what == "method" and name == "plot" and hasattr(obj, "__module__") and "torchmetrics.image.ssim" in getattr(obj, "__module__", ""):
        return True

    return False


def get_current_date():
    """Return the current date in the format 'DD Month YYYY'."""
    return datetime.datetime.now().strftime("%d %B %Y")


def setup(app):
    """Set up the Sphinx application.

    This function is called when the Sphinx application is initialized.
    It is used to connect event listeners and perform other setup tasks.

    Args:
        app: The Sphinx application object.
    """
    app.connect("autodoc-skip-member", skip_member)
    app.add_config_value("current_date", get_current_date(), "env")
