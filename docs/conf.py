# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys
import importlib.util
from sphinx_gallery.sorting import FileNameSortKey

# Improve module path setup for proper imports
# First, add the root directory containing the kaira package
sys.path.insert(0, os.path.abspath(".."))

# Check for graphviz installation and configure it
try:
    import graphviz
    import shutil

    has_graphviz = True
    print(f"Successfully imported Python graphviz package version {graphviz.__version__}")

    # Check if the 'dot' command is available
    dot_path = shutil.which("dot")
    if dot_path:
        print(f"Found graphviz 'dot' command at {dot_path}")
    else:
        print("WARNING: graphviz 'dot' command not found in PATH. Continuing without graphviz dot command.")
except ImportError:
    has_graphviz = False
    print("WARNING: Python graphviz package not found. Inheritance diagrams may not render properly.")

# Debug information - print the current path
print("Python sys.path:")
for p in sys.path:
    print(f"  - {p}")

# Verify we can import kaira modules
try:
    import kaira

    print(f"Successfully imported kaira version {kaira.__version__}")

    # Check if data module exists
    if importlib.util.find_spec("kaira.data") is not None:
        print("kaira.data module found")
    else:
        print("WARNING: kaira.data module not found")

    # List available kaira modules
    print("Available kaira modules:")
    for module_name in dir(kaira):
        if not module_name.startswith("_"):
            print(f"  - {module_name}")

except ImportError as e:
    print(f"Error importing kaira: {e}")
    print("Documentation may not build correctly")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Kaira"
author = "Kaira Team"
version = "0.2.1"
release = "0.2.1"

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
    "sphinx.ext.inheritance_diagram",  # Add inheritance diagrams
    "sphinx_rtd_theme",
    "sphinx_gallery.gen_gallery",  # Add sphinx-gallery for plot directive
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.todo",  # Add support for TODOs
    "sphinx.ext.ifconfig",  # Add support for conditional content
    "sphinx_design",  # Add sphinx-design for better UI components
    # "sphinx_copybutton",  # Add copy button to code blocks
    # "hoverxref.extension"
]

# Enhanced sphinx-gallery configuration
# Note: This configuration contains function objects (like FileNameSortKey) that cannot be cached
# The warning about "cannot cache unpickable configuration value" is expected and can be ignored
sphinx_gallery_conf = {
    # Directory paths
    "examples_dirs": [
        "../examples/channels",
        "../examples/constraints",
        "../examples/modulation",
        "../examples/metrics",
        "../examples/data",
        "../examples/losses",
        "../examples/models",
        "../examples/models_fec",
        "../examples/benchmarks",
    ],
    "gallery_dirs": [
        "auto_examples/channels",
        "auto_examples/constraints",
        "auto_examples/modulation",
        "auto_examples/metrics",
        "auto_examples/data",
        "auto_examples/losses",
        "auto_examples/models",
        "auto_examples/models_fec",
        "auto_examples/benchmarks",
    ],
    # File patterns and organization
    "filename_pattern": r"\.py$",
    "ignore_pattern": r"__init__\.py|utils\.py",
    # Content display options
    "download_all_examples": True,
    "plot_gallery": True,  # Enable gallery plots
    "thumbnail_size": (400, 280),
    "remove_config_comments": True,
    "min_reported_time": 1,
    "show_memory": False,
    "matplotlib_animations": True,
    "show_signature": True,
    # Enable Binder integration with JupyterLab interface
    "binder": {
        "org": "ipc-lab",
        "repo": "kaira",
        "branch": "main",
        "binderhub_url": "https://mybinder.org",
        "dependencies": "../requirements.txt",
        "use_jupyter_lab": True,
    },
    # Image scrapers
    "image_scrapers": ("matplotlib",),
    # Execution settings
    "junit": "_build/junit-results.xml",  # Save test results
    "inspect_global_variables": True,  # Inspect global variables
    # Custom template
    "default_thumb_file": "_static/logo.png",
    # Build settings
    "only_warn_on_example_error": True,  # Only warn on example errors, don't fail
    "abort_on_example_error": False,  # Don't abort on example errors
    "reset_modules": ("matplotlib",),
    # Backreference settings
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": "kaira",
    # Reference configurations
    "reference_url": {
        "kaira": None,  # The module has no reference URL
    },
    "capture_repr": ("_repr_html_", "__repr__"),  # Capture representations for objects
    # Gallery organization
    "within_subsection_order": FileNameSortKey,  # Sort by filename within subsections
    # First cell in generated notebooks - improving the note about downloading and Binder
    "first_notebook_cell": "# %% [markdown]\n# # {title}\n# {descr}\n\n*This notebook demonstrates {title}*\n\n**Quick Links:**\n- Download this example as a Jupyter notebook or Python script using the buttons below\n- Run this example interactively in your browser via Binder\n",
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
# Set recursive to true to document items in submodules automatically
autosummary_recursive = True
# Configure template mapping for different types of objects
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
bibtex_tooltips = False  # disable default tooltips
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

# Configuration for inheritance diagrams
inheritance_graph_attrs = {
    "rankdir": "TB",
    "size": '"12.0, 8.0"',  # already quoted
    "bgcolor": "transparent",
    "splines": "ortho",
    "ratio": "compress",
    "fontsize": "12",
    # escape the comma so it isn’t seen as an attr‐separator by dot
    "fontname": '"Roboto\\, sans-serif"',
}

# Add custom JavaScript files
html_js_files = [
    "gallery-custom.js",
    ("https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js", {"defer": "defer"}),
    ("https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js", {"defer": "defer"}),
]

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

suppress_warnings = [
    "autodoc.duplicate_object",
    "config.cache",  # Suppress warnings about unpicklable configuration values like sphinx_gallery_conf
]
# Commented out # Add setting to prevent duplicate documentation of enum members
# "app.add_directive", "app.add_role", "app.add_generic_role", "app.add_transform",
# "app.add_post_transform", "app.add_domain", "app.add_builder", "misc.highlighting_failure",
# "autosummary", "autosectionlabel.*",


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

    # Add config value only if it doesn't exist
    try:
        app.add_config_value("current_date", get_current_date(), "env")
    except AttributeError:  # More specific exception
        pass  # Config value already exists

    # Download auto_examples before building if needed
    app.connect("config-inited", download_auto_examples_if_needed)


def download_auto_examples_if_needed(app, config):
    """Download auto_examples if needed during Sphinx build."""
    import subprocess  # nosec B404 - subprocess is needed for legitimate script execution
    import sys
    from pathlib import Path

    # Get the project root directory
    docs_dir = Path(app.srcdir).resolve()
    project_root = docs_dir.parent
    # Ensure download_script is a string representation of the path
    download_script = str(project_root / "scripts" / "download_auto_examples.py")
    python_executable = str(sys.executable)  # Ensure python executable is a string

    if Path(download_script).exists():
        try:
            print("[Sphinx] Running auto_examples download script...")
            # Ensure all parts of the command are strings
            # Validate that we're only executing our own script with trusted python executable
            if not Path(download_script).is_file() or not Path(python_executable).is_file():
                raise ValueError("Invalid script or python executable path")
            command = [python_executable, download_script]
            result = subprocess.run(command, capture_output=True, text=True, cwd=str(project_root), check=False)  # nosec B603 - controlled execution of our own script
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode != 0:
                print(f"[Sphinx] Warning: auto_examples download script failed with code {result.returncode}")
        except Exception as e:
            print(f"[Sphinx] Warning: Could not run auto_examples download script: {e}")
    else:
        print(f"[Sphinx] Warning: Download script not found at {download_script}")
