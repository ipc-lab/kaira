# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath('../kaira'))

print(os.path.abspath('../kaira'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Kaira'
author = "Selim F. Yilmaz, Imperial IPC Lab"

copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser", 
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    "sphinx.ext.napoleon"
    
]

bibtex_bibfiles = ['refs.bib']

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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Switch to ReadTheDocs theme
html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']

html_css_files = [
    #'custom.css',
]

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Options for source files ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-source-files

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Include README.md in the documentation
master_doc = 'index'
