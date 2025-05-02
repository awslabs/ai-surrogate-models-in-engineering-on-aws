# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import sys
import os

# insert ML4Simkit path into the system
sys.path.insert(0, os.path.abspath(".."))

import mlsimkit

project = "AI Surrogate Models for Engineering on AWS"
copyright = "Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved."
author = "AWS AutoMfg Product Engineering"
version = mlsimkit.__version__

# a release is the version without the commit/build
release = ".".join(map(str, mlsimkit._version.__version_tuple__[0:3]))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # allow google-style docstrings for code signatures
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
]

latex_engine = "lualatex"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
]
source_suffix = [".rst", ".md"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
}

# -- HTML output configuration -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/theming.html
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "body_min_width": "800",
    "body_max_width": "auto",
    "note_bg": "#D6EAF8",
    "note_border": "#B7DDF7",
    "warn_bg": "#F7D9C4",
    "warn_border": "#F7C6A3",
    "show_powered_by": False,
}

html_sidebars = {
    "index": ["sidebarintro.html", "relations.html", "searchbox.html"],
    "**": [
        "sidebarintro.html",
        "sidebar_localtoc.html",
        "relations.html",
        "searchbox.html",
    ],
}

# Set the TOC depth for the local TOC in the sidebar
html_context = {
    "localtoc_depth": 1,  # Adjust the depth as needed
}

# Add copyright to PDF
# See https://stackoverflow.com/questions/54147210/how-to-add-copyright-notice-to-sphinx-generated-latex-documentation
latex_elements = {
    "extraclassoptions": "oneside, openany, 11pt, letterpaper",
    "preamble": r"""
\makeatletter
   \fancypagestyle{normal}{
% this is the stuff in sphinx.sty
    \fancyhf{}
    \fancyfoot[LE,RO]{{\py@HeaderFamily\thepage}}
% we comment this out and
    %\fancyfoot[LO]{{\py@HeaderFamily\nouppercase{\rightmark}}}
    %\fancyfoot[RE]{{\py@HeaderFamily\nouppercase{\leftmark}}}
% add copyright stuff
    \fancyfoot[LO,RE]{{\textcopyright\ Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.}}
% again original stuff
    \fancyhead[LE,RO]{{\py@HeaderFamily \@title\sphinxheadercomma\py@release}}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0.4pt}
    }
% this is applied to each opening page of a chapter
   \fancypagestyle{plain}{
    \fancyhf{}
    \fancyfoot[LE,RO]{{\py@HeaderFamily\thepage}}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0.4pt}
% add copyright stuff for example at left of footer on odd pages,
% which is the case for chapter opening page by default
    \fancyfoot[LO,RE]{{\textcopyright\ Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.}}
    }
\makeatother
""",
}
