"""Sphinx configuration for spikecurate's documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import spikecurate  # noqa: E402

project = "spikecurate"
copyright = "2026, steeve laquitaine"
author = "steeve laquitaine"
release = spikecurate.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "spikecurate"
