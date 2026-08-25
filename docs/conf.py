"""Sphinx configuration for the heliox documentation."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

from heliox import __version__  # noqa: E402

# -- Project information -----------------------------------------------------
project = "heliox"
author = "Pragyaan Gaur"
copyright = f"{date.today().year}, {author}"
release = __version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Numpydoc-style docstrings throughout, which is the scientific Python norm.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "none"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Warnings that are not worth failing the build over.
nitpicky = False
suppress_warnings = ["autodoc.import_object"]

# -- HTML output -------------------------------------------------------------
html_theme = "alabaster"
html_static_path = []
html_title = f"heliox {version}"

html_theme_options = {
    "description": "A solar physics toolkit for Python",
    "github_user": "pragyaangaur",
    "github_repo": "heliox",
    "fixed_sidebar": True,
    "page_width": "1000px",
}

# -- Doctests ----------------------------------------------------------------
doctest_global_setup = """
import matplotlib
matplotlib.use('Agg')
import numpy as np
import astropy.units as u
"""
