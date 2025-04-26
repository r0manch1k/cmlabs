import os
import sys
import matplotlib
import matplotlib.pyplot as plt

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cmlabs"
copyright = "2025, Roman Sokolovsky"
author = "Roman Sokolovsky"
release = "0.0.1"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../cmlabs"))
sys.path.insert(0, os.path.abspath("../../cmlabs/interpolate"))
sys.path.insert(0, os.path.abspath("../../cmlabs/interpolate/tests"))

matplotlib.use("agg")
plt.ioff()

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_favicon",
]

templates_path = ["_templates"]
source_suffix = ".rst"
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
favicons = [
    {"rel": "icon", "sizes": "16x16", "href": "favicon-16x16.png", "type": "image/png"},
    {"rel": "icon", "sizes": "32x32", "href": "favicon-32x32.png", "type": "image/png"},
    {
        "rel": "apple-touch-icon",
        "sizes": "180x180",
        "href": "apple-touch-icon.png",
        "type": "image/png",
    },
]
