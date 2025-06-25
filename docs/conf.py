import os
import sys
sys.path.insert(0, os.path.abspath('../syndat'))

project = 'syndat'
copyright = '2025, Fraunhofer SCAI'
author = 'Tim Adams'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']