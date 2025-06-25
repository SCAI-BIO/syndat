import os
import sys
import subprocess

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

def get_version_from_git():
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--match", "v*.*.*", "--abbrev=0"]
        ).decode().strip()
        if tag.startswith('v'):
            return tag[1:]
        return tag
    except Exception:
        return "latest"

release = get_version_from_git()
version = release  
