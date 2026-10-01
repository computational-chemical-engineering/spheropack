"""Sphinx configuration for spheropack."""

import shutil
import subprocess
from pathlib import Path

import spheropack

project = "spheropack"
author = "Frank Peters"
copyright = "2026, Frank Peters"
release = spheropack.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_nb",
    "breathe",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
nb_execution_mode = "off"  # notebooks are committed with their outputs (some take minutes)
nb_execution_timeout = 600
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# C++ API: run Doxygen, then let breathe read its XML output.
_here = Path(__file__).parent
(_here / "_build" / "doxygen").mkdir(parents=True, exist_ok=True)
subprocess.run(["doxygen", "Doxyfile"], cwd=_here, check=True)
breathe_projects = {"spheropack": str(_here / "_build" / "doxygen" / "xml")}
breathe_default_project = "spheropack"
# Copies of the notebooks next to their pages, for the "Download notebook" badges.
_extra = _here / "_build" / "extra"
(_extra / "notebooks").mkdir(parents=True, exist_ok=True)
for _nb in (_here / "notebooks").glob("*.ipynb"):
    shutil.copy2(_nb, _extra / "notebooks" / _nb.name)
html_extra_path = [str(_extra)]

html_theme = "pydata_sphinx_theme"
html_title = "spheropack"
html_theme_options = {
    "github_url": "https://github.com/computational-chemical-engineering/spheropack",
    "show_toc_level": 2,
}
