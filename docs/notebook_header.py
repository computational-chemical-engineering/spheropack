"""Inserts (or refreshes) the header cells of the example notebooks.

Every notebook in docs/notebooks gets, right after its title cell:

- a Markdown cell with an "Open in Colab" badge and a "Download notebook" badge;
- a code cell that installs spheropack when it is not importable (Colab, fresh
  environments) and does nothing otherwise.

Run after adding a notebook: ``python docs/notebook_header.py``. The cells are tagged
in their metadata, so running the script again replaces them instead of adding more.
"""

from pathlib import Path

import nbformat

REPO = "computational-chemical-engineering/spheropack"
SITE = "https://computational-chemical-engineering.github.io/spheropack"
TAG = "spheropack-header"

INSTALL = """# Install spheropack when it is not available (for example on Google Colab).
import importlib.util

if importlib.util.find_spec("spheropack") is None:
    %pip install -q spheropack"""


def header_cells(name: str) -> list:
    colab = f"https://colab.research.google.com/github/{REPO}/blob/main/docs/notebooks/{name}"
    download = f"{SITE}/notebooks/{name}"
    badges = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab}) "
        f"[![Download notebook](https://img.shields.io/badge/download-notebook-orange?logo=jupyter)]({download})"
    )
    md = nbformat.v4.new_markdown_cell(badges, metadata={TAG: True})
    code = nbformat.v4.new_code_cell(INSTALL, metadata={TAG: True})
    return [md, code]


def main() -> None:
    for path in sorted((Path(__file__).parent / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        cells = [c for c in nb.cells if not c.metadata.get(TAG)]
        nb.cells = cells[:1] + header_cells(path.name) + cells[1:]
        nbformat.write(nb, path)
        print("header:", path.name)


if __name__ == "__main__":
    main()
