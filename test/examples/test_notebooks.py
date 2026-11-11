"""Every notebook in ``examples/`` runs from top to bottom without an error.

Each runs from its own directory, as the documentation build runs it, since a notebook
may read a file beside it.
"""
import runpy
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.py"))


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_a_notebook_runs_without_an_error(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    runpy.run_path(str(path))
    plt.close("all")
