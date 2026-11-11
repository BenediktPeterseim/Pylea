"""The top-level documentation links to every notebook, and every link resolves.

pdoc turns a backticked dotted name into a link only where the name resolves, and leaves a
name that does not as plain text, so a renamed module would otherwise go unnoticed.
"""
import importlib
import re
from pathlib import Path

import pytest

import pylea
import pylea.examples

_DOTTED = re.compile(r"`(pylea(?:\.\w+)+)`")


def _resolves(name: str) -> bool:
    """Whether `name` is a module, or an attribute reached from the longest module prefix."""
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for attribute in parts[split:]:
                obj = getattr(obj, attribute)
        except AttributeError:
            return False
        return True
    return False


@pytest.mark.parametrize("module", [pylea, pylea.examples], ids=lambda m: m.__name__)
def test_every_dotted_name_in_the_documentation_resolves(module):
    names = set(_DOTTED.findall(module.__doc__ or ""))
    assert names
    assert [name for name in names if not _resolves(name)] == []


def test_the_top_level_documentation_links_every_notebook_and_nothing_else():
    notebooks = (Path(__file__).resolve().parents[2] / "examples").glob("*.py")
    linked = set(re.findall(r"\]\(examples/(\w+)\.html\)", pylea.__doc__ or ""))
    assert linked == {path.stem for path in notebooks}
