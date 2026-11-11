"""Build the HTML API reference into ``docs/_build``.

Every module under ``pylea/`` is named explicitly rather than discovered by recursion:
several packages re-export a function under its defining module's name via ``__all__``
(``pylea.simplify.lift`` the module, ``lift`` the function), and pdoc's automatic recursion
follows ``__all__``, silently dropping the shadowed module's page.

Doctests are highlighted as Python sessions.  Docstrings write them as indented blocks,
which doctest reads and Markdown renders without highlighting, so `fenced_doctests` turns
each into a fenced ``pycon`` block while the page is rendered; the docstrings are unchanged.

Diagrams the docstrings refer to are drawn into the same directory, so that a page can show
one with an ordinary Markdown image reference.

Usage::

    python docs/make.py
"""
from pathlib import Path

import pdoc
import pdoc.render
from jinja2 import pass_context

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "_build"


def fenced_doctests(markdown: str) -> str:
    """`markdown` with every indented doctest block written as a fenced ``pycon`` block.

    A doctest block is a run of non-blank lines following a blank line, the first starting
    with ``>>>`` after an indentation of at least four spaces, and every one indented at
    least that far.  The fence takes four spaces less indentation than the block had, so a
    block inside a list item stays inside it.  A doctest at the margin is left as it is,
    since pdoc's Markdown renderer already highlights it.

        >>> fenced_doctests("Adding:\\n\\n    >>> 1 + 1\\n    2\\n\\nDone.").split("\\n")
        ['Adding:', '', '```pycon', '>>> 1 + 1', '2', '```', '', 'Done.']
    """
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip(" "))
        starts_block = (indent >= 4 and line[indent:].startswith(">>>")
                        and (i == 0 or not lines[i - 1].strip()))
        end = i
        while end < len(lines) and lines[end].strip():
            end += 1
        block = lines[i:end]
        if not starts_block or any(not b.startswith(" " * indent) for b in block):
            out.append(line)
            i += 1
            continue
        margin = " " * (indent - 4)
        out.append(margin + "```pycon")
        out.extend(margin + b[indent:] for b in block)
        out.append(margin + "```")
        i = end
    return "\n".join(out)


def highlight_doctests() -> None:
    """Nothing, having made pdoc's `to_markdown` template filter apply `fenced_doctests`.

    The filter is what every template passes a docstring through before rendering it as
    HTML, and `pdoc.render.env` is the Jinja environment pdoc offers for such changes.
    """
    convert = pdoc.render.env.filters["to_markdown"]

    @pass_context
    def to_markdown(context, docstring: str) -> str:
        return fenced_doctests(convert(context, docstring))

    pdoc.render.env.filters["to_markdown"] = to_markdown


def modules() -> list[str]:
    """The dotted name of every module in the package, packages before their contents."""
    names = []
    for path in sorted((ROOT / "pylea").rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if any(part.startswith("_") for part in parts):
            continue                    # private modules, such as subprocess workers
        names.append(".".join(parts))
    return names


def illustrate() -> None:
    """Nothing, having drawn the diagrams the docstrings refer to into the output.

    A page refers to one by a relative Markdown image reference, so each is written beside
    the page that shows it.  The colours are fixed rather than following the reader's
    setting, these pages being of one colour whatever that setting is.  The font size sets
    the size of the whole picture.
    """
    from pylea.draw import Style
    from pylea.examples.disease import test
    from pylea.examples.sprinkler import sprinkler_network

    OUT.mkdir(parents=True, exist_ok=True)
    style = Style(font_size=15.0)
    for name, picture in (("test", test.diagram(style=style)),
                          ("sprinkler_network",
                           sprinkler_network.dependency_graph(style=style))):
        (OUT / f"{name}.svg").write_text(picture.svg(theme="light"), encoding="utf-8")


def notebooks() -> None:
    """Nothing, having run every notebook in ``examples/`` and written it into the output.

    Each ``examples/<name>.py``, a notebook in jupytext's percent format, is written as
    ``examples/<name>.ipynb`` with its outputs, for download, and as
    ``examples/<name>.html``.  A cell that raises stops the build.  The page follows the
    reader's colour setting, as the diagrams in it do.  Every other file of ``examples/``,
    such as a network a notebook reads, is copied beside them, so that a page can link it.
    """
    import shutil
    from string import Template

    import jupytext
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from pygments.formatters import HtmlFormatter

    out = OUT / "examples"
    out.mkdir(parents=True, exist_ok=True)
    page = Template((ROOT / "docs" / "notebook_page.html").read_text(encoding="utf-8"))
    highlight = (HtmlFormatter(style="default").get_style_defs(".highlight")
                 + "\n@media (prefers-color-scheme: dark) {\n"
                 + HtmlFormatter(style="github-dark").get_style_defs(".highlight")
                 + "\n}\n.highlight, .highlight pre { background: var(--code-bg); }")
    exporter = HTMLExporter(template_name="basic")
    for data in sorted((ROOT / "examples").iterdir()):
        if data.is_file() and data.suffix != ".py" and not data.name.startswith("."):
            shutil.copy(data, out / data.name)
    for source in sorted((ROOT / "examples").glob("*.py")):
        notebook = jupytext.read(source)
        NotebookClient(notebook, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(source.parent)}}).execute()
        nbformat.write(notebook, out / f"{source.stem}.ipynb")
        body, _ = exporter.from_notebook_node(notebook)
        title = source.stem.replace("_", " ").capitalize()
        (out / f"{source.stem}.html").write_text(
            page.substitute(title=title, body=body, highlight=highlight,
                            notebook=f"{source.stem}.ipynb"),
            encoding="utf-8")


if __name__ == "__main__":
    pdoc.render.configure(docformat="markdown", footer_text="Pylea")
    highlight_doctests()
    pdoc.pdoc(*modules(), output_directory=OUT)
    illustrate()
    notebooks()
    print(f"wrote {sum(1 for _ in OUT.rglob('*.html'))} pages to {OUT}")
