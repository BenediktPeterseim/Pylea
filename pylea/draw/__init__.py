"""Drawing a DPP, as a dot diagram or as a graph over its variables.

A `pylea.ir.body.Body`, one definition of a DPP, is a dot diagram already: a set of
variables, a list of assignments, and two interface lists.  So a picture of one needs no
translation, only a rendering.  Every variable is a dot, every assignment is a box, and the
picture reads bottom to top, from the inputs to the outputs.

`diagram_of` draws one definition of a `Program`; `pylea.syntax.dpp.DPP.diagram` draws the
main function of a DPP.
A variable of several bits is several wires of the body, drawn as one dot wherever the
body keeps them together; `pylea.draw.sketch` says when it does not.  There is no nesting:
a call is a box carrying the name of the definition it calls, and a called DPP is drawn
by asking it for its own diagram.

`dependency_graph_of` draws the other picture, where the assignments are left out and the
variables of the interface are the nodes: one box per variable the DPP is given or returns,
and an arrow to each variable computed from it.  For a DPP whose outputs are the variables
of a network, that is the network.  It reads from the top down, the other way
round from the dot diagram.  Both pictures are laid out and rendered by the same code,
differing only in the sketch they are built from.

The result carries its own display: `Diagram.svg` for the markup, `Diagram.tikz` for a
figure to include in a LaTeX document, `Diagram.save` for a file, `Diagram.show` to open one
in a browser, and inline rendering in a notebook.  Nothing here needs a package outside the
standard library, and no layout program is called.

    >>> from pylea.examples.disease import test
    >>> d = diagram_of(test.program(), title="test")
    >>> d.svg().startswith("<svg")
    True
"""
from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

from pylea.draw.dependency import dependency_sketch_of
from pylea.draw.glyph import GlyphKind, Style
from pylea.draw.layout import Placement, place
from pylea.draw.sketch import Sketch, sketch_of
from pylea.draw.svg import to_svg
from pylea.draw.tikz import to_tikz
from pylea.ir.program import Program

__all__ = ["Diagram", "diagram_of", "dependency_graph_of", "Style", "GlyphKind", "Sketch",
           "Placement", "sketch_of", "dependency_sketch_of", "place", "to_svg", "to_tikz"]


class Diagram:
    """A laid-out picture of one definition, and the ways of looking at it."""

    __slots__ = ("sketch", "placement", "style")

    def __init__(self, sketch: Sketch, placement: Placement, style: Style) -> None:
        self.sketch = sketch
        self.placement = placement
        self.style = style

    @property
    def size(self) -> tuple[float, float]:
        """The width and height of the picture, in the units the glyphs are drawn in."""
        return self.placement.width, self.placement.height

    def svg(self, theme: str = "auto") -> str:
        """This picture as a complete ``<svg>`` element.

        `theme` is ``"auto"`` to follow the reader's setting, or ``"light"`` or ``"dark"``
        to fix the colours; a page of a fixed colour needs the latter, since a file loaded
        through an ``img`` element cannot see the colours of the page holding it.
        """
        return to_svg(self.sketch, self.placement, self.style, theme)

    def tikz(self, standalone: bool = False) -> str:
        """This picture as a `tikzpicture`, for a LaTeX document to include.

        `standalone` wraps it in a document that compiles on its own.
        """
        return to_tikz(self.sketch, self.placement, self.style, standalone)

    def save(self, path, theme: str = "auto") -> Path:
        """This picture written to `path`, which is returned.  The suffix picks the format.

        ``.svg`` writes the markup and ``.tex`` a standalone LaTeX document, since a file
        written on its own is one meant to be looked at rather than included.  Any other
        suffix raises `ValueError`.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".svg":
            path.write_text(self.svg(theme), encoding="utf-8")
        elif suffix == ".tex":
            path.write_text(self.tikz(standalone=True), encoding="utf-8")
        else:
            raise ValueError(f"cannot write {path.suffix or 'a file with no suffix'}; "
                             "the diagram is written as .svg or as .tex")
        return path

    def show(self) -> Path:
        """The path of a temporary copy of this picture, now opened in a browser."""
        handle = tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False,
                                             encoding="utf-8")
        with handle:
            handle.write(self.svg())
        path = Path(handle.name)
        webbrowser.open(path.as_uri())
        return path

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        """The picture and a one-line summary, for a notebook to render."""
        width, height = self.size
        return {"image/svg+xml": self.svg(),
                "text/plain": f"<diagram {round(width)} x {round(height)}>"}

    def __repr__(self) -> str:
        width, height = self.size
        return (f"<Diagram {self.sketch.title or 'unnamed'}: "
                f"{len(self.sketch.nodes)} nodes, {round(width)} x {round(height)}>")


def diagram_of(prog: Program, body: int | None = None, body_names=None, op_names=None,
               input_names=None, output_names=None, title: str = "",
               style: Style | None = None,
               sweeps: int | None = None,
               wire_groups=(), input_widths=None, output_widths=None) -> Diagram:
    """The dot diagram of one definition of `prog`, the main function where `body` is
    left out.

    `body_names` and `op_names` label a called definition and an assignment of this
    definition by index.  `wire_groups` names tuples of wires each standing for one
    variable, drawn as a single dot wherever the definition keeps them together;
    `input_widths` and `output_widths` give the number of wires of each interface
    variable, one wire each where left out; and `input_names` and `output_names` label
    the variable at each interface position.  `style` sets the sizes and `sweeps` bounds
    the ordering step of the layout.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> diagram_of(coin.program(), title="coin")
        <Diagram coin: 3 nodes, ... x ...>
    """
    style = style or Style()
    sketch = sketch_of(prog, body, body_names, op_names, input_names,
                       output_names, title, wire_groups, input_widths, output_widths)
    return Diagram(sketch, place(sketch, style, sweeps), style)


def dependency_graph_of(prog: Program, body: int | None = None, input_names=None,
                        output_names=None, input_widths=None, output_widths=None,
                        title: str = "", style: Style | None = None,
                        sweeps: int | None = None) -> Diagram:
    """The dependency graph of one definition of `prog`, the main function where `body`
    is left out.

    One box per variable the definition is given or returns, and an arrow from each
    variable to the ones computed from it; see `pylea.draw.dependency` for what counts
    as one.
    `input_widths` and `output_widths` give the number of wires of each interface
    variable, `input_names` and `output_names` label them by position, `style` sets the
    sizes and `sweeps` bounds the ordering step of the layout.

        >>> from pylea.examples.sprinkler import sprinkler_network
        >>> dependency_graph_of(sprinkler_network.program(), title="sprinkler_network",
        ...                     output_widths=(1, 1, 2, 2))
        <Diagram sprinkler_network: 4 nodes, ... x ...>
    """
    style = style or Style()
    sketch = dependency_sketch_of(prog, body, input_names, output_names,
                                  input_widths, output_widths, title)
    return Diagram(sketch, place(sketch, style, sweeps), style)
