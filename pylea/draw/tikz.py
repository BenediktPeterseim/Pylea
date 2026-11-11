r"""Rendering a placed sketch as TikZ.

`to_tikz` writes one `tikzpicture` for a document to include, needing nothing in the
preamble beyond ``\usepackage{tikz}``.  Every shape is drawn from the same outlines
`pylea.draw.glyph` gives the SVG backend, so the two agree on what a diagram looks like.
The gates are drawn as paths rather than taken from a circuit library, which is what keeps
the output to base TikZ.

One unit of a placement is one point here, so a diagram is the size on the page that it is
on the screen.  Both axes point up, as a placement's do, so no coordinate is turned over.

Four styles carry the appearance and a document may redefine any of them: `pylea wire`,
`pylea outline`, `pylea solid` and `pylea label`.

    >>> from pylea.examples.disease import test
    >>> tikz = test.diagram().tikz()
    >>> tikz.startswith("\\begin{tikzpicture}"), tikz.rstrip().endswith("\\end{tikzpicture}")
    (True, True)
"""
from __future__ import annotations

from pylea.draw.glyph import Style
from pylea.draw.layout import Placement
from pylea.draw.sketch import Sketch

# Stroke width, as a fraction of the font size, matching the weight the SVG backend draws
# at.  A document wanting another weight redefines the styles rather than changing this.
_STROKE = 0.075

_ESCAPES = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "×": r"$\times$",
            **{c: "\\" + c for c in "{}_#&%$"}}


def _n(value: float) -> str:
    """`value` rounded to a hundredth, without a trailing zero."""
    return f"{round(value, 2):g}"


def _escape(text: str) -> str:
    """`text` with the characters LaTeX reads as markup written out, and the multiplication
    sign set in maths mode, where every LaTeX font has it.

        >>> _escape("body#0"), _escape("CPT_Rain"), _escape("×0.3")
        ('body\\\\#0', 'CPT\\\\_Rain', '$\\\\times$0.3')
    """
    return "".join(_ESCAPES.get(c, c) for c in text)


def _path(path, x: float, y: float) -> str:
    """One glyph outline, moved to ``(x, y)``, as a TikZ path."""
    out = []
    for seg in path:
        head, rest = seg[0], [(x + seg[i], y + seg[i + 1]) for i in range(1, len(seg), 2)]
        points = [f"({_n(px)},{_n(py)})" for px, py in rest]
        if head == "M":
            out.append(points[0])
        elif head == "L":
            out.append(f" -- {points[0]}")
        elif head == "C":
            out.append(f" .. controls {points[0]} and {points[1]} .. {points[2]}")
        else:
            out.append(" -- cycle")
    return "".join(out)





def to_tikz(sketch: Sketch, placement: Placement, style: Style | None = None,
            standalone: bool = False) -> str:
    """`sketch`, laid out as `placement`, as a `tikzpicture`.

    `standalone` wraps it in a document of the `standalone` class, which compiles on its
    own and is what a caller checking a figure wants.
    """
    style = style or Style()
    stroke = _n(_STROKE * style.font_size)
    band = style.font_size * 2.4 if sketch.title else 0.0
    # A title wider than the picture would overflow the bounding box, so the box grows
    # symmetrically around the centred title where it needs to.
    over = 0.0
    if sketch.title:
        title_width = style.text_width(sketch.title) * 1.15 + 2 * style.margin
        over = max(0.0, (title_width - placement.width) / 2)

    lines = [
        "\\begin{tikzpicture}[x=1pt,y=1pt,",
        f"    pylea wire/.style={{draw=black,line width={stroke}pt,line cap=round}},",
        f"    pylea outline/.style={{draw=black,fill=white,line width={stroke}pt,"
        "line join=round},",
        "    pylea solid/.style={draw=none,fill=black},",
        "    pylea label/.style={inner sep=0pt,text=black,font=\\ttfamily"
        f"\\fontsize{{{_n(style.font_size)}}}{{{_n(style.font_size * 1.2)}}}\\selectfont}}]",
        f"  \\useasboundingbox ({_n(-over)},{_n(-band)}) rectangle "
        f"({_n(placement.width + over)},{_n(placement.height)});",
    ]

    for wire in placement.routes:
        lines.append(f"  \\draw[pylea wire] {_path(wire.path, 0.0, 0.0)};")
        if wire.head:
            lines.append(f"  \\path[pylea solid] {_path(wire.head, 0.0, 0.0)};")

    for i, node in enumerate(sketch.nodes):
        glyph = placement.glyphs[i]
        x, y = placement.x[i], placement.y[i]
        for shape in glyph.shapes:
            kind = "pylea solid" if shape.solid else "pylea outline"
            lines.append(f"  \\path[{kind}] {_path(shape.path, x, y)};")
        for label in glyph.labels:
            lines.append(f"  \\node[pylea label] at ({_n(x + label.x)},{_n(y + label.y)}) "
                         f"{{{_escape(label.text)}}};")

    if sketch.title:
        lines.append(f"  \\node[pylea label] at ({_n(placement.width / 2)},"
                     f"{_n(-band / 2)}) {{{_escape(sketch.title)}}};")

    lines.append("\\end{tikzpicture}")
    picture = "\n".join(lines)

    if not standalone:
        return picture
    return ("\\documentclass[tikz,border=2pt]{standalone}\n"
            "\\begin{document}\n" + picture + "\n\\end{document}\n")
