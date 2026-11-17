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

Labels are written as they are, apart from the characters LaTeX reads as markup.  A label
with a character outside Latin-1, such as a Greek or Chinese name, needs a Unicode engine,
LuaLaTeX or XeLaTeX, and a typewriter font that has the character; pdfLaTeX stops at it.
The standalone document chooses its font setup by engine: the T1 font encoding under
pdfLaTeX and `fontspec` under the other two, where a document may then set a font with
`\setmonofont`.

    >>> from pylea.examples.disease import test
    >>> tikz = test.diagram().tikz()
    >>> tikz.startswith("\\begin{tikzpicture}"), tikz.rstrip().endswith("\\end{tikzpicture}")
    (True, True)
"""
from __future__ import annotations

from pylea.draw.glyph import Style, number_text
from pylea.draw.layout import Placement
from pylea.draw.sketch import Sketch

# Stroke width, as a fraction of the font size, matching the weight the SVG backend draws
# at.  A document wanting another weight redefines the styles rather than changing this.
_STROKE = 0.075

_ESCAPES = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "×": r"$\times$",
            **{c: "\\" + c for c in "{}_#&%$"}}


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
        points = [f"({number_text(px)},{number_text(py)})" for px, py in rest]
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
    own and is what a caller checking a figure wants.  See the module docstring for labels
    outside Latin-1.
    """
    style = style or Style()
    stroke = number_text(_STROKE * style.font_size)
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
        f"\\fontsize{{{number_text(style.font_size)}}}"
        f"{{{number_text(style.font_size * 1.2)}}}\\selectfont}}]",
        f"  \\useasboundingbox ({number_text(-over)},{number_text(-band)}) rectangle "
        f"({number_text(placement.width + over)},{number_text(placement.height)});",
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
            lines.append(f"  \\node[pylea label] at ({number_text(x + label.x)},"
                         f"{number_text(y + label.y)}) {{{_escape(label.text)}}};")

    if sketch.title:
        lines.append(f"  \\node[pylea label] at ({number_text(placement.width / 2)},"
                     f"{number_text(-band / 2)}) {{{_escape(sketch.title)}}};")

    lines.append("\\end{tikzpicture}")
    picture = "\n".join(lines)

    if not standalone:
        return picture
    return ("\\documentclass[tikz,border=2pt]{standalone}\n"
            "\\usepackage{iftex}\n"
            "\\ifPDFTeX\n  \\usepackage[T1]{fontenc}\n"
            "\\else\n  \\usepackage{fontspec}\n\\fi\n"
            "\\begin{document}\n" + picture + "\n\\end{document}\n")
