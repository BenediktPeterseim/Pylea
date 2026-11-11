"""Rendering a placed sketch as SVG.

`to_svg` writes one self-contained element with no external references, so the result can be
saved, opened in a browser, or shown inline in a notebook.  Colours are two custom
properties, and by default a dark-scheme override follows the reader's setting.

A page of a fixed colour asks for a fixed `theme` instead.  A file loaded through an ``img``
element is a document of its own and cannot see the colours of the page holding it, so it
would follow the reader's setting while the page did not, and a light diagram would land on
a light page.

Wires are drawn before glyphs, and a glyph's outline is filled with the background, so a
wire passing behind a box is hidden by it.  A wire, an arrow head and a glyph outline are
all drawn from the same kind of path, so one emitter serves for the three.

The y axis of a placement points up and the y axis of SVG points down, so every ordinate is
subtracted from the height of the picture as it is written.

The ``viewBox`` is in drawing units, and the ``width`` and ``height`` attributes are those
units times `pylea.draw.glyph.Style.scale`, which is where the size the element is shown at
is set.
"""
from __future__ import annotations

from pylea.draw.glyph import Style
from pylea.draw.layout import Placement
from pylea.draw.sketch import Sketch

_FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

_LIGHT = "--ink:#16181d;--paper:#fff"
_DARK = "--ink:#e6e8ec;--paper:#1b1d22"

_STYLE = """\
.pylea{{{palette}}}
{override}
.pylea .wire{{fill:none;stroke:var(--ink);stroke-width:{stroke}}}
.pylea .outline{{fill:var(--paper);stroke:var(--ink);stroke-width:{stroke};\
stroke-linejoin:round}}
.pylea .solid{{fill:var(--ink);stroke:none}}
.pylea text{{fill:var(--ink);font-family:{font};text-anchor:middle;\
dominant-baseline:central}}\
"""


def _n(value: float) -> str:
    """`value` rounded to a hundredth, without a trailing zero."""
    return f"{round(value, 2):g}"





def _shape(path, x: float, y: float, flip) -> str:
    """One glyph outline, moved to `(x, y)` and turned the right way up."""
    out = []
    for seg in path:
        head, rest = seg[0], seg[1:]
        pairs = [(x + rest[i], flip(y + rest[i + 1])) for i in range(0, len(rest), 2)]
        out.append(head + " ".join(f"{_n(px)} {_n(py)}" for px, py in pairs))
    return "".join(out)


def to_svg(sketch: Sketch, placement: Placement, style: Style | None = None,
           theme: str = "auto") -> str:
    """`sketch`, laid out as `placement`, as a complete ``<svg>`` element.

    `theme` is ``"auto"`` to follow the reader's setting, or ``"light"`` or ``"dark"`` to
    fix the colours, which is what a page of a fixed colour needs.  Any other value raises
    `ValueError`.
    """
    style = style or Style()
    if theme not in ("auto", "light", "dark"):
        raise ValueError(f"theme must be 'auto', 'light' or 'dark'; got {theme!r}")
    band = style.font_size * 2.4 if sketch.title else 0.0
    width, height = placement.width, placement.height
    # A title wider than the picture would overflow the viewBox, so the box grows
    # symmetrically around the centred title where it needs to.
    over = 0.0
    if sketch.title:
        title_width = style.text_width(sketch.title) * 1.15 + 2 * style.margin
        over = max(0.0, (title_width - width) / 2)

    def flip(value: float) -> float:
        return height - value

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="pylea" '
        f'viewBox="{_n(-over)} 0 {_n(width + 2 * over)} {_n(height + band)}" '
        f'width="{_n((width + 2 * over) * style.scale)}" '
        f'height="{_n((height + band) * style.scale)}">',
        f"<style>{_STYLE.format(stroke=1.1, font=_FONT, **_palette(theme))}</style>",
    ]

    wires = "".join(f'<path class="wire" d="{_shape(wire.path, 0.0, 0.0, flip)}"/>'
                    for wire in placement.routes)
    wires += "".join(f'<path class="solid" d="{_shape(wire.head, 0.0, 0.0, flip)}"/>'
                     for wire in placement.routes if wire.head)
    parts.append(f"<g>{wires}</g>")

    for i, node in enumerate(sketch.nodes):
        glyph = placement.glyphs[i]
        if not glyph.shapes and not glyph.labels:
            continue
        x, y = placement.x[i], placement.y[i]
        drawn = [f'<path class="{"solid" if shape.solid else "outline"}" '
                 f'd="{_shape(shape.path, x, y, flip)}"/>'
                 for shape in glyph.shapes]
        drawn += [f'<text x="{_n(x + label.x)}" y="{_n(flip(y + label.y))}" '
                  f'font-size="{_n(label.size)}">{_escape(label.text)}</text>'
                  for label in glyph.labels]
        parts.append("".join(drawn))

    if sketch.title:
        parts.append(f'<text x="{_n(width / 2)}" y="{_n(height + band / 2)}" '
                     f'font-size="{_n(style.font_size * 1.15)}">'
                     f"{_escape(sketch.title)}</text>")

    parts.append("</svg>")
    return "\n".join(parts)


def _palette(theme: str) -> dict:
    """The colours to define, and the override to follow the reader's setting where asked."""
    if theme == "dark":
        return {"palette": _DARK, "override": ""}
    return {
        "palette": _LIGHT,
        "override": ("@media (prefers-color-scheme:dark){.pylea{" + _DARK + "}}"
                     if theme == "auto" else ""),
    }


def _escape(text: str) -> str:
    """`text` with the three characters that cannot appear literally in XML replaced."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
