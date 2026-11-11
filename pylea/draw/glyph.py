"""The symbol drawn for each kind of operation.

A `Glyph` is a size, a set of port positions, and an outline, all in local coordinates with
the origin at the centre and the y axis pointing up.  Outlines are built from straight lines
and cubic Béziers only, so that every backend can draw them without a curve primitive of its
own; a circular arc is approximated by quarter-circle cubics, which is exact to within a
thousandth of the radius.

`GlyphKind` names the symbols.  The logic gates take their standard shapes, a coin flip is a
circle carrying its probability, and conditioning is an eye.  A variable of a DPP, which
is what `pylea.draw.dependency` draws, is a rounded rectangle carrying its name, with a
second outline inside the first where the DPP is given the variable rather than computing
it.  Everything else is a box carrying a label.

`Style` holds the sizes.  Text widths are estimated as a fixed fraction of the font size per
character, which holds because the backends set a monospace family.

    >>> g = glyph_of(GlyphKind.AND, "", 2, 1, Style())
    >>> g.width, g.height, g.in_ports, g.out_ports
    (26.0, 20.0, ((-5.0, -10.0), (5.0, -10.0)), ((0.0, 10.0),))
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import sqrt

# Control point offset, as a fraction of the radius, for a quarter circle drawn as a cubic.
_KAPPA = 4.0 / 3.0 * (sqrt(2.0) - 1.0)


class GlyphKind(Enum):
    """Which symbol an operation is drawn as."""

    AND = auto()
    OR = auto()
    NOT = auto()
    FLIP = auto()           # a circle carrying its probability
    OBSERVE = auto()        # an eye
    BOX = auto()            # anything else: a labelled rectangle
    CALL = auto()           # another body, drawn as a labelled rectangle
    DOT = auto()            # a wire
    BOUNDARY = auto()       # where a dangling wire meets the edge of the picture
    VARIABLE = auto()       # a variable of a DPP, drawn as a rounded rectangle
    PARAMETER = auto()      # a variable a DPP is given, drawn with a doubled outline


@dataclass(frozen=True, slots=True)
class Style:
    """The sizes every glyph and the layout are built from, and the scale the picture is shown at."""

    font_size: float = 9.0
    dot_radius: float = 2.6
    gate_width: float = 26.0
    gate_height: float = 20.0
    box_height: float = 20.0
    port_sep: float = 10.0          # distance between adjacent ports on one edge
    port_inset: float = 4.0         # clearance between the outermost port and the corner
    node_sep: float = 12.0          # clearance between neighbours on one rank
    rank_gap: float = 10.0          # least clearance between one rank and the next
    slope: float = 0.45             # rise over run a wire between ranks aims for
    max_stretch: float = 12.0       # most one gap may grow, as a multiple of `rank_gap`
    corner: float = 11.0            # radius a wire turns a corner through
    arrow_head: float = 7.0         # length of the head a wire of a directed picture ends in
    margin: float = 16.0
    # Width of one character, as a fraction of the font size.  Above the advance of the
    # monospace faces both backends set, 0.602 for Menlo and 0.525 for LaTeX's
    # typewriter, so a box is roomy rather than overflowing its label.
    text_aspect: float = 0.6
    # Factor the SVG element is displayed at.  The layout and every size above are in
    # drawing units and stay as they are; only the displayed size changes.  The TikZ output
    # is set in points and does not use it.
    scale: float = 1.0

    def text_width(self, text: str) -> float:
        """The width `text` takes at this style's font size."""
        return self.text_aspect * self.font_size * len(text)


# A path segment, in the four forms every backend can draw.
#   ("M", x, y) ("L", x, y) ("C", x1, y1, x2, y2, x, y) ("Z",)
type Seg = tuple


@dataclass(frozen=True, slots=True)
class Shape:
    """A closed outline.  `filled` paints it with the background, hiding the wires behind."""

    path: tuple[Seg, ...]
    filled: bool = True
    solid: bool = False             # painted in the foreground colour rather than outlined


@dataclass(frozen=True, slots=True)
class Label:
    """Text centred on a point."""

    x: float
    y: float
    text: str
    size: float


@dataclass(frozen=True, slots=True)
class Glyph:
    """The symbol for one node: its extent, where wires attach, and what to draw.

    `flat` marks a glyph whose ports lie on an edge, which a wire must meet at a right
    angle.  A dot has no edge, so a wire runs through it without being bent.

    `corner` is the radius the outline is rounded by, which `entry_point` needs to stop a
    wire on it; zero is a square corner, and the value is ignored for a glyph whose
    outline is not a rectangle.
    """

    width: float
    height: float
    in_ports: tuple[tuple[float, float], ...]
    out_ports: tuple[tuple[float, float], ...]
    shapes: tuple[Shape, ...] = ()
    labels: tuple[Label, ...] = ()
    flat: bool = True
    corner: float = 0.0


def _spread(n: int, y: float, style: Style) -> tuple[tuple[float, float], ...]:
    """`n` ports evenly spaced about the origin at height `y`."""
    return tuple(((i - (n - 1) / 2.0) * style.port_sep, y) for i in range(n))


def _port_span(n: int, style: Style) -> float:
    """The width `n` ports need, including the clearance at both ends."""
    return max(n - 1, 0) * style.port_sep + 2.0 * style.port_inset


def _box_width(label: str, n_in: int, n_out: int, style: Style) -> float:
    """The width a labelled rectangle needs to hold its label and its ports."""
    return max(style.gate_width, style.text_width(label) + style.font_size,
               _port_span(n_in, style), _port_span(n_out, style))


def _circle(cx: float, cy: float, r: float) -> tuple[Seg, ...]:
    """A full circle, as four quarter-circle cubics."""
    k = _KAPPA * r
    return (
        ("M", cx + r, cy),
        ("C", cx + r, cy + k, cx + k, cy + r, cx, cy + r),
        ("C", cx - k, cy + r, cx - r, cy + k, cx - r, cy),
        ("C", cx - r, cy - k, cx - k, cy - r, cx, cy - r),
        ("C", cx + k, cy - r, cx + r, cy - k, cx + r, cy),
        ("Z",),
    )


def _rectangle(w: float, h: float) -> tuple[Seg, ...]:
    return (("M", -w / 2, -h / 2), ("L", w / 2, -h / 2),
            ("L", w / 2, h / 2), ("L", -w / 2, h / 2), ("Z",))


def _rounded_rectangle(w: float, h: float, r: float) -> tuple[Seg, ...]:
    """A rectangle of `w` by `h` whose corners are quarter circles of radius `r`."""
    r = min(r, w / 2, h / 2)
    x, y, k = w / 2, h / 2, _KAPPA * r
    return (
        ("M", -x + r, -y), ("L", x - r, -y),
        ("C", x - r + k, -y, x, -y + r - k, x, -y + r),
        ("L", x, y - r),
        ("C", x, y - r + k, x - r + k, y, x - r, y),
        ("L", -x + r, y),
        ("C", -x + r - k, y, -x, y - r + k, -x, y - r),
        ("L", -x, -y + r),
        ("C", -x, -y + r - k, -x + r - k, -y, -x + r, -y),
        ("Z",),
    )


def _and_gate(w: float, h: float) -> tuple[Seg, ...]:
    """A flat bottom and straight sides, closed by a semicircle of radius ``w / 2``."""
    r = w / 2
    k = _KAPPA * r
    shoulder = h / 2 - r                    # where the straight sides give way to the arc
    return (
        ("M", -r, -h / 2), ("L", r, -h / 2), ("L", r, shoulder),
        ("C", r, shoulder + k, k, h / 2, 0.0, h / 2),
        ("C", -k, h / 2, -r, shoulder + k, -r, shoulder),
        ("Z",),
    )


def _or_gate(w: float, h: float) -> tuple[Seg, ...]:
    """A bottom curving up into the body, and sides sweeping to a point."""
    r, t = w / 2, h / 2
    return (
        ("M", -r, -t),
        ("C", -0.46 * w, -0.28 * h, 0.46 * w, -0.28 * h, r, -t),      # the concave bottom
        ("C", r, -0.05 * h, 0.35 * w, 0.3 * h, 0.0, t),               # the right side
        ("C", -0.35 * w, 0.3 * h, -r, -0.05 * h, -r, -t),             # the left side
        ("Z",),
    )


def _eye(w: float, h: float) -> tuple[Seg, ...]:
    """Two arcs meeting at the corners, the outline of an eye."""
    r, t = w / 2, h / 2
    return (
        ("M", -r, 0.0),
        ("C", -0.27 * w, 1.33 * t, 0.27 * w, 1.33 * t, r, 0.0),
        ("C", 0.27 * w, -1.33 * t, -0.27 * w, -1.33 * t, -r, 0.0),
        ("Z",),
    )


def glyph_of(kind: GlyphKind, label: str, n_in: int, n_out: int, style: Style) -> Glyph:
    """The glyph for an operation of `kind` with `n_in` arguments and `n_out` results.

    A box widens to hold its label and its ports, so a call to a long name stays legible:

        >>> narrow = glyph_of(GlyphKind.CALL, "f", 1, 1, Style())
        >>> wide = glyph_of(GlyphKind.CALL, "positive_test", 1, 1, Style())
        >>> narrow.width < wide.width
        True
    """
    if kind is GlyphKind.BOUNDARY:
        # A marker carries the one port the wire ending there needs, which says which end of
        # the picture it is at: a wire leaves an input marker and enters an output one.  The
        # label goes on the far side, clear of the picture.
        ports = (((0.0, 0.0),) if n_in else (), ((0.0, 0.0),) if n_out else ())
        if not label:
            return Glyph(0.0, 0.0, *ports, flat=False)
        away = style.font_size * 0.95 * (-1.0 if n_out else 1.0)
        return Glyph(style.text_width(label), 2 * (abs(away) + style.font_size / 2), *ports,
                     labels=(Label(0.0, away, label, style.font_size),), flat=False)

    if kind is GlyphKind.DOT:
        r = style.dot_radius
        return Glyph(2 * r, 2 * r, ((0.0, 0.0),), ((0.0, 0.0),),
                     shapes=(Shape(_circle(0.0, 0.0, r), filled=True, solid=True),),
                     flat=False)

    if kind in (GlyphKind.AND, GlyphKind.OR):
        w = max(style.gate_width, _port_span(n_in, style))
        h = style.gate_height
        path = _and_gate(w, h) if kind is GlyphKind.AND else _or_gate(w, h)
        return Glyph(w, h, _spread(n_in, -h / 2, style), _spread(n_out, h / 2, style),
                     shapes=(Shape(path),))

    if kind is GlyphKind.NOT:
        w, h = style.gate_width * 0.7, style.gate_height
        bubble = h * 0.175
        apex = h / 2 - 2 * bubble
        return Glyph(w, h, ((0.0, -h / 2),), ((0.0, h / 2),), shapes=(
            Shape((("M", -w / 2, -h / 2), ("L", w / 2, -h / 2), ("L", 0.0, apex), ("Z",))),
            Shape(_circle(0.0, apex + bubble, bubble)),
        ))

    if kind is GlyphKind.OBSERVE:
        w, h = style.gate_width, style.gate_height * 0.6
        pupil = h * 0.29
        return Glyph(w, h, ((0.0, -h / 2),), (), shapes=(
            Shape(_eye(w, h)),
            Shape(_circle(0.0, 0.0, pupil), filled=True, solid=True),
        ))

    if kind is GlyphKind.FLIP:
        r = max(style.gate_height / 2,
                style.text_width(label) / 2 + style.font_size * 0.25)
        return Glyph(2 * r, 2 * r, (), _spread(n_out, r, style),
                     shapes=(Shape(_circle(0.0, 0.0, r)),),
                     labels=(Label(0.0, 0.0, label, style.font_size),) if label else ())

    w, h = _box_width(label, n_in, n_out, style), style.box_height
    text = (Label(0.0, 0.0, label, style.font_size),) if label else ()

    if kind in (GlyphKind.VARIABLE, GlyphKind.PARAMETER):
        corner = h * 0.35
        outlines = [Shape(_rounded_rectangle(w, h, corner))]
        if kind is GlyphKind.PARAMETER:
            # A second outline within the first, which is what marks a variable the
            # DPP is given rather than one it computes.  Both are filled, so the inner
            # one is drawn after the outer one.
            inset = h * 0.13
            outlines.append(
                Shape(_rounded_rectangle(w - 2 * inset, h - 2 * inset, corner - inset)))
        # The ports are on an edge, but the glyph is not flat: a wire between two
        # variables is drawn as the straight line the arrow head at its end points along,
        # rather than meeting the edge at a right angle as a wire of a circuit does.
        return Glyph(w, h, _spread(n_in, -h / 2, style), _spread(n_out, h / 2, style),
                     shapes=tuple(outlines), labels=text, flat=False, corner=corner)

    return Glyph(w, h, _spread(n_in, -h / 2, style), _spread(n_out, h / 2, style),
                 shapes=(Shape(_rectangle(w, h)),), labels=text)


# Bisection steps to find where a wire meets an outline.  Thirty halvings put the point
# within a millionth of a picture a million units across, which no backend can draw apart.
_CLIP_STEPS = 30


def entry_point(glyph: Glyph, centre, aim, coming_from) -> tuple[float, float]:
    """Where the line from `coming_from` to `aim` first meets the outline of `glyph`.

    `centre` is where the glyph is drawn and `aim` a point within it, the two being given
    in the coordinates of the picture.  A wire drawn to this point rather than to a port on
    one edge meets whichever side of the glyph faces where it comes from, so that its arrow
    head stands off the outline instead of lying along it, which is what a wire arriving at
    a shallow angle would otherwise do.

    `aim` itself is returned where the line meets nothing, which is where `coming_from` is
    within the glyph already or `aim` is not.  The outline is a rectangle with corners of
    radius `Glyph.corner`, which is convex, so the points of the line within it are one
    run and bisection finds where that run starts.

        >>> box = glyph_of(GlyphKind.VARIABLE, "x", 1, 1, Style())
        >>> round(entry_point(box, (0.0, 0.0), (0.0, 0.0), (0.0, -50.0))[1], 3)
        -10.0
    """
    a, b = glyph.width / 2, glyph.height / 2
    r = min(glyph.corner, a, b)
    inside = _within(aim[0] - centre[0], aim[1] - centre[1], a, b, r)
    if not inside or _within(coming_from[0] - centre[0], coming_from[1] - centre[1], a, b, r):
        return aim

    outside, within = 0.0, 1.0
    for _ in range(_CLIP_STEPS):
        half = (outside + within) / 2
        point = (coming_from[0] + (aim[0] - coming_from[0]) * half,
                 coming_from[1] + (aim[1] - coming_from[1]) * half)
        if _within(point[0] - centre[0], point[1] - centre[1], a, b, r):
            within = half
        else:
            outside = half
    return (coming_from[0] + (aim[0] - coming_from[0]) * within,
            coming_from[1] + (aim[1] - coming_from[1]) * within)


def _within(x: float, y: float, a: float, b: float, r: float) -> bool:
    """Whether ``(x, y)`` lies within a rectangle of half-widths `a` and `b`, corners `r`.

    Measured from the centre of the rectangle.  A point past the corner in both directions
    is within it where it is within the corner's own circle.
    """
    past_x, past_y = abs(x) - (a - r), abs(y) - (b - r)
    if past_x <= 0.0 or past_y <= 0.0:
        return abs(x) <= a and abs(y) <= b
    return past_x * past_x + past_y * past_y <= r * r
