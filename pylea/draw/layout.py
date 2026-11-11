"""Where each node and wire of a sketch goes.

`place` assigns a point to every node and a route to every wire, reading bottom to top: the
inputs are on the lowest rank and the outputs on the highest.  The result is measured in the
units the glyphs are, with the origin at the bottom left.  A sketch marked `downward` is
turned over at the end, so that it reads from the top down instead.

The steps are the usual layered ones, less the two a straight-line definition does not need.
Ranking is a longest path, exact here because a `pylea.ir.body.Body` is already
topologically sorted and acyclic.  A wire crossing more than one rank is broken into one
dummy node per rank it passes.  Ranks are then ordered by repeated median sweeps, keeping
the ordering that crosses least.  Coordinates come from `pylea.draw.coordinates`, which
places the dummies carrying one wire as a single block so that the wire is straight.

Argument order shows in the drawing because a port's offset within its box enters both the
ordering and the coordinates.

A wire crossing several ranks is drawn straight, the coordinates placing its dummies as one
block, so it bends only where it meets the glyph at either end.

A wire of a directed sketch ends in an arrow head, and the ranks of such a picture are set
apart far enough for the head to stand in the gap rather than fill it.

An operation reading nothing, a coin flip or a constant, goes two ranks below the operation
that first reads it rather than at the bottom.  Pinning one to the bottom rank stretches
every wire leaving it and makes the lowest rank as wide as the number of coin flips in the
definition.

    >>> from pylea import DPP, Bool, flip
    >>> from pylea.draw.sketch import sketch_of
    >>> @DPP
    ... def either() -> Bool:
    ...     return flip(0.5) | flip(0.25)
    >>> p = place(sketch_of(either.program()))
    >>> len(p.routes), p.height > p.width / 4
    (6, True)
"""
from __future__ import annotations

from dataclasses import dataclass

from pylea.draw.coordinates import coordinates
from pylea.draw.glyph import Glyph, GlyphKind, Style, entry_point, glyph_of
from pylea.draw.sketch import Sketch

# Sweeps of the ordering step, and the node budget that cuts them back.  One sweep costs a
# pass over every node, so a sketch of tens of thousands of nodes still settles in about a
# second.
_MAX_ORDER_SWEEPS = 8
_SWEEP_BUDGET = 400_000
_COORD_SWEEPS = 8


@dataclass(frozen=True, slots=True)
class Wire:
    """One wire: the points it runs through, and the path drawn through them.

    The path is the straight run between each pair of points, with the corners rounded off,
    so the wire never leaves the line it is following.  A curve fitted through the points
    instead would swing wide of a corner where the two sides of it are of very different
    lengths, which is exactly the corner a wire crossing many ranks turns.

    `head` is the outline of the arrow head a wire of a directed sketch ends in, and is
    empty otherwise.  Both it and `path` hold the same segments a glyph outline does, so
    both backends draw them with what they already have; the head is a closed triangle, to
    be filled rather than stroked.
    """

    points: tuple[tuple[float, float], ...]
    path: tuple[tuple, ...]
    head: tuple[tuple, ...] = ()


@dataclass(frozen=True, slots=True)
class Placement:
    """The glyph and centre of every node, the route of every wire, and the total extent."""

    glyphs: tuple[Glyph, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    routes: tuple[Wire, ...]
    width: float
    height: float


def place(sketch: Sketch, style: Style | None = None,
          sweeps: int | None = None) -> Placement:
    """`sketch` laid out, reading bottom to top.

    `sweeps` bounds the ordering step; left out, it follows the size of the picture.
    """
    style = style or Style()
    glyphs = tuple(glyph_of(n.kind, n.label, n.n_in, n.n_out, style) for n in sketch.nodes)

    g = _Layered(sketch, _ranks(sketch), glyphs, style)
    _order(g, sweeps)
    x = coordinates(g, style)
    y = _ordinates(g, style, x, sketch.directed)
    placement = _measure(sketch, g, glyphs, x, y, style)
    return _turned_over(placement) if sketch.downward else placement


# -- ranking ---------------------------------------------------------------------------

def _ranks(sketch: Sketch) -> list[int]:
    """The rank of every node: 0 at the bottom, one above the highest rank feeding it."""
    n = len(sketch.nodes)
    succs: list[list[int]] = [[] for _ in range(n)]
    indegree = [0] * n
    for e in sketch.edges:
        succs[e.src].append(e.dst)
        indegree[e.dst] += 1

    rank = [0] * n
    ready = [i for i in range(n) if indegree[i] == 0]
    while ready:
        u = ready.pop()
        for v in succs[u]:
            rank[v] = max(rank[v], rank[u] + 1)
            indegree[v] -= 1
            if indegree[v] == 0:
                ready.append(v)

    outs = {i for i, node in enumerate(sketch.nodes)
            if node.kind is GlyphKind.BOUNDARY and not succs[i]}
    if outs:
        # Every output wire reaches one line, so the top of the picture is straight.
        top = max((r for i, r in enumerate(rank) if i not in outs), default=-1) + 1
        for i in outs:
            rank[i] = top

    for i, node in enumerate(sketch.nodes):
        if node.op is None or node.n_in:
            continue
        readers = [rank[w] for d in succs[i] for w in succs[d]]
        if readers:
            rank[i] = max(0, min(readers) - 2)
            for d in succs[i]:
                rank[d] = rank[i] + 1
    return rank


# -- the graph with dummies ------------------------------------------------------------

class _Layered:
    """The sketch with a dummy node on every rank a wire passes through.

    Nodes below `n_real` are the sketch's own; the rest carry a wire across a rank.  `chains`
    holds, per sketch edge, the nodes it runs through, its own ends included.

    A neighbour is recorded together with two offsets of the port the wire meets it at: one
    as a fraction of the box, which the ordering adds to an integer position, and one as a
    distance, which the coordinates add to an x.
    """

    __slots__ = ("rank", "layers", "pos", "chains", "width", "height", "is_dummy",
                 "is_edge", "n_real", "preds", "succs", "by_rank")

    def __init__(self, sketch: Sketch, rank: list[int], glyphs, style: Style) -> None:
        self.n_real = len(sketch.nodes)
        self.rank = list(rank)
        self.width = [g.width for g in glyphs]
        self.height = [g.height for g in glyphs]
        self.is_dummy = [False] * self.n_real
        self.is_edge = [n.kind is GlyphKind.BOUNDARY for n in sketch.nodes]
        self.chains: list[list[int]] = []
        self.preds: list[list[tuple[int, float, float]]] = [[] for _ in range(self.n_real)]
        self.succs: list[list[tuple[int, float, float]]] = [[] for _ in range(self.n_real)]
        self.by_rank: dict[int, list[tuple[int, float, int, float]]] = {}

        def add(r: int) -> int:
            self.rank.append(r)
            self.width.append(0.0)
            self.height.append(0.0)
            self.is_dummy.append(True)
            self.is_edge.append(False)
            self.preds.append([])
            self.succs.append([])
            return len(self.rank) - 1

        def link(u, off_u, dx_u, v, off_v, dx_v) -> None:
            self.succs[u].append((v, off_v, dx_v))
            self.preds[v].append((u, off_u, dx_u))
            self.by_rank.setdefault(self.rank[u], []).append((u, off_u, v, off_v))

        for e in sketch.edges:
            src, dst = sketch.nodes[e.src], sketch.nodes[e.dst]
            off_src = (e.src_port + 1) / (src.n_out + 1) - 0.5
            off_dst = (e.dst_port + 1) / (dst.n_in + 1) - 0.5
            dx_src = glyphs[e.src].out_ports[e.src_port][0]
            dx_dst = glyphs[e.dst].in_ports[e.dst_port][0]

            chain = [e.src]
            for r in range(rank[e.src] + 1, rank[e.dst]):
                chain.append(add(r))
            chain.append(e.dst)
            last = len(chain) - 1
            for k in range(last):
                link(chain[k], off_src if k == 0 else 0.0, dx_src if k == 0 else 0.0,
                     chain[k + 1], off_dst if k + 1 == last else 0.0,
                     dx_dst if k + 1 == last else 0.0)
            self.chains.append(chain)

        self.layers: dict[int, list[int]] = {}
        for i, r in enumerate(self.rank):
            self.layers.setdefault(r, []).append(i)
        self.pos = [0] * len(self.rank)
        self.renumber()

    def renumber(self) -> None:
        for nodes in self.layers.values():
            for i, node in enumerate(nodes):
                self.pos[node] = i


# -- ordering --------------------------------------------------------------------------

def _order(g: _Layered, sweeps: int | None) -> None:
    """Order every rank so that few wires cross, by repeated median sweeps."""
    if sweeps is None:
        sweeps = max(1, min(_MAX_ORDER_SWEEPS, _SWEEP_BUDGET // max(len(g.rank), 1)))
    ranks = sorted(g.layers)

    for r in ranks:                                 # one pass upward, to start from
        _sort_by(g, r, g.preds)                     # something better than the build order

    best, fewest = [list(g.layers[r]) for r in ranks], _crossings(g)
    for _ in range(sweeps):
        for r in ranks:
            _sort_by(g, r, g.preds)
        for r in reversed(ranks):
            _sort_by(g, r, g.succs)
        crossings = _crossings(g)
        if crossings < fewest:
            best, fewest = [list(g.layers[r]) for r in ranks], crossings

    for r, nodes in zip(ranks, best):
        g.layers[r] = nodes
    g.renumber()


def _sort_by(g: _Layered, r: int, adjacency) -> None:
    """Sort rank `r` by the median position of each node's neighbours in `adjacency`."""
    nodes = g.layers[r]

    def key(node: int) -> float:
        near = sorted(g.pos[m] + off for m, off, _ in adjacency[node])
        return near[len(near) // 2] if near else g.pos[node]

    nodes.sort(key=key)                             # stable, so a node with no neighbours
    for i, node in enumerate(nodes):                # keeps its place
        g.pos[node] = i


def _crossings(g: _Layered) -> int:
    """The number of pairs of wires that cross, summed over adjacent ranks."""
    total = 0
    for segments in g.by_rank.values():
        ordered = sorted((g.pos[u] + ou, g.pos[v] + ov) for u, ou, v, ov in segments)
        total += _inversions([v for _, v in ordered])
    return total


def _inversions(values: list[float]) -> int:
    """The number of out-of-order pairs in `values`, counted with a Fenwick tree.

        >>> _inversions([0.0, 1.0, 2.0]), _inversions([2.0, 1.0, 0.0])
        (0, 3)
    """
    n = len(values)
    place_of = [0] * n
    for i, k in enumerate(sorted(range(n), key=values.__getitem__)):
        place_of[k] = i + 1

    tree = [0] * (n + 1)
    total = 0
    for i in reversed(range(n)):
        j = place_of[i] - 1
        while j > 0:
            total += tree[j]
            j -= j & -j
        j = place_of[i]
        while j <= n:
            tree[j] += 1
            j += j & -j
    return total


# -- coordinates -----------------------------------------------------------------------

def _separation(g: _Layered, a: int, b: int, style: Style) -> float:
    """The least distance between the centres of neighbours `a` and `b` on one rank.

    An interface marker is drawn past either end of the picture rather than on the rank it
    was ranked onto, so it can never meet anything but another marker and needs no room
    against one.  Charging it the width of its own label would push the wire it ends off
    the vertical.
    """
    if g.is_edge[a] != g.is_edge[b]:
        return 0.0
    gap = style.node_sep * (0.6 if g.is_dummy[a] and g.is_dummy[b] else 1.0)
    return (g.width[a] + g.width[b]) / 2 + gap


# Which of the wires leaving a rank the gap above it is set by.  The widest would let one
# wire stretch a whole picture; the middle one would leave most of a busy rank flat.
_SPREAD_QUANTILE = 0.8


def _travel(g: _Layered, x: list[float]) -> dict[int, float]:
    """How far the wires leaving each rank reach across, at `_SPREAD_QUANTILE`."""
    reach = {}
    for r, segments in g.by_rank.items():
        spans = sorted(abs(x[u] - x[v]) for u, _, v, _ in segments)
        reach[r] = spans[min(int(len(spans) * _SPREAD_QUANTILE), len(spans) - 1)]
    return reach


def _ordinates(g: _Layered, style: Style, x: list[float],
               directed: bool = False) -> list[float]:
    """A y for every node: the ranks stacked upward, each as tall as its tallest node.

    The gap above a rank follows how far the wires leaving it reach across, so that they
    meet the next rank at a readable angle.  A fixed gap makes the angle depend on the width
    of the picture: on a wide rank the wires flatten until the crossings between them cannot
    be told apart, and no amount of reordering helps, since the crossings are in the graph
    rather than in the arrangement.  `Style.slope` is the rise over run aimed for and
    `Style.max_stretch` bounds what one gap may cost.

    An arrow head stands in the gap rather than beside it, so a `directed` picture adds its
    length to the least gap; without that the head takes most of a short wire and the two
    ranks read as joined rather than as one pointing at the other.
    """
    ranks = sorted(g.layers)
    tall = {r: max((g.height[n] for n in g.layers[r]), default=0.0) for r in ranks}
    reach = _travel(g, x)
    least = style.rank_gap + (style.arrow_head if directed else 0.0)

    at, level = 0.0, {}
    for k, r in enumerate(ranks):
        if not k:
            at = tall[r] / 2
        else:
            below = ranks[k - 1]
            gap = min(max(least, style.slope * reach.get(below, 0.0)),
                      style.rank_gap * style.max_stretch)
            at += tall[below] / 2 + gap + tall[r] / 2
        level[r] = at
    return [level[r] for r in g.rank]


# -- routes and extent -----------------------------------------------------------------

def _measure(sketch: Sketch, g: _Layered, glyphs, x, y, style: Style) -> Placement:
    """The placement, shifted so that the whole picture sits inside its margins.

    An interface marker is moved to the edge of the picture first.  Left on the centre line
    of its rank, a dangling wire would stop level with whatever else shares that rank
    instead of running to the boundary.
    """
    drawn = [i for i, node in enumerate(sketch.nodes) if node.kind is not GlyphKind.BOUNDARY]
    floor = min((y[i] - glyphs[i].height / 2 for i in drawn), default=0.0)
    ceiling = max((y[i] + glyphs[i].height / 2 for i in drawn), default=0.0)
    for i, node in enumerate(sketch.nodes):
        if node.kind is GlyphKind.BOUNDARY:
            y[i] = floor - style.rank_gap if g.succs[i] else ceiling + style.rank_gap

    routes = [_wire(_route(sketch, e, chain, glyphs, x, y),
                    glyphs[e.src].flat, glyphs[e.dst].flat, style, sketch.directed)
              for e, chain in zip(sketch.edges, g.chains)]

    left = min((x[i] - glyphs[i].width / 2 for i in range(g.n_real)), default=0.0)
    right = max((x[i] + glyphs[i].width / 2 for i in range(g.n_real)), default=0.0)
    for wire in routes:
        left = min(left, min(px for px, _ in wire.points))
        right = max(right, max(px for px, _ in wire.points))
    top = max((y[i] + glyphs[i].height / 2 for i in range(g.n_real)), default=0.0)
    bottom = min((y[i] - glyphs[i].height / 2 for i in range(g.n_real)), default=0.0)

    dx, dy = style.margin - left, style.margin - bottom
    return Placement(
        tuple(glyphs),
        tuple(x[i] + dx for i in range(g.n_real)),
        tuple(y[i] + dy for i in range(g.n_real)),
        tuple(_mapped(wire, lambda px, py: (px + dx, py + dy)) for wire in routes),
        right - left + 2 * style.margin,
        top - bottom + 2 * style.margin,
    )



def _route(sketch: Sketch, e, chain, glyphs, x, y) -> list[tuple[float, float]]:
    """The points one wire runs through, its two ends included.

    A wire of an undirected picture runs from port to port.  A wire of a directed one is
    aimed at the centre of the glyph at either end, at the offset of its port, and stopped
    where it meets the outline: a wire arriving at a shallow angle then meets the side of
    the box facing it rather than lying along the edge its port is on, which is where its
    arrow head would otherwise sit flat against the outline.  Aiming from centre to centre
    rather than from port to port keeps the two ends on one line.
    """
    src, dst = (x[e.src], y[e.src]), (x[e.dst], y[e.dst])
    from_port = glyphs[e.src].out_ports[e.src_port]
    to_port = glyphs[e.dst].in_ports[e.dst_port]
    middle = [(x[d], y[d]) for d in chain[1:-1]]
    if not sketch.directed:
        return [(src[0] + from_port[0], src[1] + from_port[1]), *middle,
                (dst[0] + to_port[0], dst[1] + to_port[1])]

    leaves, enters = (src[0] + from_port[0], src[1]), (dst[0] + to_port[0], dst[1])
    return [entry_point(glyphs[e.src], src, leaves, middle[0] if middle else enters),
            *middle,
            entry_point(glyphs[e.dst], dst, enters, middle[-1] if middle else leaves)]


def _mapped(wire: Wire, move) -> Wire:
    """`wire` with `move` applied to every point of it, its path and its head included."""
    def moved(seg):
        points = [move(seg[i], seg[i + 1]) for i in range(1, len(seg), 2)]
        return (seg[0],) + tuple(value for point in points for value in point)

    return Wire(tuple(move(px, py) for px, py in wire.points),
                tuple(moved(seg) for seg in wire.path),
                tuple(moved(seg) for seg in wire.head))


def _turned_over(placement: Placement) -> Placement:
    """`placement` mirrored about its own horizontal centre line.

    What a picture reading from the top down is: the ranking, the ordering and the routing
    are the same either way round, and turning the result over sends a wire out of the
    lower edge of the node it starts at, into the upper edge of the node it ends at, and
    its arrow head with it.  The glyphs are not turned over, so every glyph of such a
    picture must be symmetric about its own horizontal axis, which the boxes of
    `pylea.draw.dependency` are.
    """
    height = placement.height

    def over(x: float, y: float) -> tuple[float, float]:
        return x, height - y

    return Placement(placement.glyphs, placement.x,
                     tuple(height - value for value in placement.y),
                     tuple(_mapped(wire, over) for wire in placement.routes),
                     placement.width, height)


def _wire(points, flat_start: bool, flat_end: bool, style: Style,
          directed: bool = False) -> Wire:
    """One wire through `points`, leaving a flat glyph edge at a right angle.

    A short length is added at an end that meets a flat edge, so that the wire leaves the
    edge before it turns.  Points that coincide are dropped, since a corner between them has
    no direction to turn through.  `directed` puts an arrow head on the end.
    """
    run = list(points)
    if flat_start:
        run.insert(1, (run[0][0], run[0][1] + _lead(run[0], run[1], style)))
    if flat_end:
        run.insert(-1, (run[-1][0], run[-1][1] - _lead(run[-1], run[-2], style)))

    kept = [run[0]]
    for point in run[1:]:
        if abs(point[0] - kept[-1][0]) > 1e-9 or abs(point[1] - kept[-1][1]) > 1e-9:
            kept.append(point)
    return Wire(tuple(points), _rounded(kept, style.corner),
                _arrow_head(kept, style) if directed else ())


def _arrow_head(points, style: Style) -> tuple[tuple, ...]:
    """A triangle at the end of `points`, its tip there and its axis along the last run.

    Empty where the wire has no length to take a direction from.  The head lies on the
    wire rather than past its end, so that its tip meets the glyph exactly where the wire
    would have.
    """
    tip = points[-1]
    along = next((d for point in reversed(points[:-1])
                  if (d := _towards(point, tip)) is not None), None)
    if along is None:
        return ()
    length = style.arrow_head
    base = (tip[0] - along[0] * length, tip[1] - along[1] * length)
    across = (-along[1] * length * 0.35, along[0] * length * 0.35)
    return (("M", tip[0], tip[1]),
            ("L", base[0] + across[0], base[1] + across[1]),
            ("L", base[0] - across[0], base[1] - across[1]),
            ("Z",))


def _lead(end, next_point, style: Style) -> float:
    """How far a wire runs straight out of a flat edge before it may turn."""
    return min(style.rank_gap * 0.6, abs(next_point[1] - end[1]) * 0.35)


def _rounded(points, radius: float) -> tuple[tuple, ...]:
    """The path through `points`, each corner replaced by a curve inside the turn.

    The curve is a cubic with both of its control points at the corner, so it stays within
    the two straight lines meeting there and cannot swing past either of them.
    """
    if len(points) < 2:
        return ()

    path: list[tuple] = [("M", points[0][0], points[0][1])]
    for before, corner, after in zip(points, points[1:], points[2:]):
        into, out = _towards(corner, before), _towards(corner, after)
        if into is None or out is None or abs(into[0] * out[1] - into[1] * out[0]) < 1e-9:
            continue                                    # the wire runs straight through
        cut = min(radius, _apart(before, corner) / 2, _apart(corner, after) / 2)
        path.append(("L", corner[0] + into[0] * cut, corner[1] + into[1] * cut))
        path.append(("C", corner[0], corner[1], corner[0], corner[1],
                     corner[0] + out[0] * cut, corner[1] + out[1] * cut))
    path.append(("L", points[-1][0], points[-1][1]))
    return tuple(path)


def _towards(origin, point):
    """The unit vector from `origin` to `point`, or nothing where they coincide."""
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    length = (dx * dx + dy * dy) ** 0.5
    return None if length < 1e-9 else (dx / length, dy / length)


def _apart(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
