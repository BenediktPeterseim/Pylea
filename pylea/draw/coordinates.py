"""Horizontal coordinates for a layered graph.

`coordinates` assigns an x to every node of a `pylea.draw.layout` graph by the method of
Brandes and Koepf [1], with the correction of [2], which is what keeps a wire crossing many
ranks straight.

The idea is to place blocks rather than nodes.  A node is aligned with the median of the
neighbours below it wherever two such alignments would not cross, which chains the dummies
carrying one wire into a single block; every node of a block then takes one x, so the wire
is drawn as a straight line.  Compaction places the blocks as close together as the order
within each rank allows.

Compaction has two phases.  The first places each block against the blocks to its left
that belong to its class, the blocks reached by going left from it.  The second shifts
whole classes: two neighbours on a rank in different classes bound how far right the left
class may go, and each class is shifted as far right as all of its bounds allow, a class
being solved only after the classes to its right.  Shifting a class while its blocks are
placed instead, against a neighbouring class as if that class kept its place, lets shifts
fail to add up along a chain of classes, and two classes can then overlap; [2] shows this
and that the classes can be solved in the order used here.  Coordinates are held per
block and copied to the block's nodes at the end, so the other flaw [2] corrects, in how
the nodes of a block take its coordinate, does not arise.

Placing one node at a time cannot do this.  A rank laid out at the least separation leaves
a node in the middle of it nowhere to go, so its coordinate follows its place in the rank
rather than the wires reaching it, and a wire crossing many such ranks bends at every one.

The method is run four times, aligning to the neighbours below and above and biasing to the
left and to the right, and each node takes the mean of its middle two coordinates.  A single
pass leans towards whichever side it was biased to; the combination is balanced, and no pass
is allowed to break a block, so the straightness survives it.  Where every pass keeps the
least separation between neighbours, so does the combination: of two neighbours `a` left
of `b`, each pass places `b` at least that far right of `a`, so the k-th smallest
coordinate of `b` is at least that far right of the k-th smallest of `a`, for every k.

Where one pass is far narrower than the combination, as a long running total drawn as a
column is against three passes drawing it as a staircase, that pass is used alone.

[1] U. Brandes, B. Koepf.  Fast and Simple Horizontal Coordinate Assignment.  Graph
    Drawing (GD 2001), Lecture Notes in Computer Science 2265, 31-44, Springer, 2002.
    https://doi.org/10.1007/3-540-45848-4_3

[2] U. Brandes, J. Walter, J. Zink.  Erratum: Fast and Simple Horizontal Coordinate
    Assignment.  arXiv:2008.01252, 2020.  https://arxiv.org/abs/2008.01252
"""
from __future__ import annotations

from pylea.draw.glyph import Style

# Most the combined passes may be wider than the narrowest pass before that pass is used
# alone.  The ratio was at most 1.9 on the shipped examples, hidden Markov chains, fault
# trees, chains of gates and 400 random DPPs, and 10.4 on a running total over 50 terms;
# 3 is a choice inside that gap, made by comparing the pictures, not a fitted value.
# Re-measure if the alignment or the ordering changes.
_MEDIAN_WIDTH_LIMIT = 3.0


def coordinates(g, style: Style) -> list[float]:
    """An x for every node of `g`, with the nodes carrying one wire in a line."""
    if not g.rank:
        return []
    conflicted = _crossed_by_a_short_wire(g)
    passes = [(rightward, _one_pass(g, style, conflicted, upward, rightward))
              for upward in (False, True) for rightward in (False, True)]
    return _average(passes)


def _neighbours(g, node: int, upward: bool):
    """The nodes one rank away that `node` aligns towards, nearest rank first."""
    return [m for m, _, _ in (g.succs[node] if upward else g.preds[node])]


def _crossed_by_a_short_wire(g) -> set[tuple[int, int]]:
    """The wires that cross a stretch of a longer wire, which may not be aligned.

    A wire running between two dummies is a stretch of one that crosses several ranks, and
    keeping those straight is the point of the method.  Where a shorter wire crosses one,
    the shorter one gives way.
    """
    marked: set[tuple[int, int]] = set()
    ranks = sorted(g.layers)
    for lower, upper in zip(ranks, ranks[1:]):
        below, above = g.layers[lower], g.layers[upper]
        last_seen, scanned = 0, 0
        for k, node in enumerate(above):
            inner = None
            if g.is_dummy[node]:
                inner = next((m for m in _neighbours(g, node, upward=False)
                              if g.is_dummy[m]), None)
            if k == len(above) - 1 or inner is not None:
                boundary = g.pos[inner] if inner is not None else len(below) - 1
                while scanned <= k:
                    for m in _neighbours(g, above[scanned], upward=False):
                        if not last_seen <= g.pos[m] <= boundary:
                            marked.add((m, above[scanned]))
                    scanned += 1
                last_seen = boundary
    return marked


def _one_pass(g, style: Style, conflicted, upward: bool, rightward: bool) -> list[float]:
    """One alignment and compaction, biased as `upward` and `rightward` say."""
    ranks = sorted(g.layers, reverse=upward)
    rows = [g.layers[r][::-1] if rightward else g.layers[r] for r in ranks]
    place = {node: i for row in rows for i, node in enumerate(row)}

    root, align = _blocks(g, rows, place, conflicted, upward)
    x = _compact(g, style, rows, place, root, align)
    return [-value for value in x] if rightward else x


def _blocks(g, rows, place, conflicted, upward: bool):
    """`root` and `align`: the nodes of one wire chained into one block.

    Each node joins the block of its median neighbour on the side being aligned to, unless
    that would cross an alignment already made or a stretch of a longer wire.
    """
    root = list(range(len(g.rank)))
    align = list(range(len(g.rank)))

    for row in rows[1:]:
        reached = -1
        for node in row:
            near = sorted(_neighbours(g, node, upward), key=place.__getitem__)
            if not near:
                continue
            middle = (len(near) - 1) // 2, len(near) // 2
            for m in dict.fromkeys(near[i] for i in middle):
                if align[node] != node:
                    break
                edge = (m, node) if not upward else (node, m)
                if edge in conflicted or place[m] <= reached:
                    continue
                align[m] = node
                root[node] = root[m]
                align[node] = root[node]
                reached = place[m]
    return root, align


def _compact(g, style: Style, rows, place, root, align) -> list[float]:
    """An x per node, blocks packed as far left as the order within each rank allows."""
    # Each node is kept apart from its neighbour on the left, and a node that is not an
    # interface marker also from the nearest such node to its left: a marker needs no
    # room against a box, so without the second pair two boxes with a marker between
    # them could be drawn on top of each other.
    left_of = {}
    lefts: dict[int, list[int]] = {}
    for row in rows:
        last_box = None
        for i, node in enumerate(row):
            if i:
                left_of[node] = row[i - 1]
                lefts[node] = [row[i - 1]]
                if (not g.is_edge[node] and g.is_edge[row[i - 1]]
                        and last_box is not None):
                    lefts[node].append(last_box)
            if not g.is_edge[node]:
                last_box = node

    blocks = [v for v in range(len(g.rank)) if root[v] == v]
    members: dict[int, list[int]] = {v: [] for v in blocks}
    for v in range(len(g.rank)):
        members[root[v]].append(v)

    # A block cannot be placed before the blocks to its left, and that order is acyclic, so
    # the blocks are placed in it rather than by recursing, which a deep drawing would
    # exhaust the stack for.
    needs = {v: {root[u] for w in members[v] for u in lefts.get(w, ())} for v in blocks}
    order, ready = [], [v for v in blocks if not needs[v]]
    waiting = {v: set(n) for v, n in needs.items()}
    blocked_by: dict[int, list[int]] = {v: [] for v in blocks}
    for v, ns in needs.items():
        for n in ns:
            blocked_by[n].append(v)
    while ready:
        v = ready.pop()
        order.append(v)
        for w in blocked_by[v]:
            waiting[w].discard(v)
            if not waiting[w]:
                ready.append(w)

    # Phase 1: each block at its least distance from the blocks to its left within its
    # class, the class being named by its sink.
    x = {v: 0.0 for v in blocks}
    sink = {v: v for v in blocks}
    for v in order:
        w = v
        while True:
            neighbour = left_of.get(w)
            if neighbour is not None:
                u = root[neighbour]
                if sink[v] == v:
                    sink[v] = sink[u]
            for neighbour in lefts.get(w, ()):
                u = root[neighbour]
                if sink[v] == sink[u]:
                    x[v] = max(x[v], x[u] + _separation(g, neighbour, w, style))
            w = align[w]
            if w == v:
                break

    shift = _class_shifts(g, style, lefts, root, sink, x)
    return [x[root[v]] + shift[sink[root[v]]] for v in range(len(g.rank))]


def _class_shifts(g, style: Style, lefts, root, sink, x) -> dict[int, float]:
    """The shift of every class, each as far right as the classes to its right allow.

    Two nodes kept apart on a rank (`lefts`), in different classes, bound the shift of the
    left class by that of the right one.  Each class is solved after the classes bounding it, by a
    depth-first search without recursion, and a class bounded by none is not shifted.

    The classes with these bounds as edges form an acyclic graph.  A sink is a block with
    no neighbour to its left on any row, so no two sinks share a row, and by Lemma 1 of [1]
    a class to the right of another on some row has its sink on an earlier row, the rows
    taken in the order the pass aligns them.  A cycle raises `RuntimeError`, as a guard on
    that invariant.

    [1] U. Brandes, J. Walter, J. Zink.  Erratum: Fast and Simple Horizontal Coordinate
        Assignment.  arXiv:2008.01252, 2020.  https://arxiv.org/abs/2008.01252
    """
    bounds: dict[int, list[tuple[int, float]]] = {}
    for w, neighbours in lefts.items():
        for neighbour in neighbours:
            u, v = root[neighbour], root[w]
            if sink[u] != sink[v]:
                bounds.setdefault(sink[u], []).append(
                    (sink[v], x[v] - x[u] - _separation(g, neighbour, w, style)))

    shift: dict[int, float] = {}
    on_path: set[int] = set()
    for start in dict.fromkeys(sink.values()):
        stack = [(start, 0)]
        while stack:
            c, i = stack.pop()
            if c in shift:
                continue
            on_path.add(c)
            right = bounds.get(c, ())
            if i < len(right):
                stack.append((c, i + 1))
                d = right[i][0]
                if d in on_path:
                    raise RuntimeError(f"the classes of blocks {c} and {d} each lie to the "
                                       "right of the other")
                stack.append((d, 0))
                continue
            shift[c] = min((shift[d] + offset for d, offset in right), default=0.0)
            on_path.discard(c)
    return shift


def _separation(g, a: int, b: int, style: Style) -> float:
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


def _average(passes) -> list[float]:
    """The four passes laid over one another, each node taking the mean of the middle two
    of its four coordinates, or the narrowest pass alone where that is over
    `_MEDIAN_WIDTH_LIMIT` times narrower.

    A pass fixes the distances between nodes and not where they start, so each is first slid
    onto the narrowest of them: a pass biased to the left is matched at its left edge and one
    biased to the right at its right edge, which is the side each of them packed against.

        >>> _average([(False, [0.0, 10.0]), (False, [0.0, 12.0]),
        ...           (True, [-14.0, 0.0]), (True, [-20.0, 0.0])])
        [-2.0, 10.0]
        >>> _average([(False, [0.0, 10.0]), (False, [0.0, 40.0]),
        ...           (True, [-60.0, 0.0]), (True, [-70.0, 0.0])])
        [0.0, 10.0]
    """
    widths = [max(x) - min(x) for _, x in passes]
    narrowest = passes[widths.index(min(widths))][1]

    aligned = []
    for rightward, x in passes:
        offset = (max(narrowest) - max(x)) if rightward else (min(narrowest) - min(x))
        aligned.append([value + offset for value in x])

    median = [sum(sorted(column)[1:3]) / 2 for column in zip(*aligned)]
    if max(median) - min(median) > _MEDIAN_WIDTH_LIMIT * min(widths):
        return list(narrowest)
    return median
