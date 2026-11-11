"""Horizontal coordinates for a layered graph.

`coordinates` assigns an x to every node of a `pylea.draw.layout` graph by the method of
Brandes and Koepf, which is what keeps a wire crossing many ranks straight.

The idea is to place blocks rather than nodes.  A node is aligned with the median of the
neighbours below it wherever two such alignments would not cross, which chains the dummies
carrying one wire into a single block; every node of a block then takes one x, so the wire
is drawn as a straight line.  Compaction places the blocks as close together as the order
within each rank allows.

Placing one node at a time cannot do this.  A rank laid out at the least separation leaves
a node in the middle of it nowhere to go, so its coordinate follows its place in the rank
rather than the wires reaching it, and a wire crossing many such ranks bends at every one.

The method is run four times, aligning to the neighbours below and above and biasing to the
left and to the right, and the four results are averaged.  A single pass leans towards
whichever side it was biased to; the average of the four is balanced, and no pass is allowed
to break a block, so the straightness survives the averaging.
"""
from __future__ import annotations

from pylea.draw.glyph import Style


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
    left_of = {}
    for row in rows:
        for i, node in enumerate(row[1:], start=1):
            left_of[node] = row[i - 1]

    blocks = [v for v in range(len(g.rank)) if root[v] == v]
    members: dict[int, list[int]] = {v: [] for v in blocks}
    for v in range(len(g.rank)):
        members[root[v]].append(v)

    # A block cannot be placed before the blocks to its left, and that order is acyclic, so
    # the blocks are placed in it rather than by recursing, which a deep drawing would
    # exhaust the stack for.
    needs = {v: {root[left_of[w]] for w in members[v] if w in left_of} for v in blocks}
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

    x = {v: 0.0 for v in blocks}
    sink = {v: v for v in blocks}
    shift = {v: float("inf") for v in blocks}

    for v in order:
        w = v
        while True:
            neighbour = left_of.get(w)
            if neighbour is not None:
                u = root[neighbour]
                if sink[v] == v:
                    sink[v] = sink[u]
                gap = _separation(g, neighbour, w, style)
                if sink[v] != sink[u]:
                    shift[sink[u]] = min(shift[sink[u]], x[v] - x[u] - gap)
                else:
                    x[v] = max(x[v], x[u] + gap)
            w = align[w]
            if w == v:
                break

    placed = [0.0] * len(g.rank)
    for v in range(len(g.rank)):
        block = root[v]
        placed[v] = x[block]
        if shift[sink[block]] < float("inf"):
            placed[v] += shift[sink[block]]
    return placed


def _separation(g, a: int, b: int, style: Style) -> float:
    """The least distance between the centres of neighbours `a` and `b` on one rank."""
    if g.is_edge[a] != g.is_edge[b]:
        return 0.0
    gap = style.node_sep * (0.6 if g.is_dummy[a] and g.is_dummy[b] else 1.0)
    return (g.width[a] + g.width[b]) / 2 + gap


def _average(passes) -> list[float]:
    """The four passes laid over one another, each node taking the middle two of the four.

    A pass fixes the distances between nodes and not where they start, so each is first slid
    onto the narrowest of them: a pass biased to the left is matched at its left edge and one
    biased to the right at its right edge, which is the side each of them packed against.
    """
    widths = [max(x) - min(x) for _, x in passes]
    narrowest = passes[widths.index(min(widths))][1]

    aligned = []
    for rightward, x in passes:
        offset = (max(narrowest) - max(x)) if rightward else (min(narrowest) - min(x))
        aligned.append([value + offset for value in x])

    return [sum(sorted(column)[1:3]) / 2 for column in zip(*aligned)]
