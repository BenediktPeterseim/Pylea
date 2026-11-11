"""The picture of a body, before any geometry.

A `Sketch` is what a `pylea.ir.body.Body` looks like as a dot diagram: one dot per drawn
wire, one box per op, and one boundary marker per interface position.  This is a direct
reading of the body rather than a translation of it, since a body already is a variable
set, a list of assignments and two interface lists.

A drawn wire is usually one wire of the body.  `wire_groups` names tuples of wires that
stand for one variable of more than one bit, and such a tuple is drawn as a single dot
wherever its wires appear only together: as one contiguous run, in order, in every op
argument list, op result list and interface list that touches any of them.  A group whose
wires are read or produced separately anywhere is drawn as separate dots instead, which is
what an op over single bits requires.

Edges all point the same way, from an argument dot into a box and from a box out to a result
dot, and carry the port they attach to, so that argument order is visible in the drawing.  A
wire read by several ops is one dot with several edges leaving it, a wire absent from the
outputs is a dot with none, and a wire named twice in the outputs is a dot with two boundary
edges.

    >>> from pylea import DPP, Bool, flip
    >>> @DPP
    ... def either() -> Bool:
    ...     return flip(0.5) | flip(0.25)
    >>> s = sketch_of(either.program())
    >>> len(s.nodes), len(s.edges)
    (7, 6)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pylea.draw.glyph import GlyphKind
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program

_GATES = {
    Tag.AND: GlyphKind.AND,
    Tag.OR: GlyphKind.OR,
    Tag.NOT: GlyphKind.NOT,
    Tag.FLIP: GlyphKind.FLIP,
    Tag.OBSERVE: GlyphKind.OBSERVE,
}


@dataclass(frozen=True, slots=True)
class Node:
    """One dot, box or boundary marker, with the number of ports on each side."""

    kind: GlyphKind
    label: str = ""
    n_in: int = 1
    n_out: int = 1
    wire: int | None = None         # the first wire a dot or a variable box stands for
    op: int | None = None           # the position in `Body.ops` a box stands for


@dataclass(frozen=True, slots=True)
class Edge:
    """A wire from port `src_port` of `src` to port `dst_port` of `dst`."""

    src: int
    src_port: int
    dst: int
    dst_port: int


@dataclass(frozen=True, slots=True)
class Sketch:
    """The nodes and edges of one body, with the name to caption it by.

    `directed` draws every wire with an arrow head at the end it enters.  A dot diagram
    leaves them out, its wires running from the bottom of the picture to the top and the
    glyphs at either end saying which way round each is; a graph over variables carries
    them, a box giving the reader nothing to read a direction from.

    `downward` turns the picture over, so that a node is drawn above the ones its wires
    lead to.  A dot diagram reads upward, from the inputs at the bottom to the outputs at
    the top; a graph of the variables of a network reads downward, which is how such a
    network is usually drawn.
    """

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    title: str = ""
    directed: bool = False
    downward: bool = False


def _prim_glyph(prim: Prim, label: str) -> tuple[GlyphKind, str]:
    """The symbol and the text for an op calling `prim`, `label` being its recorded name."""
    gate = _GATES.get(prim.tag)
    if gate is not None:
        return gate, format(prim.params[0], "g") if prim.tag is Tag.FLIP else ""
    if prim.tag is Tag.MATRIX:
        return (GlyphKind.BOX, label) if label else _table_glyph(prim)
    if prim.tag is Tag.CONST:
        # The bit pattern only reads as the value for a type whose values are 0, 1, 2, ...
        return GlyphKind.BOX, label or str(prim.params[0])
    if prim.tag is Tag.UNIFORM:
        return GlyphKind.BOX, "unif"
    if prim.tag is Tag.EQUATE:
        return GlyphKind.BOX, "="
    return GlyphKind.BOX, prim.tag.name.lower()


def _table_glyph(prim: Prim) -> tuple[GlyphKind, str]:
    """The symbol and the text for a table with no recorded name.

    A table reading nothing and producing one bit whose column sums to one is a coin flip,
    and is drawn as one carrying its probability of 1.  A table reading nothing and
    producing nothing multiplies the weight of every assignment by its one entry, and is
    drawn as a box carrying that entry after a multiplication sign.  Simplification
    produces both, by folding the coin flips of a block into one and an observation of a
    coin flip into its weight.  Any other table is a box with no text.

        >>> _table_glyph(Prim.from_matrix([[0.25], [0.75]]))
        (<GlyphKind.FLIP: 4>, '0.75')
        >>> _table_glyph(Prim.from_matrix([[0.3]]))
        (<GlyphKind.BOX: 6>, '×0.3')
        >>> _table_glyph(Prim.from_matrix([[0.5, 0.0], [0.5, 1.0]]))
        (<GlyphKind.BOX: 6>, '')
    """
    entries = prim.matrix.ravel()
    if prim.in_bits == 0 and prim.out_bits == 1 and abs(float(entries.sum()) - 1.0) <= 1e-9:
        return GlyphKind.FLIP, format(float(entries[1]), "g")
    if prim.in_bits == 0 and prim.out_bits == 0:
        return GlyphKind.BOX, "×" + format(float(entries[0]), "g")
    return GlyphKind.BOX, ""


def display_name(name: str) -> str:
    """`name` without the underscores that mark an internal callable.

        >>> display_name("_CPT_Rain_"), display_name("positive_test")
        ('CPT_Rain', 'positive_test')
    """
    return name.strip("_") or name


def segments(wires, widths, what: str) -> list[tuple[int, ...]]:
    """`wires` cut into one tuple per interface variable, `widths` giving the sizes.

    Left out, every variable is one wire.  Raises `ValueError` where the widths do not
    cover the wires exactly.  `what` names the interface in that message.

        >>> segments((3, 4, 5), (1, 2), "output")
        [(3,), (4, 5)]
    """
    widths = tuple(widths) if widths is not None else (1,) * len(wires)
    if sum(widths) != len(wires):
        raise ValueError(f"{what} widths {widths} cover {sum(widths)} wires; "
                         f"the interface has {len(wires)}")
    cut, at = [], 0
    for width in widths:
        cut.append(tuple(wires[at:at + width]))
        at += width
    return cut


def _accepted_groups(b, candidates: Iterable[tuple[int, ...]],
                     in_segments, out_segments) -> list[tuple[int, ...]]:
    """The groups of `candidates` that can each be drawn as one dot.

    A group qualifies where its wires appear only together, as one contiguous run in its
    own order, in every op argument list, op result list and interface list touching any
    of them.  A group failing that is dropped and its wires drawn as separate dots, which
    is what an op reading or producing a single bit of it requires.  Of two groups sharing
    a wire the first is kept.  An interface segment of several wires that no surviving
    group equals is drawn as separate dots too, so any group reaching into one is dropped
    as well, until nothing more changes.
    """
    groups: list[tuple[int, ...]] = []
    taken: set[int] = set()
    for g in candidates:
        g = tuple(g)
        if len(g) < 2 or len(set(g)) != len(g):
            continue
        if any(w in taken for w in g) or not all(0 <= w < len(b.bits) for w in g):
            continue
        groups.append(g)
        taken.update(g)

    lists = [b.inputs, b.outputs] + [ws for op in b.ops for ws in (op.args, op.outs)]

    def together(g: tuple[int, ...]) -> bool:
        member = frozenset(g)
        for ws in lists:
            i = 0
            while i < len(ws):
                if ws[i] in member:
                    if tuple(ws[i:i + len(g)]) != g:
                        return False
                    i += len(g)
                else:
                    i += 1
        return True

    accepted = {g for g in groups if together(g)}

    changed = True
    while changed:
        changed = False
        for segment in in_segments + out_segments:
            if len(segment) < 2 or segment in accepted:
                continue
            bits = set(segment)
            for g in [g for g in accepted if bits & set(g)]:
                accepted.discard(g)
                changed = True
    return [g for g in groups if g in accepted]


def _runs(ws, group_of) -> list[int]:
    """The first wire of each run of `ws`, an accepted group passing as a single run."""
    runs, i = [], 0
    while i < len(ws):
        g = group_of.get(ws[i])
        runs.append(ws[i])
        i += len(g) if g is not None else 1
    return runs


def sketch_of(prog: Program, body: int | None = None, body_names=None, op_names=None,
              input_names=None, output_names=None, title: str = "",
              wire_groups: Iterable[tuple[int, ...]] = (),
              input_widths=None, output_widths=None) -> Sketch:
    """The dot diagram of one body of `prog`, the root where `body` is left out.

    `body_names` names a called body and `op_names` an op of this body, both by index; a
    call whose body is unnamed is captioned by its index instead.  `wire_groups` names
    tuples of wires each standing for one variable, drawn as a single dot wherever the
    body keeps them together; see the module docstring for when it does not.
    `input_widths` and `output_widths` give the number of wires of each interface
    variable, every variable being one wire where they are left out, and `input_names`
    and `output_names` label the variable at each interface position.  A several-wire
    variable drawn as separate dots labels each of them by position, as ``name[0]``.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> [n.label for n in sketch_of(both.program()).nodes if n.op is not None]
        ['body#0', 'body#0', '']
    """
    body_names, op_names = body_names or {}, op_names or {}
    input_names, output_names = input_names or {}, output_names or {}
    b = prog.bodies[prog.root if body is None else body]

    in_segments = segments(b.inputs, input_widths, "input")
    out_segments = segments(b.outputs, output_widths, "output")
    accepted = _accepted_groups(b, wire_groups, in_segments, out_segments)
    group_of = {w: g for g in accepted for w in g}

    nodes: list[Node] = []
    edges: list[Edge] = []
    node_of: dict[int, int] = {}
    for w in range(len(b.bits)):
        if w in node_of:
            continue
        run = group_of.get(w, (w,))
        for u in run:
            node_of[u] = len(nodes)
        nodes.append(Node(GlyphKind.DOT, wire=run[0]))

    for j, op in enumerate(b.ops):
        if isinstance(op.callee, Prim):
            kind, label = _prim_glyph(op.callee, display_name(op_names.get(j, "")))
        else:
            kind = GlyphKind.CALL
            label = display_name(body_names.get(op.callee, f"body#{op.callee}"))
        arg_runs, out_runs = _runs(op.args, group_of), _runs(op.outs, group_of)
        box = len(nodes)
        nodes.append(Node(kind, label, len(arg_runs), len(out_runs), op=j))
        edges += [Edge(node_of[a], 0, box, p) for p, a in enumerate(arg_runs)]
        edges += [Edge(box, p, node_of[o], 0) for p, o in enumerate(out_runs)]

    # An interface wire ends at a marker that draws nothing, so the layout routes it to the
    # edge of the picture the way it routes any other wire.
    def marker(name: str, entering: bool, dot: int) -> None:
        i = len(nodes)
        if entering:
            nodes.append(Node(GlyphKind.BOUNDARY, name, n_in=0))
            edges.append(Edge(i, 0, dot, 0))
        else:
            nodes.append(Node(GlyphKind.BOUNDARY, name, n_out=0))
            edges.append(Edge(dot, 0, i, 0))

    for interface, names, entering in ((in_segments, input_names, True),
                                       (out_segments, output_names, False)):
        for position, segment in enumerate(interface):
            name = names.get(position, "")
            if len(segment) == 1 or group_of.get(segment[0]) == segment:
                marker(name, entering, node_of[segment[0]])
            else:
                for i, w in enumerate(segment):
                    marker(f"{name}[{i}]" if name else "", entering, node_of[w])

    return Sketch(tuple(nodes), tuple(edges), title)
