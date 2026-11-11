"""The picture of a DPP as a graph over its variables.

A `pylea.draw.sketch.Sketch` of one rounded box per variable a definition is given or
returns, and one arrow from a variable to each variable computed from it.  It reads from
the top down, a variable sitting above the ones computed from it, which is how such a
network is usually drawn and the other way round from the dot diagram of
`pylea.draw.sketch`.  The sketch is directed, so each wire ends in a head at the variable
it points into.  A variable the DPP is given carries a second outline inside the first,
since it has no dependencies to show whether or not the DPP has any.

The edges come from `pylea.ir.dependency.direct_dependencies`, which works over wires;
here they are lifted to variables, a variable being a segment of the interface that
`input_widths` and `output_widths` mark off.  Variable `Y` depends on variable `X` where
any wire of `Y` depends on any wire of `X`.  Two positions naming the same wires are one
variable, so a DPP returning one value twice draws one box.

A dependency is what a value is computed from, which for a DPP whose outputs are the
variables of a network is that network.  It is not a statement about statistical
dependence in either direction: two outputs drawn from one internal coin flip the DPP does
not return are correlated and are drawn apart, and conditioning is not drawn at all,
since the wires an `OBSERVE` reads are not interface wires.

    >>> from pylea.examples.sprinkler import sprinkler_network
    >>> s = dependency_sketch_of(sprinkler_network.program(),
    ...                          output_names={0: "c", 1: "s", 2: "r", 3: "w"},
    ...                          output_widths=(1, 1, 2, 2))
    >>> [n.label for n in s.nodes], len(s.edges)
    (['c', 's', 'r', 'w'], 4)
"""
from __future__ import annotations

from collections import Counter

from pylea.draw.glyph import GlyphKind
from pylea.draw.sketch import Edge, Node, Sketch, segments
from pylea.ir.dependency import direct_dependencies
from pylea.ir.program import Program


def dependency_sketch_of(prog: Program, body: int | None = None, input_names=None,
                         output_names=None, input_widths=None, output_widths=None,
                         title: str = "") -> Sketch:
    """The dependency graph of one definition of `prog`, the main function where `body`
    is left out.

    `input_widths` and `output_widths` give the number of wires of each interface
    variable, every variable being one wire where they are left out, and `input_names` and
    `output_names` label the variable at each interface position.  A position with no name
    is labelled by the interface it is on and its position there, and a label two variables
    would share carries the index of the variable as well.

    Raises `ValueError` where the variables have no acyclic order, which happens where two
    of them are computed from each other bit by bit.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def noisy(x: Bool) -> Bool:
        ...     return x & flip(0.9)
        >>> s = dependency_sketch_of(noisy.program(), input_names={0: "x"},
        ...                          output_names={0: "y"})
        >>> [(n.kind.name, n.label) for n in s.nodes]
        [('PARAMETER', 'x'), ('VARIABLE', 'y')]
    """
    b = prog.bodies[prog.root if body is None else body]
    wires, labels, given = _variables(
        segments(b.inputs, input_widths, "input"), input_names or {},
        segments(b.outputs, output_widths, "output"), output_names or {})

    parents = _parents(wires, direct_dependencies(prog, body))
    children = _children(parents)
    _refuse_a_cycle(parents, children, labels)

    port_of = {(i, j): port for i, js in enumerate(children) for port, j in enumerate(js)}

    nodes = tuple(
        Node(GlyphKind.PARAMETER if given[i] else GlyphKind.VARIABLE, labels[i],
             n_in=len(parents[i]), n_out=len(children[i]), wire=wires[i][0])
        for i in range(len(wires)))
    edges = tuple(Edge(i, port_of[(i, j)], j, port)
                  for j, ps in enumerate(parents) for port, i in enumerate(ps))
    return Sketch(nodes, edges, title, directed=True, downward=True)


def _variables(in_segments, input_names, out_segments,
               output_names) -> tuple[list[tuple[int, ...]], list[str], list[bool]]:
    """The variables of a body's interface: their wires, their labels, which are given.

    Positions naming the same wires are one variable, which takes the first label offered
    for it and is given where any of those positions is an input.  A label two variables
    would share, which is what a caller naming a position by its type offers for two
    positions of one type, carries the index of the variable as well, since a node of this
    picture is nothing but its label.
    """
    wires: list[tuple[int, ...]] = []
    labels: list[str] = []
    given: list[bool] = []
    at: dict[tuple[int, ...], int] = {}

    for is_input, interface, names in ((True, in_segments, input_names),
                                       (False, out_segments, output_names)):
        for position, segment in enumerate(interface):
            name = names.get(position) or f"{'in' if is_input else 'out'} {position}"
            i = at.get(segment)
            if i is None:
                at[segment] = len(wires)
                wires.append(segment)
                labels.append(name)
                given.append(is_input)
            else:
                given[i] = given[i] or is_input

    shared = {name for name, n in Counter(labels).items() if n > 1}
    labels = [f"{name} {i}" if name in shared else name
              for i, name in enumerate(labels)]
    return wires, labels, given


def _parents(wires: list[tuple[int, ...]],
             deps: dict[int, frozenset[int]]) -> list[list[int]]:
    """The variables each variable of `wires` is computed from, in order.

    A wire shared by two variables, as one bit of a wider variable returned on its own is,
    credits both of them.
    """
    holders: dict[int, list[int]] = {}
    for i, segment in enumerate(wires):
        for w in segment:
            holders.setdefault(w, []).append(i)

    parents = []
    for i, segment in enumerate(wires):
        found = {j for w in segment for v in deps[w] for j in holders.get(v, ())}
        parents.append(sorted(found - {i}))
    return parents


def _children(parents: list[list[int]]) -> list[list[int]]:
    """The variables computed from each variable, which is `parents` the other way round."""
    children: list[list[int]] = [[] for _ in parents]
    for j, ps in enumerate(parents):
        for i in ps:
            children[i].append(j)
    return children


def _refuse_a_cycle(parents: list[list[int]], children: list[list[int]],
                    labels: list[str]) -> None:
    """Nothing, or `ValueError` where the variables cannot be ordered by dependency.

    The relation over wires is acyclic, but a variable holds several of them, so two
    variables whose bits are computed in turn from each other do form a cycle.  There is
    no picture reading one way down the page for that, and a caller is told so rather than
    shown an arbitrary one.
    """
    waiting = [len(ps) for ps in parents]
    ready = [i for i, n in enumerate(waiting) if not n]
    settled = 0
    while ready:
        i = ready.pop()
        settled += 1
        for j in children[i]:
            waiting[j] -= 1
            if not waiting[j]:
                ready.append(j)

    if settled != len(parents):
        stuck = ", ".join(labels[i] for i, n in enumerate(waiting) if n)
        raise ValueError(f"the variables {stuck} are computed from one another bit by "
                         "bit, so they have no order to draw a dependency graph in")
