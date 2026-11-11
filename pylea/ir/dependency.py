"""Direct dependencies between the interface wires of a body.

`direct_dependencies` says, for every wire a body takes in or returns, which of the others
its value is computed from.  Wire `v` is a direct dependency of wire `w` where some op
reading `v` leads to `w` through wires that are not themselves part of the interface; a
route passing through a third interface wire is not one, and is left to that wire's own
entry.  The relation is acyclic over wires, a wire's dependencies being its ancestors, and
`pylea.draw.dependency` draws it as the graph of a DPP whose outputs are the variables of a
network.

Calls are summarised rather than spliced in.  Each body is summarised once, by this same
relation over its own interface, and a call site resolves that summary against its
arguments and results, so the work is one pass per body rather than one pass per call site.
Splicing instead costs a copy of the callee at every site, which is quadratic in the size
of the DPP where a helper is called many times.

A summary relates a body's outputs to its inputs and to its other outputs.  The second is
what a caller returning several results of one call needs: the relation between two such
results lies inside the callee, so without it a DPP that returns what a DPP it calls
returned would show no dependencies at all.

A callee may hand one value back at several of its interface positions, by returning an
argument or by returning one wire twice.  At a call site these are several wires of the
caller carrying one value, and a summary records which of its interface positions carry
one value, so that this holds through any depth of calls.  Reading any of those wires is
reading the first of them in the caller's interface, just as a wire appearing twice in an
interface is named by its first position; where none of them is part of the interface,
the route passes through to what reaches the value.

An op calling a primitive has all of its results depend on all of its arguments, since a
primitive carries a dense matrix whose entries may relate any input bit to any output bit.
`pylea.ir.prim.Tag.ITE` is the one primitive whose bits are related positionally, and the
bits it relates belong to one variable of the DPP wherever it is used, so nothing is
lost by not separating them.

    >>> from pylea.examples.sprinkler import sprinkler_network
    >>> deps = direct_dependencies(sprinkler_network.program())
    >>> {w: sorted(d) for w, d in deps.items()}
    {0: [], 1: [0], 2: [0], 3: [0], 4: [1, 2, 3], 5: [1, 2, 3]}

Wire 0 carries the cloudiness, 1 the sprinkler, 2 and 3 the rain and 4 and 5 the wetness of
the grass, so this is the network the DPP was written from.
"""
from __future__ import annotations

from dataclasses import dataclass

from pylea.ir.bits import set_bits
from pylea.ir.body import Body
from pylea.ir.prim import Prim
from pylea.ir.program import Program, reachable


def direct_dependencies(prog: Program,
                        body: int | None = None) -> dict[int, frozenset[int]]:
    """Every interface wire of one body of `prog`, and the interface wires it comes from.

    The root where `body` is left out.  A key for each distinct wire of ``inputs +
    outputs``, in that order; an input wire depends on nothing, an output wire on the
    interface wires reaching it without another interface wire in between.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def chain(x: Bool) -> tuple[Bool, Bool]:
        ...     y = x & flip(0.5)
        ...     z = y & flip(0.5)
        ...     return y, z
        >>> direct_dependencies(chain.program())
        {0: frozenset(), 2: frozenset({0}), 4: frozenset({2})}

    Wire 4 carries `z` and wire 2 carries `y`, so `z` depends on `y` alone: its route back
    to the input `x` passes through `y`, which has its own entry.
    """
    index = prog.root if body is None else body
    b = prog.bodies[index]

    summaries: dict[int, _Summary] = {}
    for i in reachable(Program(prog.bodies, index)):     # callees before callers
        if i != index:
            summaries[i] = _summary(prog, i, summaries)

    interface = b.inputs + b.outputs
    values = _values(b, summaries)
    marks, first, _ = _marks(b, values)
    into = _into(b, marks, values, summaries)
    given = set(b.inputs)
    return {w: frozenset() if w in given
               else frozenset(interface[first[k]] for k in set_bits(into[w]))
            for w in dict.fromkeys(interface)}


@dataclass(frozen=True, slots=True)
class _Summary:
    """How each output of one body is computed from that body's own interface.

    A bit of a mask stands for one interface wire.  `carriers` gives, for each bit, the
    interface positions of every wire carrying that wire's value, in order: an index into
    `Body.inputs` below `n_in`, and `n_in` plus an index into `Body.outputs` above it.
    `masks` gives, for each output position, the bits it is computed from directly, and
    `order` lists the output positions so that one comes after every output it is computed
    from.
    """

    carriers: tuple[tuple[int, ...], ...]
    masks: tuple[int, ...]
    order: tuple[int, ...]
    n_in: int


def _summary(prog: Program, i: int, summaries: dict[int, _Summary]) -> _Summary:
    """The summary of body `i`, whose callees must be summarised in `summaries` already."""
    b = prog.bodies[i]
    values = _values(b, summaries)
    marks, _, carriers = _marks(b, values)
    into = _into(b, marks, values, summaries)
    given = set(b.inputs)
    # An output that is an input carries that input's value, rather than anything reaching
    # the wire, since no op produces it.
    masks = tuple(marks[w] if w in given else into[w] for w in b.outputs)
    return _Summary(carriers, masks, _output_order(b, summaries), len(b.inputs))


def _values(b: Body, summaries: dict[int, _Summary]) -> list[int]:
    """For each wire of `b`, one wire standing for every wire carrying the same value.

    Two wires carry one value where a call hands one value of its callee to both, as an
    argument it returns or as a wire it returns twice.
    """
    value = list(range(len(b.bits)))

    def find(w: int) -> int:
        while value[w] != w:
            value[w] = value[value[w]]
            w = value[w]
        return w

    for op in b.ops:
        if isinstance(op.callee, Prim):
            continue
        s = summaries[op.callee]
        for positions in s.carriers:
            wires = [op.args[p] if p < s.n_in else op.outs[p - s.n_in] for p in positions]
            for w in wires[1:]:
                value[find(w)] = find(wires[0])
    return [find(w) for w in range(len(b.bits))]


def _marks(b: Body, values: list[int]) -> tuple[dict[int, int], tuple[int, ...],
                                                tuple[tuple[int, ...], ...]]:
    """One bit per distinct interface wire of `b`, the position naming each bit's wire,
    and the positions of every interface wire carrying each bit's value.

    A position indexes ``b.inputs + b.outputs``.  A wire appearing at several positions is
    named by the first of them, so a wire that is both an input and an output is named by
    its input position, which is where its value comes from.
    """
    marks: dict[int, int] = {}
    first: list[int] = []
    of_value: dict[int, list[int]] = {}
    for k, w in enumerate(b.inputs + b.outputs):
        if w not in marks:
            marks[w] = 1 << len(first)
            first.append(k)
        of_value.setdefault(values[w], []).append(k)
    interface = b.inputs + b.outputs
    carriers = tuple(tuple(of_value[values[interface[k]]]) for k in first)
    return marks, tuple(first), carriers


def _into(b: Body, marks: dict[int, int], values: list[int],
          summaries: dict[int, _Summary]) -> list[int]:
    """The marked wires reaching each wire of `b`, as a mask, blocked at a marked wire.

    A marked wire passes its own mark on rather than what reaches it, which is what makes
    the relation direct: a route through it belongs to that wire's own entry.  A wire
    carrying the value of a marked wire passes that mark on, the first such wire's where
    there are several, a result of a call never passing on its own.
    """
    into = [0] * len(b.bits)
    named: dict[int, list[int]] = {}
    for w in dict.fromkeys(b.inputs + b.outputs):
        named.setdefault(values[w], []).append(w)

    def carried(w: int, own: int = -1) -> int:
        for v in named.get(values[w], ()):
            if v != own:
                return marks[v]
        return into[w]

    for op in b.ops:
        if isinstance(op.callee, Prim):
            joint = 0
            for w in op.args:
                joint |= carried(w)
            for w in op.outs:
                into[w] = joint
            continue
        s = summaries[op.callee]
        for j in s.order:
            out = op.outs[j]
            mask = 0
            for k in set_bits(s.masks[j]):
                (w, *_) = [w for w in (op.args[p] if p < s.n_in else op.outs[p - s.n_in]
                                       for p in s.carriers[k]) if w != out]
                mask |= carried(w, own=out)
            into[out] = mask
    return into


def _output_order(b: Body, summaries: dict[int, _Summary]) -> tuple[int, ...]:
    """The output positions of `b`, one coming after every output it is computed from.

    `_into` computes the results of the ops in op order, and the results of one call in
    the order of the callee's summary, since one result of a call may be computed from
    another; that order is therefore one.  An output that is an input is available from
    the start.
    """
    rank: dict[int, tuple[int, int]] = {}
    for k, op in enumerate(b.ops):
        if isinstance(op.callee, Prim):
            rank.update((w, (k, 0)) for w in op.outs)
        else:
            rank.update((op.outs[j], (k, r))
                        for r, j in enumerate(summaries[op.callee].order))
    return tuple(sorted(range(len(b.outputs)), key=lambda j: rank.get(b.outputs[j], (-1, 0))))
