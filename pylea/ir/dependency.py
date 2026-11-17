"""Direct dependencies between the interface wires of a body.

`direct_dependencies` says, for every wire a body takes in or returns, which of the others
its value is computed from.  Wire `v` is a direct dependency of wire `w` where some op
reading `v` leads to `w` through wires that are not themselves part of the interface; a
route passing through a third interface wire is not one, and is left to that wire's own
entry.  A wire carrying the same value as an earlier interface wire, which a call to a
definition returning its argument gives, depends on that earlier wire alone, although no op
reads it.  The relation is acyclic over wires, a wire's dependencies being its ancestors,
and `pylea.draw.dependency` draws it as the graph of a DPP whose outputs are the variables
of a network.

Calls are summarised rather than spliced in.  Each body is summarised once, by this same
relation over its own interface, and a call site resolves that summary against its
arguments and results, so the work is one pass per body rather than one pass per call site.
Splicing instead costs a copy of the callee at every site, which is quadratic in the size
of the DPP where a helper is called many times.

A summary relates a body's outputs to its inputs and to its other outputs.  The second is
what a caller returning several results of one call needs: the relation between two such
results lies inside the callee, so without it a DPP that returns what a DPP it calls
returned would show no dependencies at all.

**Wires carrying one value form a class, and each class is created once.**  A callee may
hand one value back at several of its interface positions, by returning an argument or by
returning one wire twice, and at a call site the wires at those positions join one class
of the caller; a summary records, for each output, the earlier position whose value it
carries, so that this holds through any depth of calls.  A class is created by an input,
by a primitive's result, or by a call result that is the first carrier of its value
inside the callee.  The first position of a class in ``inputs + outputs`` stands for it:

* reading any wire of the class reads that position, or, where the class has no interface
  wire, what reaches the class where it is created;
* that position depends on what reaches the op creating the class, and on nothing for an
  input;
* a later position of the class depends on the first alone, as a copy of it.

A class is created only from classes created before it, so no value can depend on itself
or on a value computed from it.  Taking the value of a class from whichever interface wire
carries it instead would let the mark of an output flow back to the wires it is computed
from, and from there into entries computed before it.

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

from pylea.ir._union_find import UnionFind
from pylea.ir.bits import set_bits
from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim
from pylea.ir.program import Program, reachable


def direct_dependencies(prog: Program,
                        body: int | None = None) -> dict[int, frozenset[int]]:
    """Every interface wire of one body of `prog`, and the interface wires it comes from.

    The root where `body` is left out.  A key for each distinct wire of ``inputs +
    outputs``, in that order; an input wire depends on nothing, an output wire carrying
    the value of an earlier interface wire on that wire, and any other output wire on the
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
    summaries: dict[int, _Summary] = {}
    for i in reachable(Program(prog.bodies, index)):     # callees before callers
        if i != index:
            summaries[i] = _summary(prog.bodies[i], summaries)

    b = prog.bodies[index]
    interface = b.inputs + b.outputs
    a = _analyse(b, summaries)
    deps: dict[int, frozenset[int]] = {}
    for k, w in enumerate(interface):
        if w in deps:
            continue
        c = a.cls[w]
        f = a.first[c]
        if k < len(b.inputs):
            deps[w] = frozenset()
        elif f < k:
            deps[w] = frozenset({interface[f]})
        else:
            deps[w] = frozenset(interface[p] for p in set_bits(a.into[c]))
    return deps


@dataclass(frozen=True, slots=True)
class _Summary:
    """How each output of one body is computed from that body's own interface.

    A position indexes ``Body.inputs + Body.outputs``: an index into `Body.inputs` below
    `n_in`, and `n_in` plus an index into `Body.outputs` from there on.  `alias` gives, for
    each output, the earlier position whose value it carries, and -1 for an output whose
    value the body creates.  `masks` gives, for each output of the second kind, the
    positions it is computed from directly as the bits of a mask, and 0 for the others.
    `order` lists the outputs of the second kind in the order the body creates them, so
    that one comes after every output in its mask.
    """

    alias: tuple[int, ...]
    masks: tuple[int, ...]
    order: tuple[int, ...]
    n_in: int


def _summary(b: Body, summaries: dict[int, _Summary]) -> _Summary:
    """The summary of `b`, whose callees must be summarised in `summaries` already."""
    a = _analyse(b, summaries)
    n_in = len(b.inputs)
    alias, masks = [], []
    for j, w in enumerate(b.outputs):
        c = a.cls[w]
        f = a.first[c]
        alias.append(f if f < n_in + j else -1)
        masks.append(a.into[c] if f == n_in + j else 0)
    order = sorted((j for j in range(len(b.outputs)) if alias[j] < 0),
                   key=lambda j: a.created[a.cls[b.outputs[j]]])
    return _Summary(tuple(alias), tuple(masks), tuple(order), n_in)


@dataclass(frozen=True, slots=True)
class _Analysis:
    """The value classes of the wires of one body, and what reaches each.

    `cls` names the class of every wire by one of its wires.  `first` gives, for a class
    with an interface wire, its first position in ``inputs + outputs``.  `into` gives, per
    class, the first positions reaching the op that creates it, as the bits of a mask, and
    `created` the place of that op: its index in `Body.ops` and the result's rank within
    the op, an input coming before every op.
    """

    cls: list[int]
    first: dict[int, int]
    into: list[int]
    created: list[tuple[int, int]]


def _at(op: Op, s: _Summary, p: int) -> int:
    """The wire of a call site standing at position `p` of the callee's interface."""
    return op.args[p] if p < s.n_in else op.outs[p - s.n_in]


def _analyse(b: Body, summaries: dict[int, _Summary]) -> _Analysis:
    """The value classes of `b` and what reaches each, its callees summarised in
    `summaries`."""
    uf = UnionFind(len(b.bits))
    # A call result carrying the value of an argument or of an earlier result joins that
    # value's class rather than starting one.
    for op in b.ops:
        if isinstance(op.callee, Prim):
            continue
        s = summaries[op.callee]
        for j, p in enumerate(s.alias):
            if p >= 0:
                uf.union(op.outs[j], _at(op, s, p))
    cls = [uf.find(w) for w in range(len(b.bits))]

    first: dict[int, int] = {}
    for k, w in enumerate(b.inputs + b.outputs):
        first.setdefault(cls[w], k)

    into = [0] * len(b.bits)
    created = [(-1, 0)] * len(b.bits)

    def read(w: int) -> int:
        c = cls[w]
        k = first.get(c)
        return into[c] if k is None else 1 << k

    for n, op in enumerate(b.ops):
        if isinstance(op.callee, Prim):
            joint = 0
            for w in op.args:
                joint |= read(w)
            for w in op.outs:
                into[cls[w]] = joint
                created[cls[w]] = (n, 0)
            continue
        s = summaries[op.callee]
        for r, j in enumerate(s.order):
            mask = 0
            for p in set_bits(s.masks[j]):
                mask |= read(_at(op, s, p))
            c = cls[op.outs[j]]
            into[c] = mask
            created[c] = (n, r)
    return _Analysis(cls, first, into, created)
