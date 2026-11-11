"""Programs: a DPP held as a table of bodies with a root.

A call names a body by index, so a body called from several places is stored once and
compiled once.  `intern` merges equal bodies and drops those the root cannot reach;
`validate` checks, for a `Program` assembled by other means, the invariants a
`pylea.ir.body.BodyBuilder` maintains.  `intern_with_remap` also reports where each body
went, which is what a caller holding a table keyed by body index needs.

`reachable` lists the bodies reachable from the root, callees before callers, which is the
order every traversal here uses.  `use_counts` counts call sites per body across the whole
DPP rather than within one caller.

Four measures describe a body.  `consumer_counts` gives the readers of each wire, counting
each output position as a read.  `shared_reads` sums the readers beyond the first, so a
wire read three times counts twice, and `multi_parent_wires` instead counts the wires
having any.  `op_count` sums the ops over distinct reachable bodies, and `wire_slots`
counts every op argument and result plus the interface, which is the number of axes in
the body's network.

`discardable` and `deterministic` lift the two `pylea.ir.prim.Prim` properties to bodies: a
body has either where all of its ops do, recursively.  Neither converse holds, so `False`
means the property was not established rather than that it fails.

    >>> from pylea.ir.body import BodyBuilder
    >>> from pylea.ir.prim import Prim
    >>> b = BodyBuilder()
    >>> x = b.fresh()
    >>> (y,) = b.emit(Prim.not_(), (x,))
    >>> (z,) = b.emit(Prim.and_(), (y, y))
    >>> prog = Program((b.finish(inputs=(x,), outputs=(z,)),), root=0)
    >>> op_count(prog), shared_reads(prog.main), wire_slots(prog.main)
    (2, 1, 7)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pylea.ir.body import Body, Op, signature
from pylea.ir.prim import Prim


@dataclass(frozen=True, slots=True)
class Program:
    """A DPP: a table of bodies, one per definition, and the index of the main function."""

    bodies: tuple[Body, ...]
    root: int

    def body(self, i: int) -> Body:
        return self.bodies[i]

    @property
    def main(self) -> Body:
        return self.bodies[self.root]


def with_bodies(prog: Program, bodies) -> Program:
    """`prog` with `bodies` in its place, or `prog` itself where they are the same.

    Returning the argument unchanged is how a pass reports that it fired nowhere, which
    lets the driver skip interning and re-measuring.
    """
    bodies = tuple(bodies)
    return prog if bodies == prog.bodies else Program(bodies, prog.root)


def reachable(prog: Program) -> list[int]:
    """The body indices reachable from the root, callees before callers.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> reachable(both.program())
        [0, 1]
    """
    # Without recursion, so that a call graph of any depth can be walked.
    seen = {prog.root}
    order: list[int] = []
    stack = [(prog.root, iter(prog.bodies[prog.root].callees()))]
    while stack:
        body, pending = stack[-1]
        callee = next(pending, None)
        if callee is None:
            stack.pop()
            order.append(body)
        elif callee not in seen:
            seen.add(callee)
            stack.append((callee, iter(prog.bodies[callee].callees())))
    return order


def use_counts(prog: Program) -> Counter:
    """The number of call sites of each body, summed over the whole DPP.

    Counted across all callers rather than within one, so a body called once by each of two
    callers has a count of 2.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> dict(use_counts(both.program()))
        {0: 2}
    """
    counts: Counter = Counter()
    for i in reachable(prog):
        for c in prog.bodies[i].callees():
            counts[c] += 1
    return counts


def op_count(prog: Program) -> int:
    """The number of ops, summed over distinct reachable bodies.

    Counted per distinct body, so a body called twice contributes its ops once:

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> op_count(both.program())
        4
    """
    return sum(len(prog.bodies[i].ops) for i in reachable(prog))


def consumer_counts(body: Body) -> Counter:
    """How many times each wire of `body` is read, counting each output as one read.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> b = BodyBuilder()
        >>> x, y = b.fresh(), b.fresh()
        >>> (z,) = b.emit(Prim.and_(), (x, y))
        >>> body = b.finish(inputs=(x, y), outputs=(z, z))
        >>> dict(consumer_counts(body))
        {0: 1, 1: 1, 2: 2}
        >>> multi_parent_wires(body), shared_reads(body), wire_slots(body)
        (1, 1, 7)
    """
    counts: Counter = Counter()
    for op in body.ops:
        for w in op.args:
            counts[w] += 1
    for w in body.outputs:
        counts[w] += 1
    return counts


def multi_parent_wires(body: Body) -> int:
    """The number of wires of `body` that are read more than once."""
    return sum(1 for n in consumer_counts(body).values() if n >= 2)


def shared_reads(body: Body) -> int:
    """The reads beyond the first, summed over the wires of `body`.

    How much sharing there is, rather than how many wires carry any: a wire read three
    times counts twice, where `multi_parent_wires` counts it once.
    """
    return sum(n - 1 for n in consumer_counts(body).values() if n >= 2)


def wire_slots(body: Body) -> int:
    """The wire positions of `body`: every op argument and result, plus its interface.

    One position is one axis of one tensor in the body's network, so this is the network's
    total rank.  It falls where an op loses an argument without being removed, and where a
    body stops returning something, neither of which `op_count` or `shared_reads` can see.
    """
    return (sum(len(op.args) + len(op.outs) for op in body.ops)
            + len(body.inputs) + len(body.outputs))


def discardable(prog: Program) -> dict[int, bool]:
    """Which bodies may be deleted at a call site whose results nothing reads.

        >>> from pylea import DPP, Bool, flip, observe
        >>> @DPP
        ... def conditioned() -> Bool:
        ...     v = flip(0.5)
        ...     observe(v)
        ...     return v
        >>> discardable(conditioned.program())
        {0: False}


    A body all of whose ops are discardable qualifies: summing out its outputs sums out the
    last op's, leaving 1, and so on down.  The converse does not hold, since a body may
    condition and still sum to 1, so `False` here means "not established" rather than
    "known to condition".
    """
    return _propagate(prog, lambda p: p.is_discardable)


def deterministic(prog: Program) -> dict[int, bool]:
    """Which bodies are a function of their inputs, so that CSE may merge two calls to them.

        >>> from pylea import DPP, Bool
        >>> @DPP
        ... def gate(a: Bool, b: Bool) -> Bool:
        ...     return a & b
        >>> deterministic(gate.program())
        {0: True}


    A body all of whose ops are deterministic qualifies: every op's outputs are fixed by
    its arguments, so at most one choice of values for the internal wires contributes, and
    it contributes 1.
    """
    return _propagate(prog, lambda p: p.is_deterministic)


def _propagate(prog: Program, holds_of) -> dict[int, bool]:
    """`holds_of` on every primitive of a body and, recursively, of every body it calls."""
    result: dict[int, bool] = {}
    for i in reachable(prog):                       # callees first
        result[i] = all(holds_of(op.callee) if isinstance(op.callee, Prim)
                        else result[op.callee]
                        for op in prog.bodies[i].ops)
    return result


def intern(prog: Program) -> Program:
    """An equivalent DPP with equal bodies merged and unreachable ones dropped.

    Merging is bottom-up, a body's identity including the indices of its callees, so two
    equal bodies compare equal only once their own callees have been merged.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> def coin_body():
        ...     b = BodyBuilder()
        ...     (x,) = b.emit(Prim.flip(0.5), ())
        ...     return b.finish(inputs=(), outputs=(x,))
        >>> caller = BodyBuilder(bodies=[coin_body(), coin_body()])
        >>> (a,) = caller.emit(0, ())
        >>> (c,) = caller.emit(1, ())        # an identical body, stored twice
        >>> (out,) = caller.emit(Prim.and_(), (a, c))
        >>> prog = Program((coin_body(), coin_body(),
        ...                 caller.finish(inputs=(), outputs=(out,))), root=2)
        >>> len(prog.bodies), len(intern(prog).bodies)
        (3, 2)
    """
    return intern_with_remap(prog)[0]


def intern_with_remap(prog: Program) -> tuple[Program, dict[int, int]]:
    """`intern(prog)`, and the map from old body index to new that it applied.

    The map covers the reachable bodies only, and several of them share an index wherever
    they were merged.  Anything a caller holds keyed by body index, such as a table of
    names, needs this to be carried across.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> merged, remap = intern_with_remap(both.program())
        >>> len(remap) >= len(merged.bodies)
        True
    """
    # Bottom-up: a body's identity includes its callees' indices, so two equal bodies only
    # compare equal once their callees have been merged.
    remap: dict[int, int] = {}
    canon: dict[Body, int] = {}
    bodies: list[Body] = []

    for i in reachable(prog):                       # callees first
        b = prog.bodies[i]
        rewritten = Body(
            b.bits, b.inputs, b.outputs,
            tuple(Op(op.callee if isinstance(op.callee, Prim) else remap[op.callee],
                     op.args, op.outs)
                  for op in b.ops),
        )
        j = canon.get(rewritten)
        if j is None:
            j = len(bodies)
            canon[rewritten] = j
            bodies.append(rewritten)
        remap[i] = j

    return Program(tuple(bodies), remap[prog.root]), remap


def validate(prog: Program) -> None:
    """Nothing, or `ValueError` if `prog` violates the invariants `BodyBuilder` maintains."""
    for i, b in enumerate(prog.bodies):
        produced: set[int] = set()
        for op in b.ops:
            in_w, out_w = signature(op.callee, prog.bodies)
            if len(op.args) != len(in_w) or len(op.outs) != len(out_w):
                raise ValueError(f"body {i}: arity mismatch in {op}")
            for w, want in zip(op.args, in_w):
                if b.bits[w] != want:
                    raise ValueError(f"body {i}: {op} argument wire {w} has width "
                                     f"{b.bits[w]}, expected {want}")
            for w in op.outs:
                if w in produced:
                    raise ValueError(f"body {i}: wire {w} produced twice (not SSA)")
                produced.add(w)
            if not isinstance(op.callee, Prim) and not 0 <= op.callee < len(prog.bodies):
                raise ValueError(f"body {i}: call to unknown body {op.callee}")
        for w in b.inputs:
            if w in produced:
                raise ValueError(f"body {i}: input wire {w} is also produced by an op")
        for w in {w for op in b.ops for w in op.args} | set(b.outputs):
            if w not in produced and w not in b.inputs:
                raise ValueError(
                    f"body {i}: wire {w} is read but neither an input nor produced")
        if len(set(b.inputs)) != len(b.inputs):
            raise ValueError(f"body {i}: input wire repeated in {b.inputs}")
    if not 0 <= prog.root < len(prog.bodies):
        raise ValueError(f"root {prog.root} is not a body index")
