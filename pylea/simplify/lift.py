"""Flip lifting.

Take a wire `e` read by several disjunctions and by nothing but `AND` and `OR` gates, and
the gate `d` through which every path from `e` up to a wire that leaves the circuit passes.
Where `e` being true forces `d` true, the value of `d` only matters while `e` is false, so
`e` may be set to false in the computation of `d`: every disjunction `OR(e, b)` becomes `b`
and every conjunction `AND(e, b)` becomes 0, and `e` moves into one new disjunction
`OR(d, e)` that takes the place of `d`.  Dually, where `e` being false forces `d` false,
`e` is set to true below `d` and re-entered once at a conjunction.  Either way `e` ends
with one reader.

Three conditions are needed.  `d` dominates `e`: a wire leaving the circuit is a body
output or an argument of any op that is not a gate, and every path from `e` up to one
passes through `d`, so the gates on those paths form a block whose only exit is `d` and
the rewrite inside it cannot be seen except through `d`.  `e` forces `d`, which is
decided by propagating the value of `e` forwards through the gates, syntactic implication
in the sense of `pylea.simplify.implication` restricted to what follows from one literal.
Every reader of `e` is an `AND` or an `OR` reading `e` once, and at least two of them are
disjunctions, or conjunctions for the dual, so that the rewrite deletes more than it adds.

`e` is a coin flip in the sense of `pylea.simplify.common.is_flip`, a primitive reading
nothing and producing `e` alone: its op has the one wire `e` and moves to the new gate at no
cost.
A primitive with further results cannot move, as they are read where they are, and lifting
a wire computed from other wires can raise the cost, as its op stays where its arguments are
while `e` has to reach `d`; what such a wire's reads admit is left to `pylea.simplify.mask`.

The region of a lift is the set of gates reached by going up from `e` and stopping at `d`,
`d`'s own included.  A wire of the region other than `d` is read by gates of the region
alone, since a reader of any other kind would be a way out of the circuit that avoids `d`.
So the rewrite changes no op outside the region, except that the op producing `e` moves and
ops left unread go.  Whether the conditions of a lift hold depends on its region, and on `d`
being read at all: a lift with a disjoint region either leaves them as they were or leaves
`d` unread, and the whole region with it, whose gates the sweep then deletes.  Each round
proposes sites from one dominator pass and applies, in one sweep, every site whose region is
disjoint from those of the sites taken before it.  A site whose region meets one of those is
checked again against the body as it then stands, and applied on its own where it still
holds.  The rounds go on until one proposes no site.
"""
from __future__ import annotations

from collections.abc import Iterator

from pylea.ir.body import Body, Op, prune_wires
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program

from pylea.simplify.common import (
    Circuit, Propagation, Sweep, constant_bits, each_body, expired, gate_tag, is_flip,
    leaving,
)

ROOT = -1                      # the virtual node every root wire hangs from


def lift(prog: Program, deadline: float | None = None) -> Program:
    """`prog` with every wire lifted wherever the conditions hold.

    `deadline` is a `time.perf_counter` reading past which no further site is looked for
    or checked again; the sites with disjoint regions found by then are applied, in one
    sweep of the body.

    Here `x` is read by two disjunctions under a third, so it comes out read once:

        >>> import numpy as np
        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import Program, consumer_counts
        >>> from pylea.ir.reference import semantics
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (p,) = b.emit(Prim.flip(0.4), ())
        >>> (q,) = b.emit(Prim.flip(0.5), ())
        >>> (a,) = b.emit(Prim.or_(), (x, p))
        >>> (c,) = b.emit(Prim.or_(), (x, q))
        >>> (d,) = b.emit(Prim.or_(), (a, c))
        >>> prog = Program((b.finish(inputs=(), outputs=(d,)),), root=0)
        >>> lifted = lift(prog)
        >>> consumer_counts(prog.main)[x], consumer_counts(lifted.main)[x]
        (2, 1)
        >>> bool(np.allclose(semantics(prog), semantics(lifted)))
        True
    """
    return each_body(prog, lambda body: _lift_body(body, deadline), deadline)


def _lift_body(body: Body, deadline: float | None) -> Body:
    """`body` after round upon round of lifts, until a round proposes no site or the
    deadline passes."""
    current = body
    while not expired(deadline):
        circuit = Circuit(current)
        taken: set[int] = set()
        disjoint, meeting = [], []
        for e, d, mode in _sites(current, circuit):
            region = _region(circuit, e, d)
            if region.isdisjoint(taken):
                taken |= region
                disjoint.append((e, d, mode))
            else:
                meeting.append((e, d, mode))
            if expired(deadline):
                break
        if not disjoint:
            break
        current = _apply(current, disjoint)
        for site in meeting:
            if expired(deadline):
                break
            if _still_a_site(current, *site):
                current = _apply(current, [site])
        current = prune_wires(current)
    return current


def _region(circuit: Circuit, e: int, d: int) -> set[int]:
    """The positions of the gates reached by going up from `e` and stopping at `d`."""
    top = circuit.made_at[d]
    region: set[int] = set()
    stack = list(circuit.readers.get(e, ()))
    while stack:
        j = stack.pop()
        if j in region:
            continue
        region.add(j)
        if j != top:
            stack.extend(circuit.readers.get(circuit.outs[j], ()))
    return region


# --------------------------------------------------------------------------- the graph

def _order(body: Body) -> list[int]:
    """The wires, every one after all of its readers' results, with `ROOT` first."""
    at: dict[int, int] = {w: -1 for w in range(len(body.bits))}
    for j, op in enumerate(body.ops):
        for w in op.outs:
            at[w] = j
    return [ROOT] + sorted(range(len(body.bits)), key=lambda w: -at[w])


def _idom(body: Body, roots: set[int]) -> dict[int, int]:
    """The immediate dominator of every wire reachable from `ROOT`.

    The graph is acyclic and `_order` visits a node after all of its predecessors, so one
    pass suffices.
    """
    predecessors: dict[int, list[int]] = {w: [] for w in range(len(body.bits))}
    for op in body.ops:
        for a in set(op.args):
            predecessors[a].extend(op.outs)
    for w in roots:
        predecessors[w].append(ROOT)

    order = _order(body)
    rank = {node: i for i, node in enumerate(order)}
    idom = {ROOT: ROOT}

    def meet(a: int, b: int) -> int:
        while a != b:
            while rank[a] > rank[b]:
                a = idom[a]
            while rank[b] > rank[a]:
                b = idom[b]
        return a

    for w in order[1:]:
        known = [p for p in predecessors[w] if p in idom]
        if not known:
            continue
        candidate = known[0]
        for p in known[1:]:
            candidate = meet(candidate, p)
        idom[w] = candidate
    return idom


def _dominates(body: Body, roots: set[int], d: int, e: int) -> bool:
    """Whether every path from a root to `e` passes through `d`, tested by deleting `d`."""
    below = {w: op.args for op in body.ops for w in op.outs}
    seen: set[int] = set()
    stack = [w for w in roots if w != d]
    while stack:
        w = stack.pop()
        if w == e:
            return False
        if w in seen or w == d:
            continue
        seen.add(w)
        stack.extend(below.get(w, ()))
    return True


# ------------------------------------------------------------------- forced values

def _forced(circuit: Circuit, e: int, value: int, d: int) -> bool:
    """Whether fixing `e` to `value` forces `d` to `value`, propagating through the gates."""
    propagation = Propagation(circuit, e, value)
    propagation.advance(circuit.made_at[d])
    return propagation.known.get(d) == value


# ------------------------------------------------------------------------ the sites

def _modes(body: Body, readers: dict[int, list[int]], e: int) -> tuple[str, ...]:
    """The kinds of lift, ``"or"`` before ``"and"``, that the readers of `e` admit.

    Every reader must be an `AND` or an `OR` reading `e` once, and at least two of them
    must be gates of the kind the lift deletes.
    """
    if e in body.outputs:
        return ()
    kinds = []
    for j in readers.get(e, ()):
        op = body.ops[j]
        tag = gate_tag(op)
        if tag not in (Tag.AND, Tag.OR) or op.args.count(e) != 1:
            return ()
        kinds.append(tag)
    return tuple(mode for mode, tag in (("or", Tag.OR), ("and", Tag.AND))
                 if kinds.count(tag) >= 2)


def _candidates(body: Body) -> list[int]:
    """The wires produced by a coin flip that is not a constant."""
    return [op.outs[0] for op in body.ops
            if is_flip(op.callee) and constant_bits(op.callee) is None]


def _sites(body: Body, circuit: Circuit) -> Iterator[tuple[int, int, str]]:
    """The lifts over `body`, as ``(e, d, mode)``, proposed from one dominator pass.

    `circuit` is the `Circuit` of `body`.  The sites are yielded one at a time so that a
    caller working to a deadline can stop part way.
    """
    readers = circuit.readers
    idom = _idom(body, leaving(body))
    for e in _candidates(body):
        modes = _modes(body, readers, e)
        if not modes:
            continue
        d = idom.get(e)
        if d is None or d == ROOT or d == e:
            continue
        for mode in modes:
            if _forced(circuit, e, 1 if mode == "or" else 0, d):
                yield e, d, mode
                break


def _still_a_site(body: Body, e: int, d: int, mode: str) -> bool:
    """Whether the lift ``(e, d, mode)``, proposed on an earlier form of `body`, holds on
    `body` as it now stands."""
    circuit = Circuit(body)
    if mode not in _modes(body, circuit.readers, e):
        return False
    at = circuit.made_at.get(d, -1)
    if at < 0 or circuit.tags[at] is None:
        return False
    if not _dominates(body, leaving(body), d, e):
        return False
    return _forced(circuit, e, 1 if mode == "or" else 0, d)


# ----------------------------------------------------------------------- the rewrite

def _apply(body: Body, sites: list[tuple[int, int, str]]) -> Body:
    """`body` with every lift in `sites`, whose regions are disjoint, applied.

    For a lift ``(e, d, mode)``, each reader of `e` of the kind the lift deletes becomes its
    other argument, and each reader of the other kind a constant.  The op producing `e`
    moves to just after the op producing `d`, followed by the new gate over `e` and the
    value of `d` computed below it.  The new gate is given the wire `d`, so nothing reading
    `d` has to change.  The op producing `d` is a gate, as the arguments of any other op
    leave the circuit and `d` could not then lie on every path from `e` to a wire that
    does.  It is rewritten to produce a fresh wire, which the new gate reads; where it is
    itself a gate the lift deletes, the new gate reads its other argument instead.
    """
    lifted = {e: mode for e, _, mode in sites}
    top = {d: e for e, d, _ in sites}
    moved: dict[int, Op] = {}
    s = Sweep(body)
    for op in body.ops:
        args = s.begin(op)
        if not op.args and op.outs and op.outs[0] in lifted:
            moved[op.outs[0]] = op              # emitted again next to the new gate
            continue
        e = next((w for w in args if w in lifted), None)
        d = op.outs[0] if op.outs and op.outs[0] in top else None
        assert e is None or d is None or top[d] == e     # the regions are disjoint
        if e is None and d is None:
            s.keep(op, args)
            continue
        if e is None:
            assert d is not None and isinstance(op.callee, Prim)
            e = top[d]
            below = s.fresh()
            s.emit(op.callee, args, (below,))
        elif gate_tag(op) is (Tag.OR if lifted[e] == "or" else Tag.AND):
            (below,) = [w for w in args if w != e]
            if d is None:
                s.alias(op.outs[0], below)
                continue
        else:
            constant = Prim.const(0 if lifted[e] == "or" else 1, 1)
            if d is None:
                s.replace(op, constant)
                continue
            below = s.fresh()
            s.emit(constant, (), (below,))
        source = moved.pop(e)
        assert isinstance(source.callee, Prim)
        s.emit(source.callee, (), source.outs)
        s.emit(Prim.or_() if lifted[e] == "or" else Prim.and_(), (below, e), (d,))
    assert not moved
    return s.finish(prune=False)
