"""Constant folding, and common subexpression elimination at the bottom of a body.

A source is a primitive that reads nothing: a `flip`, a constant, or a table with no
parents.  `const_fold` multiplies a source into the one op reading its results, so that a
`flip` feeding a table becomes a column of the table and the negation of a `flip` becomes
the `flip` of the opposite probability.  A gate is folded this way only when all of its
arguments are such sources, the result then reading nothing; a gate with one source
argument and one other keeps its tag, which the rules matching on tags rely on.

`block_fold` folds a block of gates computed from sources of one bit alone, the coin flips
of `pylea.simplify.common.is_flip`, into one source.  The block is the cone of gates below
one wire `y`, and it must be foldable: the two arguments of each `AND` depend on no common
source, and the two arguments of each `OR` either depend on no common source or are never
both true, which is tested by unit propagation on the clauses of the body's gates.  The
weight of `y` is then computed bottom up, as in a circuit in deterministic decomposable
negation normal form.  No wire of the block but `y` may be read outside it.  A source of the
block may be, provided it is a distribution and the block is guarded: a literal guards a
wire if the wire only matters while the literal is true, and the guards of `y` together with
the guards of the source, taken over the source's readers outside the block, must be
contradictory under unit propagation.  The block is then evaluated on a private copy of that
source, and the source stays for its other readers.  Sources read nowhere else go with the
block, and their weight goes into the new source, so a source whose column does not sum to 1
is folded exactly.

`bottom_cse` merges several gates with the same callee and the same arguments whose joint
cone is such a block, with every source of it read inside the block alone, into one source
that every reader of the gates now reads.  The gates are equal for every choice of values
that contributes, so the merge is sound, and folding the cone at the same time is what keeps
the contraction cost from rising.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from pylea.ir.bits import set_bits
from pylea.ir.body import Body, Op, prune_wires
from pylea.ir.prim import Prim, Tag, is_stochastic
from pylea.ir.program import Program

from pylea.simplify.common import (
    canonical, composed, constant_bits, each_body, expired, gate_tag, is_flip, is_gate,
    is_source, neg, pos,
)


def const_fold(prog: Program) -> Program:
    """`prog` with each source multiplied into the one op reading its results.

    A `flip` feeding one table becomes a column of the table:

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import op_count
        >>> b = BodyBuilder()
        >>> (c,) = b.emit(Prim.flip(0.3), ())
        >>> (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (c,))
        >>> prog = Program((b.finish(inputs=(), outputs=(m,)),), root=0)
        >>> op_count(prog), op_count(const_fold(prog))
        (2, 1)
    """
    return each_body(prog, _fold_sources)


def block_fold(prog: Program, deadline: float | None = None) -> Program:
    """`prog` with every foldable block of gates over sources folded into one source.

    `deadline` is a `time.perf_counter` reading past which no further block is
    considered; the blocks folded before it are complete rewrites.

    Two coin flips feeding a chain of gates that no other op reads become one coin flip:

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import op_count
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.5), ())
        >>> (y,) = b.emit(Prim.flip(0.5), ())
        >>> (n,) = b.emit(Prim.not_(), (y,))
        >>> (a,) = b.emit(Prim.and_(), (x, n))
        >>> prog = Program((b.finish(inputs=(), outputs=(a,)),), root=0)
        >>> op_count(prog), block_fold(prog).main.ops[0].callee
        (4, Prim(tag=<Tag.FLIP: 5>, in_bits=0, out_bits=1, params=(0.25,)))
    """
    return each_body(prog, lambda body: _fold_blocks(body, deadline))


def bottom_cse(prog: Program, deadline: float | None = None) -> Program:
    """`prog` with equal gates at the bottom of a body merged and their cone folded.

    `deadline` is a `time.perf_counter` reading past which no further gates are
    considered; the merges made before it are complete rewrites.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import op_count
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.5), ())
        >>> (y,) = b.emit(Prim.flip(0.5), ())
        >>> (a,) = b.emit(Prim.and_(), (x, y))
        >>> (c,) = b.emit(Prim.and_(), (x, y))
        >>> prog = Program((b.finish(inputs=(), outputs=(a, c)),), root=0)
        >>> op_count(prog), op_count(bottom_cse(prog))
        (4, 1)
    """
    return each_body(prog, lambda body: _merge_bottom(body, deadline))


# --------------------------------------------------------------- sources into readers

def _fold_sources(body: Body) -> Body:
    readers: dict[int, set[int]] = defaultdict(set)
    for k, op in enumerate(body.ops):
        for w in op.args:
            readers[w].add(k)
    outputs = set(body.outputs)

    # host op -> the sources folded into it.  A constant is folded into every reader that
    # admits it, each reader taking a private copy; any other source into its one reader.
    by_host: dict[int, list[int]] = defaultdict(list)
    for k, op in enumerate(body.ops):
        if not is_source(op.callee):
            continue
        hosts = set().union(*(readers[w] for w in op.outs))
        constant = constant_bits(op.callee) is not None
        if not constant and (len(hosts) != 1 or any(w in outputs for w in op.outs)
                             or any(not readers[w] for w in op.outs)):
            continue
        for j in hosts:
            if _admits(body.ops[j], constant):
                by_host[j].append(k)
    for j, ks in list(by_host.items()):
        host = body.ops[j]
        fed = {w for k in ks for w in body.ops[k].outs}
        if is_gate(host.callee) and not all(w in fed for w in host.args):
            del by_host[j]
    if not by_host:
        return body

    gone = {k for ks in by_host.values() for k in ks}
    for k in list(gone):                            # a constant stays for its other readers
        op = body.ops[k]
        if any(w in outputs for w in op.outs) or any(
                j not in by_host or k not in by_host[j]
                for w in op.outs for j in readers[w]):
            gone.discard(k)

    ops: list[Op] = []
    for j, op in enumerate(body.ops):
        if j in gone:
            continue
        for k in by_host.get(j, ()):
            source = body.ops[k]
            args = tuple(w for w in op.args if w not in source.outs)
            op = Op(composed(source, op, op.outs, args), args, op.outs)
        ops.append(op)
    return prune_wires(Body(body.bits, body.inputs, body.outputs, tuple(ops)))


def _admits(host: Op, constant: bool) -> bool:
    """Whether `host` may absorb a source, a constant one where `constant`.

    A call cannot absorb a source.  A constant feeding a gate or a selection is the
    Boolean rules' case, and folding it here would leave a matrix where those rules expect
    a tag.  A gate absorbs a `flip` only together with its every other argument, so that
    the result reads nothing, since a gate reading a `flip` and something else is what the
    rules over gates match on; that condition is checked by the caller over all of the
    gate's arguments.
    """
    if isinstance(host.callee, int):
        return False
    if constant:
        return not (is_gate(host.callee) or host.callee.tag is Tag.ITE)
    return True


# ------------------------------------------------------------------ unit propagation

class _Clauses:
    """The clauses of a body's gates, and unit propagation over them.

    Literals are numbered by `pylea.simplify.common.pos` and `pylea.simplify.common.neg`.
    `conflict` says whether propagating `units` derives both values of some wire, with the
    clauses of the ops in `excluded` left out.
    """

    def __init__(self, body: Body) -> None:
        self.wires = len(body.bits)
        self.clauses: list[tuple[int, ...]] = []
        self.owner: list[int] = []
        self.watching: dict[int, list[int]] = defaultdict(list)
        for k, op in enumerate(body.ops):
            tag = gate_tag(op)
            if tag is None:
                continue
            (y,) = op.outs
            if tag is Tag.NOT:
                (x,) = op.args
                self._add(k, (pos(y), pos(x)), (neg(y), neg(x)))
            elif tag is Tag.AND:
                a, b = op.args
                self._add(k, (neg(y), pos(a)), (neg(y), pos(b)), (pos(y), neg(a), neg(b)))
            else:
                a, b = op.args
                self._add(k, (pos(y), neg(a)), (pos(y), neg(b)), (neg(y), pos(a), pos(b)))

    def _add(self, owner: int, *clauses: tuple[int, ...]) -> None:
        for clause in clauses:
            index = len(self.clauses)
            self.clauses.append(clause)
            self.owner.append(owner)
            for lit in clause:
                self.watching[lit ^ 1].append(index)     # falsified when `lit ^ 1` holds

    def propagated(self, units, excluded: int = 0) -> list[int] | None:
        """The wire values unit propagation reaches from `units`, or `None` on a
        conflict."""
        value = [-1] * self.wires
        return value if self._propagate(units, value, excluded) else None

    def conflict(self, units, excluded: int = 0, start: list[int] | None = None) -> bool:
        """Whether propagating `units` on top of the wire values `start` derives a
        conflict."""
        value = [-1] * self.wires if start is None else list(start)
        return not self._propagate(units, value, excluded)

    def _propagate(self, units, value: list[int], excluded: int) -> bool:
        queue: list[int] = []
        for lit in units:
            if not self._assign(lit, value, queue):
                return False
        clauses, owner, watching = self.clauses, self.owner, self.watching
        while queue:
            lit = queue.pop()
            for index in watching.get(lit, ()):
                if excluded and (excluded >> owner[index]) & 1:
                    continue
                unassigned = None
                satisfied = False
                for other in clauses[index]:
                    v = value[other >> 1]
                    if v < 0:
                        if unassigned is not None:
                            unassigned = -1          # two free literals, nothing to do
                            break
                        unassigned = other
                    elif v == (other & 1) ^ 1:
                        satisfied = True
                        break
                if satisfied or unassigned == -1:
                    continue
                if unassigned is None:
                    return False
                if not self._assign(unassigned, value, queue):
                    return False
        return True

    @staticmethod
    def _assign(lit: int, value: list[int], queue: list[int]) -> bool:
        wire, v = lit >> 1, (lit & 1) ^ 1
        known = value[wire]
        if known < 0:
            value[wire] = v
            queue.append(lit)
            return True
        return known == v


# ------------------------------------------------------------------------- guards

def _dependents(body: Body) -> list[int]:
    """For each wire, the wires depending on it, itself included, as a wire bitset."""
    above = [0] * len(body.bits)
    for k in reversed(range(len(body.ops))):
        op = body.ops[k]
        reached = 0
        for w in op.outs:
            reached |= above[w] | (1 << w)
        for w in op.args:
            above[w] |= reached
    for w in range(len(body.bits)):
        above[w] |= 1 << w
    return above


def _guard_of(op: Op, v: int) -> tuple[int, int] | None:
    """The literal `op` contributes to the guards of its argument `v`, and `op`'s result.

    `None` where `op` is not a gate.  The literal is given as ``(wire, negated)`` and is
    `None` for a `NOT` and for a gate reading `v` twice, which contribute the guards of
    their result alone.
    """
    tag = gate_tag(op)
    if tag is None:
        return None
    if tag is Tag.NOT or op.args[0] == op.args[1]:
        return (-1, 0)
    u = op.args[1] if op.args[0] == v else op.args[0]
    return (u, 0 if tag is Tag.AND else 1)


class _Guards:
    """The guards of every wire of a body, as two wire bitsets per wire.

    ``positive[v]`` holds the wires `u` such that `u` guards `v`, and ``negative[v]`` those
    such that `NOT u` does.  `of` gives the two for a wire over a subset of its readers,
    with a subset of the ops absent, which is what the guards of a source look like once a
    block it fed has been folded.
    """

    def __init__(self, body: Body, readers: dict[int, list[int]]) -> None:
        self.body = body
        self.readers = readers
        self.outputs = set(body.outputs)
        self.above = _dependents(body)
        self.positive = [0] * len(body.bits)
        self.negative = [0] * len(body.bits)
        produced_at = {w: k for k, op in enumerate(body.ops) for w in op.outs}
        # Readers first: a wire's guards are read off the guards of its readers' results.
        for v in sorted(range(len(body.bits)), key=lambda w: -produced_at.get(w, -1)):
            self.positive[v], self.negative[v] = self.of(v)

    def of(self, v: int, excluded: int = 0) -> tuple[int, int]:
        """The guards of `v` over its readers outside the ops in `excluded`, which must be
        ops reading no result of a reader of `v`; the dependents of `v` are then the wires
        reached through those readers alone."""
        if v in self.outputs:
            return 0, 0
        positive = negative = -1
        above = 1 << v
        for k in self.readers[v]:
            if (excluded >> k) & 1:
                continue
            op = self.body.ops[k]
            guard = _guard_of(op, v)
            if guard is None:
                return 0, 0
            u, negated = guard
            (out,) = op.outs
            p, n = self.positive[out], self.negative[out]
            if u >= 0:
                if negated:
                    n |= 1 << u
                else:
                    p |= 1 << u
            positive &= p
            negative &= n
            above |= self.above[out]
        if positive == -1:                          # no reader at all
            return 0, 0
        return positive & ~above, negative & ~above


def _literals(positive: int, negative: int):
    """The literals of two guard bitsets."""
    for w in set_bits(positive):
        yield pos(w)
    for w in set_bits(negative):
        yield neg(w)


# ------------------------------------------------------------------------- blocks

class _Blocks:
    """The gates of a body computed from coin flips alone, with their cones and weights.

    A source here is a coin flip in the sense of `pylea.simplify.common.is_flip`.

    For each such gate's result wire `y`: `at` is the position of the op producing it;
    `dep` the set of sources it depends on, as a bitset over source numbers; `cone` the
    ops below it, itself included, as a bitset over op positions; `inside` the readers of
    every wire of the cone but `y`, as a bitset over op positions, with bit `external`
    standing for the body's outputs; `foldable` whether the cone is a circuit whose weight
    can be computed bottom up; and `probability` the probabilities of `y` being false and
    true, each source's column taken divided by its sum.  The weight of each value of `y`
    is its probability times the product of the sums of its sources' columns.

    The two probabilities are carried separately rather than one as 1 minus the other, so
    that each is a sum of products of non-negative numbers, as in the contraction the fold
    replaces, and keeps its relative precision however close the other comes to 1.  The one
    exception is a disjunction of two arguments never both true, whose probability of being
    false is a difference; it is not negative, and a difference below 0 by round-off is
    taken as 0.

    `deadline` is a `time.perf_counter` reading past which no further gate is recorded;
    the gates recorded before it are a prefix of the body, each with its whole cone.
    """

    def __init__(self, body: Body, deadline: float | None = None) -> None:
        self.body = body
        self.external = len(body.ops)
        self.readers: dict[int, list[int]] = defaultdict(list)
        self.reading: dict[int, int] = defaultdict(int)     # wire -> reader ops, as bits
        for k, op in enumerate(body.ops):
            for w in op.args:
                self.readers[w].append(k)
                self.reading[w] |= 1 << k
        for w in body.outputs:
            self.reading[w] |= 1 << self.external
        self.clauses: _Clauses | None = None
        self.guards: _Guards | None = None

        self.source_bit: dict[int, int] = {}                # source wire -> its number
        self.source_wire: list[int] = []                    # number -> source wire
        self.source_mass: list[float] = []
        self.source_normalised: list[bool] = []
        self.at: dict[int, int] = {}
        self.dep: dict[int, int] = {}
        self.cone: dict[int, int] = {}
        self.inside: dict[int, int] = {}
        self.foldable: dict[int, bool] = {}
        self.probability: dict[int, tuple[float, float]] = {}
        self.gates: list[int] = []                          # the grounded gates' wires

        for k, op in enumerate(body.ops):
            if expired(deadline):
                break
            if is_flip(op.callee):
                self._source(k, op)
            elif gate_tag(op) is not None and all(w in self.dep for w in op.args):
                self._gate(k, op)

    def _source(self, k: int, op: Op) -> None:
        assert isinstance(op.callee, Prim)
        (w,) = op.outs
        column = np.asarray(op.callee.matrix, dtype=float)[:, 0]
        mass = float(column.sum())
        number = len(self.source_mass)
        self.source_bit[w] = number
        self.source_wire.append(w)
        self.source_mass.append(mass)
        self.source_normalised.append(is_stochastic(op.callee.matrix))
        self.at[w] = k
        self.dep[w] = 1 << number
        self.cone[w] = 0
        self.inside[w] = 0
        self.foldable[w] = True
        # A source of no weight at all leaves its block none; it is never kept for other
        # readers, not being a distribution, so `folded` multiplies this by 0.
        self.probability[w] = ((float(column[0]) / mass, float(column[1]) / mass) if mass
                               else (0.0, 0.0))

    def _gate(self, k: int, op: Op) -> None:
        tag = gate_tag(op)
        (y,) = op.outs
        self.gates.append(y)
        self.at[y] = k
        self.dep[y] = _union(self.dep[x] for x in op.args)
        self.cone[y] = (1 << k) | _union(self.cone[x] for x in op.args)
        self.inside[y] = _union(self.reading[x] | self.inside[x]
                                for x in op.args if x not in self.source_bit)
        foldable = all(self.foldable[x] for x in op.args)
        p = self.probability
        if tag is Tag.NOT:
            (x,) = op.args
            false, true = p[x]
            p[y] = (true, false)
        else:
            a, b = op.args
            (fa, ta), (fb, tb) = p[a], p[b]
            disjoint = not (self.dep[a] & self.dep[b])
            if tag is Tag.AND:
                foldable = foldable and disjoint
                p[y] = (fa * fb + fa * tb + ta * fb, ta * tb)
            elif disjoint:
                p[y] = (fa * fb, ta * tb + ta * fb + fa * tb)
            elif foldable and self._conflict((pos(a), pos(b))):
                p[y] = (max(fa - tb, 0.0), ta + tb)         # never both true
            else:
                foldable = False
                p[y] = (0.0, 0.0)
        self.foldable[y] = foldable

    def _conflict(self, units) -> bool:
        if self.clauses is None:
            self.clauses = _Clauses(self.body)
        return self.clauses.conflict(units)

    def closed(self, y: int, cone: int | None = None) -> bool:
        """Whether no wire of the cone of `y` but `y` itself is read outside it."""
        cone = self.cone[y] if cone is None else cone
        return not (self.inside[y] & ~cone)

    def shared_sources(self, y: int, cone: int | None = None) -> list[int]:
        """The sources of `y` with a reader outside its cone."""
        cone = self.cone[y] if cone is None else cone
        return [w for w in self.sources(self.dep[y]) if self.reading[w] & ~cone]

    def sources(self, dep: int) -> list[int]:
        """The wires of the sources in `dep`."""
        return [self.source_wire[number] for number in set_bits(dep)]

    def guarded(self, y: int, sources, folded: int = 0) -> bool:
        """Whether `y` never matters at the same time as any of `sources`, once the cone
        of `y` and the cones in `folded` are folded.

        The guards of `y` are propagated once, and the guards of each source on top of
        that; a source whose guards hold the negation of one of them, or of each other,
        needs no propagation.
        """
        if self.guards is None:
            self.guards = _Guards(self.body, self.readers)
        cone = self.cone[y] | folded
        positive_y, negative_y = self.guards.positive[y], self.guards.negative[y]
        pending: list[tuple[int, int]] = []
        for e in sources:
            positive, negative = self.guards.of(e, excluded=cone)
            if not positive and not negative:
                return False
            if not (positive & negative_y or negative & positive_y or positive & negative):
                pending.append((positive, negative))
        if not pending:
            return True
        if self.clauses is None:
            self.clauses = _Clauses(self.body)
        start = self.clauses.propagated(_literals(positive_y, negative_y), excluded=cone)
        if start is None:
            return True                             # `y` never matters at all
        return all(self.clauses.conflict(_literals(*guards), excluded=cone, start=start)
                   for guards in pending)

    def folded(self, y: int, dep: int | None = None, cone: int | None = None) -> Prim:
        """The source standing for `y` once its cone is folded, its weight over the sources
        of the cone that are read inside it alone."""
        cone = self.cone[y] if cone is None else cone
        dep = self.dep[y] if dep is None else dep
        # A source kept for its other readers is a distribution, its column summing to 1,
        # so the weight of `y` over all of its sources is the weight over the removed ones.
        mass = 1.0
        for w in self.sources(dep):
            if not self.reading[w] & ~cone:
                mass *= self.source_mass[self.source_bit[w]]
        false, true = self.probability[y]
        return canonical(Prim.from_matrix([[mass * false], [mass * true]]))

    def removed(self, cone: int, dep: int) -> int:
        """The ops of `cone` together with the sources in `dep` read inside it alone."""
        for w in self.sources(dep):
            if not self.reading[w] & ~cone:
                cone |= 1 << self.at[w]
        return cone


def _union(masks) -> int:
    total = 0
    for m in masks:
        total |= m
    return total


def _fold_blocks(body: Body, deadline: float | None = None) -> Body:
    blocks = _Blocks(body, deadline)
    qualifying: list[int] = []
    covered = 0
    # Top down, so that a block inside a qualifying one is never considered on its own.
    for y in reversed(blocks.gates):
        if expired(deadline):
            break
        if (blocks.cone[y] & covered) or not blocks.foldable[y] or not blocks.closed(y):
            continue
        # The guards are read off the body as it stands once the blocks accepted so far
        # are folded, since a clause of one of those blocks no longer holds afterwards.
        shared = blocks.shared_sources(y)
        if any(not blocks.source_normalised[blocks.source_bit[e]] for e in shared):
            continue
        if shared and not blocks.guarded(y, shared, covered):
            continue
        qualifying.append(y)
        covered |= blocks.cone[y]
    if not qualifying:
        return body

    replacement: dict[int, Op] = {}
    removed = 0
    for y in qualifying:
        replacement[blocks.at[y]] = Op(blocks.folded(y), (), (y,))
        removed |= blocks.removed(blocks.cone[y], blocks.dep[y])
    return _rebuilt(body, replacement, removed)


def _rebuilt(body: Body, replacement: dict[int, Op], removed: int,
             aliased: dict[int, int] | None = None) -> Body:
    """`body` with the ops in `removed` gone, those in `replacement` replaced, and the wires
    in `aliased` read as their values."""
    aliased = aliased or {}
    ops = []
    for k, op in enumerate(body.ops):
        if k in replacement:
            ops.append(replacement[k])
        elif not (removed >> k) & 1:
            ops.append(Op(op.callee, tuple(aliased.get(w, w) for w in op.args), op.outs))
    outputs = tuple(aliased.get(w, w) for w in body.outputs)
    return prune_wires(Body(body.bits, body.inputs, outputs, tuple(ops)))


def _merge_bottom(body: Body, deadline: float | None = None) -> Body:
    blocks = _Blocks(body, deadline)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for y in blocks.gates:
        op = body.ops[blocks.at[y]]
        groups[(op.callee, op.args)].append(blocks.at[y])

    replacement: dict[int, Op] = {}
    aliased: dict[int, int] = {}
    removed = 0
    for members in groups.values():
        if expired(deadline):
            break
        if len(members) < 2:
            continue
        wires = [body.ops[k].outs[0] for k in members]
        first = wires[0]
        cone = _union(blocks.cone[w] for w in wires)
        if (cone & removed) or not blocks.foldable[first] or not blocks.closed(first, cone):
            continue
        if blocks.shared_sources(first, cone):
            continue
        replacement[members[0]] = Op(blocks.folded(first, cone=cone), (), (first,))
        for w in wires[1:]:
            aliased[w] = first
        removed |= blocks.removed(cone, blocks.dep[first]) & ~(1 << members[0])
    if not replacement:
        return body
    return _rebuilt(body, replacement, removed, aliased)
