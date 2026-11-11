"""What the rules share: reading an op, composing two ops, and rebuilding a body op by op.

`Sweep` rebuilds one body in op order.  A rule that deletes an op aliases the op's result
wire to the wire that now stands for it, and every later op reads through the alias.  The
sweep keeps count of how often each wire is still read, which is what a rule needs to know
whether a wire is internal to the pattern it matched: a pattern may delete a wire only if
nothing outside the pattern reads it.

`is_gate` says whether a callee is one of `AND`, `OR` and `NOT`, the only ops the Boolean
rules match on.  `is_source` says whether a callee is a primitive that reads nothing, and
`is_flip` whether it moreover produces one bit, which is what a coin flip is to the rules:
a `flip`, a one-bit constant, or a table over one bit with no parents, whatever its weights.
A source with several results is not a coin flip, since its results are produced together.
`constant_bits` reads the value a source always produces off its matrix, so a `flip` of
exactly 0 or 1 and a table concentrated on one pattern count as constants alongside a
written one.  `admitted_pattern` reads, off a primitive that produces nothing, the one
argument pattern it gives weight to, which is what an `observe` is to the rules.

`composed` multiplies the matrices of two ops that share wires into one primitive, and
`canonical` turns a primitive that reads nothing into the `flip` or constant whose matrix it
exactly is, so that one matrix is written one way whichever rule produced it.  `evaluated`
rewrites a gate or a selection some of whose arguments have known values, which is how both
the constants of `pylea.simplify.boolean` and the observed values of
`pylea.simplify.observe` are propagated.

`leaving`, `Circuit` and `Propagation` are what the rules over the gates of a body share:
the wires through which those gates are read, the gates as tables, and what the value of
one wire decides through them.  A literal, in `pos` and `neg`, is a one-bit wire or its
negation, numbered so that the two literals of one wire differ in the lowest bit.
"""
from __future__ import annotations

import heapq
import string
from collections import Counter
from collections.abc import Mapping
from time import perf_counter

import numpy as np

from pylea.ir.body import Body, Op, prune_wires
from pylea.ir.prim import Prim, Tag, constant_value
from pylea.ir.program import Program, consumer_counts, reachable, with_bodies

GATES = frozenset({Tag.AND, Tag.OR, Tag.NOT})

# Per gate, the argument value leaving the other argument alone.  The opposite value is the
# one deciding the result on its own.
UNIT = {Tag.AND: 1, Tag.OR: 0}

_ALPHABET = string.ascii_letters


def each_body(prog: Program, rewrite, deadline: float | None = None) -> Program:
    """`prog` with `rewrite`, a `Body -> Body` function, applied to every reachable body.

    A body `rewrite` returns unchanged is kept by identity, and `prog` is returned
    itself when no body changed, which is how a rule reports that it fired nowhere.
    `deadline` is a `time.perf_counter` reading past which no further body is started.
    """
    bodies = list(prog.bodies)
    for i in reachable(prog):
        if expired(deadline):
            break
        rebuilt = rewrite(prog.bodies[i])
        bodies[i] = prog.bodies[i] if rebuilt == prog.bodies[i] else rebuilt
    return with_bodies(prog, bodies)


def expired(deadline: float | None) -> bool:
    """Whether `deadline`, a `time.perf_counter` reading or `None` for none, has passed."""
    return deadline is not None and perf_counter() >= deadline


def is_gate(callee: Prim | int) -> bool:
    """Whether `callee` is `AND`, `OR` or `NOT`."""
    return isinstance(callee, Prim) and callee.tag in GATES


def gate_tag(op: Op) -> Tag | None:
    """The tag of `op`'s callee where it is a gate, else `None`."""
    callee = op.callee
    return callee.tag if isinstance(callee, Prim) and callee.tag in GATES else None


def is_source(callee: Prim | int) -> bool:
    """Whether `callee` is a primitive reading nothing and producing something."""
    return isinstance(callee, Prim) and callee.in_bits == 0 and callee.out_bits > 0


def is_flip(callee: Prim | int) -> bool:
    """Whether `callee` is a primitive reading nothing and producing one bit.

        >>> is_flip(Prim.flip(0.3)), is_flip(Prim.from_matrix([[0.2], [0.5]]))
        (True, True)
        >>> is_flip(Prim.from_matrix([[0.1], [0.2], [0.3], [0.4]]))
        False
    """
    return isinstance(callee, Prim) and callee.in_bits == 0 and callee.out_bits == 1


def is_effect(callee: Prim | int) -> bool:
    """Whether `callee` is a primitive producing nothing and reading something."""
    return isinstance(callee, Prim) and callee.out_bits == 0 and callee.in_bits > 0


def constant_bits(callee: Prim | int) -> tuple[int, ...] | None:
    """The bit each output wire of `callee` always carries, or `None` if it is not constant.

    Read off the matrix, so a `flip` of exactly 0 or 1 and a table concentrated on one
    pattern count as constants:

        >>> constant_bits(Prim.const(2, 2)), constant_bits(Prim.flip(1.0))
        ((1, 0), (1,))
        >>> constant_bits(Prim.flip(0.5)) is None
        True
    """
    if not isinstance(callee, Prim) or callee.in_bits:
        return None
    value = constant_value(callee)
    if value is None:
        return None
    n = callee.out_bits
    return tuple((value >> (n - 1 - i)) & 1 for i in range(n))


def admitted_pattern(callee: Prim | int) -> tuple[int, ...] | None:
    """The one argument pattern an effect `callee` admits, a bit per argument, or `None`.

    Read off the matrix rather than the tag, so an opaque matrix with no outputs counts
    alongside an `observe`.  The weight of the surviving entry does not matter: an op
    worth some weight at one pattern and zero elsewhere admits only that pattern.

        >>> admitted_pattern(Prim.observe())
        (1,)
        >>> admitted_pattern(Prim.and_()) is None
        True
    """
    if not is_effect(callee):
        return None
    assert isinstance(callee, Prim)
    row = np.asarray(callee.matrix, dtype=float)[0]
    (nonzero,) = np.nonzero(row)
    if len(nonzero) != 1:
        return None
    pattern = int(nonzero[0])
    return tuple((pattern >> (callee.in_bits - 1 - i)) & 1 for i in range(callee.in_bits))


def canonical(prim: Prim) -> Prim:
    """`prim`, or the `flip` or constant whose matrix is exactly that of `prim`.

    Only a primitive reading nothing is rewritten.  A column summing to 1 up to round-off
    is not a `flip`'s, whose false entry is ``1 - p`` computed in floating point, so such a
    column stays a table and its matrix stays as it was:

        >>> canonical(Prim.from_matrix([[0.75], [0.25]]))
        Prim(tag=<Tag.FLIP: 5>, in_bits=0, out_bits=1, params=(0.25,))
        >>> canonical(Prim.from_matrix([[0.0], [1.0]]))
        Prim(tag=<Tag.CONST: 4>, in_bits=0, out_bits=1, params=(1, 1))
        >>> canonical(Prim.from_matrix([[0.3], [0.7]])).tag      # 1 - 0.7 is not 0.3
        <Tag.MATRIX: 7>
    """
    if prim.tag is not Tag.MATRIX or prim.in_bits:
        return prim
    value = constant_value(prim)
    if value is not None:
        return Prim.const(value, prim.out_bits)
    if prim.out_bits == 1:
        p = float(prim.matrix[1, 0])
        if p <= 1.0 and float(prim.matrix[0, 0]) == 1.0 - p:
            return Prim.flip(p)
    return prim


def pos(w: int) -> int:
    """The literal that wire `w` is true."""
    return 2 * w


def neg(w: int) -> int:
    """The literal that wire `w` is false."""
    return 2 * w + 1


def leaving(body: Body) -> set[int]:
    """The wires through which the gates of `body` are read: its outputs, and every wire
    an op other than a gate reads."""
    out = set(body.outputs)
    for op in body.ops:
        if not is_gate(op.callee):
            out.update(op.args)
    return out


class Circuit:
    """The gates of a body as tables indexed by op position.

    `tags` holds each op's gate tag, or `None` for an op that is not a gate; `deciding`
    the argument value that decides a conjunction or disjunction on its own, -1 for a
    negation and `None` for an op that is not a gate; `outs` the one result of each op, or
    -1 for an op with another number of results; `readers` the ops reading each wire, each
    once, in op order; and `made_at` the position of the op producing each wire, or -1
    for an input.
    """

    def __init__(self, body: Body) -> None:
        self.tags = [gate_tag(op) for op in body.ops]
        self.deciding = [None if tag is None else -1 if tag is Tag.NOT else 1 - UNIT[tag]
                         for tag in self.tags]
        self.args = [op.args for op in body.ops]
        self.outs = [op.outs[0] if len(op.outs) == 1 else -1 for op in body.ops]
        self.readers: dict[int, list[int]] = {}
        for j, op in enumerate(body.ops):
            for w in dict.fromkeys(op.args):
                self.readers.setdefault(w, []).append(j)
        self.made_at = {w: -1 for w in body.inputs}
        self.made_at.update({w: j for j, op in enumerate(body.ops) for w in op.outs})


class Propagation:
    """The wires that wire `x` being `value` decides through the gates of a circuit.

    A gate is decided where an argument decides it, or where both arguments are decided;
    gates in `excluded` are left out, and `exclude` leaves out one more.  The gates are
    visited in op order, so a gate is visited after every gate it reads, and only as far as
    `advance` has been asked to go: after ``advance(k)``, `known` holds the value of every
    decided wire produced at or before position `k`, every decided wire is produced at or
    before position `last`, and where `done` holds, the propagation has decided every wire
    it ever will.

    Where `bit` maps some readers of `x` to bits, `needs` holds, for each decided wire, the
    bits of the readers without whose gates it would not be decided.  A wire decided by any
    one deciding argument needs what all of those arguments need in common; a wire decided
    by both of its arguments needs what either needs; and a gate in `bit` adds its own.
    """

    def __init__(self, circuit: Circuit, x: int, value: int,
                 excluded: set[int] | None = None, bit: dict[int, int] | None = None) -> None:
        self.circuit = circuit
        self.excluded = set() if excluded is None else excluded
        self.bit = bit
        self.known = {x: value}
        self.needs = {x: 0}
        self.last = circuit.made_at.get(x, -1)
        self.reached = -1
        self.queue = [j for j in circuit.readers.get(x, ()) if j not in self.excluded]
        heapq.heapify(self.queue)

    @property
    def done(self) -> bool:
        return not self.queue

    def advance(self, until: int) -> None:
        """Nothing, every gate at a position up to `until` now visited."""
        known, queue, excluded = self.known, self.queue, self.excluded
        readers = self.circuit.readers
        self.reached = max(self.reached, until)
        while queue and queue[0] <= until:
            j = heapq.heappop(queue)
            out = self.circuit.outs[j]
            if out in known:
                continue
            decided = self._evaluated(j)
            if decided is None:
                continue
            known[out], need = decided
            self.last = j
            if self.bit is not None:
                self.needs[out] = need
            for k in readers.get(out, ()):
                if k not in excluded:
                    heapq.heappush(queue, k)

    def exclude(self, j: int) -> None:
        """Nothing, gate `j` now left out as well, and every gate visited so far that reads
        what changes visited again.

        `excluded` must be a set the propagation may add to.  The result is the propagation
        that would have left `j` out from the start, advanced as far.  Only the wires above
        the result of `j` change, so only those are visited.
        """
        c, known, needs = self.circuit, self.known, self.needs
        self.excluded.add(j)
        if j > self.reached:
            self.queue = [k for k in self.queue if k != j]
            heapq.heapify(self.queue)
            return
        again = [j]
        while again:
            k = heapq.heappop(again)
            while again and again[0] == k:
                heapq.heappop(again)
            out = c.outs[k]
            if out not in known:
                continue
            decided = None if k in self.excluded else self._evaluated(k)
            if decided is None:
                del known[out]
                needs.pop(out, None)
            elif self.bit is not None and decided[1] != needs[out]:
                needs[out] = decided[1]
            else:
                continue
            for r in c.readers.get(out, ()):
                if r <= self.reached:
                    heapq.heappush(again, r)

    def _evaluated(self, j: int) -> tuple[int, int] | None:
        """The value and the needs of the result of gate `j` given its arguments, or `None`
        where they do not decide it."""
        c, known, needs = self.circuit, self.known, self.needs
        decisive = c.deciding[j]
        if decisive is None:
            return None
        own = 0 if self.bit is None else self.bit.get(j, 0)
        if decisive < 0:
            (a,) = c.args[j]
            va = known.get(a)
            return None if va is None else (1 - va, needs.get(a, 0) | own)
        a, b = c.args[j]
        va, vb = known.get(a), known.get(b)
        if va == decisive or vb == decisive:
            need = -1
            if va == decisive:
                need &= needs.get(a, 0)
            if vb == decisive:
                need &= needs.get(b, 0)
            return decisive, need | own
        if va is None or vb is None:
            return None
        return 1 - decisive, needs.get(a, 0) | needs.get(b, 0) | own


def composed(a: Op, b: Op, outs: tuple[int, ...], args: tuple[int, ...]) -> Prim:
    """The matrices of `a` and `b` multiplied together, laid out over `outs` then `args`.

    Both ops must call primitives.  A wire naming two slots of one op is that op's
    diagonal, and a wire the two ops share is summed away; `outs` and `args` list the
    wires that survive.  The result may name a wire twice, in which case only its
    diagonal is meaningful, that being all the lowerer ever reads of a repeated wire.
    """
    symbol: dict[int, str] = {}
    for w in a.outs + a.args + b.outs + b.args:
        symbol.setdefault(w, _ALPHABET[len(symbol)])

    slots = outs + args
    first: dict[int, int] = {}
    for i, w in enumerate(slots):
        first.setdefault(w, i)

    spec = (f"{''.join(symbol[w] for w in a.outs + a.args)},"
            f"{''.join(symbol[w] for w in b.outs + b.args)}->"
            f"{''.join(symbol[w] for w in first)}")
    result = np.einsum(spec, _tensor(a), _tensor(b))
    if len(first) < len(slots):
        grid = np.indices((2,) * len(slots))
        result = result[tuple(grid[first[w]] for w in first)]
    return canonical(Prim.from_matrix(result.reshape(1 << len(outs), 1 << len(args))))


def _tensor(op: Op) -> np.ndarray:
    """`op`'s matrix reshaped to one axis per wire slot, its results then its arguments."""
    assert isinstance(op.callee, Prim)
    return np.asarray(op.callee.matrix, dtype=float).reshape(
        (2,) * (len(op.outs) + len(op.args)))


class Sweep:
    """A body being rebuilt op by op, with the wires of deleted ops aliased to others.

    For each op of the original body in turn, `begin` resolves its arguments through the
    aliases and returns them, after which the rule keeps the op with `keep`, replaces it
    with `replace`, adds ops of its own with `emit`, or deletes it: with `alias` naming a
    wire for each of its results, or with `drop` where it has none.  A rule may also leave
    an op reading nothing out where it stands and `emit` it again further on, before any
    op it keeps that reads its results, which moves the op.  `finish` returns the rebuilt
    body.

    `reads` counts, for a wire, the reads that remain once the current op's own have been
    taken out: those of ops not yet reached, those of ops already kept, and the body's
    outputs.  A wire with no such reads is read by the current op alone, which is the
    condition a pattern needs for a wire it is about to delete.

    The rule is done with an op when the next op begins, or the body finishes.  An op kept
    earlier whose results are then read by nothing, because the op reading them was
    deleted or rewritten to read something else, is deleted as well where summing out its
    results leaves 1, and so on down.  The ops internal to a matched pattern go with the
    pattern this way, and an op that conditions stays whatever reads it.  Until then every
    op the current one reads stays, so that a rule may alias several results of one op,
    or emit several ops over its arguments, in any order.

    `producer` gives, for a wire of the body being built, the kept op producing it with its
    arguments resolved, and `constant` the value of every one-bit wire produced by a
    constant primitive.
    """

    def __init__(self, body: Body) -> None:
        self.body = body
        self.bits = list(body.bits)
        self.aliased: dict[int, int] = {}
        self.remaining: Counter = consumer_counts(body)
        self.kept_reads: Counter = Counter()
        self.ops: list[Op | None] = []
        self.made_at: dict[int, int] = {}
        self.producer: dict[int, Op] = {}
        self.constant: dict[int, int] = {}
        self.changed = False
        self._current: tuple[int, ...] = ()

    def resolve(self, w: int) -> int:
        while w in self.aliased:
            w = self.aliased[w]
        return w

    def begin(self, op: Op) -> tuple[int, ...]:
        """The arguments of `op` resolved through the aliases, their reads now consumed."""
        if self._current:
            self._settle()
        args = tuple(self.resolve(w) for w in op.args)
        for w in args:
            self.remaining[w] -= 1
        self._current = args
        return args

    def reads(self, w: int) -> int:
        return self.remaining[w] + self.kept_reads[w]

    def keep(self, op: Op, args: tuple[int, ...], callee: Prim | int | None = None) -> None:
        """Nothing, `op` now being part of the rebuilt body, reading `args`.

        `callee` replaces the op's own where given, which is how a rule rewrites an op to
        a smaller one over fewer wires.
        """
        if callee is None and args == op.args:
            kept = op
        else:
            kept = Op(op.callee if callee is None else callee, args, op.outs)
            self.changed = self.changed or kept != op
        for w in args:
            self.kept_reads[w] += 1
        for w in op.outs:
            self.producer[w] = kept
            self.made_at[w] = len(self.ops)
        bits = constant_bits(kept.callee)
        if bits is not None:
            for w, bit in zip(kept.outs, bits):
                if self.bits[w] == 1:
                    self.constant[w] = bit
        self.ops.append(kept)

    def emit(self, callee: Prim, args: tuple[int, ...], outs: tuple[int, ...] = ()) -> None:
        """Nothing, a new op ``outs = callee(args)`` now being part of the rebuilt body."""
        self.keep(Op(callee, args, outs), args)
        self.changed = True

    def alias(self, out: int, target: int) -> None:
        """Nothing, `out` now standing for `target`.

        The op producing `out` is deleted once each of its results has been aliased.
        """
        self.aliased[out] = target
        self.remaining[target] += self.remaining[out]
        self.changed = True

    def drop(self) -> None:
        """Nothing, the current op, which has no results, now deleted."""
        self.changed = True

    def replace(self, op: Op, callee: Prim, args: tuple[int, ...] = ()) -> None:
        """Nothing, `op` now being `callee` over `args` with the same result wires."""
        self.keep(op, args, callee)
        self.changed = True

    def fresh(self, bits: int = 1) -> int:
        self.bits.append(bits)
        return len(self.bits) - 1

    def delete(self, w: int) -> None:
        """Nothing, the kept op producing `w` now deleted, its results all being unread.

        For an op a rule has absorbed into another, whatever its matrix; an op left unread
        by chance is deleted once the rule is done with the current op, and only where it
        does not condition.
        """
        at = self.made_at.get(w)
        if at is None:
            return                              # already gone, having been discardable
        op = self.ops[at]
        assert op is not None and not any(self.reads(x) for x in op.outs)
        self._forget(at, op)
        for a in dict.fromkeys(op.args):
            self._drop_dead(a)

    def _settle(self) -> None:
        """Nothing, the rule being done with the current op, whose arguments may now go."""
        current, self._current = self._current, ()
        for w in dict.fromkeys(current):
            self._drop_dead(w)

    def _drop_dead(self, w: int) -> None:
        """Nothing, the kept op producing `w` deleted where nothing reads its results."""
        at = self.made_at.get(w)
        if at is None or self.reads(w):
            return
        op = self.ops[at]
        if op is None or any(self.reads(x) for x in op.outs):
            return
        if not isinstance(op.callee, Prim) or not op.callee.is_discardable:
            return
        self._forget(at, op)
        for a in dict.fromkeys(op.args):
            self._drop_dead(a)

    def _forget(self, at: int, op: Op) -> None:
        self.ops[at] = None
        self.changed = True
        for x in op.outs:
            self.producer.pop(x, None)
            self.made_at.pop(x, None)
        for a in op.args:
            self.kept_reads[a] -= 1

    def finish(self, prune: bool = True) -> Body:
        """The rebuilt body, or the original where nothing changed.

        With `prune`, the wires nothing names any longer are dropped and the others
        renumbered, as `pylea.ir.body.prune_wires` does.  A rule rebuilding one body
        several times over and naming its wires across the rebuilds leaves that to the
        last.
        """
        self._settle()
        if not self.changed:
            return self.body
        outputs = tuple(self.resolve(w) for w in self.body.outputs)
        ops = tuple(op for op in self.ops if op is not None)
        body = Body(tuple(self.bits), self.body.inputs, outputs, ops)
        return prune_wires(body) if prune else body


def evaluated(s: Sweep, op: Op, args: tuple[int, ...], known: Mapping[int, int]) -> bool:
    """Whether `op` was rewritten through `s` from the values `known` gives some of its
    arguments, `args` being those arguments as `Sweep.begin` returned them.

    A negation of a known wire becomes a constant, and so does a gate a known argument
    decides.  An `AND` or `OR` with a known argument at the value leaving the other alone
    becomes that other argument, or a constant where the other is known as well and is not
    itself produced by a constant.  A selection on a known condition becomes the branch it
    selects.  Anything else is left to the caller.  A known wire that is not a constant is
    thus read no more by the rewritten op, which is what lets an observed value stand for
    every read of an observed wire.
    """
    tag = gate_tag(op)
    if tag is Tag.NOT:
        if args[0] not in known:
            return False
        s.replace(op, Prim.const(1 - known[args[0]], 1))
        return True
    if tag is not None:
        unit = UNIT[tag]
        a, b = args
        va, vb = known.get(a), known.get(b)
        if (1 - unit) in (va, vb):
            s.replace(op, Prim.const(1 - unit, 1))
        elif unit in (va, vb):
            other = b if va == unit else a
            if other in known and other not in s.constant:
                s.replace(op, Prim.const(unit, 1))
            else:
                s.alias(op.outs[0], other)
        else:
            return False
        return True
    if isinstance(op.callee, Prim) and op.callee.tag is Tag.ITE and args[0] in known:
        k = len(op.outs)
        branch = args[1:1 + k] if known[args[0]] else args[1 + k:]
        for out, w in zip(op.outs, branch):
            s.alias(out, w)
        return True
    return False
