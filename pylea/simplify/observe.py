"""Observation splitting, observation propagation, and repeated observations.

An effect is an op with arguments and no results, such as an `observe`.  `observe_rules`
applies three rewrites to the effects of each body.

Observation splitting replaces the observation of a conjunction by one observation of each
conjunct, where nothing else reads the conjunction.  For the output of any other primitive
read by an effect alone, the effect and the primitive are multiplied into one effect over
the primitive's arguments: the observation of a negation becomes an effect admitting the
negated wire at 0, and the observation of a table's output becomes the observed row.  A
disjunction is left as it is, since the observation of a disjunction says nothing about
either disjunct on its own.

Observation propagation reads, off an effect, the one argument pattern it admits, and
evaluates every other op reading one of those wires at the admitted value.  A gate becomes
its other argument or a constant, a selection on a pinned condition becomes the branch
selected, and any other primitive becomes its matrix restricted to the value, over one
argument fewer.  That includes an effect that admits no single pattern, such as a soft
observation, which weights several, or a table of zeros, which admits none; one reading
pinned wires alone becomes a constant weight.  A call is left alone, since evaluating a body
at a value would specialise it.  The effect keeps its own read, which is what conditions; so
does every other effect admitting one pattern, so that a wire two effects pin stays pinned.
A wire that two effects pin to different values is not propagated at all, the body being
zero everywhere.

A repeated effect, calling a deterministic callee on the same arguments as an earlier one,
is deleted: it contributes a weight of 0 or 1 that the earlier call already contributes.
"""
from __future__ import annotations

from collections import Counter

from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program, deterministic

from pylea.simplify.common import (
    Sweep, admitted_pattern, composed, each_body, evaluated, gate_tag,
)


def observe_rules(prog: Program) -> Program:
    """`prog` with observations split and propagated and repeated ones deleted.

    An observed coin flip keeps its own read, but the gate reading it reads nothing: an
    `AND` with an argument known to be 1 is its other argument.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import op_count
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.5), ())
        >>> (y,) = b.emit(Prim.flip(0.5), ())
        >>> _ = b.emit(Prim.observe(), (x,))
        >>> (out,) = b.emit(Prim.and_(), (x, y))
        >>> prog = Program((b.finish(inputs=(), outputs=(out,)),), root=0)
        >>> op_count(prog), op_count(observe_rules(prog))
        (4, 3)
    """
    is_det = deterministic(prog)
    return each_body(prog, lambda body: _rewrite(body, is_det))


def _pinned(body: Body) -> dict[int, int]:
    """The value each wire is pinned to by the effects admitting one pattern, leaving out
    a wire that two effects pin to different values."""
    pinned: dict[int, int] = {}
    conflicting: set[int] = set()
    for op in body.ops:
        pattern = admitted_pattern(op.callee)
        if pattern is None:
            continue
        for w, bit in zip(op.args, pattern):
            if pinned.setdefault(w, bit) != bit:
                conflicting.add(w)
    for w in conflicting:
        del pinned[w]
    return pinned


def _rewrite(body: Body, is_det: dict[int, bool]) -> Body:
    pinned = _pinned(body)
    s = Sweep(body)
    seen: set[tuple] = set()
    for op in body.ops:
        args = s.begin(op)
        evaluable = isinstance(op.callee, Prim) and any(w in pinned for w in args)
        if op.outs or not op.args:
            if evaluable:
                _evaluate(s, op, args, pinned)
            else:
                s.keep(op, args)
            continue
        det = op.callee.is_deterministic if isinstance(op.callee, Prim) else is_det[op.callee]
        if det:
            key = (op.callee, args)
            if key in seen:
                s.drop()                    # a repeated effect
                continue
            seen.add(key)
        if evaluable and admitted_pattern(op.callee) is None:
            _evaluate(s, op, args, pinned)
        elif not _split(s, op, args):
            s.keep(op, args)
    return s.finish()


def _split(s: Sweep, op: Op, args: tuple[int, ...]) -> bool:
    """Whether `op` was replaced by effects on the arguments of the one op it reads.

    The arguments of `op` must all be results of one primitive op, and that op's results
    must all be read by `op` alone.  The effect then becomes two observations where the
    op is an `AND` under an `observe`, nothing where it is an `OR`, and otherwise the
    product of the two matrices over the op's arguments, which is then an effect of the
    same kind again and is absorbed further where it can be.
    """
    if not isinstance(op.callee, Prim):
        return False
    callee: Prim = op.callee
    reads = args
    absorbed: list[Op] = []
    held: Counter = Counter()           # the reads of the absorbed ops, which go with them
    split = False
    while (p := _sole_producer(s, reads, held)) is not None:
        tag = gate_tag(p)
        if tag is Tag.OR:
            break
        absorbed.append(p)
        if tag is Tag.AND and callee.tag is Tag.OBSERVE:
            split = True
            break
        callee = composed(p, Op(callee, reads, ()), (), p.args)
        reads = p.args
        held = Counter(p.args)
    if not absorbed:
        return False
    if split:
        for w in absorbed[-1].args:
            s.emit(Prim.observe(), (w,))
    else:
        s.replace(op, callee, reads)
    for p in absorbed:
        s.delete(p.outs[0])
    return True


def _sole_producer(s: Sweep, args: tuple[int, ...], held: Counter) -> Op | None:
    """The primitive op producing every one of `args`, where its results are read by the
    current op alone, else `None`.

    `held` counts the reads of each wire by kept ops that go with the current one.
    """
    producers = {s.producer.get(w) for w in args}
    if len(producers) != 1:
        return None
    (p,) = producers
    if p is None or isinstance(p.callee, int):
        return None
    if set(p.outs) != set(args) or any(s.reads(w) != held[w] for w in p.outs):
        return None
    return p


def _evaluate(s: Sweep, op: Op, args: tuple[int, ...], pinned: dict[int, int]) -> None:
    """Nothing, `op` now being itself evaluated at the pinned values among its arguments."""
    if evaluated(s, op, args, pinned):
        return
    held = tuple(dict.fromkeys(w for w in args if w in pinned))
    value = 0
    for w in held:
        value = (value << 1) | pinned[w]
    rest = tuple(w for w in args if w not in pinned)
    source = Op(Prim.const(value, len(held)), (), held)
    s.replace(op, composed(source, Op(op.callee, args, op.outs), op.outs, rest), rest)
