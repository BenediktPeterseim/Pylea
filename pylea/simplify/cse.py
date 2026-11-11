"""Common subexpression elimination over a private coin flip.

Two ops with the same deterministic callee over the same arguments compute the same
values, so the second may be deleted and its results replaced by the first's.  Done
anywhere, this can raise the contraction cost: the merged op ties together everything
either copy's results fed.  `flip_cse` merges such copies only where one of their shared
arguments is a coin flip read by nothing but the copies, and a coin flip here is any
primitive that reads nothing and produces one bit, `pylea.simplify.common.is_flip`.  After
the merge the coin flip is read once, so a contraction may always take it into the
surviving copy first, at no extra cost, and the one result of the merged op then runs where
the coin flip ran before.  An op with two or more results is left alone: its results would
all have to run where the coin flip alone did, and that can cost more.  So is a source with
two or more results, which is not a coin flip: its other results are read where they are,
so it cannot be taken into the surviving copy for nothing.

A callee is deterministic where it is a primitive whose matrix is deterministic, or a body
every op of which is; a coin flip is not, so two coin flips never merge.  The rule leaves
effects alone, which `pylea.simplify.observe` merges, and copies sharing no private
coin flip, which it cannot merge safely.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from pylea.ir.body import Body
from pylea.ir.prim import Prim
from pylea.ir.program import Program, consumer_counts, deterministic

from pylea.simplify.common import Sweep, each_body, is_flip


def flip_cse(prog: Program) -> Program:
    """`prog` with equal deterministic ops merged wherever they share a private coin flip.

    Here the two conjunctions read the same coin flip `y`, which nothing else reads, so the
    second is deleted and `d` reads the first:

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import op_count
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.5), ())
        >>> (y,) = b.emit(Prim.flip(0.5), ())
        >>> (z,) = b.emit(Prim.flip(0.5), ())
        >>> (a,) = b.emit(Prim.and_(), (x, y))
        >>> (c,) = b.emit(Prim.and_(), (x, y))
        >>> (d,) = b.emit(Prim.or_(), (a, z))
        >>> (e,) = b.emit(Prim.or_(), (c, x))
        >>> prog = Program((b.finish(inputs=(), outputs=(d, e)),), root=0)
        >>> op_count(prog), op_count(flip_cse(prog))
        (7, 6)
    """
    is_det = deterministic(prog)
    return each_body(prog, lambda body: _merge(body, is_det))


def _merge(body: Body, is_det: dict[int, bool]) -> Body:
    reads = consumer_counts(body)
    producer = {w: op for op in body.ops for w in op.outs}

    groups: dict[tuple, list[int]] = defaultdict(list)
    for k, op in enumerate(body.ops):
        if not op.args or len(op.outs) != 1:
            continue
        det = op.callee.is_deterministic if isinstance(op.callee, Prim) else is_det[op.callee]
        if det:
            groups[(op.callee, op.args)].append(k)

    alias: dict[int, int] = {}
    for (_, args), members in groups.items():
        if len(members) < 2:
            continue
        if not any(_private_flip(a, producer, reads, members, body) for a in set(args)):
            continue
        (target,) = body.ops[members[0]].outs
        for k in members[1:]:
            alias[body.ops[k].outs[0]] = target
    if not alias:
        return body

    s = Sweep(body)
    for op in body.ops:
        args = s.begin(op)
        if op.outs and op.outs[0] in alias:
            s.alias(op.outs[0], s.resolve(alias[op.outs[0]]))
        else:
            s.keep(op, args)
    return s.finish()


def _private_flip(a: int, producer, reads: Counter, members: list[int], body: Body) -> bool:
    """Whether `a` comes from a coin flip and is read by the `members`, once each, and by
    nothing else."""
    op = producer.get(a)
    if op is None or not is_flip(op.callee):
        return False
    return reads[a] == len(members) and all(body.ops[k].args.count(a) == 1 for k in members)
