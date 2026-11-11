"""Absorption by implication.

A disjunction `OR(u, v)` in which `v` syntactically implies `u` equals `u`, and a
conjunction `AND(u, v)` in which `v` implies `u` equals `v`; `absorb` deletes such a gate
and makes its readers read the argument it equals.  Implication is
`pylea.simplify.implication.Implications`, and since a wire implies itself, a gate
reading one wire twice is that wire.

The relation is computed once per body, and every gate it absorbs is rewritten in one
sweep.  Deleting a gate that equals one of its arguments leaves every implication between
the remaining wires in place, so the later gates of the sweep are rewritten against the
same relation.

`idempotence` is the one instance that needs no relation, a gate reading one wire twice,
for a caller that wants it without the cost of the relation, which grows with the square
of the body.
"""
from __future__ import annotations

from pylea.ir.body import Body
from pylea.ir.prim import Tag
from pylea.ir.program import Program

from pylea.simplify.common import Sweep, each_body, gate_tag, pos
from pylea.simplify.implication import Implications


def absorb(prog: Program, deadline: float | None = None) -> Program:
    """`prog` with every gate that equals one of its arguments by implication deleted.

    `deadline` is a `time.perf_counter` reading past which no further body is started and
    the relation of the current one is no longer extended.  The relation of a body takes
    ``n * n / 8`` bytes for its `n` literals, taken as it is computed, so a deadline bounds
    the memory along with the time; without one, the whole relation is held.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (y,) = b.emit(Prim.flip(0.4), ())
        >>> (a,) = b.emit(Prim.and_(), (x, y))
        >>> (o,) = b.emit(Prim.or_(), (a, x))            # a implies x, so this is x
        >>> prog = Program((b.finish(inputs=(), outputs=(o,)),), root=0)
        >>> [op.callee.tag.name for op in absorb(prog).main.ops]
        ['FLIP']
    """
    return each_body(prog, lambda body: _absorbed(body, deadline), deadline)


def _absorbed(body: Body, deadline: float | None) -> Body:
    if not any(gate_tag(op) in (Tag.AND, Tag.OR) for op in body.ops):
        return body
    return _rewrite(body, Implications(body, deadline))


def idempotence(prog: Program) -> Program:
    """`prog` with every gate reading one wire twice deleted, its readers reading the wire.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (a,) = b.emit(Prim.and_(), (x, x))
        >>> prog = Program((b.finish(inputs=(), outputs=(a,)),), root=0)
        >>> [op.callee.tag.name for op in idempotence(prog).main.ops]
        ['FLIP']
    """
    return each_body(prog, lambda body: _rewrite(body, None))


def _rewrite(body: Body, relation: Implications | None) -> Body:
    """`body` with every gate absorbed that `relation` shows equal to an argument, or with
    only the gates reading one wire twice deleted where there is no relation."""
    s = Sweep(body)
    for op in body.ops:
        args = s.begin(op)
        tag = gate_tag(op)
        if tag not in (Tag.AND, Tag.OR):
            s.keep(op, args)
            continue
        u, v = args
        (out,) = op.outs
        if u == v:
            s.alias(out, u)
        elif relation is None:
            s.keep(op, args)
        elif relation.implies(pos(v), pos(u)):
            s.alias(out, u if tag is Tag.OR else v)
        elif relation.implies(pos(u), pos(v)):
            s.alias(out, v if tag is Tag.OR else u)
        else:
            s.keep(op, args)
    return s.finish()
