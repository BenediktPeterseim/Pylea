"""Equational simplifications by the Boolean identities.

`bool_rules` sweeps each body once, forwards, rewriting a gate by the first of the
following that matches.  `c` stands for a constant wire, and a wire written `n`, `h`, `m`,
`p` or `q` is internal to the pattern: it is read by the gate being rewritten and by
nothing else.  Every pattern also matches with the arguments of any `AND` or `OR` in it
swapped.

    identity            AND(1, x) -> x                   OR(0, x) -> x
    annihilation        AND(0, x) -> 0                   OR(1, x) -> 1
    negated constant    NOT(c) -> 1 - c
    double negation     NOT(NOT(x)) -> x
    complement          AND(x, n) -> 0, n = NOT(x)       OR(x, n) -> 1, n = NOT(x)
    absorption with NOT AND(x, h) -> AND(x, y), h = OR(n, y), n = NOT(x)
                        OR(x, h) -> OR(x, y), h = AND(n, y), n = NOT(x)
                        AND(n, h) -> AND(n, y), n = NOT(x), h = OR(x, y)
                        OR(n, h) -> OR(n, y), n = NOT(x), h = AND(x, y)
    consensus           OR(p, q) -> p, p = OR(m, m'), m = AND(x, y), m' = AND(n, z),
                                       n = NOT(x), q = AND(y, z)
                        AND(p, q) -> p, p = AND(m, m'), m = OR(x, y), m' = OR(n, z),
                                       n = NOT(x), q = OR(y, z)
    equal branches      OR(m, m') -> a, m = AND(c, a), m' = AND(n, a), n = NOT(c)
    negated condition   OR(m, m') -> OR(n, a), m = AND(c, a), m' = AND(n, n), n = NOT(c)

A selection between two values, which a `match` records as an `ITE`, is rewritten by the
same two identities that the circuit form of a selection has: a constant condition selects
one branch, and two equal branches are that branch.  A selection whose two branches are
the constants of a constant condition is covered by the identities above, since the
constant propagates through its circuit form.

A constant is any wire produced by a primitive that reads nothing and always produces the
same value, which is read off the matrix.  A constant may be shared between several gates,
each identity standing for its instance over a private copy of the constant.

The identities delete gates and make their readers read an argument instead.  An op left
with no readers by that, such as the `NOT` of a complement, goes in the same sweep where
summing out its result leaves 1, as `pylea.simplify.common.Sweep` describes.  Every rewrite
here produces another `AND`, `OR`, `NOT` or a constant, so the rules matching on those tags
still match what this one leaves.
"""
from __future__ import annotations

from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program

from pylea.simplify.common import UNIT, Sweep, each_body, evaluated, gate_tag

_DUAL = {Tag.AND: Tag.OR, Tag.OR: Tag.AND}


def bool_rules(prog: Program) -> Program:
    """`prog` with the Boolean identities applied to every reachable body.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import op_count
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (one,) = b.emit(Prim.const(1, 1), ())
        >>> (y,) = b.emit(Prim.and_(), (one, x))       # AND(1, x) is x
        >>> prog = Program((b.finish(inputs=(), outputs=(y,)),), root=0)
        >>> op_count(prog), op_count(bool_rules(prog))
        (3, 1)
    """
    return each_body(prog, _rewrite)


def _rewrite(body: Body) -> Body:
    s = Sweep(body)
    for op in body.ops:
        args = s.begin(op)
        if evaluated(s, op, args, s.constant):
            continue
        tag = gate_tag(op)
        if tag is Tag.NOT:
            _not(s, op, args[0])
        elif tag is not None:
            _gate(s, op, tag, args)
        elif isinstance(op.callee, Prim) and op.callee.tag is Tag.ITE:
            _ite(s, op, args)
        else:
            s.keep(op, args)
    return s.finish()


def _not(s: Sweep, op: Op, x: int) -> None:
    (out,) = op.outs
    inner = s.producer.get(x)
    if inner is not None and gate_tag(inner) is Tag.NOT:
        s.alias(out, inner.args[0])
        return
    s.keep(op, (x,))


def _gate(s: Sweep, op: Op, tag: Tag, args: tuple[int, ...]) -> None:
    (out,) = op.outs
    a, b = args
    unit = UNIT[tag]
    if _complement(s, a, b) or _complement(s, b, a):
        s.replace(op, Prim.const(1 - unit, 1))
    elif (y := _absorbed(s, tag, a, b)) is not None:
        s.keep(op, (a, y))
    elif (y := _absorbed(s, tag, b, a)) is not None:
        s.keep(op, (b, y))
    elif (y := _absorbed_negated(s, tag, a, b)) is not None:
        s.keep(op, (a, y))
    elif (y := _absorbed_negated(s, tag, b, a)) is not None:
        s.keep(op, (b, y))
    elif _consensus(s, tag, a, b):
        s.alias(out, a)
    elif _consensus(s, tag, b, a):
        s.alias(out, b)
    elif tag is Tag.OR and (branch := _equal_branches(s, a, b)) is not None:
        s.alias(out, branch)
    elif tag is Tag.OR and (negated := _negated_condition(s, a, b)) is not None:
        s.keep(op, negated)
    else:
        s.keep(op, args)


def _internal(s: Sweep, w: int, tag: Tag, held: int = 0) -> Op | None:
    """The op producing `w` where it is a `tag` gate read by the current op and by `held`
    kept ops of the pattern alone."""
    p = s.producer.get(w)
    return p if p is not None and gate_tag(p) is tag and s.reads(w) == held else None


def _negation_of(s: Sweep, n: int, held: int = 0) -> int | None:
    """`x` where `n` is `NOT(x)` read by the current op and by `held` kept ops of the
    pattern alone, else `None`."""
    p = s.producer.get(n)
    if p is None or gate_tag(p) is not Tag.NOT or s.reads(n) != held:
        return None
    return p.args[0]


def _complement(s: Sweep, x: int, n: int) -> bool:
    return _negation_of(s, n) == x


def _absorbed(s: Sweep, tag: Tag, x: int, h: int) -> int | None:
    """`y` where `h` is the dual gate of `NOT(x)` and `y`, both internal, else `None`."""
    inner = _internal(s, h, _DUAL[tag])
    if inner is None:
        return None
    for n, y in (inner.args, inner.args[::-1]):
        if n != y and _negation_of(s, n, held=1) == x:
            return y
    return None


def _absorbed_negated(s: Sweep, tag: Tag, n: int, h: int) -> int | None:
    """`y` where `n` is `NOT(x)` and `h` the dual gate of `x` and `y`, both internal."""
    x = _negation_of(s, n)
    inner = _internal(s, h, _DUAL[tag])
    if x is None or inner is None:
        return None
    for u, y in (inner.args, inner.args[::-1]):
        if u == x:
            return y
    return None


def _consensus(s: Sweep, tag: Tag, p: int, q: int) -> bool:
    """Whether `q` is the consensus term of `p`, both internal to the pattern."""
    dual = _DUAL[tag]
    outer = _internal(s, p, tag)
    term = _internal(s, q, dual)
    if outer is None or term is None:
        return False
    for m, m_ in (outer.args, outer.args[::-1]):
        left, right = _internal(s, m, dual, held=1), _internal(s, m_, dual, held=1)
        if left is None or right is None:
            continue
        for x, y in (left.args, left.args[::-1]):
            for n, z in (right.args, right.args[::-1]):
                if _negation_of(s, n, held=1) == x and set(term.args) == {y, z} and y != z:
                    return True
    return False


def _selections(s: Sweep, m: int, m_: int):
    """``(c, a, right)`` for each way `m` is `AND(c, a)` and `m_` another internal `AND`."""
    for first, second in ((m, m_), (m_, m)):
        left, right = _internal(s, first, Tag.AND), _internal(s, second, Tag.AND)
        if left is None or right is None:
            continue
        for c, a in (left.args, left.args[::-1]):
            if c != a:
                yield c, a, right


def _equal_branches(s: Sweep, m: int, m_: int) -> int | None:
    """`a` where `m` and `m_` are `AND(c, a)` and `AND(NOT(c), a)`, all internal."""
    for c, a, right in _selections(s, m, m_):
        for n, a_ in (right.args, right.args[::-1]):
            if a_ == a and n != a and _negation_of(s, n, held=1) == c:
                return a
    return None


def _negated_condition(s: Sweep, m: int, m_: int) -> tuple[int, int] | None:
    """``(n, a)`` where `m` and `m_` are `AND(c, a)` and `AND(n, n)` with `n = NOT(c)`."""
    for c, a, right in _selections(s, m, m_):
        n, n_ = right.args
        if n != n_:
            continue
        if _negation_of(s, n, held=2) == c:
            return n, a
    return None


def _ite(s: Sweep, op: Op, args: tuple[int, ...]) -> None:
    k = len(op.outs)
    then_, else_ = args[1:1 + k], args[1 + k:]
    if then_ == else_:
        for out, w in zip(op.outs, then_):
            s.alias(out, w)
    else:
        s.keep(op, args)
