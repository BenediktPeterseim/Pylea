"""Body splicing.

`inline` replaces one call with the callee's ops.  The callee's wires are appended to the
caller and its interface wires are then merged with the call's arguments and results, the
lower-numbered wire of each pair surviving so that a class containing a caller wire is named
by one.  A callee that returns an argument, or returns the same wire twice, therefore needs
no extra op: the merge already identifies the wires involved.

`inline_all` splices every qualifying call of one caller, and `inline_calls` does so for
every body of a DPP that stays, that is the root and each body not itself to be spliced.
Both expand in place: on reaching a qualifying call they append the callee's wires and go
on through the callee's ops, splicing its qualifying calls in turn, to any depth.  Each body
that stays is built once, expanded until no qualifying call remains, and a body that is
spliced everywhere is never built on its own, so the work is the size of the result.
`flatten_wide_calls` uses `inline_calls` to remove every callee too wide to hold as a single
matrix.

Splicing because it is profitable rather than necessary belongs to `pylea.simplify.inline`.
"""
from __future__ import annotations

from pylea.ir._union_find import UnionFind
from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern, reachable


def inline(prog: Program, caller: int, site: int) -> Body:
    """The body `caller` with its `site`-th op, which must be a call, spliced in.

    Raises `ValueError` where that op calls a primitive rather than a body.

        >>> import numpy as np
        >>> from pylea import DPP, Bool, flip
        >>> from pylea.ir.program import Program
        >>> from pylea.ir.reference import semantics
        >>> @DPP
        ... def helper() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def uses() -> Bool:
        ...     return helper()
        >>> prog = uses.program()
        >>> spliced = inline(prog, prog.root, site=0)
        >>> bool(np.allclose(semantics(prog),
        ...                  semantics(Program(prog.bodies[:-1] + (spliced,), prog.root))))
        True
    """
    body = prog.bodies[caller]
    op = body.ops[site]
    if isinstance(op.callee, Prim):
        raise ValueError(f"body {caller} op {site} calls a primitive, not a body")
    callee = prog.bodies[op.callee]

    base = len(body.bits)
    bits = list(body.bits) + list(callee.bits)
    uf = UnionFind(len(bits))
    for w, a in zip(callee.inputs, op.args):
        uf.union(base + w, a)
    for w, o in zip(callee.outputs, op.outs):
        uf.union(base + w, o)

    spliced = [Op(o.callee, tuple(base + w for w in o.args), tuple(base + w for w in o.outs))
               for o in callee.ops]
    ops = list(body.ops[:site]) + spliced + list(body.ops[site + 1:])
    return _compact(bits, uf, ops, body.inputs, body.outputs)


def _compact(bits, uf: UnionFind, ops, inputs, outputs) -> Body:
    """The body with every wire replaced by its class in `uf`, and the wires naming a class
    renumbered consecutively."""
    renumber: dict[int, int] = {}
    new_bits: list[int] = []
    rep = [uf.find(w) for w in range(len(bits))]
    for w in range(len(bits)):
        if rep[w] == w:
            renumber[w] = len(new_bits)
            new_bits.append(bits[w])
    new = [renumber[rep[w]] for w in range(len(bits))]
    return Body(
        tuple(new_bits),
        tuple(new[w] for w in inputs),
        tuple(new[w] for w in outputs),
        tuple(Op(o.callee, tuple(new[w] for w in o.args), tuple(new[w] for w in o.outs))
              for o in ops),
    )


def inline_all(prog: Program, caller: int, should_inline) -> Body:
    """The body `caller` with every call of a body `i` with ``should_inline(i)`` spliced in,
    including the calls that splicing brings in.

    No call left in the result qualifies.  `caller` itself is returned, the same object,
    where none of its calls qualifies.  The result is built once, in time linear in its
    size.
    """
    body = prog.bodies[caller]
    if not any(not isinstance(op.callee, Prim) and should_inline(op.callee)
               for op in body.ops):
        return body

    bits = list(body.bits)
    uf = UnionFind(len(bits))
    ops: list[Op] = []
    # One entry per body being expanded: the number of its first wire in `bits`, and its
    # ops not yet visited.  The caller's own wires keep their numbers.
    stack = [(0, iter(body.ops))]
    while stack:
        base, pending = stack[-1]
        op = next(pending, None)
        if op is None:
            stack.pop()
            continue
        if base:
            args = tuple(base + w for w in op.args)
            outs = tuple(base + w for w in op.outs)
        else:
            args, outs = op.args, op.outs
        if isinstance(op.callee, Prim) or not should_inline(op.callee):
            ops.append(Op(op.callee, args, outs) if base else op)
            continue
        callee = prog.bodies[op.callee]
        start = len(bits)
        bits.extend(callee.bits)
        uf.grow(len(callee.bits))
        for w, a in zip(callee.inputs, args):
            uf.union(start + w, a)
        for w, o in zip(callee.outputs, outs):
            uf.union(start + w, o)
        stack.append((start, iter(callee.ops)))

    return _compact(bits, uf, ops, body.inputs, body.outputs)


def inline_calls(prog: Program, should_inline) -> Program:
    """`prog` with every call of a body `i` with ``should_inline(i)`` spliced in, including
    the calls that splicing brings in.

    The bodies that stay are the root and every reachable body `i` with
    ``not should_inline(i)``; each is rebuilt by `inline_all` once, and is kept as the same
    object where none of its calls qualifies.  A body spliced at all of its sites is left
    in the table unchanged, where `pylea.ir.program.intern` drops it.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.ir.program import intern, reachable
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def pair() -> Bool:
        ...     return coin() & coin()
        >>> @DPP
        ... def top() -> Bool:
        ...     return pair() | coin()
        >>> prog = top.program()
        >>> len(reachable(prog)), len(reachable(intern(inline_calls(prog, lambda _: True))))
        (3, 1)
    """
    bodies = list(prog.bodies)
    for i in reachable(prog):
        if i == prog.root or not should_inline(i):
            bodies[i] = inline_all(prog, i, should_inline)
    return Program(tuple(bodies), prog.root)


def flatten_wide_calls(prog: Program, max_interface_bits: int) -> Program:
    """An equivalent DPP whose every remaining callee fits in ``2**max_interface_bits``.

    A callee is spliced in where its `interface_bits` exceed `max_interface_bits`.  The
    decision is taken once per body, on the DPP as given: splicing leaves the interface of
    every body that stays unchanged, as each interface position keeps a wire of its own
    width.
    """
    too_wide = {i for i in reachable(prog)
                if prog.bodies[i].interface_bits > max_interface_bits}
    return intern(inline_calls(prog, too_wide.__contains__))
