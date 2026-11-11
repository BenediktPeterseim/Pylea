"""Body splicing.

`inline` replaces one call with the callee's ops.  The callee's wires are appended to the
caller and its interface wires are then merged with the call's arguments and results, the
lower-numbered wire of each pair surviving so that a class containing a caller wire is named
by one.  A callee that returns an argument, or returns the same wire twice, therefore needs
no extra op: the merge already identifies the wires involved.

`inline_all` repeats that over a caller until no call site qualifies, so calls uncovered by
a splice are considered too.  `flatten_wide_calls` uses it to remove every callee too wide
to hold as a single matrix, working callees first so that what is spliced in has already had
its own wide calls removed.

Splicing because it is profitable rather than necessary belongs to `pylea.simplify.inline`.
"""
from __future__ import annotations

from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern, reachable


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # The lower wire wins, so a class containing caller wires is named by one.
            self.parent[max(ra, rb)] = min(ra, rb)


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
    uf = _UnionFind(len(bits))
    for w, a in zip(callee.inputs, op.args):
        uf.union(base + w, a)
    for w, o in zip(callee.outputs, op.outs):
        uf.union(base + w, o)

    spliced = [Op(o.callee, tuple(base + w for w in o.args), tuple(base + w for w in o.outs))
               for o in callee.ops]
    ops = list(body.ops[:site]) + spliced + list(body.ops[site + 1:])

    rep = [uf.find(w) for w in range(len(bits))]
    return _compact(bits, rep,
                    [Op(o.callee, tuple(rep[w] for w in o.args),
                        tuple(rep[w] for w in o.outs)) for o in ops],
                    [rep[w] for w in body.inputs],
                    [rep[w] for w in body.outputs])


def _compact(bits, rep, ops, inputs, outputs) -> Body:
    """The body with its surviving wires renumbered consecutively, aliased ones dropped."""
    renumber: dict[int, int] = {}
    new_bits: list[int] = []
    for w in range(len(bits)):
        if rep[w] == w:
            renumber[w] = len(new_bits)
            new_bits.append(bits[w])
    r = renumber.__getitem__
    return Body(
        tuple(new_bits),
        tuple(r(w) for w in inputs),
        tuple(r(w) for w in outputs),
        tuple(Op(o.callee, tuple(r(w) for w in o.args), tuple(r(w) for w in o.outs))
              for o in ops),
    )


def inline_all(prog: Program, caller: int, should_inline) -> Body:
    """The body `caller` with every call satisfying `should_inline(body_index)` spliced in.

    Repeats until no call site qualifies, so calls exposed by an inline are considered too.
    """
    bodies = list(prog.bodies)
    body = bodies[caller]
    while True:
        for i, op in enumerate(body.ops):
            if not isinstance(op.callee, Prim) and should_inline(op.callee):
                bodies[caller] = body
                body = inline(Program(tuple(bodies), prog.root), caller, i)
                break
        else:
            return body


def flatten_wide_calls(prog: Program, max_interface_bits: int) -> Program:
    """An equivalent DPP whose every remaining callee fits in ``2**max_interface_bits``.

    Bodies are flattened callees first, so a callee spliced into a call site has already
    had its own wide calls removed.
    """
    bodies = list(prog.bodies)
    too_wide = set()
    for i in reachable(prog):
        bodies[i] = inline_all(Program(tuple(bodies), prog.root), i, too_wide.__contains__)
        if bodies[i].interface_bits > max_interface_bits:
            too_wide.add(i)
    return intern(Program(tuple(bodies), prog.root))
