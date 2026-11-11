"""Brute-force semantics, as a test oracle.

`semantics` returns the semantics of one body by running over every combination of values of
its wires and summing the product of the ops' matrix entries at those values.  A called body
is resolved to its own matrix first and then treated as a leaf.

The cost is exponential in the number of wires, so this applies to small DPPs only and
raises `TooLarge` beyond `max_wires`.  Everything the compiler does is checked against it.
"""
from __future__ import annotations

import numpy as np

from pylea.ir.prim import Prim
from pylea.ir.program import Program


class TooLarge(ValueError):
    """Raised when a body has more wires than `max_wires`."""


def semantics(prog: Program, body: int | None = None, max_wires: int = 22) -> np.ndarray:
    """The semantics of `body`, or of the root, as a ``(2**out_bits, 2**in_bits)`` matrix.

    Raises `TooLarge` where the body has more than `max_wires` wires, the enumeration being
    over all ``2**n`` combinations of their values.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> semantics(both.program()).round(4).tolist()
        [[0.875], [0.125]]
    """
    b = prog.bodies[prog.root if body is None else body]
    n = len(b.bits)
    if n > max_wires:
        raise TooLarge(f"{n} wires is {2 ** n} combinations of values; "
                       "raise max_wires deliberately")

    # A called body is itself a matrix, so resolve callees first and treat them as leaves.
    matrices: list[np.ndarray] = []
    for op in b.ops:
        if isinstance(op.callee, Prim):
            matrices.append(op.callee.matrix)
        else:
            matrices.append(semantics(prog, op.callee, max_wires))

    widths = [b.bits[w] for w in range(n)]
    out = np.zeros((1 << _bits(b, b.outputs), 1 << _bits(b, b.inputs)))

    for assignment in range(1 << sum(widths)):
        vals = _unpack(assignment, widths)
        weight = 1.0
        for op, f in zip(b.ops, matrices):
            i = _pattern(vals, b, op.args)
            o = _pattern(vals, b, op.outs)
            weight *= f[o, i]
            if weight == 0.0:
                break
        if weight:
            out[_pattern(vals, b, b.outputs), _pattern(vals, b, b.inputs)] += weight
    return out


def _bits(body, wires) -> int:
    return sum(body.bits[w] for w in wires)


def _unpack(value: int, widths: list[int]) -> list[int]:
    """Packed values of all wires split into one per wire, first wire most significant."""
    vals = []
    for w in reversed(widths):
        vals.append(value & ((1 << w) - 1))
        value >>= w
    return list(reversed(vals))


def _pattern(vals, body, wires) -> int:
    """The values of `wires` packed MSB-first, the order `pylea.semantics.matrices` uses."""
    p = 0
    for w in wires:
        p = (p << body.bits[w]) | vals[w]
    return p
