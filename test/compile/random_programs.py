"""A generator of small random `Program`s, for differential testing.

`random_program(rng)` builds a few bodies bottom-up, each calling primitives and the bodies
below it, and returns the program rooted at the last one.  Programs are kept small enough
that `pylea.ir.reference.semantics` can enumerate them.

The generator covers what the lowerer has to get right: wires wider than one bit, several
inputs and outputs, outputs that repeat or name an input, arguments that repeat, results
nothing reads, inputs nothing reads, conditioning, and calls shared between sites.
"""
from __future__ import annotations

import numpy as np

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program

MAX_WIRES = 14


def _prims(rng) -> list[Prim]:
    """Candidate primitives, with random parameters where they have any."""
    return [
        Prim.and_(), Prim.or_(), Prim.not_(),
        Prim.flip(round(rng.uniform(0.05, 0.95), 3)),
        Prim.uniform(rng.choice([1, 2])),
        Prim.const(rng.choice([0, 1]), 1),
        Prim.observe(),
        Prim.equate(1),
        Prim.ite(1),
        Prim.from_matrix(rng.random((2, 4))),
        Prim.from_matrix(rng.random((4, 2))),
        Prim.from_matrix(np.eye(2)[:, rng.permutation(2)]),
    ]


def random_body(rng, bodies: list, n_ops: int) -> "object":
    builder = BodyBuilder(bodies)
    pool: dict[int, list[int]] = {}

    def add(w: int, width: int) -> None:
        pool.setdefault(width, []).append(w)

    for _ in range(rng.integers(0, 3)):
        width = int(rng.choice([1, 1, 2]))
        add(builder.fresh(width), width)

    callees: list = _prims(rng) + list(range(len(bodies)))
    for _ in range(n_ops):
        if len(builder.bits) >= MAX_WIRES:
            break
        rng.shuffle(callees)
        for callee in callees:
            in_w = ((1,) * callee.in_bits if isinstance(callee, Prim)
                    else tuple(bodies[callee].bits[w] for w in bodies[callee].inputs))
            out_w = ((1,) * callee.out_bits if isinstance(callee, Prim)
                     else tuple(bodies[callee].bits[w] for w in bodies[callee].outputs))
            if len(builder.bits) + len(out_w) > MAX_WIRES:
                continue
            if not all(pool.get(w) for w in in_w):
                continue
            args = [pool[w][int(rng.integers(len(pool[w])))] for w in in_w]
            for w, out in zip(out_w, builder.emit(callee, args)):
                add(out, w)
            break

    every = [w for ws in pool.values() for w in ws]
    inputs = [w for w in range(len(builder.bits))
              if not any(w in op.outs for op in builder.ops)]
    n_out = int(rng.integers(0, min(4, len(every)) + 1)) if every else 0
    outputs = [every[int(rng.integers(len(every)))] for _ in range(n_out)]
    return builder.finish(inputs, outputs)


def random_program(rng, n_bodies: int = 3, n_ops: int = 5) -> Program:
    bodies: list = []
    for _ in range(n_bodies):
        bodies.append(random_body(rng, bodies, n_ops))
    return Program(tuple(bodies), len(bodies) - 1)
