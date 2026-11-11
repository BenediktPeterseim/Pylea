"""The interpreter's handling of the call graph.

Each body a plan reaches is evaluated once, callees first, without recursion, so the depth
of the call graph is not bounded by Python's recursion limit.
"""
import sys

import numpy as np

from pylea import DPP, Bool, flip


def test_a_call_chain_deeper_than_the_recursion_limit_is_built_and_evaluated():
    depth = sys.getrecursionlimit() + 100

    @DPP
    def step(s: Bool) -> Bool:
        return s & flip(0.5) | ~s & flip(0.25)
    level = step
    for _ in range(depth - 1):
        inner = level

        @DPP
        def outer(s: Bool) -> Bool:
            return inner(s)
        level = outer

    @DPP
    def chain() -> Bool:
        return level(Bool.const(True))

    array, scale = chain.compiled().matrix()
    assert np.allclose(np.asarray(array).reshape(-1) * np.exp(scale), [0.5, 0.5])
