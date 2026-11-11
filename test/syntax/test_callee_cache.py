"""Resolving a callee to a body must not depend on an identity that can be reused.

`at_least` builds a fresh `@DPP` on every call, and that DPP becomes unreachable as soon as
the call returns.  A cache keyed on the address would therefore answer for an address since
handed to an unrelated callee, silently confusing two callees of the same arity.
"""
import numpy as np
import pytest

from test.syntax.helpers import matrix_of

from pylea.syntax.dpp import DPP, Bool, at_least
from pylea.syntax.flip import flip


def _p_true(dpp) -> float:
    v = matrix_of(dpp).ravel()
    return float(v[1] / v.sum())


def _at_least_exactly(k: int, ps: list[float]) -> float:
    """P(at least `k` of the coins land true), by enumeration."""
    total = 0.0
    for mask in range(1 << len(ps)):
        if bin(mask).count("1") < k:
            continue
        w = 1.0
        for i, p in enumerate(ps):
            w *= p if (mask >> i) & 1 else 1.0 - p
        total += w
    return total


def test_gates_of_different_arity_in_one_body_are_not_confused():
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]

    @DPP
    def mixed() -> Bool:
        xs = [flip(p) for p in ps]
        a = at_least(2, *xs[:3])
        b = at_least(3, *xs)
        c = at_least(1, *xs[:4])
        return a & b & c

    assert _p_true(mixed) == pytest.approx(_joint(ps), rel=1e-9)


def _joint(ps: list[float]) -> float:
    """P(a & b & c) for the three gates above, by enumeration over all assignments."""
    total = 0.0
    for mask in range(1 << len(ps)):
        w = 1.0
        for i, p in enumerate(ps):
            w *= p if (mask >> i) & 1 else 1.0 - p
        bits = [(mask >> i) & 1 for i in range(len(ps))]
        a = sum(bits[:3]) >= 2
        b = sum(bits) >= 3
        c = sum(bits[:4]) >= 1
        if a and b and c:
            total += w
    return total


@pytest.mark.parametrize("k,n", [(1, 2), (2, 3), (3, 5), (1, 4), (4, 4)])
def test_a_voting_gate_agrees_with_enumeration_over_all_assignments(k, n):
    ps = [0.1 * (i + 1) for i in range(n)]

    @DPP
    def gate() -> Bool:
        return at_least(k, *[flip(p) for p in ps])

    assert _p_true(gate) == pytest.approx(_at_least_exactly(k, ps), rel=1e-9)


def test_the_callee_cache_is_keyed_on_the_callee_object_not_its_address():
    @DPP
    def inner() -> Bool:
        return flip(0.5)

    @DPP
    def outer() -> Bool:
        return inner()

    cache = getattr(outer, "_DPP__callee_index")
    assert cache, "nothing was absorbed, so this test is not checking anything"
    assert all(isinstance(k, DPP) for k in cache), f"keys are {[type(k) for k in cache]}"
