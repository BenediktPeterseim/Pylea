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


# ----------------------------------------------------------------- number range

import math

import pytest

from pylea import observe
from pylea.compile.lower import compile_program
from pylea.compile.order import GreedyOrderFinder
from pylea.eval.backend import ExtendedRangeBackend, NumpyBackend
from pylea.eval.interp import evaluate, evaluate_certified


def _two_reports(p: float):
    @DPP
    def two_reports() -> Bool:
        a = flip(0.5)
        b = (flip(p) & a) | (flip(3 * p) & ~a)
        c = (flip(p) & a) | (flip(3 * p) & ~a)
        observe(b & c)
        return a
    return two_reports


def _posterior(dpp, **kwargs) -> float:
    plans, root = compile_program(dpp.program(), **kwargs)
    array, _scale = evaluate(plans, root)
    column = array[:, 0]
    return float(column[1] / column.sum())


@pytest.mark.parametrize("p", [1e-161, 1e-162, 1e-170, 1e-250])
def test_products_of_rare_coin_flips_below_the_float_range_are_not_lost(p):
    assert _posterior(_two_reports(p)) == pytest.approx(0.1, rel=1e-12)


@pytest.mark.parametrize("likelihood", [1e-200, 1.0, 1e160, 1e300])
def test_extreme_likelihoods_neither_underflow_nor_overflow(likelihood):
    @DPP
    def reports() -> Bool:
        x, y = flip(0.5), flip(0.5)
        z = x & y
        observe(z, if_true=3 * likelihood, if_false=likelihood)
        observe(z, if_true=3 * likelihood, if_false=likelihood)
        return x
    assert _posterior(reports) == pytest.approx(0.8333333333333334, rel=1e-12)


def test_a_result_float64_cannot_certify_is_recomputed_with_an_exponent_per_entry():
    @DPP
    def rare_pair() -> Bool:
        a = flip(0.5)
        b = flip(1e-200) & a
        c = flip(1e-200) & a
        observe(b & c)
        return a

    result = evaluate_certified(*compile_program(rare_pair.program()))
    assert result.path == "extended"
    assert result.array[0, 0] == 0.0 and result.array[1, 0] == 1.0


def test_ordinary_dpps_are_certified_without_recomputation():
    from test.sample.grid import grid_model

    for dpp in (grid_model(8), _two_reports(1e-3)):
        assert evaluate_certified(*compile_program(dpp.program())).path == "backend"


def test_the_extended_range_evaluation_agrees_with_enumeration():
    from pylea.ir.program import intern
    from pylea.ir.reference import TooLarge, semantics
    from test.compile.random_programs import random_program

    rng = np.random.default_rng(3)
    checked = 0
    while checked < 30:
        prog = intern(random_program(rng, n_bodies=int(rng.integers(1, 3)),
                                     n_ops=int(rng.integers(1, 6))))
        try:
            want = semantics(prog)
        except TooLarge:
            continue
        for kappa in (None, 3):
            plans, root = compile_program(prog, GreedyOrderFinder(), kappa=kappa)
            from pylea.eval.interp import _run_all
            array, scale, _ = _run_all(plans, root, ExtendedRangeBackend())
            np.testing.assert_allclose(array * math.exp(scale), want, rtol=1e-12,
                                       atol=1e-300)
        checked += 1


def test_the_certificate_never_accepts_a_wrong_entry():
    """Likelihoods spread from 1e-200 to 1e200 over random DPPs; a certified float64
    result must equal the extended-range one to 1e-10 in every entry."""
    from pylea.eval.interp import _run_all

    rng = np.random.default_rng(7)
    accepted = 0
    for _trial in range(60):
        n = int(rng.integers(2, 6))
        scales = 10.0 ** rng.uniform(-200, 200, size=(n, 2))
        coin = [float(c) for c in rng.uniform(0.05, 0.95, size=n)]

        @DPP
        def random_dpp() -> tuple[Bool, Bool]:
            xs = [flip(c) for c in coin]
            for i in range(n):
                observe(xs[i] ^ xs[(i + 1) % n], if_true=float(scales[i, 0]),
                        if_false=float(scales[i, 1]))
            return xs[0], xs[-1]

        for kappa in (None, 3):
            plans, root = compile_program(random_dpp.program(), GreedyOrderFinder(),
                                          kappa=kappa)
            array, scale, certified = _run_all(plans, root, NumpyBackend())
            if not certified:
                continue
            want, want_scale = _run_all(plans, root, ExtendedRangeBackend())[:2]
            positive = want > 0
            assert np.array_equal(array > 0, positive)
            got = np.log(array[positive]) + scale
            ref = np.log(want[positive]) + want_scale
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)
            accepted += 1
    assert accepted > 20


def test_the_result_never_shares_memory_with_a_cached_leaf():
    @DPP
    def coin() -> Bool:
        return flip(0.25)

    plans, root = compile_program(coin.program())
    first, _ = evaluate(plans, root)
    first[0, 0] = 99.0
    second, _ = evaluate(plans, root)
    assert second[0, 0] == 0.75 and first.flags.writeable
