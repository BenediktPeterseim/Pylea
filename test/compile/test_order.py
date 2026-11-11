"""The sliced-best-of finder's local improvement step.

Reconfiguration is an optimisation over slicing and tree structure together, so what must
hold is what the ladder relies on everywhere: it never makes the winner worse, it never
changes the answer, and declining it is always possible.  How much it improves any given
network is a property of the instance and is not asserted here.

The bounded search of `AdaptiveOrderFinder` runs in a subprocess; what must hold is that it
finds the same kind of order as the unbounded search, and that it stops at its bound.
"""
import numpy as np
import pytest

from pylea.compile.cost import contraction_cost
from pylea.compile.lower import compile_program
from pylea.compile.network import build_network
import pylea.compile.order as order
from pylea.compile.order import AdaptiveOrderFinder, SlicedBestOfOrderFinder
from pylea.eval.interp import evaluate
from pylea.ir.program import intern
from pylea.ir.reference import TooLarge, semantics
from test.compile.random_programs import random_program

KAPPA = 8       # small enough that these networks actually slice


def _network(seed: int):
    prog = intern(random_program(np.random.default_rng(seed)))
    return prog, build_network(prog, prog.root)


@pytest.mark.parametrize("seed", range(8))
def test_reconfiguration_never_makes_the_winner_worse(seed):
    prog, net = _network(seed)
    plain = SlicedBestOfOrderFinder(reconfigure=False)
    plain.find(net.axes, net.output, net.size, kappa=KAPPA)
    improved = SlicedBestOfOrderFinder()
    improved.find(net.axes, net.output, net.size, kappa=KAPPA)
    assert improved.best_cost <= plain.best_cost


@pytest.mark.parametrize("seed", range(8))
def test_the_reconfigured_plan_computes_the_right_factor(seed):
    prog = intern(random_program(np.random.default_rng(seed)))
    plans, root = compile_program(prog, SlicedBestOfOrderFinder(), kappa=KAPPA)
    array, scale = evaluate(plans, root)
    got = np.asarray(array, dtype=float) * float(np.exp(scale))
    try:
        want = semantics(prog)
    except TooLarge:
        pytest.skip("reference enumeration too large for this draw")
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-300)


def test_the_reconfigured_plan_respects_the_slicing_bound():
    prog, _net = _network(3)
    plans, _ = compile_program(prog, SlicedBestOfOrderFinder(), kappa=KAPPA)
    assert all(p.peak_entries() <= 2 ** KAPPA or not p.sliced_dims
               for p in plans.values())


def test_costs_agree_between_finder_and_plan():
    """The finder's score is the quantity the plan pays, so the two must not drift."""
    prog, _net = _network(5)
    finder = SlicedBestOfOrderFinder()
    plans, _ = compile_program(prog, finder, kappa=KAPPA)
    assert contraction_cost(plans) == pytest.approx(finder.best_cost, rel=0.05)


def test_a_bounded_search_finds_an_order_computing_the_right_factor():
    for seed in range(20):
        prog = intern(random_program(np.random.default_rng(seed)))
        try:
            want = semantics(prog)
        except TooLarge:
            continue
        finder = AdaptiveOrderFinder(escalate_cost=0.0, max_time=5.0)
        plans, root = compile_program(prog, finder, kappa=KAPPA)
        array, scale = evaluate(plans, root)
        got = np.asarray(array, dtype=float) * float(np.exp(scale))
        assert np.allclose(got, want, atol=1e-12)
        return
    pytest.fail("no random program small enough to check")


def test_a_bounded_search_is_stopped_at_its_bound(monkeypatch):
    """cotengra tests its budget only between trials, so the bound must be enforced from
    outside; a bound of zero stops any search before it answers."""
    monkeypatch.setattr(order, "BOUND_GRACE_S", 0.0)
    _prog, net = _network(0)
    with pytest.raises(TimeoutError):
        AdaptiveOrderFinder(escalate_cost=0.0, max_time=0.0).find(
            net.axes, net.output, net.size, kappa=KAPPA)


def test_a_network_past_55156_indices_gets_distinct_symbols_cotengra_can_encode():
    n = 56_000
    ins, out, sizes, label = order._symbols([(i, i + 1) for i in range(n)], (0, n),
                                            {i: 2 for i in range(n + 1)})
    symbols = {s for t in ins for s in t}
    assert len(symbols) == n + 1 == len(label)
    for s in symbols:
        s.encode("utf-8")
