"""The cost model the planner spends against.

These pin the counting the planner's decisions rest on, and the shape of the estimate.
Nothing here asserts that a prediction is close to a wall-clock time, since the constants
belong to one machine.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from pylea.compile.cost import (
    MS_PER_STEP, OPS_PER_MS, contraction_cost, predicted_ms, predicted_ms_from_cost,
    step_count, tree_cost,
)
from pylea.compile.lower import compile_program
from pylea.compile.order import GreedyOrderFinder
from pylea.compile.plan import Contract
from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program


def _chain(n: int) -> Program:
    b = BodyBuilder()
    (w,) = b.emit(Prim.flip(0.3), ())
    for _ in range(n):
        (p,) = b.emit(Prim.flip(0.4), ())
        (w,) = b.emit(Prim.and_(), (w, p))
    return Program((b.finish((), (w,)),), 0)


def _plans(prog, kappa=None):
    plans, _root = compile_program(prog, GreedyOrderFinder(), kappa=kappa)
    return plans


def test_cost_counts_every_contraction():
    plans = _plans(_chain(4))
    want = sum(s.batch * s.left * s.sum * s.right
               for p in plans.values() for s in p.steps if isinstance(s, Contract))
    assert contraction_cost(plans) == want


def test_steps_are_counted_per_slice_only_where_they_vary():
    """A step hoisted out of the slice loop runs once, so charging it per slice overcharges."""
    plans = _plans(_chain(12), kappa=2)
    total = 0
    for p in plans.values():
        slices = max(p.n_slices, 1)
        total += sum(slices if p.varying[i] else 1 for i in range(len(p.steps)))
    assert step_count(plans) == total
    assert any(p.n_slices > 1 for p in plans.values()), "no plan was sliced"
    assert any(not v for p in plans.values() for v in p.varying), "nothing was hoisted"


@pytest.mark.parametrize("kappa", [1, 2, 3])
def test_the_tree_cost_does_not_depend_on_slicing(kappa):
    """The greedy tree is the same with and without slicing, so its cost must be too, while
    the multiplications performed grow with the slices."""
    program = _chain(12)
    held, sliced = _plans(program), _plans(program, kappa=kappa)
    assert any(p.n_slices > 1 for p in sliced.values()), "no plan was sliced"
    assert tree_cost(sliced) == tree_cost(held)
    assert contraction_cost(sliced) > contraction_cost(held)


def test_counts_past_the_float_range_are_infinite_rather_than_an_error():
    """A plan sliced into more pieces than a float can count still compiles; its estimate
    must say it cannot run instead of raising `OverflowError`."""
    plans = _plans(_chain(12), kappa=2)
    plans = {i: replace(p, sliced_dims=(2,) * 1100) if p.n_slices > 1 else p
             for i, p in plans.items()}
    assert contraction_cost(plans) == math.inf
    assert step_count(plans) == math.inf
    assert predicted_ms(plans) == math.inf


def test_a_bigger_program_costs_more():
    assert contraction_cost(_plans(_chain(8))) > contraction_cost(_plans(_chain(2)))
    assert step_count(_plans(_chain(8))) > step_count(_plans(_chain(2)))


# --------------------------------------------------------------------- the estimate

def test_a_small_program_is_charged_for_its_steps():
    """Below the overhead floor the arithmetic is irrelevant and the step count is not."""
    plans = _plans(_chain(4))
    assert predicted_ms(plans) == pytest.approx(step_count(plans) * MS_PER_STEP)


def test_the_estimate_is_never_below_the_step_count():
    for n in (1, 4, 16):
        plans = _plans(_chain(n))
        assert predicted_ms(plans) >= step_count(plans) * MS_PER_STEP


def test_the_power_law_takes_over_above_the_floor():
    small, large = 1e5, 1e12
    assert predicted_ms_from_cost(large) > predicted_ms_from_cost(small)
    # Far above the floor the estimate follows the arithmetic, not a per-step constant.
    assert predicted_ms_from_cost(large) > large / OPS_PER_MS / 10


def test_zero_cost_predicts_zero():
    assert predicted_ms_from_cost(0.0) == 0.0
    assert predicted_ms_from_cost(-1.0) == 0.0


def test_the_estimate_is_monotone_in_cost():
    costs = np.logspace(0, 14, 40)
    ms = [predicted_ms_from_cost(c) for c in costs]
    assert all(a <= b for a, b in zip(ms, ms[1:]))
