"""Decision correctness of the ``method="auto"`` portfolio, and deadline enforcement.

The lane decisions are pinned on a small grid pair, one trivially exact and one whose
greedy prediction is hopeless; the hedge is exercised with a mocked exact prediction so
that neither commit rule fires.  Subprocess-based tests each pay one interpreter start.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from test.sample.grid import grid_model
from pylea import DPP, Bool, flip, prob
from pylea.syntax.estimate import Estimate


def test_a_trivial_query_commits_exact_through_the_fast_path():
    model = grid_model(4)
    exact = float(model.compiled().distribution()[1])
    p = prob(model, method="auto", rel_err=0.05, seed=0)
    assert isinstance(p, Estimate)
    assert p.engine == "exact" and p.target_met
    assert p.low == p.high == float(p) == exact
    assert any("fast path, exact lane" in line for line in p.trace)


def test_a_wide_grid_commits_to_the_sampling_lane():
    model = grid_model(24)
    p = prob(model, method="auto", rel_err=0.05, seed=1)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert 0.0 < p.low < p.high < 1.0
    assert any("sampling lane" in line for line in p.trace)


def test_a_tree_far_wider_than_kappa_is_ruled_out_by_its_bound_without_slicing(monkeypatch):
    """Slicing such a tree took 6 s and 1.6 GB for twenty people, past 5 GB for thirty."""
    import pylea.sample.portfolio as portfolio
    from pylea.examples.birthday import two_with_same_birthday

    def no_slicing(*args, **kwargs):
        raise AssertionError("the greedy plan was sliced")
    monkeypatch.setattr(portfolio, "_greedy_prediction", no_slicing)
    p = prob(two_with_same_birthday(n_people=20), method="auto", rel_err=0.05, seed=0)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert any("lower bound decides" in line for line in p.trace)


@pytest.mark.parametrize("n", [3, 8, 16, 24])
def test_the_unsliced_bound_never_exceeds_the_sliced_prediction(n):
    import pylea.sample.portfolio as portfolio

    model = grid_model(n)
    kwargs = {"kappa": 6}
    assert portfolio._greedy_bound(model, kwargs, []) <= \
        portfolio._greedy_prediction(model, kwargs, [])


def test_a_disputed_prediction_hedges_and_the_race_terminates(monkeypatch):
    import pylea.sample.portfolio as portfolio

    def pessimistic(dpp, compile_kwargs, trace):
        trace.append("auto: greedy exact prediction 50s (mocked)")
        return 50.0

    monkeypatch.setattr(portfolio, "_greedy_prediction", pessimistic)
    model = grid_model(10)
    exact = float(model.compiled().distribution()[1])
    p = prob(model, method="auto", rel_err=0.02, seed=2)
    assert isinstance(p, Estimate)
    assert any("hedge" in line for line in p.trace)
    assert p.low - 1e-9 <= exact <= p.high + 1e-9
    assert p.engine in ("smc", "exact")


def test_the_unfolding_guard_sends_a_wide_deep_program_exact_without_unfolding():
    # The interfaces are too wide to collapse, so the exponential unfolding stands and
    # the guard must fire; the exact engine answers hierarchically without unfolding.
    width = 10

    @DPP
    def leaf(x, types=(Bool,) * width):
        return tuple(bit & flip(0.5) for bit in x)

    level = leaf
    for _ in range(20):
        previous = level

        @DPP
        def wider(x, types=(Bool,) * width):
            return previous(*previous(*x))

        level = wider

    @DPP
    def deep() -> Bool:
        ys = level(*[flip(0.5) for _ in range(width)])
        acc = ys[0]
        for y in ys[1:]:
            acc = acc & y
        return acc

    started = time.monotonic()
    p = prob(deep, method="auto", rel_err=0.05, seed=0)
    elapsed = time.monotonic() - started
    assert isinstance(p, Estimate)
    assert p.engine == "exact"
    assert any("unfolding guard" in line for line in p.trace)
    # The whole point: deciding and answering never unfolds the 2**20-call program.
    assert elapsed < 15.0


def test_a_deep_narrow_program_is_collapsed_rather_than_refused():
    # An exponential unfolding behind narrow interfaces collapses to a few table ops,
    # so the portfolio decides on the collapsed count and answers cheaply.
    @DPP
    def leaf() -> Bool:
        return flip(0.9) & flip(0.9)

    level = leaf
    for _ in range(26):
        previous = level

        @DPP
        def level() -> Bool:
            return previous() & previous()

    started = time.monotonic()
    p = prob(level, method="auto", rel_err=0.05, seed=0)
    elapsed = time.monotonic() - started
    assert isinstance(p, Estimate)
    assert not any("unfolding guard" in line for line in p.trace)
    assert elapsed < 10.0


def test_method_exact_with_a_tight_deadline_raises_timeout_by_killing_the_worker():
    model = grid_model(30)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        prob(model, method="exact", max_time=1.0)
    assert time.monotonic() - started < 15.0


def test_method_exact_with_a_generous_deadline_returns_the_exact_value():
    model = grid_model(4)
    exact = float(model.compiled().distribution()[1])
    assert prob(model, method="exact", max_time=120.0) == pytest.approx(exact, abs=0.0)


def test_auto_under_a_deadline_still_answers_a_bn_shaped_query_fast():
    @DPP
    def tiny() -> Bool:
        x = flip(0.2)
        return x | flip(0.3)

    started = time.monotonic()
    p = prob(tiny, method="auto", rel_err=0.05, max_time=60.0, seed=0)
    elapsed = time.monotonic() - started
    assert isinstance(p, Estimate)
    assert p.target_met
    assert p.low - 1e-9 <= 0.44 <= p.high + 1e-9
    # A generous deadline must not force the slow path on a trivial query; the fast
    # exact commit runs in a killable subprocess, whose start dominates the seconds.
    assert elapsed < 15.0


def test_a_failed_exact_worker_leaves_the_sampling_lane_running(monkeypatch):
    import pylea.sample.portfolio as portfolio

    class DeadLane:
        def __init__(self, program, compile_kwargs, mode="evaluate"):
            self.started = time.monotonic()

        def done(self):
            return True

        def finished_ok(self):
            return False

        def response(self, timeout=None):
            return {"error": "boom"}

        def kill(self):
            pass

    def pessimistic(dpp, compile_kwargs, trace):
        trace.append("auto: greedy exact prediction 50s (mocked)")
        return 50.0

    monkeypatch.setattr(portfolio, "ExactLane", DeadLane)
    monkeypatch.setattr(portfolio, "_greedy_prediction", pessimistic)
    model = grid_model(10)
    exact = float(model.compiled().distribution()[1])
    p = prob(model, method="auto", rel_err=0.05, seed=3)
    assert isinstance(p, Estimate)
    # The worker failed instantly, but the sampler ran on to certification instead of
    # returning an early uncertified interval.
    assert p.engine == "smc" and p.target_met
    assert p.low - 1e-9 <= exact <= p.high + 1e-9
    assert any("exact lane failed" in line for line in p.trace)


def test_an_all_zero_pilot_with_no_deadline_routes_exact_instead_of_crashing():
    # Over 2000 traced ops, so the pilot probes first; it sees no reachable mass, the
    # sampling prediction is infinite, and with no deadline the exact lane must answer.
    @DPP
    def rare_chain() -> Bool:
        acc = flip(1e-12)
        for _ in range(1100):
            acc = acc & flip(0.99)
        return acc

    p = prob(rare_chain, method="auto", rel_err=0.05, seed=0)
    assert isinstance(p, Estimate)
    assert p.engine == "exact"
    exact = float(rare_chain.compiled().distribution()[1])
    assert float(p) == exact
    assert any("no reachable mass" in line for line in p.trace)


def test_seeded_auto_results_use_the_chosen_lane_reproducibly():
    model = grid_model(24)
    first = prob(model, method="auto", rel_err=0.05, seed=7)
    second = prob(model, method="auto", rel_err=0.05, seed=7)
    assert isinstance(first, Estimate) and isinstance(second, Estimate)
    assert first.engine == second.engine == "smc"
    assert float(first) == float(second)
    assert np.isfinite(first.low) and np.isfinite(first.high)


def test_a_committed_exact_lane_that_overruns_its_deadline_share_falls_back_to_sampling(
        monkeypatch):
    import pylea.sample.portfolio as portfolio

    class StuckLane:
        def __init__(self, program, compile_kwargs, mode="evaluate"):
            self.started = time.monotonic()

        def response(self, timeout=None):
            time.sleep(min(timeout or 0.0, 0.2))
            raise TimeoutError("stuck")

        def kill(self):
            pass

    monkeypatch.setattr(portfolio, "ExactLane", StuckLane)
    model = grid_model(4)                    # the fast path commits exact
    exact = float(model.compiled().distribution()[1])
    p = prob(model, method="auto", rel_err=0.05, max_time=20.0, seed=4)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert p.low - 1e-9 <= exact <= p.high + 1e-9
    assert any("missed its" in line for line in p.trace)
    with pytest.raises(TimeoutError, match="max_time"):
        prob(model, method="exact", max_time=1.0)


def test_the_subprocess_prediction_is_computed_once_per_model_and_kappa(monkeypatch):
    import pylea.sample.portfolio as portfolio

    spawned = []

    class QuickLane:
        def __init__(self, program, compile_kwargs, mode="evaluate"):
            spawned.append(mode)
            self.started = time.monotonic()

        def response(self, timeout=None):
            return {"predicted_ms": 1500.0}

        def kill(self):
            pass

        @property
        def elapsed(self):
            return 0.0

    monkeypatch.setattr(portfolio, "ExactLane", QuickLane)
    model = grid_model(6)
    trace: list[str] = []
    first = portfolio._budgeted_prediction(model, model.program(), {}, 1.0, trace)
    second = portfolio._budgeted_prediction(model, model.program(), {}, 1.0, trace)
    assert first == second == 1.5
    assert spawned == ["predict"]
    assert any("cached" in line for line in trace)
    portfolio._budgeted_prediction(model, model.program(), {"kappa": 20}, 1.0, trace)
    assert spawned == ["predict", "predict"]
