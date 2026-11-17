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
    assert any("exact engine chosen; its predicted time" in line for line in p.trace)


def test_a_wide_grid_commits_to_the_sampling_lane():
    model = grid_model(24)
    p = prob(model, method="auto", rel_err=0.05, seed=1)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert 0.0 < p.low < p.high < 1.0
    assert any("sampling engine" in line for line in p.trace)


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
    # The sampling engine's prediction is below the fast threshold, or the bound decides.
    assert any("sampling engine chosen; its predicted time is below" in line
               or "lower bound on the exact engine's predicted time decides" in line
               for line in p.trace)


@pytest.mark.parametrize("n", [3, 8, 16, 24])
def test_the_unsliced_bound_never_exceeds_the_sliced_prediction(n):
    import pylea.sample.portfolio as portfolio

    model = grid_model(n)
    kwargs = {"kappa": 6}
    assert portfolio._greedy_bound(model, kwargs, []) <= \
        portfolio._greedy_prediction(model, kwargs, [])


def test_a_disputed_prediction_hedges_and_the_race_terminates(monkeypatch):
    import pylea.sample.portfolio as portfolio

    # Neither engine's prediction may commit at once, so that the two run in parallel.
    monkeypatch.setattr(portfolio, "T_FAST_S", 0.0)

    def pessimistic(dpp, compile_kwargs, trace):
        trace.append("auto: greedy exact prediction 50s (mocked)")
        return 50.0

    monkeypatch.setattr(portfolio, "_greedy_prediction", pessimistic)
    model = grid_model(10)
    exact = float(model.compiled().distribution()[1])
    p = prob(model, method="auto", rel_err=0.02, seed=2)
    assert isinstance(p, Estimate)
    assert any("run in parallel" in line for line in p.trace)
    assert p.low - 1e-9 <= exact <= p.high + 1e-9
    assert p.engine in ("smc", "exact")


def test_the_unfolding_guard_sends_a_wide_deep_program_exact_without_unfolding():
    # The interfaces are too wide to collapse, so the exponential unfolding stands and
    # the guard must fire; the exact engine answers hierarchically without unfolding.
    width = 10

    # `^` rather than `&`, so that the answer stays within the float range; a probability
    # of about 2**-(2**20) is certified only by the much slower evaluation with an
    # exponent per entry, which is not what this test is about.
    @DPP
    def leaf(x, types=(Bool,) * width):
        return tuple(bit ^ flip(0.5) for bit in x)

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
    assert any("more than the sampling engine accepts" in line for line in p.trace)
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
    assert not any("more than the sampling engine accepts" in line for line in p.trace)
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

    # Neither engine's prediction may commit at once, so that the two run in parallel.
    monkeypatch.setattr(portfolio, "T_FAST_S", 0.0)

    class DeadLane:
        def __init__(self, program, compile_kwargs, mode="evaluate", name=None):
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
    assert any("exact engine failed (boom)" in line for line in p.trace), p.trace


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
    assert any("predicted sampling time is unbounded" in line for line in p.trace)


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
    import pylea.sample.exact_lane as exact_lane

    class StuckLane:
        def __init__(self, program, compile_kwargs, mode="evaluate", name=None):
            self.started = time.monotonic()

        def response(self, timeout=None):
            time.sleep(min(timeout or 0.0, 0.2))
            from pylea.compile._process_tree import DeadlineExceeded

            raise DeadlineExceeded("stuck")

        def kill(self):
            pass

    monkeypatch.setattr(exact_lane, "ExactLane", StuckLane)
    model = grid_model(4)                    # the fast path commits exact
    exact = float(model.compiled().distribution()[1])
    p = prob(model, method="auto", rel_err=0.05, max_time=20.0, seed=4)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert p.low - 1e-9 <= exact <= p.high + 1e-9
    assert any("share of max_time" in line for line in p.trace)
    with pytest.raises(TimeoutError, match="max_time"):
        prob(model, method="exact", max_time=1.0)


def test_the_subprocess_prediction_is_computed_once_per_model_and_kappa(monkeypatch):
    import pylea.sample.portfolio as portfolio

    spawned = []

    class QuickLane:
        def __init__(self, program, compile_kwargs, mode="evaluate", name=None):
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
    assert any("computed earlier" in line for line in trace)
    portfolio._budgeted_prediction(model, model.program(), {"kappa": 20}, 1.0, trace)
    assert spawned == ["predict", "predict"]


_COIN_PROGRAM_SCRIPT = """
import pickle, subprocess, sys
from pylea import DPP, Bool, flip
@DPP
def coin() -> Bool:
    return flip(0.3)
worker = subprocess.Popen([sys.executable, "-m", "pylea.sample._exact_worker"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
pickle.dump(("evaluate", coin.program(), {}, "coin"), worker.stdin)
worker.stdin.flush()
answer = pickle.load(worker.stdout)
worker.wait(timeout=60)
assert sorted(answer) == ["array", "scale", "trace"], answer
assert worker.returncode == 0, worker.returncode
assert b"Fatal Python error" not in worker.stderr.read()
"""


def test_an_exact_worker_that_answers_exits_with_code_zero_and_reports_no_fatal_error():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", _COIN_PROGRAM_SCRIPT], check=True, timeout=120)


_CUSTOM_FINDER_SCRIPT = """
from pylea import DPP, Bool, flip, prob
from pylea.compile.order import GreedyOrderFinder
class MyFinder(GreedyOrderFinder):
    pass
@DPP
def coin() -> Bool:
    return flip(0.3)
try:
    prob(coin, method="exact", max_time=60, order_finder=MyFinder())
except RuntimeError as exc:
    assert "MyFinder" in str(exc) and "defined in the script" in str(exc), exc
    assert "_exact_worker" not in str(exc), exc
else:
    raise AssertionError("no error")
"""


def test_a_request_that_cannot_be_unpickled_is_reported_as_an_error_response():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", _CUSTOM_FINDER_SCRIPT], check=True, timeout=120)


def test_an_exact_query_with_a_deadline_answers_a_coin_in_under_a_second():
    @DPP
    def coin() -> Bool:
        return flip(0.3)

    prob(coin, method="exact", max_time=60)          # imports in this process
    started = time.monotonic()
    assert float(prob(coin, method="exact", max_time=60)) == pytest.approx(0.3)
    # A fresh worker answers in about 0.1 s; waiting for its exit took over a second.
    assert time.monotonic() - started < 1.0


def test_a_predictor_budget_beyond_the_thread_timeout_limit_waits_without_overflow():
    import pylea.sample.portfolio as portfolio

    model = grid_model(3)
    trace: list[str] = []
    prediction = portfolio._budgeted_prediction(model, model.program(), {"kappa": 7},
                                                1e55, trace)
    assert 0.0 <= prediction < 1.0


def test_an_exact_query_with_a_deadline_of_1e10_seconds_answers():
    @DPP
    def coin() -> Bool:
        return flip(0.3)

    assert float(prob(coin, method="exact", max_time=1e10)) == pytest.approx(0.3)


def test_a_parallel_run_that_ends_at_the_deadline_says_so(monkeypatch):
    import pylea.sample.portfolio as portfolio

    class SilentLane:
        def __init__(self, program, compile_kwargs, mode="evaluate", name=None):
            self.started = time.monotonic()

        def done(self):
            return False

        def finished_ok(self):
            return False

        def kill(self):
            pass

    def pessimistic(dpp, compile_kwargs, trace):
        return 50.0

    monkeypatch.setattr(portfolio, "ExactLane", SilentLane)
    monkeypatch.setattr(portfolio, "_greedy_prediction", pessimistic)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = prob(grid_model(10), method="auto", rel_err=1e-4, max_time=1.0, seed=3)
    assert not p.target_met
    assert any("neither engine finished" in line for line in p.trace), p.trace
    assert not any("answered first" in line for line in p.trace), p.trace


def test_a_prediction_does_not_send_the_callers_order_finder():
    from pylea.compile.order import GreedyOrderFinder
    from pylea.sample.exact_lane import ExactLane

    class ScriptFinder(GreedyOrderFinder):
        pass

    ScriptFinder.__module__ = "__main__"          # as if defined in the user's script
    model = grid_model(3)
    lane = ExactLane(model.program(), {"order_finder": ScriptFinder(), "kappa": 20},
                     mode="predict")
    try:
        assert "predicted_ms" in lane.response(60)
    finally:
        lane.kill()


def test_a_timeout_error_of_the_callers_own_is_not_taken_for_the_deadline(monkeypatch):
    import pylea.sample.exact_lane as exact_lane

    class Interrupted:
        def __init__(self, program, compile_kwargs, mode="evaluate", name=None):
            pass

        def response(self, timeout=None):
            raise TimeoutError("raised by the caller's own alarm")

        def kill(self):
            pass

    monkeypatch.setattr(exact_lane, "ExactLane", Interrupted)
    with pytest.raises(TimeoutError, match="caller's own alarm"):
        prob(grid_model(3), method="exact", max_time=30)


# A refusal as the exact engine's worker reports it.
_REFUSAL = ("exact inference on 'grid' is estimated to take about 3 years; the limit is "
            "11.6 days.  Sample with method=\"smc\" or the default method=\"auto\".")


class _RefusingLane:
    """An exact lane whose worker refused the DPP."""

    def __init__(self, program, compile_kwargs, mode="evaluate", name=None):
        self.started = time.monotonic()

    def done(self):
        return True

    def finished_ok(self):
        return False

    def response(self, timeout=None):
        from pylea.compile.lower import ExactInferenceTooSlow

        return {"error": f"ExactInferenceTooSlow: {_REFUSAL}",
                "infeasible": ExactInferenceTooSlow(_REFUSAL)}

    def kill(self):
        pass


def test_auto_samples_a_dpp_the_exact_engine_refuses(monkeypatch):
    import pylea.compile.planner as planner
    import pylea.sample.exact_lane as exact_lane

    exact = float(grid_model(4).compiled().distribution()[1])
    model = grid_model(4)                    # the fast path commits exact
    monkeypatch.setattr(planner, "MAX_PREDICTED_MS", 1e-9)
    p = prob(model, method="auto", rel_err=0.05, seed=5)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert p.low - 1e-9 <= exact <= p.high + 1e-9
    assert any("the exact engine refused (exact inference on 'grid' is estimated to take"
               in line for line in p.trace), p.trace

    # Under a deadline the refusal comes from the worker.
    monkeypatch.setattr(exact_lane, "ExactLane", _RefusingLane)
    p = prob(model, method="auto", rel_err=0.05, max_time=20.0, seed=6)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert any("the exact engine refused (exact inference on 'grid'" in line
               for line in p.trace), p.trace


def test_auto_reports_the_refusal_where_no_sampling_lane_exists(monkeypatch):
    import pylea.compile.planner as planner
    import pylea.sample.exact_lane as exact_lane
    import pylea.sample.portfolio as portfolio
    from pylea.compile.lower import ExactInferenceTooSlow

    monkeypatch.setattr(portfolio, "UNFOLDED_OPS_CAP", 0)
    monkeypatch.setattr(planner, "MAX_PREDICTED_MS", 1e-9)
    with pytest.raises(ExactInferenceTooSlow) as refusal:
        prob(grid_model(4), method="auto", rel_err=0.05, seed=0)
    message = str(refusal.value)
    assert message.startswith("exact inference on 'grid' is estimated to take about ")
    assert message.endswith("The sampling engine cannot answer instead: sampling would "
                            "run 73 assignments, more than the sampling engine accepts "
                            "(0).")
    assert 'method="auto"' not in message
    monkeypatch.setattr(exact_lane, "ExactLane", _RefusingLane)
    with pytest.raises(ExactInferenceTooSlow, match="cannot answer instead"):
        prob(grid_model(4), method="auto", rel_err=0.05, max_time=20.0, seed=0)


def test_an_exact_query_under_a_deadline_raises_the_refusal_itself(monkeypatch):
    import pylea.sample.exact_lane as exact_lane
    from pylea.compile.lower import ExactInferenceTooSlow

    monkeypatch.setattr(exact_lane, "ExactLane", _RefusingLane)
    with pytest.raises(ExactInferenceTooSlow) as refusal:
        prob(grid_model(4), method="exact", max_time=20.0)
    assert str(refusal.value) == _REFUSAL


def test_a_committed_exact_lane_whose_worker_fails_falls_back_to_sampling(monkeypatch):
    import pylea.sample.exact_lane as exact_lane

    class FailingLane(_RefusingLane):
        def response(self, timeout=None):
            return {"error": "boom"}

    monkeypatch.setattr(exact_lane, "ExactLane", FailingLane)
    model = grid_model(4)                    # the fast path commits exact
    p = prob(model, method="auto", rel_err=0.05, max_time=20.0, seed=7)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert any("auto: the exact engine failed: boom; the sampling engine answers" in line
               for line in p.trace), p.trace
    with pytest.raises(RuntimeError, match="boom"):
        prob(model, method="exact", max_time=20.0)


def test_a_parallel_run_names_a_refusal_as_such(monkeypatch):
    import pylea.sample.portfolio as portfolio

    # Neither engine's prediction may commit at once, so that the two run in parallel.
    monkeypatch.setattr(portfolio, "T_FAST_S", 0.0)

    def pessimistic(dpp, compile_kwargs, trace):
        return 50.0

    monkeypatch.setattr(portfolio, "ExactLane", _RefusingLane)
    monkeypatch.setattr(portfolio, "_greedy_prediction", pessimistic)
    p = prob(grid_model(10), method="auto", rel_err=0.05, seed=3)
    assert p.engine == "smc" and p.target_met
    assert any("auto: the exact engine refused (exact inference on 'grid' is estimated to "
               "take about 3 years; the limit is 11.6 days); the sampling engine answered"
               in line for line in p.trace), p.trace


def _returning_one_coin(n: int):
    def same():
        x = flip(0.5)
        return tuple(x for _ in range(n))

    same.__annotations__ = {"return": tuple[(Bool,) * n]}
    return DPP(same)


def test_a_prediction_refused_for_its_size_is_unbounded():
    import math

    import pylea.sample.portfolio as portfolio

    wide = _returning_one_coin(40)
    trace: list[str] = []
    assert portfolio._greedy_prediction(wide, {"kappa": 28}, trace) == math.inf
    # The same refusal from the subprocess, which tags it.
    assert portfolio._budgeted_prediction(wide, wide.program(), {"kappa": 28}, 60.0,
                                          trace) == math.inf
    assert trace == ["auto: exact inference would exceed the size limit; its time is "
                     "taken as unbounded"] * 2


def test_the_exact_worker_answers_a_refusal_with_the_exception_itself(monkeypatch):
    import pickle

    import pylea.compile.planner as planner
    from pylea.compile.lower import ExactInferenceTooSlow
    from pylea.sample._exact_worker import _handle

    monkeypatch.setattr(planner, "MAX_PREDICTED_MS", 1e-9)
    model = grid_model(4)
    response = pickle.loads(pickle.dumps(_handle(("evaluate", model.program(), {},
                                                  model.name()))))
    refusal = response["infeasible"]
    assert isinstance(refusal, ExactInferenceTooSlow)
    assert response["error"] == f"ExactInferenceTooSlow: {refusal}"
    assert str(refusal).startswith("exact inference on 'grid' is estimated to take about ")
    assert refusal.trace[0].startswith("compilation: greedy contraction tree")


def _as_distribution(event: DPP) -> DPP:
    from pylea import Enum

    class Outcome(Enum):
        elements = ["no", "yes"]

    @DPP
    def distribution() -> Outcome:
        return event().match({True: "yes", False: "no"}, data_type=Outcome)

    return distribution


def test_a_wide_grid_distribution_under_auto_is_answered_by_sampling():
    from pylea import probability_mass_function

    pmf = probability_mass_function(_as_distribution(grid_model(24)), rel_err=0.05,
                                    seed=0)
    assert all(p.engine == "smc" and p.target_met for p in pmf.values())
    assert pmf["no"].low <= pmf["no"] <= pmf["no"].high


def test_a_trivial_distribution_under_auto_is_exact():
    from pylea import probability_mass_function

    model = grid_model(4)
    pmf = probability_mass_function(_as_distribution(model), seed=0)
    assert all(p.engine == "exact" for p in pmf.values())
    assert float(pmf["yes"]) == pytest.approx(float(model.compiled().distribution()[1]))


def test_a_table_race_won_by_the_exact_lane_returns_the_exact_matrix(monkeypatch):
    import pylea.sample.portfolio as portfolio
    from pylea import probability_mass_function

    monkeypatch.setattr(portfolio, "T_FAST_S", 0.0)
    monkeypatch.setattr(portfolio, "_lane", lambda *args: "hedge")
    model = grid_model(4)
    # A relative error the sampling engine needs minutes for, so the exact engine wins.
    pmf = probability_mass_function(_as_distribution(model), rel_err=1e-4, seed=0)
    assert all(p.engine == "exact" for p in pmf.values())
    assert any("the exact engine answered first" in line for line in pmf["yes"].trace)
    assert float(pmf["yes"]) == pytest.approx(float(model.compiled().distribution()[1]))


def test_an_evaluated_dpp_under_auto_stays_exact():
    table = grid_model(4).evaluated(method="auto")
    assert type(table).__name__ == "LikelihoodTable"


def test_a_small_dpp_whose_sampling_is_predicted_below_the_fast_threshold_commits_without_a_race():
    p = prob(grid_model(16), rel_err=0.05, seed=0)
    assert p.engine == "smc"
    assert any("sampling engine chosen; its predicted time is below" in line
               for line in p.trace)
    assert not any("in parallel" in line for line in p.trace)



def test_seeded_auto_results_with_different_compile_options_are_cached_apart():
    from pylea.compile.order import GreedyOrderFinder

    model = grid_model(4)
    first = prob(model, method="auto", rel_err=0.1, seed=4)
    again = prob(model, method="auto", rel_err=0.1, seed=4)
    other = prob(model, method="auto", rel_err=0.1, seed=4,
                 order_finder=GreedyOrderFinder())
    assert again.trace[-1].startswith("result cache")
    assert not other.trace[-1].startswith("result cache")
    # The other contraction tree rounds differently, in the last digits only.
    assert float(other) == pytest.approx(float(first), rel=1e-12)
