"""The escalation ladder.

What matters is that spending more never changes the answer and never makes the plan worse,
which is what lets a stage run under a threshold or a budget that is only roughly right.

The thresholds are not asserted against wall clock anywhere.  They are passed in, so a test
can force a stage to run rather than wait for an instance expensive enough to trigger it.
The dominator lift's budget is not timed here either, since a test asserting how far a pass
gets in a given number of milliseconds asserts the speed of the machine it runs on.
"""
from unittest import mock

import numpy as np
import pytest

import pylea.compile.planner as planner
from pylea.compile.planner import (
    BEST_OF, GREEDY, LONGER_SEARCH, SIMPLIFIED, Compilation, plan_compilation,
)
from pylea.eval.interp import evaluate
from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern
from pylea.ir.reference import TooLarge, semantics
from test.compile.random_programs import random_program

def _forced(prog, **kwargs):
    """`plan_compilation` with every threshold at zero and no allowance for the search's
    own greedy trees, so the whole ladder runs."""
    with mock.patch.object(planner, "C_PROBES", 0):
        return plan_compilation(prog, t_search=0.0, t_escalate=0.0, **kwargs)


@pytest.fixture
def free_probes(monkeypatch):
    """Report every greedy compile as instantaneous, so stage B is reached on any instance.

    Whether stage B runs is decided by wall clock, which a test cannot arrange for
    by choosing an input.  Holding the clock still asks the counterfactual the stage is about
    -- were scoring free, would it run -- rather than asserting a speed.
    """
    monkeypatch.setattr("pylea.compile.planner.perf_counter", lambda: 0.0)


def _chain(n: int) -> Program:
    b = BodyBuilder()
    (w,) = b.emit(Prim.flip(0.3), ())
    for _ in range(n):
        (p,) = b.emit(Prim.flip(0.4), ())
        (w,) = b.emit(Prim.and_(), (w, p))
    return Program((b.finish((), (w,)),), 0)


def _value(c: Compilation) -> np.ndarray:
    array, scale = evaluate(c.plans, c.root)
    return np.asarray(array, dtype=float).reshape(-1) * float(np.exp(scale))


def test_a_cheap_instance_pays_for_no_search():
    """The whole point: an instance that evaluates in microseconds pays for no order search."""
    ran = " ".join(plan_compilation(_chain(6), kappa=28).trace)
    assert BEST_OF not in ran and LONGER_SEARCH not in ran, ran


def test_simplification_is_skipped_when_scoring_it_would_cost_more_than_evaluating():
    """Stage B has to be scored by a second greedy compile, and that compile is not free.

    A chain of six coins evaluates in microseconds and takes milliseconds to compile, so no
    amount of rewriting can repay itself and the ladder stops after the probe.
    """
    c = plan_compilation(_chain(6), kappa=28)
    assert c.predicted_ms < c.probe_ms                  # the premise, not an assumption
    assert SIMPLIFIED not in " ".join(c.trace), c.trace


def test_simplification_runs_regardless_once_asked_to_always():
    """The gate weighs the phase against ONE evaluation, which a repeated caller does not.

    The chain of six coins is the instance the gate declines, so this pins that ``"always"``
    overrides that decision and not merely that the stage can run at all.
    """
    prog = _chain(6)
    assert SIMPLIFIED not in " ".join(plan_compilation(prog, kappa=28).trace)
    forced = plan_compilation(prog, kappa=28, simplify="always")
    assert SIMPLIFIED in " ".join(forced.trace), forced.trace


def test_simplification_is_not_otherwise_gated(free_probes):
    """Nothing else decides whether it runs: the budget sets how long, never whether."""
    ran = " ".join(plan_compilation(_chain(6), kappa=28).trace)
    assert SIMPLIFIED in ran, ran


def test_every_stage_runs_when_the_thresholds_are_zero(free_probes):
    c = _forced(_chain(6), kappa=28)
    ran = " ".join(c.trace)
    for stage in (GREEDY, SIMPLIFIED, BEST_OF, LONGER_SEARCH):
        assert stage in ran, c.trace


def test_a_stage_that_does_not_help_is_discarded():
    """Try-and-keep-if-better is what makes a badly set threshold merely wasteful."""
    c = _forced(_chain(6), kappa=28)
    assert any(line.endswith("not kept") for line in c.trace), c.trace
    assert c.cost <= plan_compilation(_chain(6), kappa=28).cost


def test_climbing_the_ladder_never_raises_the_cost():
    for n in (2, 6, 12):
        cheap = plan_compilation(_chain(n), kappa=28)
        spent = _forced(_chain(n), kappa=28)
        assert spent.cost <= cheap.cost


def test_the_answer_does_not_depend_on_what_was_spent():
    for n in (2, 6, 12):
        cheap = plan_compilation(_chain(n), kappa=28)
        spent = _forced(_chain(n), kappa=28)
        assert np.allclose(_value(cheap), _value(spent), atol=1e-14)


def test_simplification_can_be_held_out():
    """`simplify=False` keeps the traced program, whatever the thresholds say."""
    prog = intern(_chain(6))
    c = _forced(prog, kappa=28, simplify=False)
    assert c.program == prog
    assert SIMPLIFIED not in " ".join(c.trace)


def test_stage_d_is_time_bounded(monkeypatch):
    """The hyper-search must never run under cotengra's own default budget.

    That default scales with the best tree's cost, so on an instance expensive enough to
    reach stage D it is effectively infinite -- the regression that kept `aralia/nus9601`
    from compiling.  What is asserted is the wiring: the planner constructs the finder with
    a wall-clock bound.
    """
    built = {}

    class Recording(planner.AdaptiveOrderFinder):
        def __init__(self, *args, **kwargs):
            built.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(planner, "AdaptiveOrderFinder", Recording)
    _forced(_chain(3), kappa=8)
    assert built.get("max_time") is not None


def test_a_failing_stage_leaves_the_earlier_plan_in_place(monkeypatch):
    """A search is an optimisation; losing one must not lose the compilation."""
    class Broken:
        def __init__(self, finder, limit):
            pass

        def find(self, indices, output, size, *, kappa=None):
            raise RuntimeError("no order today")

    monkeypatch.setattr(planner, "BoundedOrderFinder", Broken)
    monkeypatch.setattr(planner, "C_PROBES", 0)
    c = plan_compilation(_chain(6), kappa=28, t_search=0.0, t_escalate=1e18)
    assert "failed" in " ".join(c.trace), c.trace
    assert np.allclose(_value(c), _value(plan_compilation(_chain(6), kappa=28)))


# --------------------------------------------------------------- against enumeration

def _corpus(seed: int = 0, n: int = 25) -> list[Program]:
    rng = np.random.default_rng(seed)
    out: list[Program] = []
    while len(out) < n:
        prog = random_program(rng, n_bodies=int(rng.integers(1, 3)),
                              n_ops=int(rng.integers(1, 6)))
        try:
            semantics(prog)
        except TooLarge:
            continue
        out.append(intern(prog))
    return out


CORPUS = _corpus()


@pytest.mark.parametrize("prog", CORPUS, ids=range(len(CORPUS)))
def test_the_planned_compilation_computes_the_right_factor(prog):
    c = _forced(prog, kappa=28)
    array, scale = evaluate(c.plans, c.root)
    got = np.asarray(array, dtype=float) * float(np.exp(scale))
    assert np.allclose(got, semantics(prog), atol=1e-12)


def test_the_compilation_trace_uses_the_terms_of_the_documentation():
    import re

    c = _forced(_chain(6), kappa=28)
    for line in c.trace:
        assert line.startswith("compilation: "), line
        assert not re.search(r"\b(stage|plan|ladder|probe)", line), line


# --------------------------------------------------------------- limits and bounds

def _xor_grid(n: int) -> Program:
    """An ``n`` by ``n`` grid of coin flips, each XOR-ed with its upper and left
    neighbours, conditioned on the last cell; its treewidth grows with ``n``."""
    from pylea import DPP, Bool, flip, observe

    @DPP
    def grid() -> Bool:
        g = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                c = flip(0.3)
                if i:
                    c = c ^ g[i - 1][j]
                if j:
                    c = c ^ g[i][j - 1]
                g[i][j] = c
        observe(g[n - 1][n - 1])
        return g[0][0]

    return grid.program()


def _same_coin_returned(n: int) -> Program:
    b = BodyBuilder()
    (w,) = b.emit(Prim.flip(0.5), ())
    return Program((b.finish((), (w,) * n),), 0)


def test_a_dpp_whose_outputs_alone_exceed_the_width_bound_is_refused_before_any_search(
        monkeypatch):
    import time

    from pylea.compile.lower import ExactInferenceTooLarge
    from pylea.compile.order import GreedyOrderFinder

    def no_search(*args, **kwargs):
        raise AssertionError("a contraction tree was searched for")

    monkeypatch.setattr(GreedyOrderFinder, "find", no_search)
    started = time.monotonic()
    with pytest.raises(ExactInferenceTooLarge) as refusal:
        plan_compilation(_same_coin_returned(40), kappa=28, name="wide")
    assert time.monotonic() - started < 1.0
    message = str(refusal.value)
    assert message.startswith("exact inference on 'wide' needs a tensor with 2**40 entries "
                              "(8 TiB of memory); the limit is 2**29 entries (4 GiB)")
    assert "output variables alone have 2**40 joint values" in message
    assert "kappa=40" in message


def test_kappa_raises_the_size_limit_and_none_removes_it():
    assert plan_compilation(_same_coin_returned(30), kappa=30).plans
    assert plan_compilation(_same_coin_returned(31), kappa=None).plans


def test_a_plan_estimated_beyond_the_time_bound_is_refused_with_its_estimate(monkeypatch):
    from pylea.compile.lower import ExactInferenceTooSlow

    monkeypatch.setattr(planner, "MAX_PREDICTED_MS", 1.0)
    prog = _xor_grid(14)
    with pytest.raises(ExactInferenceTooSlow) as refusal:
        plan_compilation(prog, kappa=28, simplify=False, t_search=float("inf"),
                         t_escalate=float("inf"), name="grid")
    message = str(refusal.value)
    assert message.startswith("exact inference on 'grid' is estimated to take about ")
    assert "; the limit is 0.001 s." in message
    assert refusal.value.trace and refusal.value.trace[0].startswith("compilation: ")


def test_a_plan_left_unsliced_by_the_cost_cap_is_refused_as_too_slow_not_too_wide(
        monkeypatch):
    from pylea.compile.lower import ExactInferenceTooSlow

    monkeypatch.setattr(planner, "cost_for_ms", lambda ms: 10.0)
    with pytest.raises(ExactInferenceTooSlow) as refusal:
        plan_compilation(_xor_grid(14), kappa=4, simplify="always",
                         t_search=float("inf"), t_escalate=float("inf"))
    assert "longer than the limit of 11.6 days" in str(refusal.value)
    trace = refusal.value.trace
    assert any(SIMPLIFIED in line for line in trace), trace
    assert all("not sliced" in line for line in trace if GREEDY in line), trace


def test_a_runnable_plan_is_preferred_to_one_left_unsliced_by_the_cost_cap(monkeypatch):
    """The cap leaves stage A's tree unsliced; the search of stage C slices its own."""
    monkeypatch.setattr(planner, "cost_for_ms", lambda ms: 10.0)
    monkeypatch.setattr(planner, "C_PROBES", 0)
    c = plan_compilation(_chain(12), kappa=1, simplify=False, t_search=0.0,
                         t_escalate=float("inf"))
    assert c.runnable
    assert c.trace[0].endswith("not sliced as its cost exceeds the time limit"), c.trace
    assert c.trace[-1].startswith(f"compilation: {BEST_OF}") and \
        c.trace[-1].endswith(", kept"), c.trace
    assert np.allclose(_value(c), _value(plan_compilation(_chain(12), kappa=None)))


def test_stage_c_is_skipped_where_no_contraction_tree_can_lower_the_estimate():
    """A chain's estimate is its step count, which every contraction tree shares."""
    from pylea.compile.cost import search_gain_ms

    c = plan_compilation(_chain(2000), kappa=28, simplify=False, t_search=1.0)
    assert c.predicted_ms > 1.0 > search_gain_ms(c.plans)       # the premise
    assert BEST_OF not in " ".join(c.trace), c.trace


def test_stages_c_and_d_end_within_their_bounds(monkeypatch):
    import time

    import pylea.compile.order as order
    from pylea.compile.lower import ExactInferenceTooSlow

    monkeypatch.setattr(planner, "T_STAGE_C_SECONDS", 1.0)
    monkeypatch.setattr(planner, "T_STAGE_D_SECONDS", 1.0)
    monkeypatch.setattr(planner, "BOUND_GRACE_S", 1.0)
    monkeypatch.setattr(order, "BOUND_GRACE_S", 1.0)
    prog = _xor_grid(40)
    started = time.monotonic()
    with pytest.raises(ExactInferenceTooSlow) as refusal:
        plan_compilation(prog, kappa=28, simplify=False)
    elapsed = time.monotonic() - started
    trace = " ".join(refusal.value.trace)
    assert BEST_OF in trace and LONGER_SEARCH in trace, trace
    assert elapsed < 10.0


def test_stage_d_runs_one_bounded_search_for_all_bodies(monkeypatch):
    import pylea.compile.order as order
    from pylea.compile._process_tree import start_worker

    started = []

    def counted(module):
        started.append(module)
        return start_worker(module)

    monkeypatch.setattr(order, "start_worker", counted)
    monkeypatch.setattr(planner, "C_PROBES", 0)
    inner = []
    for k in range(4):
        b = BodyBuilder()
        (p,) = b.emit(Prim.flip(0.1 * (k + 1)), ())
        (q,) = b.emit(Prim.flip(0.6), ())
        (x,) = b.emit(Prim.and_(), (p, q))
        (y,) = b.emit(Prim.and_(), (x, p))
        inner.append(b.finish((), (y,)))
    outer = BodyBuilder(inner)
    outs = [outer.emit(i, ())[0] for i in range(4)]
    (w,) = outer.emit(Prim.and_(), (outs[0], outs[1]))
    (v,) = outer.emit(Prim.and_(), (outs[2], outs[3]))
    (r,) = outer.emit(Prim.and_(), (w, v))
    prog = Program(tuple(inner) + (outer.finish((), (r,)),), 4)
    c = plan_compilation(prog, kappa=28, simplify=False, t_search=float("inf"),
                         t_escalate=0.0)
    assert len(c.plans) == 5
    assert LONGER_SEARCH in " ".join(c.trace), c.trace
    assert started == ["pylea.compile._order_worker"]


def test_a_failed_escalation_keeps_the_greedy_tree(monkeypatch):
    import cotengra as ctg

    from pylea.compile.lower import compile_program
    from pylea.compile.order import AdaptiveOrderFinder

    def fails(self, *args, **kwargs):
        raise KeyError("tree")

    monkeypatch.setattr(ctg.AutoHQOptimizer, "search", fails)
    monkeypatch.setattr(ctg.AutoOptimizer, "search", fails)
    prog = _chain(6)
    plans, root = compile_program(prog, AdaptiveOrderFinder(escalate_cost=0.0), kappa=28)
    array, scale = evaluate(plans, root)
    np.testing.assert_allclose(np.asarray(array, dtype=float).reshape(-1) * np.exp(scale),
                               semantics(prog).reshape(-1), rtol=1e-12)
