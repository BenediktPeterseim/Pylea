"""The escalation ladder.

What matters is that spending more never changes the answer and never makes the plan worse,
which is what lets a stage run under a threshold or a budget that is only roughly right.

The thresholds are not asserted against wall clock anywhere.  They are passed in, so a test
can force a stage to run rather than wait for an instance expensive enough to trigger it.
The dominator lift's budget is not timed here either, since a test asserting how far a pass
gets in a given number of milliseconds asserts the speed of the machine it runs on.
"""
import numpy as np
import pytest

from pylea.compile.planner import Compilation, plan_compilation
from pylea.eval.interp import evaluate
from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern
from pylea.ir.reference import TooLarge, semantics
from test.compile.random_programs import random_program

def _forced(prog, **kwargs):
    """`plan_compilation` with every threshold at zero, so the whole ladder runs."""
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
    assert "C best-of" not in ran and "D auto-hq" not in ran, ran


def test_simplification_is_skipped_when_scoring_it_would_cost_more_than_evaluating():
    """Stage B has to be scored by a second greedy compile, and that compile is not free.

    A chain of six coins evaluates in microseconds and takes milliseconds to compile, so no
    amount of rewriting can repay itself and the ladder stops after the probe.
    """
    c = plan_compilation(_chain(6), kappa=28)
    assert c.predicted_ms < c.probe_ms                  # the premise, not an assumption
    assert "B simplify" not in " ".join(c.trace), c.trace


def test_simplification_runs_regardless_once_asked_to_always():
    """The gate weighs the phase against ONE evaluation, which a repeated caller does not.

    The chain of six coins is the instance the gate declines, so this pins that ``"always"``
    overrides that decision and not merely that the stage can run at all.
    """
    prog = _chain(6)
    assert "B simplify" not in " ".join(plan_compilation(prog, kappa=28).trace)
    forced = plan_compilation(prog, kappa=28, simplify="always")
    assert "B simplify" in " ".join(forced.trace), forced.trace


def test_simplification_is_not_otherwise_gated(free_probes):
    """Nothing else decides whether it runs: the budget sets how long, never whether."""
    ran = " ".join(plan_compilation(_chain(6), kappa=28).trace)
    assert "B simplify" in ran, ran


def test_every_stage_runs_when_the_thresholds_are_zero(free_probes):
    c = _forced(_chain(6), kappa=28)
    ran = " ".join(c.trace)
    for stage in ("A greedy", "B simplify", "C best-of", "D auto-hq"):
        assert stage in ran, c.trace


def test_a_stage_that_does_not_help_is_discarded():
    """Try-and-keep-if-better is what makes a badly set threshold merely wasteful."""
    c = _forced(_chain(6), kappa=28)
    assert "  discarded" in c.trace, c.trace
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
    assert "B simplify" not in " ".join(c.trace)


def test_stage_d_is_time_bounded(monkeypatch):
    """The hyper-search must never run under cotengra's own default budget.

    That default scales with the best tree's cost, so on an instance expensive enough to
    reach stage D it is effectively infinite -- the regression that kept `aralia/nus9601`
    from compiling.  What is asserted is the wiring: the planner constructs the finder with
    a wall-clock bound.
    """
    import pylea.compile.planner as planner

    built = {}

    class Recording(planner.AdaptiveOrderFinder):
        def __init__(self, *args, **kwargs):
            built.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(planner, "AdaptiveOrderFinder", Recording)
    _forced(_chain(3), kappa=8)
    assert built.get("max_time") is not None


def test_a_failing_stage_leaves_the_earlier_plan_in_place():
    """A search is an optimisation; losing one must not lose the compilation."""
    class Broken:
        def find(self, indices, output, size, *, kappa=None):
            raise RuntimeError("no order today")

    import pylea.compile.planner as planner
    original = planner.SlicedBestOfOrderFinder
    planner.SlicedBestOfOrderFinder = Broken
    try:
        c = plan_compilation(_chain(6), kappa=28, t_search=0.0, t_escalate=1e18)
    finally:
        planner.SlicedBestOfOrderFinder = original
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
