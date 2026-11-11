"""Compilation planning.

`plan_compilation` returns a `Compilation`: the plans it settled on, their contraction cost,
and a record of which stages ran and whether each was kept.

Four stages run in order.  Stage A always runs; each later one is entered only when the
estimated cost of evaluating the plan in hand exceeds what that stage costs:

    A   a greedy contraction tree             always
    B   simplification, time-budgeted         estimate above stage A's own compile time
    C   the best of many greedy trees,        estimate above `t_search`
        scored after slicing, then locally
        reconfigured
    D   cotengra's hyper-optimiser, under     estimate above `t_escalate`
        a wall-clock bound

The estimate comes from `pylea.compile.cost` and is good to an order of magnitude, which is
what deciding one side or the other of a multi-second search needs.

A stage's result replaces the one in hand only when it lowers the contraction cost, and a
stage raising an exception is noted and skipped.  A threshold set too low therefore costs
compile time and cannot make the plan worse.

Stage B takes a time budget where stages C and D take a threshold.  A threshold presumes the
stage costs about what the threshold says, which holds for a contraction-tree search and
not for simplification: its rules fire as often as the DPP admits, and some of them re-read
the whole body once per site they rewrite, so the phase costs what the DPP admits rather
than what it contains.  The budget is one evaluation of the plan in hand, so the stage
cannot spend more than it stands to save.  `pylea.simplify.simplify` may stop at any
point and still return a valid DPP, which is what allows a budget in place of an estimate.

Stage B is skipped outright when evaluation is cheaper than stage A's compile, since a
stage costs at least one further compile and could not repay itself.  Passing
``simplify="always"`` drops both that test and the budget, for a caller compiling once and
evaluating many times.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from pylea.compile.cost import contraction_cost, predicted_ms
from pylea.compile.lower import compile_program
from pylea.compile.order import (
    AdaptiveOrderFinder, GreedyOrderFinder, OrderFinder, SlicedBestOfOrderFinder,
)
from pylea.compile.plan import Plan
from pylea.ir.program import Program
from pylea.simplify import simplify as simplify_program

# Estimated evaluation, in milliseconds, above which stages C and D are entered.  Each holds
# what its own stage adds to a compile, timed over the fault-tree corpus on an Apple M3 Pro.
# Re-time on another machine or after changing the contraction-tree search or the evaluator.
T_SEARCH = 500.0
T_ESCALATE = 107000.0

# Wall-clock bound on stage D's search, in seconds.  cotengra's own budget scales with the
# contraction cost and so never binds at this stage.  The finder stops the search at the
# bound plus `pylea.compile.order.BOUND_GRACE_S`; a stopped search fails the stage, which
# leaves the plan in hand from the earlier stages.
T_STAGE_D_SECONDS = 60.0


@dataclass(frozen=True)
class Compilation:
    """Plans for a DPP, their contraction cost, and a record of the stages that ran."""

    program: Program
    plans: dict[int, Plan]
    root: int
    cost: float
    trace: tuple[str, ...]
    probe_ms: float = 0.0                   # wall clock of the greedy compile producing it

    @property
    def predicted_ms(self) -> float:
        return predicted_ms(self.plans)


def plan_compilation(prog: Program, kappa: int | None = None, *,
                     t_search: float = T_SEARCH, t_escalate: float = T_ESCALATE,
                     simplify: bool | Literal["always"] = True) -> Compilation:
    """`prog` compiled, each optional stage entered only where it is estimated to pay.

    `kappa` bounds one intermediate to ``2**kappa`` entries by slicing.  `t_search` and
    `t_escalate` are the thresholds, in milliseconds of estimated evaluation, for stages C
    and D.

    `simplify` governs stage B alone, the contraction-tree searches running either way.
    `False` leaves the DPP as it was traced, and ``"always"`` runs simplification with
    neither the comparison against stage A's compile time nor the budget.

    `Compilation.trace` shows which stages ran.  A DPP this small stops at stage A:

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plan_compilation(both.program(), kappa=28).trace
        ('A greedy: cost 1.200e+01',)
    """
    # Stage A is the one stage whose failure is not survivable, every later stage being an
    # optimisation over a plan that already runs.
    best = _compile(prog, GreedyOrderFinder(), kappa, "A greedy")

    # Stage A's own compile time is what stage B is worth entering against.  Simplification
    # is scored by a further compile of at least that cost, so where the whole evaluation is
    # cheaper than one compile, no optional stage can repay itself.
    always = simplify == "always"
    if simplify and (always or best.predicted_ms > best.probe_ms):
        budget = None if always else best.predicted_ms
        best = _better(best, _attempt(simplify_program(prog, budget_ms=budget),
                                      GreedyOrderFinder(), kappa, "B simplify"))

    if best.predicted_ms > t_search:
        best = _better(best, _attempt(best.program, SlicedBestOfOrderFinder(), kappa,
                                      "C best-of"))

    if best.predicted_ms > t_escalate:
        # `escalate_cost=0.0` makes the finder search unconditionally: the decision has been
        # taken here, in estimated milliseconds, and the finder's own cost threshold would
        # otherwise take it again.
        best = _better(best, _attempt(
            best.program,
            AdaptiveOrderFinder(escalate_cost=0.0, max_time=T_STAGE_D_SECONDS),
            kappa, "D auto-hq"))
    return best


def _compile(prog: Program, finder: OrderFinder, kappa: int | None, label: str) -> Compilation:
    """`prog` compiled with `finder`, timed, and labelled for the trace."""
    start = perf_counter()
    plans, root = compile_program(prog, finder, kappa=kappa)
    elapsed = (perf_counter() - start) * 1e3
    cost = contraction_cost(plans)
    return Compilation(prog, plans, root, cost, (f"{label}: cost {cost:.3e}",), elapsed)


def _attempt(prog: Program, finder: OrderFinder, kappa: int | None,
             label: str) -> Compilation | _Failed:
    """`prog` compiled with `finder`, or a `_Failed` carrying the exception it raised.

    A stage that fails is one the caller declines to use rather than an error, since the
    stage before it has already produced a runnable plan.
    """
    try:
        return _compile(prog, finder, kappa, label)
    except Exception as exc:                          # noqa: BLE001, any failed search
        return _Failed(label, exc)


class _Failed:
    """A stage that produced no plan, and the note recording that in the trace."""

    def __init__(self, label: str, exc: BaseException) -> None:
        self.note = f"{label}: failed, {type(exc).__name__}"


def _better(best: Compilation, candidate) -> Compilation:
    """Whichever of `best` and `candidate` has the lower contraction cost.

    The trace records both outcomes, so a compilation can be read back rather than guessed at.
    """
    if isinstance(candidate, _Failed):
        return replace(best, trace=best.trace + (candidate.note,))
    if candidate.cost < best.cost:
        return replace(candidate, trace=best.trace + candidate.trace + ("  kept",))
    return replace(best, trace=best.trace + candidate.trace + ("  discarded",))
