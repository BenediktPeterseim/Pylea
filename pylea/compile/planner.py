"""Compilation planning.

`plan_compilation` returns a `Compilation`: the plans it settled on, their contraction cost,
and a record of which stages ran and whether each was kept.

Four stages run in order.  Stage A always runs; each later one is entered only where the
estimated cost of evaluating the plan in hand exceeds what that stage costs:

    A   a greedy contraction tree             always
    B   simplification, time-budgeted         estimate above stage A's own compile time
    C   the best of many greedy trees,        gain above `t_search` and above the time
        scored after slicing, then locally    of `C_PROBES` greedy trees
        reconfigured
    D   cotengra's hyper-optimiser            gain above `t_escalate` and above the time
                                              of `C_PROBES` greedy trees

The estimate comes from `pylea.compile.cost` and is good to an order of magnitude, which is
what deciding one side or the other of a multi-second search needs.

A stage's result replaces the one in hand only when it lowers the contraction cost, and a
stage raising an exception is noted and skipped.  A threshold set too low therefore costs
compile time and cannot make the plan worse.

**Stages C and D are gated on what a contraction tree can change.**  The gain,
`pylea.compile.cost.search_gain_ms`, leaves out the steps every tree has, since no search
can remove them; a long chain whose estimate is all such steps stops after stage A.  A
search draws at least `C_PROBES` greedy trees, so the gain must also exceed that many times
what stage A's tree took to find.  Stage C's budget is the gain, up to
`T_STAGE_C_SECONDS`, so that it cannot spend more than it stands to save.  A search that
stops on its budget returns a tree that depends on the machine and its load, and the trace
says when that happened.  Each of the two runs its whole search, for every definition of
the DPP, in one subprocess that is stopped at the stage's bound plus
`pylea.compile.order.BOUND_GRACE_S`.

Stage B takes a time budget where stages C and D take a threshold.  A threshold presumes the
stage costs about what the threshold says, which holds for a contraction-tree search and
not for simplification: its rules fire as often as the DPP admits, and some of them re-read
the whole body once per site they rewrite, so the phase costs what the DPP admits rather
than what it contains.  The budget is one evaluation of the plan in hand, up to
`T_STAGE_B_MAX_MS`, so the stage cannot spend more than it stands to save.
`pylea.simplify.simplify` may stop at any point and still return a valid DPP, which is
what allows a budget in place of an estimate.

Stage B is skipped outright when evaluation is cheaper than stage A's compile, since a
stage costs at least one further compile and could not repay itself.  Passing
``simplify="always"`` drops both that test and the budget, for a caller compiling once and
evaluating many times.

**Exact inference may be refused.**  With `kappa` set, a tensor that no contraction tree
and no slicing avoid, above ``pylea.compile.lower.entry_limit(kappa)`` entries, raises
`pylea.compile.lower.ExactInferenceTooLarge` before any search.  After the last stage, a
plan estimated above `MAX_PREDICTED_MS` raises `pylea.compile.lower.ExactInferenceTooSlow`.
So does a plan that only exists unsliced: the greedy trees of stages A and B are not
sliced where their cost alone exceeds what `MAX_PREDICTED_MS` allows
(`pylea.compile.order.GreedyOrderFinder`'s `slice_cost_cap`), and such a plan serves as the
estimate later stages are compared with, while any runnable plan is preferred to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from pylea.compile.cost import (
    contraction_cost, cost_for_ms, predicted_ms, search_gain_ms,
)
from pylea.compile.lower import (
    ExactInferenceTooSlow, compile_program, entry_limit, subject,
)
from pylea.compile.order import (
    BOUND_GRACE_S, AdaptiveOrderFinder, BoundedOrderFinder, GreedyOrderFinder,
    OrderFinder, SlicedBestOfOrderFinder, check_finder_and_kappa, find_orders,
)
from pylea.compile.plan import Plan
from pylea.ir.program import Program
from pylea.simplify import simplify as simplify_program

# The least gain, `pylea.compile.cost.search_gain_ms` in milliseconds, for which stages C
# and D are entered; each is a floor beside the time of `C_PROBES` greedy trees.  Each is
# the wall clock its own stage added to a compile on an Apple M3 Pro: for stage C, the
# searches of the three fault trees on which it ran, between the largest gain it was
# declined at and the smallest it was entered at; for stage D, the search of one fault
# tree.  Not reproducible from this repository.  Re-time on another machine, or after
# changing the contraction tree finders or `pylea.eval`.
T_SEARCH = 500.0
T_ESCALATE = 107000.0

# The number of greedy trees a contraction tree search draws at least before it can stop:
# the patience of `SlicedBestOfOrderFinder`.  Derived, not fitted.
C_PROBES = SlicedBestOfOrderFinder().patience

# Upper bound on stage B's budget, in milliseconds.  Chosen, not fitted.
T_STAGE_B_MAX_MS = 30000.0

# Upper bound on stage C's budget, in seconds.  Chosen, not fitted.
T_STAGE_C_SECONDS = 20.0

# Wall-clock bound on stage D's search, in seconds.  Chosen, not fitted.  cotengra is given
# the same budget, which it tests only between trials, so the search is also stopped at the
# bound plus `pylea.compile.order.BOUND_GRACE_S`; a stopped search fails the stage, which
# leaves the plan in hand from the earlier stages.
T_STAGE_D_SECONDS = 60.0

# The estimated evaluation, in milliseconds, above which exact inference is refused: about
# 11.6 days.  Chosen, not fitted: three orders of magnitude above any wait a user accepts,
# so that an estimate good to an order of magnitude does not refuse a query that would
# finish in such a wait.
MAX_PREDICTED_MS = 1e9


@dataclass(frozen=True)
class Compilation:
    """Plans for a DPP, their contraction cost, and a record of the stages that ran.

    `runnable` is false for plans left unsliced over the cost cap of
    `pylea.compile.order.GreedyOrderFinder`, which may hold a tensor above the size limit
    slicing aims for and serve only as an estimate.
    """

    program: Program
    plans: dict[int, Plan]
    root: int
    cost: float
    trace: tuple[str, ...]
    probe_ms: float = 0.0                   # wall clock of the compile producing it
    order_ms: float = 0.0                   # of which in the contraction tree finder
    runnable: bool = True

    @property
    def predicted_ms(self) -> float:
        return predicted_ms(self.plans)


# The trace's names of the four stages, in the terms of the user documentation.
GREEDY = "greedy contraction tree"
SIMPLIFIED = "simplified DPP"
BEST_OF = "best of several contraction trees"
LONGER_SEARCH = "longer contraction tree search"


def plan_compilation(prog: Program, kappa: int | None = None, *,
                     t_search: float = T_SEARCH, t_escalate: float = T_ESCALATE,
                     simplify: bool | Literal["always"] = True,
                     name: str | None = None) -> Compilation:
    """`prog` compiled, each optional stage entered only where it is estimated to pay.

    `kappa` is the size limit slicing aims for, ``2**kappa`` entries per intermediate, and
    sets the size limit of exact inference (`pylea.compile.lower.entry_limit`); `None`
    leaves both unlimited.  `t_search` and `t_escalate` are the thresholds, in milliseconds
    of estimated evaluation that a contraction tree can change, for stages C and D.

    `simplify` governs stage B alone, the contraction-tree searches running either way.
    `False` leaves the DPP as it was traced, and ``"always"`` runs simplification with
    neither the comparison against stage A's compile time nor the budget.

    `Compilation.trace` shows which stages ran.  A DPP this small stops at stage A:

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plan_compilation(both.program(), kappa=28).trace
        ('compilation: greedy contraction tree, cost 1.200e+01',)

    Raises `pylea.compile.lower.ExactInferenceTooLarge` and
    `pylea.compile.lower.ExactInferenceTooSlow` as the module docstring says, their
    messages naming the DPP by `name`, and what
    `pylea.compile.order.check_finder_and_kappa` raises for an invalid `kappa`.
    """
    kappa = check_finder_and_kappa(None, kappa)
    limit = entry_limit(kappa)
    greedy = GreedyOrderFinder(slice_cost_cap=cost_for_ms(MAX_PREDICTED_MS))
    # Stage A is the one stage whose failure is not survivable, every later stage being an
    # optimisation over a plan that already runs.
    best = _compile(prog, greedy, kappa, limit, name, GREEDY)

    # Stage A's own compile time is what stage B is worth entering against.  Simplification
    # is scored by a further compile of at least that cost, so where the whole evaluation is
    # cheaper than one compile, no optional stage can repay itself.
    always = simplify == "always"
    if simplify and (always or best.predicted_ms > best.probe_ms):
        budget = None if always else min(best.predicted_ms, T_STAGE_B_MAX_MS)
        best = _better(best, _attempt(simplify_program(prog, budget_ms=budget), greedy,
                                      kappa, limit, name, SIMPLIFIED))

    gain = search_gain_ms(best.plans)
    if gain > max(t_search, C_PROBES * best.order_ms):
        budget_s = min(T_STAGE_C_SECONDS, gain / 1e3)
        best_of = SlicedBestOfOrderFinder(budget=budget_s)
        best = _better(best, _attempt(
            best.program, BoundedOrderFinder(best_of, budget_s + BOUND_GRACE_S), kappa,
            limit, name, BEST_OF, stopped=lambda: best_of.starved))

    gain = search_gain_ms(best.plans)
    if gain > max(t_escalate, C_PROBES * best.order_ms):
        # `escalate_cost=0.0` makes the finder search unconditionally: the decision has been
        # taken here, in estimated milliseconds, and the finder's own cost threshold would
        # otherwise take it again.
        best = _better(best, _attempt(
            best.program,
            AdaptiveOrderFinder(escalate_cost=0.0, max_time=T_STAGE_D_SECONDS),
            kappa, limit, name, LONGER_SEARCH))

    if not best.runnable or best.predicted_ms > MAX_PREDICTED_MS:
        raise ExactInferenceTooSlow(_too_slow_message(best, name), best.trace)
    return best


def _too_slow_message(best: Compilation, name: str | None) -> str:
    """The refusal of `best` as too slow.  The estimate of a plan left unsliced is a lower
    bound, slicing adding steps and never removing a multiplication."""
    limit = duration(MAX_PREDICTED_MS)
    if math.isfinite(best.predicted_ms) and best.predicted_ms > MAX_PREDICTED_MS:
        about = "about" if best.runnable else "at least"
        took = (f"is estimated to take {about} {duration(best.predicted_ms)}; the limit is "
                f"{limit}")
    else:
        took = f"is estimated to take longer than the limit of {limit}"
    return (f"exact inference on {subject(name)} {took}.  Sample with method=\"smc\" or "
            f"the default method=\"auto\".")


def duration(ms: float) -> str:
    """`ms` milliseconds in the largest unit, up to years, that it fills at least once.

        >>> duration(1e9), duration(2.5e3), duration(5.3e13), duration(9.5e48)
        ('11.6 days', '2.5 s', '1.68e3 years', '3.01e38 years')
    """
    seconds = ms / 1e3
    for unit, size in (("years", 365.25 * 86400), ("days", 86400.0), ("hours", 3600.0),
                       ("minutes", 60.0)):
        if seconds >= size:
            return f"{_number(seconds / size)} {unit}"
    return f"{_number(seconds)} s"


def _number(x: float) -> str:
    """`x` to three significant digits, an exponent written without sign or padding."""
    mantissa, _, exponent = f"{x:.3g}".partition("e")
    return mantissa if not exponent else f"{mantissa}e{int(exponent)}"


class _Timed:
    """`finder`, with the wall clock its searches take added up in `ms`, and whether any
    order it found was left unsliced over its cost cap in `over_cost_cap`."""

    def __init__(self, finder: OrderFinder):
        self.finder = finder
        self.ms = 0.0
        self.over_cost_cap = False

    def find(self, indices, output, size, *, kappa: int | None = None):
        return self.find_all([(indices, output, size)], kappa=kappa)[0]

    def find_all(self, networks, *, kappa: int | None = None):
        start = perf_counter()
        try:
            orders = find_orders(self.finder, networks, kappa=kappa)
        finally:
            self.ms += (perf_counter() - start) * 1e3
        self.over_cost_cap = self.over_cost_cap or any(o.over_cost_cap for o in orders)
        return orders


def _compile(prog: Program, finder: OrderFinder, kappa: int | None, limit: int | None,
             name: str | None, label: str, stopped=lambda: False) -> Compilation:
    """`prog` compiled with `finder`, timed, and labelled for the trace.

    `stopped` says, once the compile has run, whether the search stopped on its budget.
    """
    timed = _Timed(finder)
    start = perf_counter()
    plans, root = compile_program(prog, timed, kappa=kappa, max_entries=limit, name=name)
    elapsed = (perf_counter() - start) * 1e3
    cost = contraction_cost(plans)
    runnable = not timed.over_cost_cap
    line = f"compilation: {label}, cost {cost:.3e}"
    if not runnable:
        line += ", not sliced as its cost exceeds the time limit"
    if stopped():
        line += ", search stopped at its time limit"
    return Compilation(prog, plans, root, cost, (line,), elapsed, timed.ms, runnable)


def _attempt(prog: Program, finder: OrderFinder, kappa: int | None, limit: int | None,
             name: str | None, label: str, stopped=lambda: False) -> Compilation | _Failed:
    """`prog` compiled with `finder`, or a `_Failed` carrying the exception it raised.

    A stage that fails is one the caller declines to use rather than an error, since the
    stage before it has already produced a plan.
    """
    try:
        return _compile(prog, finder, kappa, limit, name, label, stopped)
    except Exception as exc:                          # noqa: BLE001, any failed search
        return _Failed(label, exc)


class _Failed:
    """A stage that produced no plan, and the note recording that in the trace."""

    def __init__(self, label: str, exc: BaseException) -> None:
        self.note = f"compilation: {label}, failed ({type(exc).__name__})"


def _better(best: Compilation, candidate) -> Compilation:
    """Whichever of `best` and `candidate` is runnable, or, where both or neither are, has
    the lower contraction cost.

    The trace records both outcomes, so a compilation can be read back rather than guessed at.
    """
    if isinstance(candidate, _Failed):
        return replace(best, trace=best.trace + (candidate.note,))
    *earlier, last = candidate.trace
    if (candidate.runnable, -candidate.cost) > (best.runnable, -best.cost):
        return replace(candidate, trace=best.trace + tuple(earlier) + (last + ", kept",))
    return replace(best, trace=best.trace + tuple(earlier) + (last + ", not kept",))
