"""The portfolio algorithm: the decision procedure between the exact engine and the
sampling engine.

`decide_scalar` answers one ``method="auto"`` scalar query.  The procedure, in order,
with every stage's own cost bounded:

  1. **Guards**, from counts alone: a DPP whose unfolding exceeds the op cap after
     the collapse pass runs the exact lane outright.  The count is the one
     `pylea.sample.query.collapsed` caches, the same count the sampling entry points
     test, so the two lanes agree on what is samplable.
  2. **Probe order by traced size.**  Greedy planning cost grows with the size of the
     DPP while the pilot's cost is linear in the unfolded size, so small DPPs run the greedy
     exact predictor first and large ones the pilot first; either way the first probe's
     fast path can commit without paying for the second.  For a small DPP the greedy
     tree is first priced unsliced, a lower bound on the prediction, and sliced only where
     the prediction's value can still change the decision: slicing a tree far wider than
     ``2**kappa`` costs more time and memory than any other probe, while such a tree's
     bound alone rules the exact lane out.
  3. **Fast path**: a probe predicting its own lane below `T_FAST_S` commits to that
     lane at once.  The regret is bounded by the threshold times the committed
     predictor's worst underestimation.  A pilot that saw no reachable mass predicts
     infinite sampling cost; with no deadline set the exact lane is then the only lane,
     and the procedure commits to it without paying for the second probe.
  4. **The remaining probe, budgeted**: the greedy predictor runs in a subprocess under
     a wall budget, capped at a `DEADLINE_HEADROOM` share of any remaining deadline so
     the probe can never spend the query's time budget, and on budget-out the exact
     lane counts as unpredicted.  A finite prediction is cached per DPP and `kappa`,
     since it is a deterministic function of the two.
  5. **Asymmetric commits.**  The exact predictor errs boundedly toward underestimation
     and unboundedly toward overestimation, because the full ladder's simplification and
     order search improve on the greedy plan by decades on exactly the hard instances,
     which no greedy-based prediction can see.  Commit exact only where even the
     underestimation margin `E_LO` leaves it ahead; commit sampling only where even the
     overestimation margin `E_HI` leaves the exact lane behind; hedge otherwise.  The
     margins are corpus quantiles, not structural bounds; the hedge is what makes a
     violated margin cost a bounded factor rather than a wrong answer.  Under
     `max_time`, committing exact additionally requires the inflated prediction to fit
     in a fraction of the remaining deadline, and the committed exact lane may spend
     only an `EXACT_COMMIT_SHARE` of it: where it overruns even that, the sampling lane
     answers from the rest, so a deadline always leaves a sampling stream to answer
     from, except where the unfolding guard left no sampling lane at all, which raises
     `TimeoutError`.
  6. **Hedge**: the exact lane runs in a fresh subprocess and the sampling lane in this
     process; the first finisher answers and the loser is killed.  A worker thread
     instead of a subprocess would hold the interpreter lock through the Python-heavy
     order search and starve the sampler; killing also makes deadline accounting exact.
     The price is one extra core while a race lasts and cold per-process caches in the
     child.  The sampling lane stops only for a worker that answered: a worker that
     failed or died leaves the sampler running to certification or the deadline, so a
     broken exact lane costs the hedge nothing but the wasted subprocess.

**A refusal of the exact engine reaches the lane choice.**  Exact inference may refuse a
DPP that needs more memory or time than its limits
(`pylea.compile.lower.ExactInferenceInfeasible`).  A probe refused for its size predicts
an unbounded exact time.  A committed exact lane that refuses, or whose subprocess fails,
hands over to the sampling lane where one exists, and otherwise raises: the refusal
itself, so that its message says what can be done.

The result records which lane answered, both predictions and every probe cost in
`Estimate.trace`, so a surprising runtime is read back rather than guessed at.  A
seeded query without `max_time` is cached like a ``method="smc"`` one; the hedge's
outcome depends on timing, so a repeated query returns the first call's certificate
rather than recomputing one.

`pylea.sample.exact_lane` runs the exact lane in its subprocess; a ``method="exact"``
query under `max_time` runs in the same subprocess and raises `TimeoutError` at the
deadline.  An exact query never falls back to sampling, since that would silently change
the requested contract.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

import numpy as np

from pylea.compile._process_tree import DeadlineExceeded
from pylea.compile.lower import ExactInferenceInfeasible, ExactInferenceTooLarge
from pylea.ir.program import op_count
from pylea.sample._exact_worker import probe_compile_kwargs
from pylea.sample.cost_model import (
    DEADLINE_HEADROOM, E_HI, E_LO, EXACT_COMMIT_SHARE, PREDICTOR_BUDGET_FACTOR,
    PREDICTOR_BUDGET_MIN_S, PROBE_ORDER_TRACED_OPS, T_FAST_S, TABLE_OUT_BITS_CAP,
    UNFOLDED_OPS_CAP,
)
from pylea.sample.exact_lane import ExactLane, exact_matrix_within
from pylea.sample.query import (
    cache_entry, certified, collapsed, origin_of, piloted, prepared, resolved_floor,
    result_cache_key, run_lane, smc_estimate, table_lane,
)
from pylea.sample.run import Deadline, RunOutcome, TableOutcome
from pylea.syntax.estimate import Estimate

if TYPE_CHECKING:
    from pylea.syntax.method import ExactTable


def decide_scalar(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
                  floor: float | None, max_time: float | None, seed: int | None,
                  what: str, exact_value, exact_from_matrix,
                  compile_kwargs: dict, forbidden: np.ndarray | None = None,
                  affine: tuple[float, float] = (1.0, 0.0)) -> Estimate:
    """One ``method="auto"`` scalar query, decided and answered.

    `exact_value` computes the answer in this process when the procedure commits to the
    exact lane without a deadline; `exact_from_matrix` maps a subprocess result's
    ``(array, scale)`` to the answer.  Relative error is met on the scale `affine` maps
    to, as `pylea.sample.query.certified_scalar` meets it.
    """
    origin = origin_of(affine)

    def compute() -> Estimate:
        started = time.monotonic()
        deadline = Deadline.after(max_time)
        choice = _choose(
            dpp, values, lambda report: report.predicted_seconds, rel_err=rel_err,
            confidence=confidence, deadline=deadline, seed=seed,
            compile_kwargs=compile_kwargs, origin=origin)
        trace = choice.trace

        def sampled(stop=None) -> RunOutcome:
            program, report, run_sequence = choice.pilot()
            return run_lane(program, values, report, rel_err=rel_err,
                            confidence=confidence, floor=floor, deadline=deadline,
                            seed_sequence=run_sequence, forbidden=forbidden, stop=stop,
                            origin=origin)

        def estimate(outcome: RunOutcome) -> Estimate:
            return smc_estimate(outcome, affine=affine, confidence=confidence,
                                trace=tuple(trace))

        def exact(value, elapsed: float) -> Estimate:
            return Estimate.from_exact(value, tuple(trace), elapsed)

        if choice.lane == "sampling":
            return estimate(sampled())
        if choice.lane == "exact":
            return _answer_exact(
                dpp, exact_value, exact_from_matrix, exact, deadline, max_time,
                compile_kwargs, trace, started,
                (lambda: estimate(sampled())) if choice.fallback else None,
                choice.alone)
        return _race(dpp, compile_kwargs, trace, started, sampled, estimate,
                     lambda array, scale: exact_from_matrix(array, scale), exact)

    key = result_cache_key(what, "auto", seed, max_time, rel_err, floor, confidence,
                           compile_kwargs)
    return certified(dpp, key, compute)


def decide_table(dpp, *, rel_err: float, confidence: float, floor: float | None,
                 max_time: float | None, seed: int | None, normalised_contract: bool,
                 what: str, compile_kwargs: dict) -> ExactTable | TableOutcome:
    """One ``method="auto"`` table query over a DPP without inputs, decided and answered
    by the procedure `decide_scalar` follows, priced by
    `pylea.sample.pilot.PilotReport.table_seconds`.

    The answer is the exact engine's `pylea.syntax.method.ExactTable`, or the
    `pylea.sample.run.TableOutcome` of a certified sampled table.  A table of more than
    ``2**pylea.sample.cost_model.TABLE_OUT_BITS_CAP`` entries has no sampling lane.
    """
    from pylea.syntax.method import ExactTable

    out_bits = sum(t.nbits() for t in dpp.output_types())
    n_entries = 1 << out_bits
    floor = resolved_floor(rel_err, floor, dpp.output_types())

    def compute() -> ExactTable | TableOutcome:
        started = time.monotonic()
        deadline = Deadline.after(max_time)
        guard = None
        if out_bits > TABLE_OUT_BITS_CAP:
            guard = (f"a sampled table certifies each of its 2**{out_bits} output values "
                     f"separately, more than the sampling engine accepts "
                     f"(2**{TABLE_OUT_BITS_CAP})")
        elif (not normalised_contract and collapsed(dpp)[1] <= UNFOLDED_OPS_CAP
              and not math.isfinite(prepared_log_scale(dpp))):
            guard = ("a sampled table holds its probabilities as float64 numbers, and "
                     "this DPP's lie outside that range")
        choice = _choose(
            dpp, np.ones(n_entries),
            lambda report: report.table_seconds(floor, rel_err, confidence,
                                                normalised_contract),
            rel_err=rel_err, confidence=confidence, deadline=deadline, seed=seed,
            compile_kwargs=compile_kwargs, guard=guard)
        trace = choice.trace

        def sampled(stop=None) -> TableOutcome:
            program, report, run_sequence = choice.pilot()
            return table_lane(program, report, rel_err=rel_err, confidence=confidence,
                              floor=floor, normalised_contract=normalised_contract,
                              deadline=deadline, seed_sequence=run_sequence, stop=stop,
                              prefix=())

        def table(outcome: TableOutcome) -> TableOutcome:
            return replace(outcome, trace=tuple(trace) + outcome.trace)

        def exact(matrix, elapsed: float) -> ExactTable:
            return ExactTable(matrix[0], matrix[1], elapsed, tuple(trace))

        if choice.lane == "sampling":
            return table(sampled())
        if choice.lane == "exact":
            return _answer_exact(
                dpp, lambda: dpp.compiled(**compile_kwargs).matrix(),
                lambda array, scale: (array, scale), exact, deadline, max_time,
                compile_kwargs, trace, started,
                (lambda: table(sampled())) if choice.fallback else None, choice.alone)
        return _race(dpp, compile_kwargs, trace, started, sampled, table,
                     lambda array, scale: (array, scale), exact)

    key = result_cache_key(what, "auto", seed, max_time, rel_err, floor, confidence,
                           compile_kwargs)
    return certified(dpp, key, compute)


def prepared_log_scale(dpp) -> float:
    """The static log factor of `dpp`'s prepared sampling form; infinite where it lies
    outside the float64 range, which a sampled table cannot fold into its entries."""
    from pylea.semantics.likelihood_table import scaled_matrix

    log_scale = prepared(dpp)[0].log_scale
    try:
        scaled_matrix(np.ones(1), log_scale)
    except ValueError:
        return math.inf
    return log_scale


@dataclass
class _Choice:
    """The lane the procedure chose and what answering it needs.

    `lane` is ``"exact"``, ``"sampling"`` or ``"hedge"``; `pilot` returns the prepared
    program, the pilot report and the run's seed sequence, running the pilot on its
    first call where the procedure did not; `fallback` says whether a committed exact
    lane may hand over to sampling, and `alone`, where it may not, why.
    """

    lane: str
    trace: list[str]
    pilot: Callable[[], tuple]
    fallback: bool = True
    alone: str = ""


def _choose(dpp, pilot_values: np.ndarray, price, *, rel_err: float, confidence: float,
            deadline: Deadline, seed: int | None, compile_kwargs: dict,
            origin: float = 0.0, guard: str | None = None) -> _Choice:
    """Stages 1 to 5 of the procedure in the module docstring.

    `pilot_values` is the pilot's [0, 1] vector over output patterns, and `price` maps
    the pilot report to the predicted sampling seconds; `guard`, where given, says why
    the query has no sampling lane besides the unfolding guard.
    """
    trace: list[str] = []
    piloted_once: list[tuple] = []

    def pilot() -> tuple:
        if not piloted_once:
            piloted_once.append(piloted(dpp, pilot_values, rel_err=rel_err,
                                        confidence=confidence, deadline=deadline,
                                        seed=seed, origin=origin))
            trace.extend(piloted_once[0][1].trace)
        return piloted_once[0]

    if guard is None:
        _program, unfolded = collapsed(dpp)
        if unfolded > UNFOLDED_OPS_CAP:
            guard = (f"sampling would run {unfolded} assignments, more than the "
                     f"sampling engine accepts ({UNFOLDED_OPS_CAP})")
    if guard is not None:
        trace.append(f"auto: exact engine chosen; {guard}")
        return _Choice("exact", trace, pilot, fallback=False, alone=guard)

    pred_exact = math.nan
    exact_bound = math.nan
    traced_ops = op_count(dpp.program())
    if traced_ops <= PROBE_ORDER_TRACED_OPS:
        exact_bound = _greedy_bound(dpp, compile_kwargs, trace)
        if exact_bound < T_FAST_S:
            pred_exact = _greedy_prediction(dpp, compile_kwargs, trace)
        if pred_exact < T_FAST_S and _exact_fits(pred_exact, deadline):
            trace.append(f"auto: exact engine chosen; its predicted time is below "
                         f"{T_FAST_S:g} s")
            return _Choice("exact", trace, pilot)

    _program, report, _run_sequence = pilot()
    if report.drawn == 0:
        trace.append("auto: the deadline passed while the DPP was prepared for sampling")
        return _Choice("sampling", trace, pilot)
    pred_mc = price(report)
    trace.append(f"auto: predicted sampling time {pred_mc:.3g} s")
    if pred_mc < T_FAST_S:
        trace.append(f"auto: sampling engine chosen; its predicted time is below "
                     f"{T_FAST_S:g} s")
        return _Choice("sampling", trace, pilot)

    if not math.isnan(exact_bound) and math.isnan(pred_exact):
        # The lane chosen moves only from exact through hedge to sampling as the
        # prediction grows, so where the bound and an infinite prediction choose the same
        # lane, every prediction at or above the bound does too.
        if _lane(exact_bound, pred_mc, deadline) == _lane(math.inf, pred_mc, deadline):
            trace.append("auto: a lower bound on the exact engine's predicted time decides")
            pred_exact = exact_bound
        else:
            pred_exact = _greedy_prediction(dpp, compile_kwargs, trace)

    if math.isnan(pred_exact):
        if math.isinf(pred_mc) and math.isinf(deadline.remaining()):
            # No reachable mass and no deadline: the exact lane is the only lane, so
            # the probe would add nothing to the decision.
            trace.append("auto: exact engine chosen; the predicted sampling time is "
                         "unbounded")
            return _Choice("exact", trace, pilot, fallback=False, alone=(
                "its predicted time is unbounded, and without max_time it would run "
                "indefinitely; pass max_time to receive the interval standing at the "
                "deadline"))
        budget = max(PREDICTOR_BUDGET_MIN_S, PREDICTOR_BUDGET_FACTOR * pred_mc)
        # The probe may never spend the query's own time budget.
        budget = min(budget, deadline.remaining() / DEADLINE_HEADROOM)
        pred_exact = _budgeted_prediction(dpp, dpp.program(), compile_kwargs, budget,
                                          trace)

    lane = _lane(pred_exact, pred_mc, deadline)
    if lane == "exact":
        trace.append("auto: exact engine chosen")
    elif lane == "sampling":
        trace.append("auto: sampling engine chosen")
    else:
        trace.append("auto: exact and sampling engines run in parallel")
    return _Choice(lane, trace, pilot)


def _race(dpp, compile_kwargs: dict, trace: list[str], started: float, sampled,
          from_sampling, from_matrix, from_exact):
    """Stage 6: the exact engine in its subprocess against the sampling engine here.

    `sampled(stop)` runs the sampling lane until `stop` says the exact engine answered;
    `from_sampling` and `from_exact` make the result from either engine's answer,
    `from_matrix` mapping the subprocess's ``(array, scale)`` to the exact value.
    """
    lane = None
    try:
        lane = ExactLane(dpp.program(), compile_kwargs, name=dpp.name())
        # The sampler stops only for a worker that answered; a failed or dead worker
        # leaves it running to certification or the deadline.
        outcome = sampled(stop=lane.finished_ok)
        if outcome.target_met or not lane.finished_ok():
            if lane.done() and not lane.finished_ok():
                trace.append(f"auto: {_failure(lane)}; the sampling engine answered")
            elif outcome.target_met:
                trace.append(f"auto: the sampling engine answered first, in "
                             f"{outcome.elapsed:.3g} s")
            else:
                trace.append("auto: neither engine finished before the deadline; the "
                             "sampling engine's interval is returned")
            return from_sampling(outcome)
        response = lane.response()
        value = from_matrix(response["array"], response["scale"])
        trace.append(f"auto: the exact engine answered first, in {lane.elapsed:.3g} s")
        return from_exact(value, time.monotonic() - started)
    finally:
        if lane is not None:
            lane.kill()


def _alone(refusal: ExactInferenceInfeasible, why: str) -> ExactInferenceInfeasible:
    """`refusal` under ``method="auto"`` where the sampling engine cannot answer instead,
    its remedy replaced by `why` that is so."""
    return type(refusal)(f"{refusal.reason}.  The sampling engine cannot answer instead: "
                         f"{why}.", refusal.trace)


def _failure(lane) -> str:
    """Why `lane`, which has ended without an answer, failed, as a trace line says it."""
    try:
        response = lane.response()
    except RuntimeError as exc:
        return f"the exact engine failed ({exc})"
    if isinstance(response.get("infeasible"), ExactInferenceInfeasible):
        return f"the exact engine refused ({response['infeasible'].reason})"
    return f"the exact engine failed ({response.get('error', 'no answer')})"


def _lane(pred_exact: float, pred_mc: float, deadline: Deadline) -> str:
    """The lane the asymmetric commits choose: ``"exact"``, ``"sampling"`` or ``"hedge"``.

    The choice moves only towards sampling as `pred_exact` grows:

        >>> d = Deadline.after(None)
        >>> [_lane(p, 1.0, d) for p in (1e-3, 1.0, 1e9)]
        ['exact', 'hedge', 'sampling']
    """
    if pred_exact / E_LO < pred_mc and _exact_fits(pred_exact, deadline):
        return "exact"
    if pred_exact / E_HI > pred_mc:
        return "sampling"
    return "hedge"


def _greedy_bound(dpp, compile_kwargs: dict, trace: list[str]) -> float:
    """A lower bound on `_greedy_prediction`, in seconds: the same greedy tree, unsliced.

    Slicing repeats steps and never removes a multiplication, and the prediction grows
    with both, so the unsliced plan predicts no more than the sliced one.
    """
    from pylea.compile.cost import predicted_ms

    started = time.monotonic()
    compiled = dpp.compiled(**{**probe_compile_kwargs(compile_kwargs), "kappa": None})
    bound = predicted_ms(compiled.plans) / 1e3
    trace.append(f"auto: lower bound on the exact engine's predicted time {bound:.3g} s, "
                 f"from a "
                 f"greedy contraction tree without slicing (found in "
                 f"{time.monotonic() - started:.3g} s)")
    return bound


def _greedy_prediction(dpp, compile_kwargs: dict, trace: list[str]) -> float:
    """The exact lane's greedy prediction in seconds, computed in this process through
    the per-DPP compiled cache."""
    from pylea.compile.cost import predicted_ms

    started = time.monotonic()
    try:
        compiled = dpp.compiled(**probe_compile_kwargs(compile_kwargs))
    except ExactInferenceTooLarge:
        trace.append(_TOO_LARGE_LINE)
        return math.inf
    prediction = predicted_ms(compiled.plans) / 1e3
    trace.append(f"auto: predicted exact engine time {prediction:.3g} s, from a greedy "
                 f"contraction tree (found in {time.monotonic() - started:.3g} s)")
    return prediction


# The trace line of a probe that exact inference refused for its size.
_TOO_LARGE_LINE = ("auto: exact inference would exceed the size limit; its time is taken "
                   "as unbounded")


def _budgeted_prediction(dpp, program, compile_kwargs: dict, budget: float,
                         trace: list[str]) -> float:
    """The greedy prediction from a subprocess under a wall budget, infinite on
    budget-out; a finite prediction is cached per DPP and `kappa`."""
    key = ("pred_exact", probe_compile_kwargs(compile_kwargs)["kappa"])
    cached = cache_entry(dpp).get(key)
    if cached is not None:
        trace.append(f"auto: predicted exact engine time {cached:.3g} s, from a greedy "
                     "contraction tree (computed earlier)")
        return cached
    lane = None
    try:
        lane = ExactLane(program, compile_kwargs, mode="predict", name=dpp.name())
        response = lane.response(timeout=budget)
    except DeadlineExceeded:
        trace.append(f"auto: predicting the exact engine's time did not finish within "
                     f"{budget:.3g} s; its time is taken as unbounded")
        return math.inf
    finally:
        if lane is not None:
            lane.kill()
    if isinstance(response.get("infeasible"), ExactInferenceTooLarge):
        trace.append(_TOO_LARGE_LINE)
        return math.inf
    if "error" in response:
        trace.append(f"auto: predicting the exact engine's time failed "
                     f"({response['error']}); its time is taken as unbounded")
        return math.inf
    prediction = response["predicted_ms"] / 1e3
    cache_entry(dpp)[key] = prediction
    trace.append(f"auto: predicted exact engine time {prediction:.3g} s, from a greedy "
                 f"contraction tree (found in {lane.elapsed:.3g} s, in a subprocess)")
    return prediction


def _exact_fits(pred_exact: float, deadline: Deadline) -> bool:
    """Whether committing to the exact lane leaves the deadline enough headroom."""
    return pred_exact / E_LO < deadline.remaining() / DEADLINE_HEADROOM


def _answer_exact(dpp, exact_value, exact_from_matrix, wrap, deadline: Deadline,
                  max_time: float | None, compile_kwargs: dict, trace: list[str],
                  started: float, fallback, alone: str = ""):
    """The committed exact lane's answer, `wrap(value, elapsed)`.

    Without a deadline the value is computed in this process.  Under one it runs in
    the killable subprocess; with a `fallback`, the sampling lane from the remaining
    time, it may spend only an `EXACT_COMMIT_SHARE` of the deadline before the
    fallback answers, and without one a miss raises `TimeoutError`.  A refusal by the
    exact engine, and under a deadline a failure of its subprocess, also hand over to the
    `fallback`; without one, they raise, a refusal saying what `alone` says: why the
    sampling engine cannot answer instead.
    """
    if max_time is None:
        try:
            value = exact_value()
        except ExactInferenceInfeasible as exc:
            if fallback is None:
                raise _alone(exc, alone) from None
            trace.append(f"auto: the exact engine refused ({exc.reason}); the sampling "
                         "engine answers")
            return fallback()
    else:
        allowance = deadline.remaining() * (EXACT_COMMIT_SHARE if fallback else 1.0)
        try:
            array, scale = exact_matrix_within(dpp, allowance, compile_kwargs)
        except DeadlineExceeded:
            if fallback is None:
                raise DeadlineExceeded(
                    "the exact engine did not finish within max_time, and the sampling "
                    f"engine cannot answer instead: {alone}; pass a larger max_time"
                ) from None
            trace.append(f"auto: the exact engine did not finish within its "
                         f"{allowance:.3g} s share of max_time; the sampling engine "
                         "answers in the remaining time")
            return fallback()
        except ExactInferenceInfeasible as exc:
            if fallback is None:
                raise _alone(exc, alone) from None
            trace.append(f"auto: the exact engine refused ({exc.reason}); the sampling "
                         "engine answers in the remaining time")
            return fallback()
        except RuntimeError as exc:
            if fallback is None:
                raise
            trace.append(f"auto: {exc}; the sampling engine answers in the remaining time")
            return fallback()
        value = exact_from_matrix(array, scale)
    return wrap(value, time.monotonic() - started)
