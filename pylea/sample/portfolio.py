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
     ``2**kappa`` costs more time and memory than any other probe (seconds and gigabytes on
     `pylea.examples.birthday`), while such a tree's bound alone rules the exact lane out.
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

The result records which lane answered, both predictions and every probe cost in
`Estimate.trace`, so a surprising runtime is read back rather than guessed at.  A
seeded query without `max_time` is cached like a ``method="smc"`` one; the hedge's
outcome depends on timing, so a repeated query returns the first call's certificate
rather than recomputing one.

`exact_matrix_within` and `exact_table_within` run a ``method="exact"`` query under
`max_time` in the same killable subprocess and raise `TimeoutError` at the deadline; an
exact query never falls back to sampling, since that would silently change the
requested contract.
"""
from __future__ import annotations

import math
import pickle
import sys
import threading
import time

import numpy as np

from pylea.compile._process_tree import ProcessTree
from pylea.ir.program import op_count
from pylea.sample._exact_worker import probe_compile_kwargs
from pylea.sample.cost_model import (
    DEADLINE_HEADROOM, E_HI, E_LO, EXACT_COMMIT_SHARE, PREDICTOR_BUDGET_FACTOR,
    PREDICTOR_BUDGET_MIN_S, PROBE_ORDER_TRACED_OPS, T_FAST_S, UNFOLDED_OPS_CAP,
)
from pylea.sample.pilot import PilotReport
from pylea.sample.query import (
    _entry, certified, collapsed, piloted, result_cache_key, run_lane, smc_estimate,
)
from pylea.sample.run import Deadline
from pylea.syntax.estimate import Estimate


class ExactLane:
    """The exact engine evaluating a DPP in a killable subprocess.

    The subprocess is a `pylea.compile._process_tree.ProcessTree`, so `kill` also stops
    the processes the worker started, and every caller calls `kill` once it no longer
    needs the lane.  A background thread drains the worker's stdout, so a large response
    can never deadlock the pipe while the parent is only polling; the payload write in the
    constructor is safe because the worker reads the request before it writes anything.  A
    request that cannot be pickled becomes an error response rather than an exception, so
    a racing caller handles it like any other worker failure.
    """

    def __init__(self, program, compile_kwargs: dict, mode: str = "evaluate"):
        self.started = time.monotonic()
        self._tree = ProcessTree([sys.executable, "-m", "pylea.sample._exact_worker"])
        self._process = self._tree.process
        self._out = b""
        self._response: dict | None = None
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        try:
            self._tree.send((mode, program, compile_kwargs))
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            self._response = {
                "error": f"the request could not be pickled for the subprocess: {exc}"}
            self.kill()

    def _drain(self) -> None:
        assert self._process.stdout is not None
        self._out = self._process.stdout.read()

    def done(self) -> bool:
        return self._response is not None or self._process.poll() is not None

    def finished_ok(self) -> bool:
        """Whether the worker exited with a well-formed, non-error response."""
        if not self.done():
            return False
        try:
            return "error" not in self.response()
        except RuntimeError:
            return False

    def response(self, timeout: float | None = None) -> dict:
        """The worker's response, blocking up to `timeout` seconds (None or an infinite
        timeout blocks).

        Raises `TimeoutError` when the worker has not finished in time, killing it, and
        `RuntimeError` where it produced nothing.
        """
        response = self._response
        if response is None:
            if timeout is not None and math.isinf(timeout):
                timeout = None
            self._reader.join(timeout)
            if self._reader.is_alive():
                self.kill()
                raise TimeoutError("the exact engine did not finish within its deadline")
            self._process.wait()
            if not self._out:
                raise RuntimeError("the exact-engine subprocess produced no response")
            response = self._response = pickle.loads(self._out)
        return response

    def kill(self) -> None:
        """Stops the worker, if it is still running, with every process it started, and
        closes the pipe to it."""
        self._tree.kill()
        self._tree.close()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


def exact_matrix_within(dpp, max_time: float | None, compile_kwargs: dict):
    """`dpp`'s root matrix and log scale from the exact engine, killed at the deadline.

    Raises `TimeoutError` at the deadline and `RuntimeError` on a worker failure.
    """
    lane = ExactLane(dpp.program(), compile_kwargs)
    try:
        response = lane.response(timeout=max_time)
    except TimeoutError:
        raise TimeoutError(
            "the exact engine did not finish within max_time; an exact query never "
            "falls back to sampling, so pass a larger deadline") from None
    finally:
        lane.kill()
    if "error" in response:
        raise RuntimeError(f"the exact engine failed: {response['error']}")
    return response["array"], response["scale"]


def exact_table_within(dpp, max_time: float | None, compile_kwargs: dict):
    """The exact `LikelihoodTable` under a `max_time` deadline."""
    from pylea.semantics.likelihood_table import scaled_table

    array, scale = exact_matrix_within(dpp, max_time, compile_kwargs)
    return scaled_table(dpp.input_types(), dpp.output_types(), array, scale)


def decide_scalar(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
                  floor: float | None, max_time: float | None, seed: int | None,
                  strict: bool, what: str, exact_value, exact_from_matrix,
                  compile_kwargs: dict, forbidden: np.ndarray | None = None,
                  affine: tuple[float, float] = (1.0, 0.0)) -> Estimate:
    """One ``method="auto"`` scalar query, decided and answered.

    `exact_value` computes the answer in this process when the procedure commits to the
    exact lane without a deadline; `exact_from_matrix` maps a subprocess result's
    ``(array, scale)`` to the answer.
    """
    def compute() -> Estimate:
        return _decide(dpp, values, rel_err=rel_err, confidence=confidence,
                       floor=floor, max_time=max_time, seed=seed,
                       exact_value=exact_value, exact_from_matrix=exact_from_matrix,
                       compile_kwargs=compile_kwargs, forbidden=forbidden,
                       affine=affine)

    key = result_cache_key(what, "auto", seed, max_time, rel_err, floor, confidence)
    return certified(dpp, key, compute, strict=strict, what=what)


def _decide(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
            floor: float | None, max_time: float | None, seed: int | None,
            exact_value, exact_from_matrix, compile_kwargs: dict,
            forbidden: np.ndarray | None, affine: tuple[float, float]) -> Estimate:
    deadline = Deadline.after(max_time)
    started = time.monotonic()
    trace: list[str] = []
    traced = dpp.program()

    def sample(program, report: PilotReport, run_sequence) -> Estimate:
        outcome = run_lane(program, values, report, rel_err=rel_err,
                           confidence=confidence, floor=floor, deadline=deadline,
                           seed_sequence=run_sequence, forbidden=forbidden)
        return smc_estimate(outcome, affine=affine, confidence=confidence,
                            trace=tuple(trace))

    def exact(fallback) -> Estimate:
        return _answer_exact(dpp, exact_value, exact_from_matrix, deadline, max_time,
                             compile_kwargs, trace, started, fallback)

    _program, unfolded = collapsed(dpp)
    if unfolded > UNFOLDED_OPS_CAP:
        trace.append(f"auto: unfolding guard ({unfolded} ops after collapse), "
                     "exact lane")
        return exact(fallback=None)

    pred_exact = math.nan
    exact_bound = math.nan
    traced_ops = op_count(traced)
    if traced_ops <= PROBE_ORDER_TRACED_OPS:
        exact_bound = _greedy_bound(dpp, compile_kwargs, trace)
        if exact_bound < T_FAST_S:
            pred_exact = _greedy_prediction(dpp, compile_kwargs, trace)
        if pred_exact < T_FAST_S and _exact_fits(pred_exact, deadline):
            trace.append("auto: fast path, exact lane")

            def pilot_then_sample() -> Estimate:
                program, report, run_sequence = piloted(
                    dpp, values, rel_err=rel_err, confidence=confidence,
                    deadline=deadline, seed=seed)
                trace.extend(report.trace)
                return sample(program, report, run_sequence)

            return exact(fallback=pilot_then_sample)

    program, report, run_sequence = piloted(dpp, values, rel_err=rel_err,
                                            confidence=confidence, deadline=deadline,
                                            seed=seed)
    trace.extend(report.trace)
    pred_mc = report.predicted_seconds
    trace.append(f"auto: predicted sampling {pred_mc:.3g}s")

    if not math.isnan(exact_bound) and math.isnan(pred_exact):
        # The lane chosen moves only from exact through hedge to sampling as the
        # prediction grows, so where the bound and an infinite prediction choose the same
        # lane, every prediction at or above the bound does too.
        if _lane(exact_bound, pred_mc, deadline) == _lane(math.inf, pred_mc, deadline):
            trace.append("auto: the lower bound decides, greedy plan not sliced")
            pred_exact = exact_bound
        else:
            pred_exact = _greedy_prediction(dpp, compile_kwargs, trace)

    if math.isnan(pred_exact):
        if pred_mc < T_FAST_S:
            trace.append("auto: fast path, sampling lane")
            return sample(program, report, run_sequence)
        if math.isinf(pred_mc) and math.isinf(deadline.remaining()):
            # No reachable mass and no deadline: the exact lane is the only lane, so
            # the probe would add nothing to the decision.
            trace.append("auto: the pilot saw no reachable mass, exact lane")
            return exact(fallback=None)
        budget = max(PREDICTOR_BUDGET_MIN_S, PREDICTOR_BUDGET_FACTOR * pred_mc)
        # The probe may never spend the query's own time budget.
        budget = min(budget, deadline.remaining() / DEADLINE_HEADROOM)
        pred_exact = _budgeted_prediction(dpp, traced, compile_kwargs, budget, trace)

    lane = _lane(pred_exact, pred_mc, deadline)
    if lane == "exact":
        trace.append("auto: committed to the exact lane")
        return exact(fallback=lambda: sample(program, report, run_sequence))
    if lane == "sampling":
        trace.append("auto: committed to the sampling lane")
        return sample(program, report, run_sequence)

    trace.append("auto: hedge, exact lane racing in a subprocess")
    lane = ExactLane(traced, compile_kwargs)
    try:
        # The sampler stops only for a worker that answered; a failed or dead worker
        # leaves it running to certification or the deadline.
        outcome = run_lane(program, values, report, rel_err=rel_err,
                           confidence=confidence, floor=floor, deadline=deadline,
                           seed_sequence=run_sequence, forbidden=forbidden,
                           stop=lane.finished_ok)
        if outcome.target_met or not lane.finished_ok():
            if lane.done() and not lane.finished_ok():
                trace.append("auto: the exact lane failed; the sampling lane answered")
            else:
                trace.append(f"auto: sampling lane answered in {outcome.elapsed:.3g}s")
            return smc_estimate(outcome, affine=affine, confidence=confidence,
                                trace=tuple(trace))
        value = exact_from_matrix(lane.response()["array"], lane.response()["scale"])
        trace.append(f"auto: exact lane answered in {lane.elapsed:.3g}s")
        return Estimate.exact(value, tuple(trace), time.monotonic() - started)
    finally:
        lane.kill()


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
    trace.append(f"auto: greedy exact lower bound {bound:.3g}s "
                 f"(probe {time.monotonic() - started:.3g}s)")
    return bound


def _greedy_prediction(dpp, compile_kwargs: dict, trace: list[str]) -> float:
    """The exact lane's greedy prediction in seconds, computed in this process through
    the per-DPP compiled cache."""
    from pylea.compile.cost import predicted_ms

    started = time.monotonic()
    compiled = dpp.compiled(**probe_compile_kwargs(compile_kwargs))
    prediction = predicted_ms(compiled.plans) / 1e3
    trace.append(f"auto: greedy exact prediction {prediction:.3g}s "
                 f"(probe {time.monotonic() - started:.3g}s)")
    return prediction


def _budgeted_prediction(dpp, program, compile_kwargs: dict, budget: float,
                         trace: list[str]) -> float:
    """The greedy prediction from a subprocess under a wall budget, infinite on
    budget-out; a finite prediction is cached per DPP and `kappa`."""
    key = ("pred_exact", probe_compile_kwargs(compile_kwargs)["kappa"])
    cached = _entry(dpp).get(key)
    if cached is not None:
        trace.append(f"auto: greedy exact prediction {cached:.3g}s (cached)")
        return cached
    lane = ExactLane(program, compile_kwargs, mode="predict")
    try:
        response = lane.response(timeout=budget)
    except TimeoutError:
        trace.append(f"auto: exact predictor exceeded its {budget:.3g}s budget, "
                     "treated as unpredicted")
        return math.inf
    finally:
        lane.kill()
    if "error" in response:
        trace.append(f"auto: exact predictor failed ({response['error']})")
        return math.inf
    prediction = response["predicted_ms"] / 1e3
    _entry(dpp)[key] = prediction
    trace.append(f"auto: greedy exact prediction {prediction:.3g}s "
                 f"(probe {lane.elapsed:.3g}s, subprocess)")
    return prediction


def _exact_fits(pred_exact: float, deadline: Deadline) -> bool:
    """Whether committing to the exact lane leaves the deadline enough headroom."""
    return pred_exact / E_LO < deadline.remaining() / DEADLINE_HEADROOM


def _answer_exact(dpp, exact_value, exact_from_matrix, deadline: Deadline,
                  max_time: float | None, compile_kwargs: dict, trace: list[str],
                  started: float, fallback) -> Estimate:
    """The committed exact lane's answer.

    Without a deadline the value is computed in this process.  Under one it runs in
    the killable subprocess; with a `fallback`, the sampling lane from the remaining
    time, it may spend only an `EXACT_COMMIT_SHARE` of the deadline before the
    fallback answers, and without one a miss raises `TimeoutError`.
    """
    if max_time is None:
        value = exact_value()
    else:
        allowance = deadline.remaining() * (EXACT_COMMIT_SHARE if fallback else 1.0)
        try:
            array, scale = exact_matrix_within(dpp, allowance, compile_kwargs)
        except TimeoutError:
            if fallback is None:
                raise TimeoutError(
                    "the exact engine did not finish within max_time, and the "
                    "unfolding guard left no sampling lane to answer from; pass a "
                    "larger deadline") from None
            trace.append(f"auto: the exact lane missed its {allowance:.3g}s share of "
                         "the deadline; the sampling lane answers from the rest")
            return fallback()
        value = exact_from_matrix(array, scale)
    return Estimate.exact(value, tuple(trace), time.monotonic() - started)
