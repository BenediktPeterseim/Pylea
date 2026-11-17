"""Sampled execution behind the public query functions.

`certified_scalar` and `certified_table` take a `pylea.syntax.dpp.DPP` and drive the full
sampling pipeline: feasibility guards, the cached schedule, the pilot, the chosen mode's
certified run, and the `pylea.sample.estimate` result types.  The public entry points in
`pylea.syntax` call these only when a query actually asks for sampling, so a Python
process that never samples never imports this package.

Guards, from counts alone before anything runs: sampling requires a zero-input DPP,
an unfolding of at most `pylea.sample.cost_model.UNFOLDED_OPS_CAP` ops after the
collapse pass, computed from the call DAG without unfolding, and, for a table, at most
``2**pylea.sample.cost_model.TABLE_OUT_BITS_CAP`` entries, since a sampled table
certifies every output pattern separately.  The portfolio algorithm reads the same
collapsed count through `collapsed`, so the two lanes agree on what is samplable.  An
explicit ``method="smc"`` query a guard excludes raises `ValueError`; the portfolio
algorithm instead routes it to the exact engine.

Caching, keyed so it cannot contradict reproducibility: the schedule is a deterministic
seed-free function of the DPP and is cached per DPP, in `cache_entry`.  Pilot
statistics are not cached; they depend on the query's seed and on `rel_err`.  Sampled
results are cached only under an explicit `seed`, keyed on the full contract, and never
when `max_time` was set, since a deadline makes the outcome timing-dependent.  Every
certified entry point, the portfolio algorithm's included, goes through `certified`, the
one place that reads and writes the result cache.  A missed target is reported by the
public query function, on the object it returns.

A ``method="smc"`` query with an explicit `seed` and no `max_time` is bit-reproducible:
the pilot and the certified run draw from separate children of one
`numpy.random.SeedSequence`, and every data-dependent choice is a function of seeded
statistics and sample counts only.
"""
from __future__ import annotations

import time
import weakref
from dataclasses import replace

import numpy as np

from pylea.ir.program import Program
from pylea.sample.collapse import collapse
from pylea.sample.cost_model import TABLE_OUT_BITS_CAP, UNFOLDED_OPS_CAP
from pylea.sample.pilot import PilotReport, run_pilot
from pylea.sample.run import (
    Deadline, RunOutcome, TableOutcome, run_iid, run_sweeps, run_table,
)
from pylea.sample.schedule import schedule
from pylea.sample.sweep import SweepProgram, unfold, unfolded_op_count
from pylea.semantics.likelihood_table import scaled_matrix
from pylea.syntax.estimate import Estimate

_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def cache_entry(dpp) -> dict:
    """The per-DPP cache of this package: a dictionary held while `dpp` lives, in which
    the collapsed DPP, the scheduled program, seeded results and the portfolio
    algorithm's exact predictions are stored under keys of their own."""
    entry = _cache.get(dpp)
    if entry is None:
        entry = _cache[dpp] = {}
    return entry


def collapsed(dpp) -> tuple[Program, int]:
    """The collapsed `Program` of `dpp` and its unfolded op count, cached per DPP.

    This is the count every feasibility decision reads, the portfolio algorithm's
    included: narrow affordable bodies are collapsed to their exact tables first
    (`pylea.sample.collapse`), so a deep call tree with small interfaces counts as a few
    table ops instead of an exponential unfolding.  The count itself is computed from
    the call DAG, so reading it costs nothing.
    """
    entry = cache_entry(dpp)
    if "collapsed" not in entry:
        program = collapse(dpp.program())
        entry["collapsed"] = (program, unfolded_op_count(program))
    return entry["collapsed"]


def prepared(dpp) -> tuple[SweepProgram, int]:
    """The scheduled `SweepProgram` of `dpp` and its op count, cached per DPP.

    Raises `ValueError` where `dpp` has inputs, or where the collapsed DPP still
    unfolds beyond the op cap.
    """
    entry = cache_entry(dpp)
    if "prepared" not in entry:
        if dpp.input_types() != tuple():
            raise ValueError(
                f"the sampling engine needs a DPP without inputs; this one has input types "
                f"{dpp.input_types()}: evaluate exactly, or wrap the input you care "
                "about into a zero-input DPP")
        program, ops = collapsed(dpp)
        if ops > UNFOLDED_OPS_CAP:
            raise ValueError(
                f"sampling this DPP would run {ops} assignments per sample, more than the "
                f"sampling engine accepts ({UNFOLDED_OPS_CAP}); use method='exact'")
        body = schedule(unfold(program))
        entry["prepared"] = (SweepProgram(body), len(body.ops))
    return entry["prepared"]


def resolved_floor(rel_err: float, floor: float | None, output_types) -> float:
    """The table contract's floor: the caller's, or ``min(rel_err, 1 / (4 n))``.

    `n` is the entry count ``2**out_bits``; below the default a uniform relative
    guarantee would be priced by the rarest entry.
    """
    if floor is not None:
        return floor
    n_entries = 1 << sum(t.nbits() for t in output_types)
    return min(rel_err, 1.0 / (4.0 * n_entries))


def origin_of(affine: tuple[float, float]) -> float:
    """The point of the sampled [0, 1] scale that ``a * x + b`` maps to 0, from which
    relative error on the returned value's scale is measured; 0 where ``a`` is 0.

        >>> origin_of((1.0, 0.0)), origin_of((2.0, -1.0))
        (0.0, 0.5)
    """
    a, b = affine
    return -b / a if a > 0.0 and b != 0.0 else 0.0


def piloted(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
            deadline: Deadline, seed: int | None, origin: float = 0.0
            ) -> tuple[SweepProgram, PilotReport, np.random.SeedSequence]:
    """The prepared `SweepProgram`, its pilot report for `values`, and the seed sequence the
    certified run draws from.

    The pilot and the run use separate children of one sequence seeded by `seed`, so a
    seeded query's run never depends on how many draws the pilot made.  Relative error
    is measured from `origin`, as `pylea.sample.certify.met_relative` states.
    """
    pilot_sequence, run_sequence = np.random.SeedSequence(seed).spawn(2)
    program, ops = prepared(dpp)
    report = run_pilot(program, values, rel_err=rel_err, confidence=confidence,
                       unfolded_ops=ops, seed_sequence=pilot_sequence, deadline=deadline,
                       origin=origin)
    return program, report, run_sequence


def run_lane(program: SweepProgram, values: np.ndarray, report: PilotReport, *,
             rel_err: float, confidence: float, floor: float | None,
             deadline: Deadline, seed_sequence: np.random.SeedSequence,
             forbidden: np.ndarray | None, stop=None, origin: float = 0.0) -> RunOutcome:
    """The pilot's chosen mode run to certification, deadline-priced by its timings,
    with relative error measured from `origin`."""
    if report.mode == "iid":
        return run_iid(program, values, rel_err=rel_err, confidence=confidence,
                       batch=report.batch, seed_sequence=seed_sequence, floor=floor,
                       deadline=deadline, forbidden=forbidden,
                       seconds_per_particle=report.iid_seconds_per_particle, stop=stop,
                       origin=origin)
    return run_sweeps(program, values, rel_err=rel_err, confidence=confidence,
                      n_particles=report.n_particles, seed_sequence=seed_sequence,
                      floor=floor, deadline=deadline, forbidden=forbidden,
                      seconds_per_particle=report.sweep_seconds_per_particle, stop=stop,
                      origin=origin, stream_rel_err=report.stream_rel_err)


def smc_estimate(outcome: RunOutcome, *, affine: tuple[float, float],
                 confidence: float, trace: tuple[str, ...]) -> Estimate:
    """The `Estimate` of a certified run, its interval mapped by ``a * x + b`` with
    ``a >= 0``, which keeps the order of the interval's ends."""
    a, b = affine
    assert a >= 0.0, affine
    return Estimate(a * outcome.point + b, low=a * outcome.low + b,
                    high=a * outcome.high + b, confidence=confidence,
                    engine="smc", n_samples=outcome.n_samples, elapsed=outcome.elapsed,
                    target_met=outcome.target_met, trace=trace + outcome.trace)


def certified(dpp, key, compute):
    """The result of `compute()`, read from or stored in the result cache under `key`.

    `key` is `result_cache_key`'s, None where caching would break the contract.  A hit
    returns a copy of the cached result with this call's `elapsed` and a trace line
    saying where it came from, so no two callers share one object.
    """
    started = time.perf_counter()
    entry = cache_entry(dpp)
    if key is None or key not in entry:
        result = compute()
        if key is not None:
            entry[key] = result
        return result
    cached = entry[key]
    trace = cached.trace + ("result cache: the result of an earlier call with the same "
                            "seed and contract",)
    elapsed = time.perf_counter() - started
    if isinstance(cached, Estimate):
        return cached.replace(elapsed=elapsed, trace=trace)
    return replace(cached, elapsed=elapsed, trace=trace)


def certified_scalar(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
                     floor: float | None, max_time: float | None, seed: int | None,
                     what: str, forbidden: np.ndarray | None = None,
                     affine: tuple[float, float] = (1.0, 0.0)) -> Estimate:
    """One scalar query run to certification on the sampling lane.

    `values` is the [0, 1] vector over output patterns whose weighted mean, or its
    quotient with the accepted mass for a conditioned DPP, is the certified quantity;
    `affine` maps the certified scale back to the caller's as ``a * x + b``, and
    relative error is met on the caller's scale; `forbidden` marks output patterns
    whose appearance makes the query undefined.
    """
    origin = origin_of(affine)

    def compute() -> Estimate:
        deadline = Deadline.after(max_time)
        program, report, run_sequence = piloted(
            dpp, values, rel_err=rel_err, confidence=confidence, deadline=deadline,
            seed=seed, origin=origin)
        outcome = run_lane(program, values, report, rel_err=rel_err,
                           confidence=confidence, floor=floor, deadline=deadline,
                           seed_sequence=run_sequence, forbidden=forbidden,
                           origin=origin)
        return smc_estimate(outcome, affine=affine, confidence=confidence,
                            trace=report.trace)

    key = result_cache_key(what, "smc", seed, max_time, rel_err, floor, confidence)
    return certified(dpp, key, compute)


def certified_table(dpp, *, rel_err: float, confidence: float, floor: float | None,
                    max_time: float | None, seed: int | None,
                    normalised_contract: bool, what: str) -> TableOutcome:
    """One table query run to certification on the sampling lane.

    A table wider than the output-width guard, or whose static log factor lies outside
    float64, raises `ValueError`.  The default `floor` is resolved here and recorded in
    the outcome, and every entry is certified jointly through `table_lane`.
    """
    out_bits = sum(t.nbits() for t in dpp.output_types())
    if out_bits > TABLE_OUT_BITS_CAP:
        raise ValueError(
            f"a sampled table certifies each of its 2**{out_bits} output values "
            f"separately, more than the sampling engine accepts "
            f"(2**{TABLE_OUT_BITS_CAP}); use the exact engine")
    n_entries = 1 << out_bits
    floor = resolved_floor(rel_err, floor, dpp.output_types())

    def compute() -> TableOutcome:
        # A table of masses folds the DPP's static log factor into its entries; a
        # factor outside float64 is refused before the pilot runs.  A normalised table
        # leaves the factor out.
        if not normalised_contract:
            scaled_matrix(np.ones(1), prepared(dpp)[0].log_scale)
        deadline = Deadline.after(max_time)
        program, report, run_sequence = piloted(
            dpp, np.ones(n_entries), rel_err=rel_err, confidence=confidence,
            deadline=deadline, seed=seed)
        return table_lane(program, report, rel_err=rel_err, confidence=confidence,
                          floor=floor, normalised_contract=normalised_contract,
                          deadline=deadline, seed_sequence=run_sequence)

    key = result_cache_key(what, "smc", seed, max_time, rel_err, floor, confidence)
    return certified(dpp, key, compute)


def table_lane(program: SweepProgram, report: PilotReport, *, rel_err: float,
               confidence: float, floor: float, normalised_contract: bool,
               deadline: Deadline, seed_sequence: np.random.SeedSequence,
               stop=None, prefix: tuple[str, ...] | None = None) -> TableOutcome:
    """The table certified by `pylea.sample.run.run_table` in the mode the pilot report
    implies, its trace prefixed with `prefix`, the report's trace by default.

    Sweeps run only in the survival case, where the i.i.d. first round finds no accepted
    mass at all but sweeps do.  The binding mass of the table contract is its smallest
    entry at or above `floor`, read from the first round's own empirical table.
    """
    binding = report.table_mass(floor)
    survival_only = binding == 0.0 and report.predicted_sweep < float("inf")
    outcome = run_table(program, rel_err=rel_err, confidence=confidence,
                        floor=floor, seed_sequence=seed_sequence,
                        mode="sweep" if survival_only else "iid",
                        batch=report.batch, n_particles=report.n_particles,
                        normalised_contract=normalised_contract, deadline=deadline,
                        seconds_per_particle=(report.sweep_seconds_per_particle
                                              if survival_only
                                              else report.iid_seconds_per_particle),
                        stop=stop)
    share = binding / report.evidence_mass if binding > 0.0 else 0.0
    found = (f"smallest share of a table entry at or above the floor of {floor:.3g} "
             f"in the first round: {share:.3g}" if binding > 0.0 else
             f"the first round found no table entry with a share at or above the "
             f"floor of {floor:.3g}")
    trace = ((report.trace if prefix is None else prefix)
             + ((f"sampling: {found}",) if report.drawn else ()) + outcome.trace)
    return replace(outcome, trace=trace)


def result_cache_key(what: str, method: str, seed: int | None, max_time: float | None,
                     rel_err: float, floor: float | None, confidence: float,
                     compile_kwargs: dict | None = None):
    """The result-cache key, or None where caching would break the contract.

    The key is the full contract: an explicit seed, no deadline, the method, every
    accuracy parameter, and the exact engine's `compile_kwargs`, which can decide the
    engine that answers.  Compile options that cannot be hashed give None.
    """
    if seed is None or max_time is not None:
        return None
    options = tuple(sorted((compile_kwargs or {}).items()))
    try:
        hash(options)
    except TypeError:
        return None
    return (what, method, seed, rel_err, floor, confidence, options)
