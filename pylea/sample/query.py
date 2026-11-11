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
seed-free function of the DPP and is cached per DPP.  Pilot statistics are not
cached; they depend on the query's seed and on `rel_err`, and cost milliseconds.  Sampled
results are cached only under an explicit `seed`, keyed on the full contract, and never
when `max_time` was set, since a deadline makes the outcome timing-dependent.  Every
certified entry point, the portfolio algorithm's included, goes through `certified`, the
one place that reads and writes the result cache and reports a missed target.

A ``method="smc"`` query with an explicit `seed` and no `max_time` is bit-reproducible:
the pilot and the certified run draw from separate children of one
`numpy.random.SeedSequence`, and every data-dependent choice is a function of seeded
statistics and sample counts only.
"""
from __future__ import annotations

import weakref
from dataclasses import replace

import numpy as np

from pylea.ir.program import Program
from pylea.sample.collapse import collapse
from pylea.sample.cost_model import TABLE_OUT_BITS_CAP, UNFOLDED_OPS_CAP
from pylea.sample.estimate import notify_missed_target
from pylea.sample.pilot import PilotReport, run_pilot
from pylea.sample.run import (
    Deadline, RunOutcome, TableOutcome, run_iid, run_sweeps, run_table,
)
from pylea.sample.schedule import schedule
from pylea.sample.sweep import SweepProgram, unfold, unfolded_op_count
from pylea.semantics.likelihood_table import scaled_matrix
from pylea.syntax.estimate import Estimate

_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _entry(dpp) -> dict:
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
    entry = _entry(dpp)
    if "collapsed" not in entry:
        program = collapse(dpp.program())
        entry["collapsed"] = (program, unfolded_op_count(program))
    return entry["collapsed"]


def prepared(dpp) -> tuple[SweepProgram, int]:
    """The scheduled `SweepProgram` of `dpp` and its op count, cached per DPP.

    Raises `ValueError` where `dpp` has inputs, or where the collapsed DPP still
    unfolds beyond the op cap.
    """
    entry = _entry(dpp)
    if "prepared" not in entry:
        if dpp.input_types() != tuple():
            raise ValueError(
                f"the sampling lane needs a zero-input DPP; this one has input types "
                f"{dpp.input_types()}: evaluate exactly, or wrap the input you care "
                "about into a zero-input DPP")
        program, ops = collapsed(dpp)
        if ops > UNFOLDED_OPS_CAP:
            raise ValueError(
                f"the sampling lane would execute {ops} unfolded ops per particle, above "
                f"the feasibility cap of {UNFOLDED_OPS_CAP}; use the exact engine")
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


def piloted(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
            deadline: Deadline, seed: int | None
            ) -> tuple[SweepProgram, PilotReport, np.random.SeedSequence]:
    """The prepared `SweepProgram`, its pilot report for `values`, and the seed sequence the
    certified run draws from.

    The pilot and the run use separate children of one sequence seeded by `seed`, so a
    seeded query's run never depends on how many draws the pilot made.
    """
    pilot_sequence, run_sequence = np.random.SeedSequence(seed).spawn(2)
    program, ops = prepared(dpp)
    report = run_pilot(program, values, rel_err=rel_err, confidence=confidence,
                       unfolded_ops=ops, seed_sequence=pilot_sequence, deadline=deadline)
    return program, report, run_sequence


def run_lane(program: SweepProgram, values: np.ndarray, report: PilotReport, *,
             rel_err: float, confidence: float, floor: float | None,
             deadline: Deadline, seed_sequence: np.random.SeedSequence,
             forbidden: np.ndarray | None, stop=None) -> RunOutcome:
    """The pilot's chosen mode run to certification, deadline-priced by its timings."""
    if report.mode == "iid":
        return run_iid(program, values, rel_err=rel_err, confidence=confidence,
                       batch=report.batch, seed_sequence=seed_sequence, floor=floor,
                       deadline=deadline, forbidden=forbidden,
                       seconds_per_particle=report.iid_seconds_per_particle, stop=stop)
    return run_sweeps(program, values, rel_err=rel_err, confidence=confidence,
                      n_particles=report.n_particles, seed_sequence=seed_sequence,
                      floor=floor, deadline=deadline, forbidden=forbidden,
                      seconds_per_particle=report.sweep_seconds_per_particle, stop=stop)


def smc_estimate(outcome: RunOutcome, *, affine: tuple[float, float],
                 confidence: float, trace: tuple[str, ...]) -> Estimate:
    """The `Estimate` of a certified run, its interval mapped by ``a * x + b``."""
    a, b = affine
    low, high = sorted((a * outcome.low + b, a * outcome.high + b))
    return Estimate(a * outcome.point + b, low=low, high=high, confidence=confidence,
                    engine="smc", n_samples=outcome.n_samples, elapsed=outcome.elapsed,
                    target_met=outcome.target_met, trace=trace + outcome.trace)


def certified(dpp, key, compute, *, strict: bool, what: str):
    """The result of `compute()`, read from or stored in the result cache under `key`,
    with a missed target reported.

    `key` is `result_cache_key`'s, None where caching would break the contract.  The
    cached object itself is returned on a hit; a missed target warns, or raises under
    `strict`, on every return.
    """
    entry = _entry(dpp)
    if key is not None and key in entry:
        result = entry[key]
    else:
        result = compute()
        if key is not None:
            entry[key] = result
    notify_missed_target(result.target_met, strict, what)
    return result


def certified_scalar(dpp, values: np.ndarray, *, rel_err: float, confidence: float,
                     floor: float | None, max_time: float | None, seed: int | None,
                     strict: bool, what: str, forbidden: np.ndarray | None = None,
                     affine: tuple[float, float] = (1.0, 0.0)) -> Estimate:
    """One scalar query run to certification on the sampling lane.

    `values` is the [0, 1] vector over output patterns whose weighted mean, or its
    quotient with the accepted mass for a conditioned DPP, is the certified quantity;
    `affine` maps the certified scale back to the caller's as ``a * x + b``; `forbidden`
    marks output patterns whose appearance makes the query undefined.
    """
    def compute() -> Estimate:
        deadline = Deadline.after(max_time)
        program, report, run_sequence = piloted(
            dpp, values, rel_err=rel_err, confidence=confidence, deadline=deadline,
            seed=seed)
        outcome = run_lane(program, values, report, rel_err=rel_err,
                           confidence=confidence, floor=floor, deadline=deadline,
                           seed_sequence=run_sequence, forbidden=forbidden)
        return smc_estimate(outcome, affine=affine, confidence=confidence,
                            trace=report.trace)

    key = result_cache_key(what, "smc", seed, max_time, rel_err, floor, confidence)
    return certified(dpp, key, compute, strict=strict, what=what)


def certified_table(dpp, *, rel_err: float, confidence: float, floor: float | None,
                    max_time: float | None, seed: int | None, strict: bool,
                    normalised_contract: bool, what: str) -> TableOutcome:
    """One table query run to certification on the sampling lane.

    Applies the output-width guard and refuses a static log factor outside float64,
    resolves the default `floor`, which the outcome records, picks the mode from the pilot (sweeps only where the i.i.d. pilot finds no
    accepted mass at all but sweeps do, which is the survival case), and certifies every
    entry jointly.
    """
    out_bits = sum(t.nbits() for t in dpp.output_types())
    if out_bits > TABLE_OUT_BITS_CAP:
        raise ValueError(
            f"a sampled table certifies every output pattern, and {2**out_bits} entries "
            f"(out_bits {out_bits} > {TABLE_OUT_BITS_CAP}) is beyond the width guard; "
            "use the exact engine")
    n_entries = 1 << out_bits
    floor = resolved_floor(rel_err, floor, dpp.output_types())

    def compute() -> TableOutcome:
        # A table folds the DPP's static log factor into its entries; a factor
        # outside float64 is refused before the pilot runs.
        scaled_matrix(np.ones(1), prepared(dpp)[0].log_scale)
        deadline = Deadline.after(max_time)
        program, report, run_sequence = piloted(
            dpp, np.ones(n_entries), rel_err=rel_err, confidence=confidence,
            deadline=deadline, seed=seed)
        # The binding mass of the table contract is its smallest above-floor entry,
        # read from the pilot's own empirical table; sweeps run only in the survival
        # case, where the i.i.d. pilot finds no accepted mass at all but sweeps do.
        binding = report.table_mass(floor)
        survival_only = binding == 0.0 and report.predicted_sweep < float("inf")
        outcome = run_table(program, rel_err=rel_err, confidence=confidence,
                            floor=floor, seed_sequence=run_sequence,
                            mode="sweep" if survival_only else "iid",
                            batch=report.batch, n_particles=report.n_particles,
                            normalised_contract=normalised_contract, deadline=deadline,
                            seconds_per_particle=(report.sweep_seconds_per_particle
                                                  if survival_only
                                                  else report.iid_seconds_per_particle))
        trace = (report.trace + (f"pilot: binding table mass {binding:.3g}",)
                 + outcome.trace)
        return replace(outcome, trace=trace)

    key = result_cache_key(what, "smc", seed, max_time, rel_err, floor, confidence)
    return certified(dpp, key, compute, strict=strict, what=what)


def result_cache_key(what: str, method: str, seed: int | None, max_time: float | None,
                     rel_err: float, floor: float | None, confidence: float):
    """The result-cache key, or None where caching would break the contract.

    The key is the full contract: an explicit seed, no deadline, the method, and every
    accuracy parameter.
    """
    if seed is None or max_time is not None:
        return None
    return (what, method, seed, rel_err, floor, confidence)
