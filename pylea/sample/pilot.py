"""The pilot: seeded statistics that pick a mode and price the sampling lane.

`run_pilot` runs a few i.i.d. batches and, where the DPP conditions, a few small
sweeps, and reduces them to a `PilotReport`: mass estimates, stage statistics, the chosen
execution mode with its particle counts, and the predicted certified cost of each mode.

Every choice here, the mode and the sweep size, is a function of seeded sample statistics
and fixed constants, never of wall clock, which is what keeps a seeded `method="smc"`
query bit-reproducible.  The pilot also measures per-particle wall time, but that feeds
only the two inherently timing-dependent consumers, the cross-engine lane decision and
`max_time`, through `PilotReport.predicted_seconds`.

Mode selection compares the two modes' predicted particle counts from
`pylea.sample.cost_model`, a packable DPP's i.i.d. count divided by the executor gap
since sweeps cannot pack.  Both modes are priced for the quantity the query certifies:
for a conditioned query that is the quotient of the numerator and the acceptance
probability, so both streams are priced, each from its own mass and relative variance,
and combined as `pylea.sample.cost_model.iid_particles` states.  A soft-weight DPP's
i.i.d. count uses its measured per-particle variances, since a Bernoulli price would
exceed the cost of a concentrated weight.  The sweep particle count follows
``4 * R_hat`` from the recorded stage fractions, clipped to the statistical minimum and
the memory cap, while the sweep cost prediction uses the pilot's own measured per-sweep
variances rather than any stage arithmetic, since state-dependent survival hides from
the stage counts.  A resampled population's stage fractions are a biased cost
statistic, acceptable for choosing the sweep size and predicting cost; they never feed a
certificate.

A `pylea.sample.run.Deadline` bounds the pilot itself.  A deadline already expired on
entry draws nothing: the report's `drawn` is 0, its predictions are infinite, and the
certified run then reports the vacuous interval.  Otherwise, under a finite deadline the
first i.i.d. batch is a calibration batch of 64 samples, whose measured time per sample
sizes the later batches to the remaining time, as the certified run sizes its steps; the
sweep section runs only where one sweep of `PILOT_SWEEP_PARTICLES` particles, priced
from that time, fits in what remains, and skipped sweeps leave the sweep prediction
infinite.  Under a deadline the pilot's statistics therefore depend on timing, which is
consistent with the reproducibility contract: bit-reproducibility is promised only
without `max_time`, and without a deadline the pilot runs its full batches.

An all-zero pilot is a decision input, not a failure: the predictions come out infinite,
which lower-bounds the certified cost, and the portfolio algorithm routes such a query to
the exact engine.

Pilot samples never enter a certified stream.  The caller passes a `SeedSequence` used
only here; the certified run starts from a fresh spawned child after every pilot-driven
choice is frozen.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pylea.sample.run import Deadline

from pylea.sample import cost_model
from pylea.sample.cost_model import (
    MEMORY_CAP_BYTES, PACKED_SPEEDUP, PILOT_IID_BATCHES, PILOT_SWEEP_PARTICLES,
    PILOT_SWEEPS, SWEEP_N_FACTOR, SWEEP_N_MIN, batch_size,
)
from pylea.sample.sweep import SweepProgram


@dataclass(frozen=True)
class PilotReport:
    """What the pilot measured and decided.

    `mass` estimates the queried quantity's mass, `evidence_mass` the acceptance
    probability, and `entry_masses` the empirical mass per output pattern;
    `relvar_numerator` and `relvar_denominator` are the two streams' measured per-sweep
    relative variances at the pilot's sweep size, infinite where no sweeps ran; for an
    unconditioned query the denominator's is zero.  `mode` is ``"iid"`` or ``"sweep"``,
    with `batch` the i.i.d. batch size and `n_particles` the sweep size; `predicted_iid`
    and `predicted_sweep` are the two modes' predicted certified particle counts,
    infinite where the pilot saw no hits; `drawn` is the number of i.i.d. samples the
    pilot drew, 0 where its deadline had expired before it started; `stream_rel_err` is
    the relative error a sweep-mode stream must resolve, finer than the query's
    `rel_err` where relative error is measured from an origin other than 0;
    `weight_relvar` is the measured relative variance of an i.i.d. sample's weight,
    `None` for 0/1 weights.
    """

    mass: float
    evidence_mass: float
    relvar_numerator: float
    relvar_denominator: float
    r_hat: float
    mode: str
    batch: int
    n_particles: int
    predicted_iid: float
    predicted_sweep: float
    entry_masses: np.ndarray
    iid_seconds_per_particle: float
    sweep_seconds_per_particle: float
    trace: tuple[str, ...]
    drawn: int
    stream_rel_err: float
    weight_relvar: float | None

    @property
    def predicted_particles(self) -> float:
        """The chosen mode's predicted certified particle count."""
        return self.predicted_iid if self.mode == "iid" else self.predicted_sweep

    @property
    def predicted_seconds(self) -> float:
        """The chosen mode's predicted certified wall clock.

        Wall time enters the design in exactly two places, the cross-engine lane
        decision and `max_time`; this is the first.  Mode and sweep size never read it.
        """
        per_particle = (self.iid_seconds_per_particle if self.mode == "iid"
                        else self.sweep_seconds_per_particle)
        return self.predicted_particles * per_particle

    def table_mass(self, floor: float) -> float:
        """The smallest entry mass the two-parameter table contract must resolve.

        The empirical pilot table stands in for the true one: entries whose normalised
        share falls below `floor` may be certified small instead, so the binding mass is
        the smallest entry at or above the floor.  Zero where the pilot saw no accepted
        mass at all.
        """
        if self.evidence_mass <= 0.0:
            return 0.0
        above = self.entry_masses[self.entry_masses / self.evidence_mass >= floor]
        return float(above.min()) if above.size else 0.0

    def table_seconds(self, floor: float, rel_err: float, confidence: float,
                      normalised: bool) -> float:
        """The predicted wall clock of `pylea.sample.run.run_table` without resampling,
        infinite where the first round found no entry at or above `floor`.

        The binding entry, `table_mass`, is priced as a stream of its own: Bernoulli for
        0/1 weights, otherwise with the relative variance of a weight times the entry's
        indicator, ``(1 + weight_relvar) / share - 1`` for the entry's `share` of the
        total, taking the weight independent of the entry.  A table that only sampling
        with resampling reaches is not priced here.
        """
        binding = self.table_mass(floor)
        if binding <= 0.0:
            return math.inf
        share = binding / self.evidence_mass
        relvar = (None if self.weight_relvar is None
                  else (1.0 + self.weight_relvar) / share - 1.0)
        particles = cost_model.table_particles(
            (binding, relvar), (self.evidence_mass, self.weight_relvar), rel_err,
            1.0 - confidence, len(self.entry_masses), self.batch, normalised)
        return particles * self.iid_seconds_per_particle


def run_pilot(program: SweepProgram, values: np.ndarray, *, rel_err: float,
              confidence: float, unfolded_ops: int,
              seed_sequence: np.random.SeedSequence,
              deadline: "Deadline | None" = None, origin: float = 0.0) -> PilotReport:
    """The pilot statistics and mode choice for one query.

    `values` is the query's [0, 1] vector over output patterns, as in
    `pylea.sample.run`; `unfolded_ops` sizes the i.i.d. batch; `deadline` bounds the
    pilot as the module docstring states.  Relative error is measured from `origin`, as
    `pylea.sample.certify.met_relative` states, so the predictions price the relative
    error `_resolution` derives from the pilot's estimate of the quantity.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.sample.sweep import SweepProgram, unfold
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.25)
        >>> report = run_pilot(SweepProgram(unfold(coin.program())),
        ...                    np.array([0.0, 1.0]), rel_err=0.05, confidence=0.95,
        ...                    unfolded_ops=1, seed_sequence=np.random.SeedSequence(0))
        >>> report.mode, abs(report.mass - 0.25) < 0.02
        ('iid', True)
    """
    from pylea.sample.run import Deadline, packed_or_byte

    deadline = deadline or Deadline.never()
    values = np.asarray(values, dtype=float)
    delta = 1.0 - confidence
    batch = batch_size(unfolded_ops)
    trace: list[str] = []
    if deadline.expired():
        return PilotReport(
            0.0, 0.0, math.inf, math.inf, 0.0, "iid", batch, SWEEP_N_MIN, math.inf,
            math.inf, np.zeros(1 << program.out_bits), math.inf, math.inf,
            ("sampling: the deadline passed before the first round; no samples drawn",),
            0, rel_err, None)

    # The i.i.d. batches run on the executor the certified run would use, so the
    # measured per-particle time prices the right lane; the draws are identical either
    # way, the executors sharing one byte stream.
    iid_executor = packed_or_byte(program)
    packable = iid_executor is not program
    hits = 0.0
    hits_sq = 0.0
    accepted = 0.0
    accepted_sq = 0.0
    drawn = 0
    counts = np.zeros(1 << program.out_bits)
    bounded = deadline.expires_at is not None
    started = time.perf_counter()
    # Under a deadline a calibration batch of 64 samples, the packed kernel's word
    # width, comes first, and its time per sample sizes the later batches, each run
    # only where it fits in whole words.
    for k in range(PILOT_IID_BATCHES + bounded):
        size = batch
        if bounded:
            took = time.perf_counter() - started
            fit = 64 if k == 0 else (batch if took <= 0.0 else
                                     int(deadline.remaining() * drawn / took))
            size = min(batch, fit - fit % 64)
        if size == 0:
            break
        rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        result = iid_executor.run(size, rng, resample=False, squares=True)
        counts += result.counts
        hits += float(values @ result.counts)
        assert result.counts_sq is not None
        hits_sq += float((values * values) @ result.counts_sq)
        accepted += result.total
        assert result.total_sq is not None
        accepted_sq += result.total_sq
        drawn += size
    iid_seconds = (time.perf_counter() - started) / drawn
    mass = hits / drawn
    evidence_mass = accepted / drawn
    # The streams the run certifies, as ``(mass, relvar)`` pairs; `None` prices a
    # Bernoulli stream, as `pylea.sample.run.run_iid` certifies 0/1 values.
    binary_values = bool(np.all((values == 0.0) | (values == 1.0)))
    iid_streams = [(mass, None if program.binary_weights and binary_values
                    else _relvar(hits_sq / drawn, mass))]
    if program.conditions:
        iid_streams.append((evidence_mass, None if program.binary_weights
                            else _relvar(accepted_sq / drawn, evidence_mass)))
    iid_resolution = _resolution(rel_err, origin, mass / evidence_mass
                                 if evidence_mass > 0.0 else 0.0)
    predicted_iid = (cost_model.iid_particles(iid_streams, iid_resolution, delta, batch)
                     if iid_resolution > 0.0 else math.inf)
    trace.append(f"sampling: first round of {drawn} samples without resampling, mean "
                 f"{mass:.3e}, acceptance probability {evidence_mass:.3e}")

    relvar_numerator = math.inf
    relvar_denominator = 0.0 if not program.conditions else math.inf
    r_hat = 0.0
    n_particles = SWEEP_N_MIN
    predicted_sweep = math.inf
    sweep_mass = 0.0
    sweep_seconds = iid_seconds
    stream_resolution = min(rel_err, iid_resolution) if iid_resolution > 0.0 else rel_err
    # One sweep costs about as many byte-lane samples as it has particles.
    sweep_estimate = PILOT_SWEEP_PARTICLES * iid_seconds * (PACKED_SPEEDUP if packable
                                                            else 1.0)
    if program.conditions and sweep_estimate <= deadline.remaining():
        den_values: list[float] = []
        num_values: list[float] = []
        fraction_sum = np.zeros(program.n_stages)
        started = time.perf_counter()
        for k in range(PILOT_SWEEPS):
            if k and deadline.expired():
                break
            rng = np.random.default_rng(seed_sequence.spawn(1)[0])
            result = program.run(PILOT_SWEEP_PARTICLES, rng, resample=True,
                                 record_stages=True)
            scale = math.exp(result.log_z) / PILOT_SWEEP_PARTICLES
            den_values.append(result.total * scale)
            num_values.append(float(values @ result.counts) * scale)
            assert result.stage_fractions is not None
            fraction_sum += result.stage_fractions
        sweep_seconds = (time.perf_counter() - started) / (len(den_values)
                                                        * PILOT_SWEEP_PARTICLES)
        den = np.array(den_values)
        num = np.array(num_values)
        mean = float(den.mean())
        sweep_mass = float(num.mean())
        if len(den_values) >= 2 and mean > 0.0 and sweep_mass > 0.0:
            # Divided by the means before squaring: a sweep's value carries the
            # normalising constant of the whole DPP, which can be far below the square
            # root of the smallest float.
            relvar_numerator = float((num / sweep_mass).var(ddof=1))
            relvar_denominator = float((den / mean).var(ddof=1))
            fractions = fraction_sum / len(den_values)
            r_hat = float(np.sum((1.0 - fractions) / np.maximum(fractions, 1e-12)))
            memory_cap = max(SWEEP_N_MIN, MEMORY_CAP_BYTES // max(program.peak_planes, 1))
            n_particles = int(np.clip(SWEEP_N_FACTOR * r_hat, SWEEP_N_MIN, memory_cap))
            # The measured per-sweep variance scales like 1/n, so the pilot's values at
            # its own n are rescaled to the production sweep size.
            rescale = PILOT_SWEEP_PARTICLES / n_particles
            sweep_resolution = _resolution(rel_err, origin, sweep_mass / mean)
            if sweep_resolution > 0.0:
                stream_resolution = min(rel_err, sweep_resolution)
                predicted_sweep = cost_model.sweep_particles(
                    [(sweep_mass, relvar_numerator * rescale),
                     (mean, relvar_denominator * rescale)], sweep_resolution, delta,
                    n_particles)
        trace.append(f"sampling: first round of {len(den_values)} runs of "
                     f"{PILOT_SWEEP_PARTICLES} particles with resampling, relative "
                     f"variance {relvar_numerator:.3g} for the queried quantity and "
                     f"{relvar_denominator:.3g} for the acceptance probability, "
                     f"{n_particles} particles per run chosen")

    # Sweeps always run the byte lane, so on a packable DPP the i.i.d. particle
    # count is charged the executor gap before the two are compared; the constant keeps
    # the mode a seed-reproducible function of sample statistics.
    effective_iid = predicted_iid / (PACKED_SPEEDUP if packable else 1.0)
    mode = "iid" if effective_iid <= predicted_sweep else "sweep"
    cheaper = (f" (a sample without resampling costing 1/{PACKED_SPEEDUP:g} of one with "
               f"resampling)" if packable else "")
    trace.append(f"sampling: predicted samples without resampling {predicted_iid:.3g}, "
                 f"with resampling {predicted_sweep:.3g}{cheaper}; "
                 f"{'without' if mode == 'iid' else 'with'} resampling chosen")
    return PilotReport(mass, evidence_mass, relvar_numerator, relvar_denominator, r_hat,
                       mode, batch, n_particles,
                       predicted_iid, predicted_sweep, counts / drawn,
                       iid_seconds, sweep_seconds, tuple(trace), drawn,
                       stream_resolution,
                       None if program.binary_weights
                       else _relvar(accepted_sq / drawn, evidence_mass))


def _resolution(rel_err: float, origin: float, estimate: float) -> float:
    """The relative error, measured from 0, that meets `rel_err` measured from `origin`
    at a quantity near `estimate`; 0 where the estimate is 0 or at the origin.

    An interval of width ``w`` around ``q`` meets `rel_err` from `origin` when
    ``w <= rel_err * |q - origin|``, which is relative error ``rel_err * |q - origin| / q``
    measured from 0.

        >>> _resolution(0.01, 0.0, 0.3), round(_resolution(0.01, 0.5, 0.52), 7)
        (0.01, 0.0003846)
    """
    if origin == 0.0:
        return rel_err
    if estimate <= 0.0:
        return 0.0
    return rel_err * abs(estimate - origin) / estimate


def _relvar(second_moment: float, mean: float) -> float | None:
    """The relative variance ``second_moment / mean**2 - 1``, at least 0, or `None`
    where `mean` is zero, where the stream prices as infinite either way.

    Divided by the mean before squaring, so that a small mean cannot underflow.
    """
    return max(second_moment / mean / mean - 1.0, 0.0) if mean > 0.0 else None
