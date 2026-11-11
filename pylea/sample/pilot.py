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
since sweeps cannot pack.  Both modes are priced on the smallest mass the query
certifies, which for a conditional query is the numerator rather than the acceptance
probability, and a soft-weight DPP's i.i.d. count on its measured per-particle
variance, since a Bernoulli price overstates a concentrated weight's cost severalfold.
The sweep particle count follows ``4 * R_hat`` from the recorded stage fractions,
clipped to the statistical minimum and the memory cap, while the sweep cost prediction
uses the pilot's own measured per-sweep variance rather than any stage arithmetic, since
state-dependent survival hides from the stage counts.  A conditioned query certifies
both the numerator and the denominator, so the variance priced is the larger of the two
streams'; the denominator's alone understates a rare event at a high acceptance
probability by the ratio of the two.  A resampled population's stage fractions are a
biased cost statistic, acceptable for choosing the sweep size and predicting cost; they
never feed a certificate.

A `pylea.sample.run.Deadline` bounds the pilot itself: the first i.i.d. batch always
runs, later batches and the sweep section are skipped once the deadline has expired, and
skipped sweeps leave the sweep prediction infinite.  Under a deadline the pilot's
statistics therefore depend on timing, which is consistent with the reproducibility
contract: bit-reproducibility is promised only without `max_time`.

An all-zero pilot is a decision input, not a failure: the predictions come out infinite,
which lower-bounds the certified cost, and `zero_hits` says so; the portfolio algorithm
routes such a query to the exact engine.

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

    `mass` estimates the smallest mass the query certifies, `evidence_mass` the
    acceptance probability, and `entry_masses` the empirical mass per output pattern;
    `relvar` is the larger of the numerator's and the denominator's measured per-sweep
    relative variance, the one that binds the certified quotient; `mode` is ``"iid"`` or
    ``"sweep"``, with `batch` the i.i.d. batch size and `n_particles` the sweep size;
    `predicted_iid` and `predicted_sweep` are the two modes' predicted certified
    particle counts, infinite where the pilot saw no hits.
    """

    mass: float
    evidence_mass: float
    relvar: float
    r_hat: float
    mode: str
    batch: int
    n_particles: int
    predicted_iid: float
    predicted_sweep: float
    zero_hits: bool
    entry_masses: np.ndarray
    iid_seconds_per_particle: float
    sweep_seconds_per_particle: float
    trace: tuple[str, ...]

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


def run_pilot(program: SweepProgram, values: np.ndarray, *, rel_err: float,
              confidence: float, unfolded_ops: int,
              seed_sequence: np.random.SeedSequence,
              deadline: "Deadline | None" = None) -> PilotReport:
    """The pilot statistics and mode choice for one query.

    `values` is the query's [0, 1] vector over output patterns, as in
    `pylea.sample.run`; `unfolded_ops` sizes the i.i.d. batch; `deadline` bounds the
    pilot as the module docstring states.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.sample.sweep import SweepProgram, unfold
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.25)
        >>> report = run_pilot(SweepProgram(unfold(coin.program())),
        ...                    np.array([0.0, 1.0]), rel_err=0.05, confidence=0.95,
        ...                    unfolded_ops=1, seed_sequence=np.random.SeedSequence(0))
        >>> report.mode, report.zero_hits, abs(report.mass - 0.25) < 0.02
        ('iid', False, True)
    """
    from pylea.sample.run import Deadline, packed_or_byte

    deadline = deadline or Deadline.never()
    values = np.asarray(values, dtype=float)
    delta = 1.0 - confidence
    batch = batch_size(unfolded_ops)
    trace: list[str] = []

    # The i.i.d. batches run on the executor the certified run would use, so the
    # measured per-particle time prices the right lane; the draws are identical either
    # way, the executors sharing one byte stream.
    iid_executor = packed_or_byte(program)
    packable = iid_executor is not program
    hits = 0.0
    hits_sq = 0.0
    accepted = 0.0
    drawn = 0
    counts = np.zeros(1 << program.out_bits)
    started = time.monotonic()
    for k in range(PILOT_IID_BATCHES):
        if k and deadline.expired():
            break
        rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        result = iid_executor.run(batch, rng, resample=False, squares=True)
        counts += result.counts
        hits += float(values @ result.counts)
        assert result.counts_sq is not None
        hits_sq += float((values * values) @ result.counts_sq)
        accepted += result.total
        drawn += batch
    iid_seconds = (time.monotonic() - started) / drawn
    mass = hits / drawn
    evidence_mass = accepted / drawn
    if program.binary_weights:
        particle_relvar = None
    else:
        variance = max(hits_sq / drawn - mass * mass, 0.0)
        particle_relvar = variance / mass**2 if mass > 0.0 else None
    predicted_iid = cost_model.iid_particles(mass, rel_err, delta, batch,
                                             relvar=particle_relvar)
    trace.append(f"pilot: {drawn} i.i.d. particles, mass {mass:.3e}, "
                 f"evidence {evidence_mass:.3e}")

    relvar = math.inf
    r_hat = 0.0
    n_particles = SWEEP_N_MIN
    predicted_sweep = math.inf
    sweep_mass = 0.0
    sweep_seconds = iid_seconds
    if program.conditions and not deadline.expired():
        den_values: list[float] = []
        num_values: list[float] = []
        fraction_sum = np.zeros(program.n_stages)
        started = time.monotonic()
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
        sweep_seconds = (time.monotonic() - started) / (len(den_values)
                                                        * PILOT_SWEEP_PARTICLES)
        den = np.array(den_values)
        num = np.array(num_values)
        mean = float(den.mean())
        sweep_mass = float(num.mean())
        if len(den_values) >= 2 and mean > 0.0 and sweep_mass > 0.0:
            # The certified quantity is the numerator, and for a conditioned query its
            # quotient with the denominator, so the binding variance is the larger of
            # the two streams'; the denominator's alone understates a rare event at
            # a high acceptance probability.
            relvar = max(float(den.var(ddof=1)) / mean**2,
                         float(num.var(ddof=1)) / sweep_mass**2)
            fractions = fraction_sum / len(den_values)
            r_hat = float(np.sum((1.0 - fractions) / np.maximum(fractions, 1e-12)))
            memory_cap = max(SWEEP_N_MIN, MEMORY_CAP_BYTES // max(program.peak_planes, 1))
            n_particles = int(np.clip(SWEEP_N_FACTOR * r_hat, SWEEP_N_MIN, memory_cap))
            # The measured per-sweep variance scales like 1/n, so the pilot's value at
            # its own n is rescaled to the production sweep size.
            relvar_at_n = relvar * PILOT_SWEEP_PARTICLES / n_particles
            predicted_sweep = cost_model.sweep_particles(
                relvar_at_n, sweep_mass, rel_err, delta, n_particles)
        trace.append(f"pilot: {len(den_values)} sweeps at {PILOT_SWEEP_PARTICLES}, "
                     f"relvar {relvar:.3g}, R^ {r_hat:.3g}, N {n_particles}")

    # Sweeps always run the byte lane, so on a packable DPP the i.i.d. particle
    # count is charged the executor gap before the two are compared; the constant keeps
    # the mode a seed-reproducible function of sample statistics.
    effective_iid = predicted_iid / (PACKED_SPEEDUP if packable else 1.0)
    mode = "iid" if effective_iid <= predicted_sweep else "sweep"
    zero_hits = mass == 0.0 and sweep_mass == 0.0
    trace.append(f"pilot: predicted particles i.i.d. {predicted_iid:.3g}, "
                 f"sweep {predicted_sweep:.3g} -> {mode}")
    return PilotReport(mass, evidence_mass, relvar, r_hat, mode, batch, n_particles,
                       predicted_iid, predicted_sweep, zero_hits, counts / drawn,
                       iid_seconds, sweep_seconds, tuple(trace))
