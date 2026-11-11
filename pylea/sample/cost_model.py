"""Constants and cost predictions of the sampling engine, in one place.

Every tunable of `pylea.sample` lives here, so a threshold is changed in exactly one
place and each carries the note saying where its value comes from.  The prediction
functions price a certified run in particles before it starts; they are order-of-magnitude
planning quantities for `pylea.sample.pilot` and `pylea.sample.portfolio`, never part of
any certificate.

The checkpointed certificates of `pylea.sample.certify` spend the failure budget
``delta / (j * (j + 1))`` on step `j`, so their log term grows with the stopping step.
Predictions therefore evaluate `log_factor` at the predicted stopping step and iterate
once; evaluating it at a small fixed step would understate the cost severalfold at
realistic step counts.
"""
from __future__ import annotations

import math

# The query contract's defaults (method and rel_err) live in `pylea.syntax.method`,
# which must stay importable without loading this package.

# Resampling triggers when the effective sample size falls below this fraction of N.
ESS_FRACTION = 0.5

# Sweep particle count: ``N = SWEEP_N_FACTOR * R_hat``, clipped to a small statistical
# minimum and the memory cap.  A larger N inflates the sweep certificate's range floor
# for no variance benefit.  Sweeps run batched, independent blocks side by side in one
# array totalling about `SWEEP_BATCH_TARGET` particles per accumulation step, so the
# statistically correct small N does not pay Python dispatch per sweep.
SWEEP_N_FACTOR = 4
SWEEP_N_MIN = 8
SWEEP_BATCH_TARGET = 8192

# Peak particle-state memory: live wire planes times particles, one byte per plane entry.
MEMORY_CAP_BYTES = 512 << 20

# i.i.d. batch size by unfolded op count.  The larger batch amortises Python dispatch;
# the smaller one keeps peak memory bounded on large DPPs.  The packed executor
# multiplies its batch by eight, its planes being an eighth the size.
BATCH_OPS_THRESHOLD = 3000
BATCH_SMALL_PROGRAM = 1 << 16
BATCH_LARGE_PROGRAM = 1 << 14

# Pilot sizes: two i.i.d. batches for triage; sixteen sweeps at a small N for sweep
# statistics, since a five-sweep variance estimate of a skewed distribution
# underestimates with high probability.
PILOT_IID_BATCHES = 2
PILOT_SWEEPS = 16
PILOT_SWEEP_PARTICLES = 512

# Feasibility guards, from counts alone.  Above the op cap one batch costs whole seconds
# on the byte lane; above the width cap a sampled table would certify over a million
# entries.
UNFOLDED_OPS_CAP = 10_000_000
TABLE_OUT_BITS_CAP = 20

# Per-particle speed of the packed executor over the byte lane.  Mode selection charges
# it to the sweep side, which cannot pack; a fixed constant rather than a per-query
# timing keeps the mode a seed-reproducible function of sample statistics.  From the
# packed gate benchmark on an Apple M3 Pro; re-measure after changing either executor.
PACKED_SPEEDUP = 10.0

# The sweep certifier's stopping constants: sweeps to a relative-error target near
# ``SWEEP_VARIANCE_CONSTANT * v * L / rel_err**2`` for per-sweep relative variance `v`,
# plus the range floor ``SWEEP_FLOOR_CONSTANT * L / (rel_err * mass)``, with
# ``L = ln(2 / delta)``.  Fitted against `pylea.sample.certify.BettingStream` on
# synthetic [0, 1] streams on an Apple M3 Pro, ranges 12-24 and 2.3-2.6; re-measure after
# changing the stream's stake or grid.
SWEEP_VARIANCE_CONSTANT = 20.0
SWEEP_FLOOR_CONSTANT = 2.5

# Thresholds of the portfolio algorithm.  Probe order flips at the traced size where
# greedy planning cost starts to dominate the pilot's; the fast path commits a lane
# predicted below `T_FAST_S` without running the second probe.  `E_LO` and `E_HI` bracket
# the exact predictor's error as predicted over actual time.  Each is set a factor beyond
# the extreme observed in a leave-one-family-out calibration over four instance families
# (the ARALIA fault trees, the BN queries, the grid family, and the wide-interface family;
# observed error ratios 0.015 to 1.2e7 on an Apple M3 Pro), since a value outside the bracket
# costs a hedge where a too-narrow bracket would cost a wrong commit.  Re-measure after
# changing the planner or `pylea.compile.cost`.
PROBE_ORDER_TRACED_OPS = 2000
T_FAST_S = 0.25
E_LO = 0.01
E_HI = 1e8

# The exact predictor runs under a wall budget of ``max(PREDICTOR_BUDGET_MIN_S,
# PREDICTOR_BUDGET_FACTOR * the pilot's prediction)``, itself capped at a
# `DEADLINE_HEADROOM` share of any remaining deadline so the probe can never spend the
# query's time budget; on budget-out the exact lane counts as unpredicted.
PREDICTOR_BUDGET_MIN_S = 1.0
PREDICTOR_BUDGET_FACTOR = 4.0

# Under `max_time`, committing to the exact lane requires its inflated prediction to fit
# in this fraction of the remaining deadline; otherwise the procedure hedges.  A
# committed exact lane may then spend `EXACT_COMMIT_SHARE` of the remaining deadline
# before the sampling lane takes over with the rest, so a prediction wrong by more than
# the headroom still leaves a certified answer.
DEADLINE_HEADROOM = 4.0
EXACT_COMMIT_SHARE = 0.5


def batch_size(unfolded_ops: int) -> int:
    """The byte-lane i.i.d. batch size for a DPP of `unfolded_ops` unfolded ops.

        >>> batch_size(100), batch_size(100_000)
        (65536, 16384)
    """
    return BATCH_SMALL_PROGRAM if unfolded_ops <= BATCH_OPS_THRESHOLD \
        else BATCH_LARGE_PROGRAM


def step_delta(delta: float, j: int) -> float:
    """The failure budget of accumulation step `j`, out of a total budget `delta`.

    The shares sum to at most `delta` over all steps, which is what makes the sequence of
    intervals anytime-valid.

        >>> round(sum(step_delta(0.05, j) for j in range(1, 10_000)), 6)
        0.049995
    """
    return delta / (j * (j + 1))


def log_factor(delta_j: float) -> float:
    """The log term ``ln(4 / delta_j)`` of one certification step.

    The 4 is one factor 2 for the bound's two sides and one from the empirical-Bernstein
    bound of Maurer and Pontil, whose constants `pylea.sample.certify` uses.
    """
    return math.log(4.0 / delta_j)


def iid_particles(mass: float, rel_err: float, delta: float, batch: int,
                  relvar: float | None = None) -> float:
    """Predicted particles for the i.i.d. mode to certify a quantity of mass `mass`.

    `mass` must be the smallest mass the query certifies; for a conditional query that is
    the numerator, not the acceptance probability.  Without `relvar` the samples are
    priced as Bernoulli draws under the Clopper-Pearson sequence; with it, as [0, 1]
    values of that measured per-particle relative variance under the empirical-Bernstein
    sequence, which is what soft weights need, a Bernoulli price overstating their cost
    by however far their variance sits below a coin flip's.  Infinite where `mass` is zero,
    which is how an all-zero pilot reports a lower bound on the cost rather than a
    failure.

        >>> 1e4 < iid_particles(0.25, 0.05, 0.05, 2**16) < 1e5
        True
        >>> iid_particles(0.25, 0.05, 0.05, 2**16, relvar=0.1) < \\
        ...     iid_particles(0.25, 0.05, 0.05, 2**16)
        True
        >>> iid_particles(0.0, 0.05, 0.05, 2**16)
        inf
    """
    if mass <= 0.0:
        return math.inf

    def particles(L: float) -> float:
        if relvar is None:
            return 3.0 * L / (rel_err**2 * mass)
        return 8.0 * relvar * L / rel_err**2 + 14.0 * L / (3.0 * rel_err * mass)

    first = particles(log_factor(delta))
    return particles(_iterated_log_factor(delta, first, batch))


def sweep_particles(relvar: float, mass: float, rel_err: float, delta: float,
                    n_particles: int) -> float:
    """Predicted particles for sweep mode, from the pilot's measured statistics.

    `relvar` is the measured per-sweep relative variance at `n_particles` particles per
    sweep, and `mass` the smallest mass the query certifies.  The prediction sums the
    betting certifier's variance term and range-floor term, times the particles each
    sweep costs; predicting the variance from stage counts instead would hide
    state-dependent survival, so the measured value is used.  The betting sequence needs
    no per-step budget, so its log factor does not grow with the stopping step.

        >>> sweep_particles(0.02, 0.03, 0.05, 0.05, 1024) < \\
        ...     sweep_particles(2.0, 0.03, 0.05, 0.05, 1024)
        True
    """
    if mass <= 0.0:
        return math.inf
    L = math.log(2.0 / delta)
    variance_term = SWEEP_VARIANCE_CONSTANT * relvar * L / rel_err**2
    floor_term = SWEEP_FLOOR_CONSTANT * L / (rel_err * mass)
    return (variance_term + floor_term) * n_particles


def _iterated_log_factor(delta: float, predicted_particles: float, per_step: int) -> float:
    """The log factor at the stopping step a first-pass prediction implies.

        >>> _iterated_log_factor(0.05, 1e305, 2**16) > _iterated_log_factor(0.05, 1e5, 2**16)
        True
        >>> _iterated_log_factor(0.05, math.inf, 2**16)
        inf
    """
    if math.isinf(predicted_particles):
        return math.inf
    j = max(1, math.ceil(predicted_particles / per_step))
    # ln(4 / step_delta(delta, j)) as a sum of logs: j * (j + 1) can pass the float range.
    return log_factor(delta) + math.log(j) + math.log(j + 1)
