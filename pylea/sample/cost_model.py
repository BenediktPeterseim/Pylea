"""Constants and cost predictions of the sampling engine, in one place.

Every tuned constant of `pylea.sample` lives here, so a threshold is changed in exactly
one place and each carries the note saying where its value comes from.  Numerical limits
that are not tuned, such as the smallest weight `pylea.sample.sweep` keeps unrenormalised
or the lower end of the betting grid, stay in the module that uses them.  The prediction
functions price a certified run in particles before it starts; they are order-of-magnitude
planning quantities for `pylea.sample.pilot` and `pylea.sample.portfolio`, never part of
any certificate.

The checkpointed certificates of `pylea.sample.certify` spend the failure budget
``delta / (j * (j + 1))`` on step `j`, so their log term grows with the stopping step.
Predictions therefore evaluate `log_factor` at the predicted stopping step and iterate
once.

[1] A. Maurer and M. Pontil.  Empirical Bernstein bounds and sample variance
    penalization.  Conference on Learning Theory (COLT), 2009.
    https://arxiv.org/abs/0907.3740
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# The query contract's defaults (method and rel_err) live in `pylea.syntax.method`,
# which must stay importable without loading this package.

# Resampling triggers when the effective sample size falls below this fraction of N.  A
# standard choice, not fitted.
ESS_FRACTION = 0.5

# Sweep particle count: ``N = SWEEP_N_FACTOR * R_hat``, clipped to `SWEEP_N_MIN` and the
# memory cap.  Chosen, not fitted; a larger N inflates the sweep certificate's range floor
# for no variance benefit.  Sweeps run batched, independent blocks side by side in one
# array totalling about `SWEEP_BATCH_TARGET` particles per accumulation step, so that a
# small N does not pay Python dispatch per sweep.  `SWEEP_BATCH_TARGET` is set by timing
# the byte lane on an Apple M3 Pro, not reproducible from this repository; re-measure
# after changing `pylea.sample.sweep`.
SWEEP_N_FACTOR = 4
SWEEP_N_MIN = 8
SWEEP_BATCH_TARGET = 8192

# Peak particle-state memory: live wire planes times particles, one byte per plane entry.
# A policy bound, not fitted.
MEMORY_CAP_BYTES = 512 << 20

# i.i.d. batch size by unfolded op count.  The larger batch amortises Python dispatch per
# op; the smaller one keeps peak memory, one byte per particle per live plane, bounded on
# large DPPs.  Set by timing the byte lane on an Apple M3 Pro; not reproducible from this
# repository.  Re-measure after changing `pylea.sample.sweep`.  The packed executor
# multiplies its batch by eight, its planes being an eighth the size.
BATCH_OPS_THRESHOLD = 3000
BATCH_SMALL_PROGRAM = 1 << 16
BATCH_LARGE_PROGRAM = 1 << 14

# Pilot sizes: two i.i.d. batches for triage; sixteen sweeps at a small N for sweep
# statistics, since a variance estimate from a few sweeps of a skewed distribution
# underestimates with high probability.  Chosen, not fitted.
PILOT_IID_BATCHES = 2
PILOT_SWEEPS = 16
PILOT_SWEEP_PARTICLES = 512

# Under a deadline, the share of the time left when the pilot starts that its batches
# after the first may fill, so that the certified run, whose samples alone enter the
# interval, has the rest.  Chosen, not fitted.
PILOT_DEADLINE_SHARE = 0.5

# Feasibility guards, from counts alone.  Policy bounds, not fitted.  Above the op cap one
# batch on the byte lane costs seconds; above the width cap a sampled table would certify
# over a million entries.  The width cap equals `pylea.syntax.method.MAX_OUTPUT_BITS`, the
# exact engine's bound on a queried output, and the bound `COLLAPSE_IN_BITS` and
# `COLLAPSE_OUT_BITS` together put on a spliced table, so that both engines accept the
# same tables; it is repeated here rather than imported, which keeps the syntax layer out
# of this module.
UNFOLDED_OPS_CAP = 10_000_000
TABLE_OUT_BITS_CAP = 20

# Per-particle speed of the packed executor over the byte lane.  Mode selection charges
# it to the sweep side, which cannot pack; a fixed constant rather than a per-query
# timing keeps the mode a seed-reproducible function of sample statistics.  Timed on DPPs
# of Boolean gates and coin flips on an Apple M3 Pro; not reproducible from this
# repository.  Re-measure after changing either executor.
PACKED_SPEEDUP = 10.0

# The sweep certifier's stopping constants: sweeps to a relative-error target near
# ``SWEEP_VARIANCE_CONSTANT * v * L / rel_err**2`` for per-sweep relative variance `v`,
# plus the range floor ``SWEEP_FLOOR_CONSTANT * L / (rel_err * mass)``, with
# ``L = ln(2 / delta)``.  Fitted against `pylea.sample.certify.BettingStream` on
# synthetic [0, 1] streams fed as `pylea.sample.run.run_sweeps` feeds them, 1024 values
# per batch with the interval read after each, over means 0.3 and 0.05, relative variances
# 0.01 to 1 and targets 0.05 and 0.02: the samples needed lay between 0.56 and 1.89
# times the prediction.  Re-measure after changing the stream's stake or grid.
SWEEP_VARIANCE_CONSTANT = 20.0
SWEEP_FLOOR_CONSTANT = 2.5

# The grid of `pylea.sample.certify.BettingStream`.  Its cells are refined to the
# relative width ``BETTING_CELL_SHARE * rel_err``, so the two ends of an interval lie
# at most two cells, 4% of the target, outside the confidence set.  The stream stores
# at most `REPLAY_SAMPLES_MAX` samples for replay, 16 MB with their stakes; it then lays
# at most `SEAL_CANDIDATES_MAX` candidates over its interval at that cell width and
# drops the history.  The two bounds fix how far a run reaches: the interval at that
# point is resolved whenever it is at most ``SEAL_CANDIDATES_MAX * BETTING_CELL_SHARE``,
# that is 82, times the target wide, and the width falls with the square root of the
# sample count, so every target is resolved that needs fewer than about
# ``82**2 * 2**20``, that is 7e9, samples.  Measured with ``2**20`` stored samples: 646
# candidates laid in 7 s at relative variance 100, and the cap of 4096 laid in 47 s at
# relative variance 1000.
BETTING_CELL_SHARE = 1.0 / 50.0
REPLAY_SAMPLES_MAX = 1 << 20
SEAL_CANDIDATES_MAX = 4096

# The starting grid of `pylea.sample.certify.BettingStream` has `BETTING_GRID_PER_DECADE`
# candidates per decade, cells 0.68% wide.  A refinement splits a cell into
# `BETTING_SPLIT` cells, and one reading of the interval refines at most
# `BETTING_MAX_PASSES` times, which bounds the replay of stored samples one reading can
# cause.  Resolution choices, not fitted.
BETTING_GRID_PER_DECADE = 341
BETTING_SPLIT = 8
BETTING_MAX_PASSES = 64

# Sampling with resampling reads its interval, which replays the stored samples for every
# candidate a reading adds, only once the sample count has grown by this share since the
# last reading, and always at the deadline.  The intervals do not depend on how often they
# are read, so the share bounds how far a run overshoots the step at which its target is
# met, and the number of readings grows with the logarithm of the sample count.  Chosen,
# not fitted.
BETTING_READ_GROWTH = 1.0 / 8.0

# Largest sweep size at which resampling draws every triggered block at once by inverting
# its distribution function, whose work grows with the square of the sweep size; larger
# sweeps draw with one generator call per block.  Set by timing the two on an Apple M3
# Pro; not reproducible from this repository.  Re-measure after changing
# `pylea.sample.sweep` or numpy.
RESAMPLE_VECTORISED_MAX = 128

# Interface caps of `pylea.sample.collapse`, as bits, and its per-body greedy
# contraction-cost cap.  The bit caps bound a spliced table to ``2**20`` entries, the same
# bound `TABLE_OUT_BITS_CAP` puts on a sampled table's width, with the output side tighter
# because the sampler's column draw costs a row gather per particle per halving of the
# row count.  The cost cap is in scalar multiplications of the greedy plan, priced by
# `pylea.compile.cost.OPS_PER_MS`: 1e6 is 0.01 ms of arithmetic at that rate, the
# overhead of a few plan steps, so an attempted collapse costs little more than its
# greedy compile.  Derived from `OPS_PER_MS` and `pylea.compile.cost.MS_PER_STEP`, whose
# note says when to re-measure.
COLLAPSE_IN_BITS = 12
COLLAPSE_OUT_BITS = 8
COLLAPSE_COST_CAP = 1e6

# Thresholds of the portfolio algorithm.  Probe order flips at the traced size where
# greedy planning cost starts to dominate the pilot's, set by timing on an Apple M3 Pro
# and not reproducible from this repository; the fast path commits a lane predicted
# below `T_FAST_S`, a choice, without running the second probe.  `E_LO` and `E_HI`
# bracket the exact predictor's error as predicted over actual time.  Each is set a
# factor beyond the extreme observed in a leave-one-family-out calibration over four
# instance families (the ARALIA fault trees, the BN queries, the grid family, and the
# wide-interface family; observed error ratios 0.015 to 1.2e7 on an Apple M3 Pro), since
# a value outside the bracket costs a hedge where a too-narrow bracket would cost a wrong
# commit.  Re-measure after changing the planner or `pylea.compile.cost`.
PROBE_ORDER_TRACED_OPS = 2000
T_FAST_S = 0.25
E_LO = 0.01
E_HI = 1e8

# The exact predictor runs under a wall budget of ``max(PREDICTOR_BUDGET_MIN_S,
# PREDICTOR_BUDGET_FACTOR * the pilot's prediction)``, itself capped at a
# `DEADLINE_HEADROOM` share of any remaining deadline so the probe can never spend the
# query's time budget; on budget-out the exact lane counts as unpredicted.  Choices of
# the procedure, not fitted.
PREDICTOR_BUDGET_MIN_S = 1.0
PREDICTOR_BUDGET_FACTOR = 4.0

# Under `max_time`, committing to the exact lane requires its inflated prediction to fit
# in this fraction of the remaining deadline; otherwise the procedure hedges.  A
# committed exact lane may then spend `EXACT_COMMIT_SHARE` of the remaining deadline
# before the sampling lane takes over with the rest, so a prediction wrong by more than
# the headroom still leaves a certified answer.  Choices of the procedure, not fitted.
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
    bound [1], whose constants `pylea.sample.certify` uses.
    """
    return math.log(4.0 / delta_j)


def iid_particles(streams: Sequence[tuple[float, float | None]], rel_err: float,
                  delta: float, batch: int) -> float:
    """Predicted particles for the i.i.d. mode to certify the query's streams.

    `streams` holds one ``(mass, relvar)`` pair per stream the query certifies: one for
    an unconditioned query; the queried quantity and the acceptance probability for a
    conditioned one.  A `relvar` of `None` prices the stream as Bernoulli draws under the
    Clopper-Pearson sequence; a number, as [0, 1] values of that measured per-particle
    relative variance under the empirical-Bernstein sequence, which is what soft weights
    need, a Bernoulli price overstating their cost by however far their variance sits
    below a coin flip's.

    A conditioned query certifies a quotient, whose relative width is about the sum of
    the two streams' relative widths, each stream holding half of `delta`.  The variance
    terms therefore combine as ``(sum of sqrt(term))**2``, and the range-floor terms add.
    Infinite where any mass is zero, which is how an all-zero pilot reports a lower bound
    on the cost rather than a failure.

        >>> 1e4 < iid_particles([(0.25, None)], 0.05, 0.05, 2**16) < 1e5
        True
        >>> iid_particles([(0.25, 0.1)], 0.05, 0.05, 2**16) < \\
        ...     iid_particles([(0.25, None)], 0.05, 0.05, 2**16)
        True
        >>> iid_particles([(0.25, None), (0.5, None)], 0.05, 0.05, 2**16) > \\
        ...     2 * iid_particles([(0.25, None)], 0.05, 0.05, 2**16)
        True
        >>> iid_particles([(0.0, None)], 0.05, 0.05, 2**16)
        inf
    """
    if any(mass <= 0.0 for mass, _relvar in streams):
        return math.inf
    share = delta / len(streams)
    first = _iid_count(streams, rel_err, log_factor(share))
    if math.isinf(first):
        return first                    # the iterated factor would be infinite as well
    return _iid_count(streams, rel_err, _iterated_log_factor(share, first, batch))


def _iid_count(streams: Sequence[tuple[float, float | None]], rel_err: float,
               L: float) -> float:
    """The i.i.d. particle count `iid_particles` prices, at log factor `L`."""
    # Divided one factor at a time, so that a product too small for a float gives an
    # infinite prediction rather than a division by zero.
    root = sum(math.sqrt(3.0 / mass) if relvar is None else math.sqrt(8.0 * relvar)
               for mass, relvar in streams)
    floor = sum(14.0 * L / 3.0 / rel_err / mass
                for mass, relvar in streams if relvar is not None)
    return root * root * L / rel_err / rel_err + floor


def table_particles(binding: tuple[float, float | None],
                    total: tuple[float, float | None], rel_err: float, delta: float,
                    n_entries: int, batch: int, normalised: bool) -> float:
    """Predicted i.i.d. particles for `pylea.sample.run.run_table` to certify a table.

    `binding` is the ``(mass, relvar)`` pair of the smallest entry the contract must
    resolve, and `total` that of the total mass, as `iid_particles` takes them.  The
    table splits `delta` evenly over its `n_entries` entries and the total, and applies
    its bounds at doubling steps, so its log factor is that of the step
    ``log2(particles / batch) + 1``; with `normalised`, the entry is certified relative
    to the total, a quotient priced as `iid_particles` prices one.  Infinite where the
    binding mass is 0.

        >>> few = table_particles((0.1, None), (1.0, None), 0.05, 0.05, 4, 2**16, False)
        >>> many = table_particles((0.1, None), (1.0, None), 0.05, 0.05, 64, 2**16, False)
        >>> rare = table_particles((0.01, None), (1.0, None), 0.05, 0.05, 4, 2**16, False)
        >>> few < many and few < rare
        True
    """
    streams = [binding, total] if normalised else [binding]
    if any(mass <= 0.0 for mass, _relvar in streams):
        return math.inf
    share = delta / (n_entries + 1)
    first = _iid_count(streams, rel_err, log_factor(share))
    if math.isinf(first):
        return first
    applications = max(1.0, math.log2(max(1.0, first / batch)) + 1.0)
    return _iid_count(streams, rel_err,
                      log_factor(share) + math.log(applications)
                      + math.log(applications + 1.0))


def sweep_particles(streams: Sequence[tuple[float, float]], rel_err: float,
                    delta: float, n_particles: int) -> float:
    """Predicted particles for sweep mode, from the pilot's measured statistics.

    `streams` holds one ``(mass, relvar)`` pair per stream the query certifies, as in
    `iid_particles`; `relvar` is the measured per-sweep relative variance at
    `n_particles` particles per sweep.  The prediction sums the betting certifier's
    variance term and range-floor term, combined over the streams as in `iid_particles`,
    times the particles each sweep costs; predicting the variance from stage counts
    instead would hide state-dependent survival, so the measured value is used.  The
    betting sequence needs no per-step budget, so its log factor does not grow with the
    stopping step.

        >>> sweep_particles([(0.03, 0.02)], 0.05, 0.05, 1024) < \\
        ...     sweep_particles([(0.03, 2.0)], 0.05, 0.05, 1024)
        True
        >>> sweep_particles([(0.03, 0.02), (0.5, 0.02)], 0.05, 0.05, 1024) > \\
        ...     sweep_particles([(0.03, 0.02)], 0.05, 0.05, 1024)
        True
    """
    if any(mass <= 0.0 for mass, _relvar in streams):
        return math.inf
    L = math.log(2.0 * len(streams) / delta)
    root = sum(math.sqrt(relvar) for _mass, relvar in streams)
    # Divided one factor at a time, so that a product too small for a float gives an
    # infinite prediction rather than a division by zero.
    variance_term = SWEEP_VARIANCE_CONSTANT * root * root * L / rel_err / rel_err
    floor_term = sum(SWEEP_FLOOR_CONSTANT * L / rel_err / mass for mass, _relvar in streams)
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
