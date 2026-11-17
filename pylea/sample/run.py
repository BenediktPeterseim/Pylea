"""Driving one scalar query to certification.

`run_iid` and `run_sweeps` connect a `pylea.sample.sweep.SweepProgram` to the streams of
`pylea.sample.certify` and loop until the target is met or the deadline expires.  The
query is described by a `values` vector over the output patterns, each entry in [0, 1]:
the certified quantity is the mean of ``weight * values[pattern]`` and, for a conditioned
DPP, its quotient with the mean weight.  A probability query puts 1 on its event's
patterns; an expectation query puts the rescaled values.

Stream dispatch follows the mode: without resampling certification is per particle, by
Clopper-Pearson where weights and values are 0/1 and by empirical Bernstein otherwise,
and the batch runs on the packed kernel whenever the DPP admits it; with resampling
certification is per sweep by the betting sequence, the sweeps running batched as
side-by-side blocks sized by `pylea.sample.cost_model.SWEEP_BATCH_TARGET`.  For a
conditioned DPP the numerator and denominator streams read the same samples and
split the failure budget evenly; the certified interval is their quotient, on which both
the relative-error target and the `floor` threshold are tested.  The scalar queries are
normalised contracts, so the static log factor cancels either way; only `run_table`
for an unnormalised contract, whose entries are masses, folds `SweepProgram.log_scale`
back in.

`Deadline` bounds a run by wall clock.  Stopping at the deadline is sound because the
streams are anytime-valid; the outcome then reports the interval standing and
`target_met` False.  The point estimate always lies in the interval, and with no samples
at all it is the middle of the vacuous interval.  The deadline is checked between steps,
and where the caller passes the pilot's measured per-particle seconds each step is
additionally sized to fit the remaining time, so the overshoot is bounded by one small
step rather than one full batch; a deadline already expired on entry runs nothing and
reports the vacuous interval.  In sweep mode the loop also stops once `_unreachable`
finds that no further sample can meet the target, and reports `target_met` False; only
a run needing more than about 7e9 sweeps reaches that point, as
`pylea.sample.certify.BettingStream` states.  Otherwise, without a deadline, the loop
runs until the target is met, indefinitely for a target no sample count meets, which is
the documented contract of `method="smc"`.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from pylea.sample.certify import (
    BernoulliStream, BettingStream, Certificate, WeightedParticleStream, cp_bounds,
    eb_bounds, met_floor, met_relative, ratio_interval,
)
from pylea.sample.cost_model import BETTING_READ_GROWTH, SWEEP_BATCH_TARGET, step_delta
from pylea.sample.packed import PackedProgram
from pylea.sample.sweep import SweepProgram
from pylea.semantics.likelihood_table import scaled_matrix


@dataclass(frozen=True)
class Deadline:
    """A wall-clock bound, or no bound.

        >>> Deadline.never().expired(), Deadline.after(0.0).expired()
        (False, True)
    """

    expires_at: float | None

    @classmethod
    def after(cls, seconds: float | None) -> Deadline:
        """The deadline `seconds` from now, or `never()` for `None`."""
        return cls(None if seconds is None else time.monotonic() + seconds)

    @classmethod
    def never(cls) -> Deadline:
        return cls(None)

    def expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() >= self.expires_at

    def remaining(self) -> float:
        """Seconds left, infinite for an unbounded deadline."""
        if self.expires_at is None:
            return float("inf")
        return max(0.0, self.expires_at - time.monotonic())


@dataclass(frozen=True)
class RunOutcome:
    """What a certified run produced.

    `point`, `low` and `high` are on the query's scale: the quotient for a conditioned
    DPP, the unnormalised mass otherwise.  `target_met` says whether the requested
    contract was reached rather than the deadline.
    """

    point: float
    low: float
    high: float
    n_samples: int
    elapsed: float
    target_met: bool
    trace: tuple[str, ...]


def run_iid(program: SweepProgram | PackedProgram, values: np.ndarray, *, rel_err: float,
            confidence: float, batch: int, seed_sequence: np.random.SeedSequence,
            floor: float | None = None, deadline: Deadline | None = None,
            forbidden: np.ndarray | None = None,
            seconds_per_particle: float | None = None, stop=None,
            origin: float = 0.0) -> RunOutcome:
    """The query certified in i.i.d. mode, batches of independent particles.

    Relative error is measured from `origin`, as `pylea.sample.certify.met_relative`
    states; `run_sweeps` takes it in the same sense.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.sample.sweep import SweepProgram, unfold
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.25)
        >>> outcome = run_iid(SweepProgram(unfold(coin.program())), np.array([0.0, 1.0]),
        ...                   rel_err=0.05, confidence=0.95, batch=1 << 14,
        ...                   seed_sequence=np.random.SeedSequence(0))
        >>> outcome.target_met and outcome.low < 0.25 < outcome.high
        True
    """
    deadline = deadline or Deadline.never()
    values = np.asarray(values, dtype=float)
    ledger = Certificate(confidence)
    conditioned = program.conditions
    values_binary = bool(np.all((values == 0.0) | (values == 1.0)))
    binary = program.binary_weights
    # With 0/1 weights the squared per-particle value equals the value times a 0/1
    # weight, so the plain counts already carry the second moment.
    need_squares = not binary
    executor, batch = _iid_executor(program, batch)

    numerator = _particle_stream(ledger, "the queried quantity",
                                 0.5 if conditioned else 1.0, binary and values_binary)
    denominator = (_particle_stream(ledger, "the acceptance probability", 0.5, binary)
                   if conditioned else None)

    def run_step(step: int, rng: np.random.Generator) -> int:
        result = executor.run(step, rng, resample=False, squares=need_squares)
        _check_forbidden(forbidden, result.counts)
        num_total = float(values @ result.counts)
        if isinstance(numerator, BernoulliStream):
            numerator.add(step, int(round(num_total)))
        else:
            counts_sq = result.counts if binary else result.counts_sq
            numerator.add(step, num_total, float((values * values) @ counts_sq))
        if denominator is not None:
            if isinstance(denominator, BernoulliStream):
                denominator.add(step, int(round(result.total)))
            else:
                assert result.total_sq is not None      # need_squares set above
                denominator.add(step, result.total, result.total_sq)
        return step

    outcome = _certify(
        executor, numerator, denominator, ledger,
        next_step=lambda: _step_particles(batch, seconds_per_particle, deadline),
        run_step=run_step, rel_err=rel_err, origin=origin, floor=floor,
        deadline=deadline,
        seed_sequence=seed_sequence, stop=stop,
        describe=lambda n, ending: f"sampling: without resampling, batches of {batch}, "
                                   f"{n} samples, {ending}")
    return outcome


def run_sweeps(program: SweepProgram, values: np.ndarray, *, rel_err: float,
               confidence: float, n_particles: int,
               seed_sequence: np.random.SeedSequence, floor: float | None = None,
               deadline: Deadline | None = None,
               forbidden: np.ndarray | None = None,
               seconds_per_particle: float | None = None, stop=None,
               origin: float = 0.0, stream_rel_err: float | None = None) -> RunOutcome:
    """The query certified in sweep mode, blocks of `n_particles`-sized sweeps per step.

    `stream_rel_err`, `rel_err` by default, is the relative error each stream resolves,
    which sets the cell width of its `pylea.sample.certify.BettingStream`; a caller
    measuring relative error from an `origin` other than 0 passes the finer value the
    query needs on the streams' own scale.

        >>> from pylea import DPP, Bool, flip, observe
        >>> from pylea.sample.sweep import SweepProgram, unfold
        >>> @DPP
        ... def either() -> Bool:
        ...     a, b = flip(0.5), flip(0.5)
        ...     observe(a | b)
        ...     return a
        >>> outcome = run_sweeps(SweepProgram(unfold(either.program())),
        ...                      np.array([0.0, 1.0]), rel_err=0.1, confidence=0.95,
        ...                      n_particles=1024,
        ...                      seed_sequence=np.random.SeedSequence(1))
        >>> outcome.target_met and outcome.low < 2 / 3 < outcome.high
        True
    """
    deadline = deadline or Deadline.never()
    values = np.asarray(values, dtype=float)
    ledger = Certificate(confidence)
    conditioned = program.conditions
    resolution = rel_err if stream_rel_err is None else stream_rel_err
    numerator = BettingStream(ledger.share("the queried quantity",
                                           0.5 if conditioned else 1.0), resolution)
    denominator = (BettingStream(ledger.share("the acceptance probability", 0.5),
                                 resolution)
                   if conditioned else None)
    blocks = max(1, SWEEP_BATCH_TARGET // n_particles)

    def run_step(step_blocks: int, rng: np.random.Generator) -> int:
        result = program.run_blocks(step_blocks, n_particles, rng)
        scales = np.exp(result.log_z) / n_particles
        _check_forbidden(forbidden, result.counts.sum(axis=0))
        numerator.add_many((result.counts @ values) * scales)
        if denominator is not None:
            denominator.add_many(result.counts.sum(axis=1) * scales)
        return step_blocks * n_particles

    streams = [s for s in (numerator, denominator) if s is not None]
    sealing_too_long = []

    def next_step() -> int:
        step_blocks = _step_blocks(blocks, n_particles, seconds_per_particle, deadline)
        # A seal cannot be interrupted, so under a deadline the step that would start
        # one runs only where the seal's estimated time fits in what remains.
        if step_blocks and deadline.expires_at is not None and sum(
                s.seal_seconds(step_blocks) for s in streams) > deadline.remaining():
            sealing_too_long.append(True)
            return 0
        return step_blocks

    def ending_of(ending: str) -> str:
        return ("stopped before the deadline, the time left being too short for the "
                "next step" if sealing_too_long and ending != "target met" else ending)

    return _certify(
        program, numerator, denominator, ledger,
        next_step=next_step,
        run_step=run_step, rel_err=rel_err, origin=origin, floor=floor,
        deadline=deadline, read_growth=BETTING_READ_GROWTH,
        seed_sequence=seed_sequence, stop=stop,
        describe=lambda n, ending: f"sampling: with resampling, {blocks} x {n_particles} "
                                   f"particles per step, {numerator.n} runs of "
                                   f"{n_particles} particles, {ending_of(ending)}",
        out_of_reach=lambda: f"sampling: stopped; this run cannot narrow the interval to "
                             f"the target of {rel_err:g}, which would need more than "
                             f"about {_about(numerator.reach * n_particles)} samples")


def _certify(program, numerator, denominator, ledger: Certificate, *, next_step,
             run_step, rel_err: float, origin: float, floor: float | None,
             deadline: Deadline,
             seed_sequence: np.random.SeedSequence, stop, describe,
             out_of_reach=None, read_growth: float | None = None) -> RunOutcome:
    """The certification loop both modes share: steps sized by `next_step` and run by
    `run_step`, each on a fresh spawned seed, until the contract is met, the deadline
    expires, `stop` says so, or `_unreachable` finds that no further sample can meet the
    contract.

    The interval is read after every step, or, with `read_growth`, once the sample count
    has grown by that share since the last reading, and always when the deadline has
    passed or `stop` says so.

    `run_step` feeds the streams and returns the particles it ran; `describe` renders
    the first trace line from the particle count and how the run ended, and
    `out_of_reach` the line saying why a run stopped short of its target.
    """
    started = time.monotonic()
    n = 0
    low, high, point = 0.0, 1.0, 0.5
    met = False
    ending = "stopped at the deadline"
    reason: tuple[str, ...] = ()
    next_read = 0.0
    unread = False
    while True:
        step = next_step()
        if step == 0:
            if unread:
                # The steps since the last reading are read before the run ends.
                low, high, point = _combine(program, numerator, denominator)
                if met_relative(low, high, rel_err, origin) or met_floor(high, floor):
                    met = True
                    ending = "target met"
            break
        rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        n += run_step(step, rng)
        unread = True
        if (read_growth is not None and n < next_read and not deadline.expired()
                and not (stop is not None and stop())):
            continue
        unread = False
        next_read = n * (1.0 + read_growth) if read_growth is not None else 0.0
        low, high, point = _combine(program, numerator, denominator)
        if met_relative(low, high, rel_err, origin) or met_floor(high, floor):
            met = True
            ending = "target met"
            break
        if _unreachable(numerator, denominator, rel_err, floor, origin):
            ending = "target not met"
            reason = (out_of_reach(),) if out_of_reach is not None else ()
            break
        if deadline.expired() or (stop is not None and stop()):
            break
    trace = (describe(n, ending),) + reason + ledger.splits
    return RunOutcome(point, low, high, n, time.monotonic() - started, met, trace)


def _unreachable(numerator, denominator, rel_err: float, floor: float | None,
                 origin: float = 0.0) -> bool:
    """Whether no further sample can meet the contract.

    Only an exhausted stream (`pylea.sample.certify.BettingStream.exhausted`) makes a
    contract unreachable; it keeps its interval for good.  Any other stream may still
    narrow to any point of its interval, and the contract is tested with each such
    stream narrowed to either end of its interval, which covers the most favourable case
    of each test.  With the numerator kept at ``[a, b]`` and the denominator narrowed to
    a point `x`, the quotient ``[a / x, min(1, b / x)]`` meets the relative target if and
    only if ``min(x, b) <= (1 + rel_err) * a``, easiest at the smallest `x`, and the
    floor if and only if ``min(1, b / x) <= floor``, easiest at the largest.  With the
    denominator kept at ``[c, d]`` and the numerator at a point `y`, the target needs
    ``min(d / y, d / c) <= 1 + rel_err``, easiest at the largest `y`, and the floor
    ``min(1, y / c) <= floor``, easiest at the smallest.  With relative error measured
    from an `origin` other than 0, each test is a ratio of two functions linear in the
    point, so it is again easiest at one end or the other, and the same ends are tried.
    A stream whose lower end is still zero is taken to be able to meet anything.
    """
    streams = [s for s in (numerator, denominator) if s is not None]
    if not any(s.exhausted for s in streams):
        return False
    choices = []
    for stream in streams:
        low, high = stream.interval()
        if stream.exhausted:
            choices.append([(low, high)])
        elif low <= 0.0:
            return False
        else:
            choices.append([(low, low), (high, high)])
    for chosen in itertools.product(*choices):
        low, high = chosen[0] if len(chosen) == 1 else ratio_interval(*chosen)
        if met_relative(low, high, rel_err, origin) or met_floor(high, floor):
            return False
    return True


def _about(x: float) -> str:
    """`x` to one significant digit, as ``7e9``.

        >>> _about(7.04e9), _about(5.6e10)
        ('7e9', '6e10')
    """
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


@dataclass(frozen=True)
class TableOutcome:
    """What a certified table run produced: entrywise masses with jointly valid bounds.

    All arrays run over the output patterns.  `total_low` and `total_high` bound the
    total mass, certified by their own stream; `floor` is the threshold the entries were
    certified against.  Under an unnormalised contract everything is on the true
    scale, the DPP's static log factor folded in; under a normalised one the factor is
    left out, since dividing by the total cancels it, so that a table whose masses lie
    outside the float64 range still gives its distribution.
    """

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    total_point: float
    total_low: float
    total_high: float
    floor: float
    n_samples: int
    elapsed: float
    target_met: bool
    trace: tuple[str, ...]


def run_table(program: SweepProgram | PackedProgram, *, rel_err: float,
              confidence: float, floor: float,
              seed_sequence: np.random.SeedSequence, mode: str = "iid",
              batch: int = 1 << 14, n_particles: int = 1024,
              normalised_contract: bool = False, deadline: Deadline | None = None,
              seconds_per_particle: float | None = None, stop=None) -> TableOutcome:
    """Every output pattern's mass certified at once, under the two-parameter contract.

    `stop`, where given, is asked after every step and ends the run with the interval
    standing when it returns True, as the portfolio algorithm's parallel run needs.

    Each entry is estimated to relative error `rel_err` or certified to lie below
    `floor` on the normalised scale, jointly at `confidence`: the failure budget is
    split evenly over the entries plus one total-mass stream.  With
    `normalised_contract` the relative-error test applies to the entry's quotient with
    the total, which is what a posterior query wants and implies the unnormalised test.

    The bounds are computed vectorised over the entries and applied at doubling
    accumulation steps, each application drawing the next share of the step budget;
    checking every step would price wide tables by the quantile function instead of the
    sampling.

    Under an unnormalised contract the entries are masses, so the DPP's static log
    factor is folded into them; this raises `ValueError` before running where that
    factor does not fit float64, and after running where a certified entry would not,
    under the criterion of `pylea.semantics.likelihood_table.scaled_matrix`.  Under
    `normalised_contract` the factor is left out, as `TableOutcome` states.
    """
    deadline = deadline or Deadline.never()
    started = time.monotonic()
    size = 1 << program.out_bits
    share = 1.0 / (size + 1)
    delta = (1.0 - confidence) * share
    log_scale = 0.0 if normalised_contract else program.log_scale
    scaled_matrix(np.ones(1), log_scale)                # refuse an unfoldable scale early
    sweeping = mode == "sweep"
    blocks = max(1, SWEEP_BATCH_TARGET // n_particles)
    need_squares = not program.binary_weights and not sweeping
    if sweeping:
        executor = program
    else:
        executor, batch = _iid_executor(program, batch)

    counts = np.zeros(size)
    counts_sq = np.zeros(size)
    total = 0.0
    total_sq = 0.0
    n = 0
    steps = 0
    applications = 0
    next_checkpoint = 1
    while True:
        if sweeping:
            assert isinstance(executor, SweepProgram)
            step_blocks = _step_blocks(blocks, n_particles, seconds_per_particle,
                                       deadline)
            ran = step_blocks * n_particles
            if ran:
                rng = np.random.default_rng(seed_sequence.spawn(1)[0])
                block_result = executor.run_blocks(step_blocks, n_particles, rng)
                rows = block_result.counts * (np.exp(block_result.log_z)[:, None]
                                              / n_particles)
                counts += rows.sum(axis=0)
                counts_sq += (rows * rows).sum(axis=0)
                row_totals = rows.sum(axis=1)
                total += float(row_totals.sum())
                total_sq += float((row_totals * row_totals).sum())
                n += ran
                steps += step_blocks
        else:
            ran = _step_particles(batch, seconds_per_particle, deadline)
            if ran:
                rng = np.random.default_rng(seed_sequence.spawn(1)[0])
                result = executor.run(ran, rng, squares=need_squares)
                counts += result.counts
                counts_sq += (result.counts_sq if result.counts_sq is not None
                              else result.counts)
                total += result.total
                total_sq += (result.total_sq if result.total_sq is not None
                             else result.total)
                n += ran
                steps += 1

        stopped = stop is not None and stop()
        if ran and steps < next_checkpoint and not deadline.expired() and not stopped:
            continue                       # bounds applied at doubling steps only
        while next_checkpoint <= steps:
            next_checkpoint *= 2
        applications += 1
        delta_j = step_delta(delta, applications)
        samples = steps if sweeping else n
        lower, upper = _entry_bounds(program, sweeping, counts, counts_sq, samples,
                                     delta_j)
        total_low, total_high = _entry_bounds(
            program, sweeping, np.array([total]), np.array([total_sq]), samples, delta_j)
        total_low, total_high = float(total_low[0]), float(total_high[0])

        if not (np.all((0.0 <= lower) & (lower <= upper) & (upper <= 1.0))
                and 0.0 <= total_low <= total_high <= 1.0):
            raise RuntimeError("certificate invariant violated: an entry's interval is "
                               "not within 0 <= low <= high <= 1")
        if normalised_contract:
            quotient_low = np.where(total_high > 0.0, lower / total_high, 0.0)
            quotient_high = (np.minimum(upper / total_low, 1.0) if total_low > 0.0
                             else np.ones(size))
            relative_met = (0.0 < quotient_low) & (quotient_low <= quotient_high) & \
                (quotient_high - quotient_low <= rel_err * quotient_low)
        else:
            relative_met = (0.0 < lower) & (lower <= upper) & \
                (upper - lower <= rel_err * lower)
        below_floor = (upper / total_low <= floor) if total_low > 0.0 \
            else np.zeros(size, dtype=bool)
        met = bool(np.all(relative_met | below_floor))
        if met or deadline.expired() or stopped:
            # With no samples the point is the middle of each vacuous interval; otherwise
            # the mean, moved into its interval where it lies outside.
            divisor = steps if sweeping else n
            if divisor:
                entries = np.clip(counts / divisor, lower, upper)
                total_point = min(max(total / divisor, total_low), total_high)
            else:
                entries = (lower + upper) / 2
                total_point = (total_low + total_high) / 2
            point, low, high, totals = (
                scaled_matrix(a, log_scale) for a in (
                    entries, lower, upper,
                    np.array([total_point, total_low, total_high])))
            return TableOutcome(
                point, low, high, float(totals[0]), float(totals[1]), float(totals[2]),
                floor, n, time.monotonic() - started, met,
                (f"sampling: table, {'with' if sweeping else 'without'} resampling, "
                 f"{n} samples, "
                 f"{'target met' if met else 'stopped' if stopped else 'stopped at the deadline'}",
                 f"failure probability split over {size} entries and the total, "
                 f"{share:.3g} of it each"))


def _entry_bounds(program: SweepProgram | PackedProgram, sweeping: bool,
                  sums: np.ndarray,
                  sums_sq: np.ndarray, samples: int, delta_j: float):
    """Vectorised per-entry bounds on the mean, on the rescaled [0, 1] sample scale.

    The certificates are `pylea.sample.certify.cp_bounds` and `eb_bounds`, the same
    bounds the scalar streams apply."""
    if samples < 2:
        return np.zeros_like(sums), np.ones_like(sums)
    if not sweeping and program.binary_weights:
        return cp_bounds(samples, np.round(sums).astype(np.int64), delta_j)
    means = sums / samples
    variances = np.maximum(0.0, (sums_sq - samples * means**2) / (samples - 1))
    return eb_bounds(samples, means, variances, delta_j)


def _check_forbidden(forbidden: np.ndarray | None, counts: np.ndarray) -> None:
    if forbidden is not None and float(forbidden @ counts) > 0.0:
        raise ValueError(
            "positive probability lies on bit patterns that encode no declared value")


def packed_or_byte(program: SweepProgram | PackedProgram):
    """The packed executor where `program` admits one, else the byte lane unchanged.

    Packing is decided per DPP: any DPP with 0/1 weights runs packed, its table
    ops handled at the kernel's boundary, and only soft weights keep the byte lane.
    The packed form is held by the `SweepProgram` (`SweepProgram.packed`), so the pilot
    and the certified runs share one preprocessing pass, and it is freed with it.
    """
    if isinstance(program, PackedProgram):
        return program
    packed = program.packed
    return program if packed is None else packed


def _iid_executor(program: SweepProgram | PackedProgram, batch: int):
    """The i.i.d. executor and its batch size.

    A packed plane is an eighth of a byte plane, so the memory ceiling behind the batch
    size admits eight times the particles, and the larger batch is what amortises the
    per-op dispatch floor over enough words."""
    executor = packed_or_byte(program)
    if isinstance(executor, PackedProgram):
        batch *= 8
    return executor, batch


def _step_particles(batch: int, per_particle: float | None,
                    deadline: Deadline) -> int:
    """The particles the next step runs: `batch`, fewer where the remaining deadline
    prices smaller, zero at an expired deadline.

    Steps are kept multiples of 64, the packed kernel's word width, with 64 as the
    smallest step so an almost-spent deadline still measures something."""
    remaining = deadline.remaining()
    if remaining <= 0.0:
        return 0
    if per_particle is None or per_particle <= 0.0 or math.isinf(remaining):
        return batch
    fit = int(remaining / per_particle)
    return min(batch, max(64, fit - fit % 64))


def _step_blocks(blocks: int, n_particles: int, per_particle: float | None,
                 deadline: Deadline) -> int:
    """The sweep blocks the next step runs; the block analogue of `_step_particles`."""
    remaining = deadline.remaining()
    if remaining <= 0.0:
        return 0
    if per_particle is None or per_particle <= 0.0 or math.isinf(remaining):
        return blocks
    fit = int(remaining / (per_particle * n_particles))
    return min(blocks, max(1, fit))


def _particle_stream(ledger: Certificate, label: str, fraction: float, bernoulli: bool):
    budget = ledger.share(label, fraction)
    return BernoulliStream(budget) if bernoulli else WeightedParticleStream(budget)


def _combine(program: SweepProgram | PackedProgram, numerator,
             denominator) -> tuple[float, float, float]:
    """The certified interval and point estimate of the normalised query.

    The scalar queries are normalised contracts, so the static log factor cancels: with
    no weighting op left, every particle's rescaled weight is exactly one, and the
    rescaled numerator mean already is the conditional value.  The point is the mean, or
    the quotient of means, moved into the interval where it lies outside, and the middle
    of the interval where the denominator's mean is zero.  An interval outside
    ``0 <= low <= high <= 1`` raises `RuntimeError`: it would be a defect of a
    certificate, never a result.
    """
    if denominator is None:
        low, high = numerator.interval()
        mean = numerator.mean
    else:
        low, high = ratio_interval(numerator.interval(), denominator.interval())
        mean = (numerator.mean / denominator.mean if denominator.mean > 0.0
                else (low + high) / 2)
    if not 0.0 <= low <= high <= 1.0:
        raise RuntimeError(f"certificate invariant violated: interval [{low!r}, {high!r}]")
    return low, high, min(max(mean, low), high)
