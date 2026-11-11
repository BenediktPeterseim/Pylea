"""Confidence sequences and the failure-budget ledger.

Every certificate here is anytime-valid: the bound is applied once per accumulation step
``j = 1, 2, ...`` with the failure budget ``delta / (j * (j + 1))``, which sums to at most
`delta`, so the interval standing at any step contains the truth with probability at least
``1 - delta`` simultaneously over all steps.  Stopping at an arbitrary step, as a
deadline does, is therefore sound.

The stream types, dispatched by the execution mode:

  * `BernoulliStream`: i.i.d. particles with 0/1 weights.  The certificate is an exact
    Clopper-Pearson sequence, which is tighter than the empirical-Bernstein bound
    severalfold at zero hits and gives two-sided intervals far earlier.
  * `WeightedParticleStream`: i.i.d. particles with weights in [0, 1].  The certificate
    is the empirical-Bernstein bound of Maurer and Pontil over particles.
  * `BettingStream`: sweep values in [0, 1] under resampling, where no exact
    distribution is available.  A betting confidence sequence, whose constants beat the
    checkpointed empirical-Bernstein bound severalfold; the range floor stated in the
    `pylea.sample` package docstring still applies.
  * `SweepStream`: the checkpointed empirical-Bernstein form of the same stream, kept
    for the entrywise table bounds, whose vectorised evaluation the betting grid does
    not fit, and as the reference the betting stream is compared against.

A conditioned query certifies a quotient: `ratio_interval` combines the two streams'
intervals, and validity needs only a union bound over the two certificates, which the
`Certificate` ledger provides by splitting the query's failure budget and recording every
split.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

from pylea.sample.cost_model import log_factor, step_delta


def eb_bounds(n: int, means: np.ndarray, variances: np.ndarray,
              delta_j: float) -> tuple[np.ndarray, np.ndarray]:
    """The two-sided empirical-Bernstein bounds for `n` i.i.d. values in [0, 1],
    vectorised over means and sample variances.

    The width is the Maurer-Pontil bound ``sqrt(2 var L / n) + 7 L / (3 (n - 1))`` with
    ``L = ln(4 / delta_j)``; the second term is the range floor that no variance can
    remove.  Requires ``n >= 2``.
    """
    L = log_factor(delta_j)
    width = np.sqrt(2.0 * np.asarray(variances) * L / n) + 7.0 * L / (3.0 * (n - 1))
    means = np.asarray(means)
    return np.maximum(0.0, means - width), np.minimum(1.0, means + width)


def eb_interval(n: int, mean: float, var: float, delta_j: float) -> tuple[float, float]:
    """The two-sided empirical-Bernstein interval for `n` i.i.d. values in [0, 1].

    The scalar form of `eb_bounds`; `var` is the sample variance.

        >>> low, high = eb_interval(10_000, 0.3, 0.21, 0.01)
        >>> 0.27 < low < 0.3 < high < 0.33
        True
    """
    if n < 2:
        return 0.0, 1.0
    low, high = eb_bounds(n, np.asarray(mean), np.asarray(var), delta_j)
    return float(low), float(high)


def cp_bounds(n: int, hits: np.ndarray, delta_j: float) -> tuple[np.ndarray, np.ndarray]:
    """The exact two-sided Clopper-Pearson bounds for `n` draws, vectorised over the hit
    counts.

    Each side receives half of `delta_j`.  The quantile arguments are clamped away from
    the edges so the branches `np.where` discards never evaluate an invalid beta.
    """
    hits = np.asarray(hits, dtype=np.int64)
    at_least_one = np.clip(hits, 1, None)
    low = np.where(hits > 0,
                   beta.ppf(delta_j / 2, at_least_one, n - at_least_one + 1), 0.0)
    at_most_all_but_one = np.clip(hits, None, n - 1)
    high = np.where(hits < n,
                    beta.ppf(1 - delta_j / 2, at_most_all_but_one + 1,
                             n - at_most_all_but_one), 1.0)
    return low, high


def cp_interval(n: int, k: int, delta_j: float) -> tuple[float, float]:
    """The exact two-sided Clopper-Pearson interval for `k` successes in `n` draws.

    The scalar form of `cp_bounds`.  At zero successes the upper bound is
    ``1 - (delta_j / 2) ** (1 / n)``, about ``ln(2 / delta_j) / n``, which is what makes
    a zero-hit run a usable threshold certificate:

        >>> low, high = cp_interval(1_000_000, 0, 0.05)
        >>> low == 0.0 and high < 4e-6
        True
    """
    low, high = cp_bounds(n, np.asarray(k), delta_j)
    return float(low), float(high)


def ratio_interval(numerator: tuple[float, float],
                   denominator: tuple[float, float]) -> tuple[float, float]:
    """The interval of a quotient of two bracketed non-negative quantities, capped to [0, 1].

    Needs no independence between the two intervals, only that each holds; the caller
    supplies that via a union bound over their budgets.  A denominator whose lower bound
    is zero yields the vacuous interval.

        >>> ratio_interval((0.2, 0.3), (0.9, 1.0))
        (0.2, 0.3333333333333333)
        >>> ratio_interval((0.2, 0.3), (0.0, 1.0))
        (0.0, 1.0)
    """
    (n_low, n_high), (d_low, d_high) = numerator, denominator
    if d_low <= 0.0:
        return 0.0, 1.0
    return max(0.0, n_low / d_high), min(1.0, n_high / d_low)


class Budget:
    """A share of a certificate's failure budget.

    `delta_at(j)` is accumulation step `j`'s share, per the schedule in the module
    docstring.
    """

    __slots__ = ("delta",)

    def __init__(self, delta: float):
        self.delta = delta

    def delta_at(self, j: int) -> float:
        return step_delta(self.delta, j)


class Certificate:
    """The failure-budget ledger of one query.

    Splits ``1 - confidence`` over the streams the query needs and records every split,
    so the final object is auditable.  Asking for more than the whole budget raises
    `ValueError`.

        >>> ledger = Certificate(confidence=0.95)
        >>> num, den = ledger.share("numerator", 0.5), ledger.share("denominator", 0.5)
        >>> round(num.delta, 4), ledger.splits
        (0.025, ('numerator: 0.025', 'denominator: 0.025'))
    """

    def __init__(self, confidence: float):
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must lie strictly in (0, 1); got {confidence}")
        self.confidence = confidence
        self.delta = 1.0 - confidence
        self._granted = 0.0
        self._splits: list[str] = []

    def share(self, label: str, fraction: float) -> Budget:
        """A `Budget` holding `fraction` of the total failure budget, now granted."""
        if self._granted + fraction > 1.0 + 1e-12:
            raise ValueError(f"failure budget exhausted: {self._granted} + {fraction} > 1")
        self._granted += fraction
        delta = self.delta * fraction
        self._splits.append(f"{label}: {delta:.4g}")
        return Budget(delta)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits)


class BernoulliStream:
    """Hit counts of i.i.d. 0/1 draws, certified by a Clopper-Pearson sequence.

        >>> stream = BernoulliStream(Budget(0.05))
        >>> stream.add(10_000, 2_600)
        >>> low, high = stream.interval()
        >>> low < 0.26 < high and high - low < 0.03
        True
    """

    __slots__ = ("budget", "n", "hits", "j")

    def __init__(self, budget: Budget):
        self.budget = budget
        self.n = 0
        self.hits = 0
        self.j = 0

    def add(self, n: int, hits: int) -> None:
        """One accumulation step of `n` draws with `hits` successes."""
        self.n += n
        self.hits += hits
        self.j += 1

    def interval(self) -> tuple[float, float]:
        if self.j == 0:
            return 0.0, 1.0
        return cp_interval(self.n, self.hits, self.budget.delta_at(self.j))

    @property
    def mean(self) -> float:
        return self.hits / self.n if self.n else 0.0


class WeightedParticleStream:
    """Sums of i.i.d. particle values in [0, 1], certified by empirical Bernstein.

        >>> stream = WeightedParticleStream(Budget(0.05))
        >>> stream.add(100_000, 30_000.0, 30_000.0)     # 0/1 values would allow CP too
        >>> low, high = stream.interval()
        >>> low < 0.3 < high
        True
    """

    __slots__ = ("budget", "n", "total", "total_sq", "j")

    def __init__(self, budget: Budget):
        self.budget = budget
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.j = 0

    def add(self, n: int, total: float, total_sq: float) -> None:
        """One accumulation step of `n` values summing to `total`, squares to `total_sq`."""
        self.n += n
        self.total += total
        self.total_sq += total_sq
        self.j += 1

    def interval(self) -> tuple[float, float]:
        if self.n < 2:
            return 0.0, 1.0
        mean = self.total / self.n
        var = max(0.0, (self.total_sq - self.n * mean * mean) / (self.n - 1))
        return eb_interval(self.n, mean, var, self.budget.delta_at(self.j))

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


class SweepStream:
    """I.i.d. sweep values in [0, 1], certified by empirical Bernstein per sweep.

        >>> import numpy as np
        >>> stream = SweepStream(Budget(0.05))
        >>> for value in np.random.default_rng(0).uniform(0.2, 0.4, size=4096):
        ...     stream.add(float(value))
        >>> low, high = stream.interval()
        >>> low < 0.3 < high
        True
    """

    __slots__ = ("budget", "n", "total", "total_sq")

    def __init__(self, budget: Budget):
        self.budget = budget
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0

    def add(self, value: float) -> None:
        """One sweep's value, which is one accumulation step."""
        self.n += 1
        self.total += value
        self.total_sq += value * value

    def interval(self) -> tuple[float, float]:
        if self.n < 2:
            return 0.0, 1.0
        mean = self.total / self.n
        var = max(0.0, (self.total_sq - self.n * mean * mean) / (self.n - 1))
        return eb_interval(self.n, mean, var, self.budget.delta_at(self.n))

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


class BettingStream:
    """[0, 1] samples certified by a hedged betting confidence sequence.

    The construction of Waudby-Smith and Ramdas: for each candidate mean `m` two capital
    processes bet against it, one gaining where the samples run above `m` and one where
    they run below, with a predictable stake truncated so capital stays positive; the
    average of the two is a non-negative martingale with unit mean at the true mean, so
    by Ville's inequality the candidates whose capital stays below ``1/delta`` form a
    time-uniform confidence sequence.  No per-step budget split is needed, which is
    where this beats the checkpointed empirical-Bernstein sequence in its constants, and
    it has no range floor tied to a per-step budget either.

    The capital is a deterministic function of the data at *every* candidate mean, so
    the confidence set is valid over all of [0, 1] and a grid is only where it is
    evaluated.  The stream starts from a coarse geometric grid and, whenever the
    rejection boundary sits in a cell wider than the target granularity, subdivides that
    cell: the new candidates' capital is computed once retroactively from the stored
    batches, and from then on they update incrementally with every batch.  Each cell is
    subdivided at most a bounded number of times, so refinement stays a vanishing share
    of the work while the endpoints reach a relative granularity of about 0.02% instead
    of the coarse cell's 0.7%.  The stake is frozen per batch from the state before it,
    which keeps it predictable, and the per-batch stakes are stored with the batches so
    a late candidate replays exactly the process it would have had.

    The stored batches exist only for that replay, so their number is capped at
    `_REPLAY_BATCHES`: on reaching the cap the current boundary is refined to the target
    granularity once and the history is dropped, after which no further cells subdivide.
    This bounds the stream's memory on an indefinitely long run; if the boundary later
    drifts into a cell never refined, the endpoints fall back to that cell's coarse
    granularity, and validity is unaffected either way.

        >>> import numpy as np
        >>> stream = BettingStream(Budget(0.05))
        >>> stream.add_many(np.random.default_rng(0).uniform(0.2, 0.4, size=4096))
        >>> low, high = stream.interval()
        >>> low < 0.3 < high and high - low < 0.02
        True
    """

    _GRID = np.concatenate(([0.0], np.geomspace(1e-15, 1.0, 15 * 341 + 1)))
    _SPLIT = 8                          # subdivisions per refinement of a boundary cell
    _GRAIN = 2e-4                       # stop refining below this relative cell width
    _REPLAY_BATCHES = 1024              # stored batches; the cap bounds memory and replay
    _MAX_PASSES = 64                    # refinement passes per `interval` call

    __slots__ = ("budget", "n", "total", "total_sq", "_candidates", "_log_up",
                 "_log_down", "_batches", "_sealed", "_low", "_high")

    def __init__(self, budget: Budget):
        self.budget = budget
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self._candidates = self._GRID.copy()
        self._log_up = np.zeros(self._candidates.size)
        self._log_down = np.zeros(self._candidates.size)
        self._batches: list[tuple[np.ndarray, float]] = []
        self._sealed = False
        self._low = 0.0
        self._high = 1.0

    def add(self, value: float) -> None:
        """One sample; see `add_many` for the batch form the run loops use."""
        self.add_many(np.array([value]))

    def add_many(self, values: np.ndarray) -> None:
        """A batch of samples in [0, 1], with the stake frozen from the state before it."""
        values = np.asarray(values, dtype=float)
        base = self._base_stake()
        up, down = _capital_gains(values, base, self._candidates)
        self._log_up += up
        self._log_down += down
        if not self._sealed:
            self._batches.append((values, base))
            if len(self._batches) > self._REPLAY_BATCHES:
                self._seal()
        self.n += len(values)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())

    def _seal(self) -> None:
        """Nothing, having refined the current boundary once and dropped the history.

        Called at the replay cap; see the class docstring for the trade."""
        self.interval()
        self._batches = []
        self._sealed = True

    def _base_stake(self) -> float:
        """The predictable stake: variance-adapted with a prior of 1/4."""
        mean = self.total / self.n if self.n else 0.5
        variance = (0.25 + self.total_sq - 2 * mean * self.total
                    + self.n * mean * mean) / (self.n + 1)
        steps = self.n + 1
        return math.sqrt(2.0 * math.log(2.0 / self.budget.delta)
                         / (variance * steps * math.log1p(steps)))

    def interval(self) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 1.0
        threshold = math.log(1.0 / self.budget.delta)
        first = last = 0
        # Each pass subdivides one boundary cell and rescans; the scan also runs once
        # after the last pass, so the endpoints are always read from the grid they
        # index into.
        for passes in range(self._MAX_PASSES + 1):
            hedged = _hedged(self._log_up, self._log_down)
            surviving = np.flatnonzero(hedged < threshold)
            if surviving.size == 0:
                return self._low, max(self._high, self._low)
            first, last = int(surviving[0]), int(surviving[-1])
            if passes == self._MAX_PASSES:
                break
            if not (self._refine(first - 1) or self._refine(last)):
                break
        candidates = self._candidates
        low = candidates[first - 1] if first else 0.0
        high = candidates[last + 1] if last + 1 < candidates.size else 1.0
        # The running intersection keeps the intervals nested, which a time-uniform
        # confidence sequence admits.  A candidate outside it can therefore never be
        # reported again, so only the survivors and the rejected neighbour on each side,
        # which are the endpoints, stay watched.
        self._low = max(self._low, float(low))
        self._high = min(self._high, float(high))
        keep = slice(max(first - 1, 0), last + 2)
        self._candidates = candidates[keep]
        self._log_up = self._log_up[keep]
        self._log_down = self._log_down[keep]
        return self._low, max(self._high, self._low)

    def _refine(self, cell: int) -> bool:
        """Whether the cell starting at candidate `cell` was subdivided.

        New candidates join the watched set with their capital replayed once from the
        stored batches; a cell already at the target granularity is left alone, as are
        the cell above zero and every cell once the stream is sealed.
        """
        if self._sealed:
            return False
        candidates = self._candidates
        if cell < 0 or cell + 1 >= candidates.size:
            return False
        a, b = candidates[cell], candidates[cell + 1]
        # The cell above zero has no relative width to refine to; it stays at the
        # grid's resolution, which is far below any floor a query can ask for.
        if a == 0.0 or b - a <= self._GRAIN * b:
            return False
        fine = np.linspace(a, b, self._SPLIT + 1)[1:-1]
        log_up = np.zeros(fine.size)
        log_down = np.zeros(fine.size)
        for values, base in self._batches:
            up, down = _capital_gains(values, base, fine)
            log_up += up
            log_down += down
        at = cell + 1
        self._candidates = np.concatenate([candidates[:at], fine, candidates[at:]])
        self._log_up = np.concatenate([self._log_up[:at], log_up, self._log_up[at:]])
        self._log_down = np.concatenate([self._log_down[:at], log_down,
                                         self._log_down[at:]])
        return True

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


def _capital_gains(values: np.ndarray, base: float, candidates: np.ndarray):
    """The two capital processes' log gains for one batch, per candidate mean."""
    with np.errstate(divide="ignore"):
        stake_up = np.minimum(base, 0.5 / np.maximum(candidates, 1e-300))
        stake_down = np.minimum(base, 0.5 / np.maximum(1.0 - candidates, 1e-300))
    centred = values[:, None] - candidates[None, :]
    return (np.log1p(stake_up[None, :] * centred).sum(axis=0),
            np.log1p(-stake_down[None, :] * centred).sum(axis=0))


def _hedged(log_up: np.ndarray, log_down: np.ndarray) -> np.ndarray:
    """The log of the hedged average capital, computed stably from the two logs."""
    top = np.maximum(log_up, log_down)
    return top + np.log1p(np.exp(-np.abs(log_up - log_down))) - math.log(2.0)


def met_relative(low: float, high: float, rel_err: float) -> bool:
    """Whether an interval certifies relative error `rel_err`.

        >>> met_relative(0.99, 1.0, 0.02), met_relative(0.0, 1e-9, 0.02)
        (True, False)
    """
    return low > 0.0 and high - low <= rel_err * low


def met_floor(high: float, floor: float | None) -> bool:
    """Whether an interval certifies the value to lie below `floor`.

        >>> met_floor(1e-9, 1e-6), met_floor(1e-9, None)
        (True, False)
    """
    return floor is not None and high <= floor
