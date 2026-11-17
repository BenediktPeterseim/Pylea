"""Confidence sequences and the failure-budget ledger.

Every certificate here is anytime-valid: the bound is applied once per accumulation step
``j = 1, 2, ...`` with the failure budget ``delta / (j * (j + 1))``, which sums to at most
`delta`, so the interval standing at any step contains the truth with probability at least
``1 - delta`` simultaneously over all steps.  Stopping at an arbitrary step, as a
deadline does, is therefore sound.

The stream types, dispatched by the execution mode:

  * `BernoulliStream`: i.i.d. particles with 0/1 weights.  The certificate is an exact
    Clopper-Pearson sequence, whose bounds are beta quantiles of the hit count, exact for
    Bernoulli draws.  Its beta quantiles are checked against the beta distribution function
    and recomputed by bisection where the library routine is inaccurate.
  * `WeightedParticleStream`: i.i.d. particles with weights in [0, 1].  The certificate
    is the empirical-Bernstein bound [1] over particles.
  * `BettingStream`: sweep values in [0, 1] under resampling, where no exact
    distribution is available.  The certificate is a betting confidence sequence [2],
    which needs no per-step split of the failure budget; the range floor stated in the
    `pylea.sample` package docstring still applies.

`eb_bounds` evaluates the empirical-Bernstein bound for many means at once, which is what
the entrywise bounds of a sampled table use.

A conditioned query certifies a quotient: `ratio_interval` combines the two streams'
intervals, and validity needs only a union bound over the two certificates, which the
`Certificate` ledger provides by splitting the query's failure budget and recording every
split.

[1] A. Maurer and M. Pontil.  Empirical Bernstein bounds and sample variance
    penalization.  Conference on Learning Theory (COLT), 2009.
    https://arxiv.org/abs/0907.3740

[2] I. Waudby-Smith and A. Ramdas.  Estimating means of bounded random variables by
    betting.  Journal of the Royal Statistical Society, Series B, 86(1), 2024.
    https://doi.org/10.1093/jrsssb/qkad009
"""
from __future__ import annotations

import math
import time

import numpy as np

from pylea.sample.cost_model import (
    BETTING_CELL_SHARE, BETTING_GRID_PER_DECADE, BETTING_MAX_PASSES, BETTING_SPLIT,
    REPLAY_SAMPLES_MAX, SEAL_CANDIDATES_MAX, log_factor, step_delta,
)


def eb_bounds(n: int, means: np.ndarray, variances: np.ndarray,
              delta_j: float) -> tuple[np.ndarray, np.ndarray]:
    """The two-sided empirical-Bernstein bounds for `n` i.i.d. values in [0, 1],
    vectorised over means and sample variances.

    The width is the bound of [1] in the module docstring,
    ``sqrt(2 var L / n) + 7 L / (3 (n - 1))`` with ``L = ln(4 / delta_j)``; the second
    term is the range floor that no variance can remove.  Requires ``n >= 2``.
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

    Each side receives half of `delta_j`.  Every bound is checked against the beta
    distribution function by `_beta_quantile`, and an interval that does not contain the
    hit rate raises `RuntimeError`: it would be a defect, never a result.
    """
    hits = np.asarray(hits, dtype=np.int64)
    q = delta_j / 2
    # The shapes are clamped away from the edges so that the entries `np.where`
    # discards never evaluate an invalid beta.
    at_least_one = np.clip(hits, 1, None)
    low = np.where(hits > 0,
                   _beta_quantile(at_least_one, n - at_least_one + 1, q, upper=False), 0.0)
    at_most_all_but_one = np.clip(hits, None, n - 1)
    high = np.where(hits < n,
                    _beta_quantile(at_most_all_but_one + 1, n - at_most_all_but_one, q,
                                   upper=True), 1.0)
    rate = hits / n
    if not np.all((low <= rate) & (rate <= high)):
        bad = int(np.flatnonzero(~((low <= rate) & (rate <= high)))[0])
        raise RuntimeError(
            f"Clopper-Pearson certificate invariant violated: {int(hits.flat[bad])} hits "
            f"in {n} draws gave the interval [{low.flat[bad]!r}, {high.flat[bad]!r}]")
    return low, high


# Relative accuracy of the tail probability at a beta quantile that `_beta_quantile`
# accepts from scipy.  Far below any failure budget a query can carry, and above the
# accuracy of `betainc`.
_QUANTILE_RTOL = 1e-6

# The smallest tail probability `_beta_quantile` accepts.  scipy's `betainc` and
# `betaincc` returned 0.0 for true tails near 1e-291 (scipy 1.16.0), and matched a
# 50-digit reference on the conservative side for every tail from 1e-100 upwards over
# n from 1 to 1e12.
_MIN_TAIL = 1e-100


def _beta_quantile(a: np.ndarray, b: np.ndarray, q: float, *, upper: bool) -> np.ndarray:
    """The `q` quantile of the beta distribution with shapes `a` and `b`, from below or,
    with `upper`, from above, on the side that keeps the tail probability at most `q`.

    For a lower bound this is the largest `x` with ``betainc(a, b, x) <= q``; for an
    upper bound the smallest `x` with ``betaincc(a, b, x) <= q``.  scipy's inverse
    functions give the answer for almost every shape, but not for all: some versions
    return a wrong quantile at particular shapes.  Each result is therefore checked by
    evaluating the distribution function at it, and accepted if its tail lies within
    `_QUANTILE_RTOL` below `q`; every other entry is found by bisection over the
    floating-point numbers in ``[0, 1]``, which ends on the exact float on the safe
    side.  An upper bound within about ``1e-12`` of 1 always takes the bisection, since
    neighbouring floats there differ by more than the tolerance allows, and is 1.0 where
    no float below 1 has a tail of at most `q`.

    `q` must be at least `_MIN_TAIL`, or `ValueError` is raised: scipy's distribution
    functions lose their relative accuracy where a tail approaches the smallest float, so
    neither the check nor the bisection could be trusted there.  A query's failure budget
    is split into shares far above that bound.

        >>> x = _beta_quantile(np.array([1]), np.array([1]), 0.25, upper=False)
        >>> bool(0.25 * (1 - 1e-6) <= x[0] <= 0.25)
        True
    """
    from scipy.special import betainc, betaincc, betainccinv, betaincinv

    if not q >= _MIN_TAIL:
        raise ValueError(f"a tail probability of {q!r} is below {_MIN_TAIL}, where the beta "
                         "distribution function is not accurate")
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    tail = betaincc if upper else betainc
    target = q * (1.0 - _QUANTILE_RTOL)
    with np.errstate(all="ignore"):
        x = np.asarray((betainccinv if upper else betaincinv)(a, b, target), dtype=float)
        found = tail(a, b, x)
        good = (np.isfinite(x) & (0.0 <= x) & (x <= 1.0)
                & (np.abs(found - target) <= _QUANTILE_RTOL * target))
    if np.all(good):
        return x
    x = np.array(x, copy=True)
    redo = np.flatnonzero(~good.ravel())
    x.ravel()[redo] = _bisect_quantile(a.ravel()[redo], b.ravel()[redo], q, upper, tail)
    return x


def _bisect_quantile(a: np.ndarray, b: np.ndarray, q: float, upper: bool,
                     tail) -> np.ndarray:
    """`_beta_quantile` by bisection over the bit patterns of the floats in ``[0, 1]``,
    which order like the floats themselves.

    The invariant is that `safe` has a tail of at most `q` and `unsafe` does not; 64
    halvings reach neighbouring floats.
    """
    zero = np.zeros(a.shape, dtype=np.int64)
    one = np.full(a.shape, np.float64(1.0).view(np.int64))
    safe, unsafe = (one, zero) if upper else (zero, one)
    for _ in range(64):
        if np.all(np.abs(safe - unsafe) <= 1):
            break
        middle = (safe + unsafe) // 2
        with np.errstate(all="ignore"):
            ok = tail(a, b, middle.view(np.float64)) <= q
        safe = np.where(ok, middle, safe)
        unsafe = np.where(ok, unsafe, middle)
    return safe.view(np.float64)


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
        >>> num = ledger.share("the queried quantity", 0.5)
        >>> den = ledger.share("the acceptance probability", 0.5)
        >>> round(num.delta, 4), ledger.splits[0]
        (0.025, 'failure probability 0.025 for the queried quantity')
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
        self._splits.append(f"failure probability {delta:.4g} for {label}")
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

    # Whether no further draw can narrow the interval; never for this stream.
    exhausted = False

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

    # Whether no further value can narrow the interval; never for this stream.
    exhausted = False

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


class BettingStream:
    """[0, 1] samples certified by a hedged betting confidence sequence.

    The construction of [2] in the module docstring: for each candidate mean `m` two capital
    processes bet against it, one gaining where the samples run above `m` and one where
    they run below, with a predictable stake truncated so capital stays positive; the
    average of the two is a non-negative martingale with unit mean at the true mean, so
    by Ville's inequality the candidates whose capital stays below ``1/delta`` form a
    time-uniform confidence sequence.  No per-step budget split is needed, so the
    sequence has no log factor growing with the step and no range floor tied to a
    per-step budget.

    The stake of each sample is fixed before the batch that carries it: the variance
    estimate comes from the samples before the batch, and the sample's own index `t`
    sets the decay ``1 / sqrt(t log(1 + t))``, so a batch bets as its samples would one
    at a time, apart from the variance estimate.  Each factor ``1 +- stake * (x - m)``
    stays at least 1/2.

    The capital is a deterministic function of the data at *every* candidate mean, so
    the confidence set is valid over all of [0, 1] and a grid is only where it is
    evaluated.  The reported interval runs between the rejected neighbours of the
    outermost surviving candidates, intersected with every earlier interval.  The stream
    starts from a coarse geometric grid with cells 0.68% wide and refines it where the
    ends of the interval are read:

      * a cell holding an end of the interval is subdivided while it is wider, in
        relative terms, than the cell width ``grain = BETTING_CELL_SHARE * rel_err``;
      * where no candidate survives, the confidence set lies between two of them, and
        the cells on both sides of the candidate with the least capital are subdivided,
        below `grain` if need be, until one survives.  The interval therefore does not
        depend on how often it is read.

    A new candidate's capital is computed once from the stored samples and their
    stakes, which is exactly the process it would have had, and from then on it updates
    with every batch.

    The stored samples exist only for that replay, so their number is capped at
    `REPLAY_SAMPLES_MAX`.  On reaching the cap the stream is *sealed*: it lays candidates
    geometrically over the whole range it still watches, one per cell of relative width
    `grain` and at most `SEAL_CANDIDATES_MAX` of them, computes their capital from the
    history, and drops the history; no cell is subdivided after that.  The seal resolves
    the range to `grain` whenever its relative width is at most
    ``SEAL_CANDIDATES_MAX * grain``, which is 82 times the target.  The width falls with
    the square root of the sample count, so the target is reachable in every run that
    needs fewer than about `reach` samples, 7e9.  A sealed stream with at most one
    surviving candidate is *exhausted*: no further sample narrows its interval.  Validity
    is unaffected by any of this, every candidate's capital being the exact process of
    its mean.

        >>> import numpy as np
        >>> stream = BettingStream(Budget(0.05), rel_err=0.01)
        >>> for batch in np.random.default_rng(0).uniform(0.2, 0.4, size=(4, 1024)):
        ...     stream.add_many(batch)
        >>> low, high = stream.interval()
        >>> low < 0.3 < high and high - low < 0.02
        True
    """

    # The starting grid: 0, and 1e-15 to 1 at `BETTING_GRID_PER_DECADE` per decade.  The
    # lower end is a numerical limit, not tuned; the cell from 0 to it is never refined.
    _GRID = np.concatenate(([0.0], np.geomspace(1e-15, 1.0,
                                                15 * BETTING_GRID_PER_DECADE + 1)))
    REPLAY_SAMPLES_MAX = REPLAY_SAMPLES_MAX
    SEAL_CANDIDATES_MAX = SEAL_CANDIDATES_MAX

    __slots__ = ("budget", "grain", "n", "total", "total_sq", "_candidates", "_log_up",
                 "_log_down", "_history", "_stored", "_sealed", "_low", "_high",
                 "_replay_seconds")

    def __init__(self, budget: Budget, rel_err: float):
        self.budget = budget
        self.grain = BETTING_CELL_SHARE * rel_err
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self._candidates = self._GRID.copy()
        self._log_up = np.zeros(self._candidates.size)
        self._log_down = np.zeros(self._candidates.size)
        # Row 0 holds the stored samples, row 1 their stakes; the first `_stored`
        # columns are in use.
        self._history = np.empty((2, 0))
        self._stored = 0
        self._sealed = False
        self._low = 0.0
        self._high = 1.0
        self._replay_seconds = 0.0      # seconds per stored sample and new candidate

    def add(self, value: float) -> None:
        """One sample; see `add_many` for the batch form the run loops use."""
        self.add_many(np.array([value]))

    def add_many(self, values: np.ndarray) -> None:
        """A batch of samples in [0, 1], each staked from the state before the batch."""
        values = np.asarray(values, dtype=float).ravel()
        if values.size == 0:
            return
        stakes = self._stakes(values.size)
        up, down = _capital_gains(values, stakes, self._candidates)
        self._log_up += up
        self._log_down += down
        self.n += values.size
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        if not self._sealed:
            self._store(values, stakes)
            if self._stored >= self.REPLAY_SAMPLES_MAX:
                self._seal()

    def _stakes(self, size: int) -> np.ndarray:
        """The predictable stakes of the next `size` samples: variance-adapted with a
        prior of 1/4, the variance taken before the batch, sample `i` at index
        ``t = n + 1 + i``."""
        mean = self.total / self.n if self.n else 0.5
        variance = (0.25 + self.total_sq - 2 * mean * self.total
                    + self.n * mean * mean) / (self.n + 1)
        t = self.n + 1 + np.arange(size, dtype=float)
        return np.sqrt(2.0 * math.log(2.0 / self.budget.delta)
                       / (variance * t * np.log1p(t)))

    def _store(self, values: np.ndarray, stakes: np.ndarray) -> None:
        """`values` and their stakes appended to the history, which grows by doubling
        up to `REPLAY_SAMPLES_MAX` and past it by the last batch only."""
        end = self._stored + values.size
        capacity = self._history.shape[1]
        if end > capacity:
            grown = np.empty((2, max(end, min(max(2 * capacity, 1024),
                                               self.REPLAY_SAMPLES_MAX))))
            grown[:, :self._stored] = self._history[:, :self._stored]
            self._history = grown
        self._history[0, self._stored:end] = values
        self._history[1, self._stored:end] = stakes
        self._stored = end

    def _seal(self) -> None:
        """Nothing, having read the interval, laid candidates over the watched range and
        dropped the history; see the class docstring."""
        self.interval()
        self._refine_inside()
        self._history = np.empty((2, 0))
        self._stored = 0
        self._sealed = True

    def _refine_inside(self) -> None:
        """Candidates laid geometrically between the smallest positive and the largest
        watched candidate, one per cell of relative width `grain`, at most
        `SEAL_CANDIDATES_MAX`."""
        candidates = self._candidates
        positive = candidates[candidates > 0.0]
        if positive.size < 2:
            return
        low, high = float(positive[0]), float(positive[-1])
        cells = math.ceil(math.log(high / low) / math.log1p(self.grain))
        count = min(cells - 1, self.SEAL_CANDIDATES_MAX)
        if count < 1:
            return
        self._insert(np.setdiff1d(np.geomspace(low, high, count + 2)[1:-1], candidates))

    @property
    def reach(self) -> float:
        """About the number of samples beyond which the seal may not resolve a target;
        see the class docstring."""
        share = self.SEAL_CANDIDATES_MAX * BETTING_CELL_SHARE
        return share * share * self.REPLAY_SAMPLES_MAX

    @property
    def exhausted(self) -> bool:
        """Whether no further sample can narrow the interval: sealed, with at most one
        surviving candidate."""
        if not self._sealed:
            return False
        hedged = _hedged(self._log_up, self._log_down)
        return int(np.count_nonzero(hedged < math.log(1.0 / self.budget.delta))) <= 1

    def interval(self) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 1.0
        threshold = math.log(1.0 / self.budget.delta)
        # Each pass subdivides cells and rescans; the scan also runs once after the last
        # pass, so the endpoints are always read from the grid they index into.
        surviving = np.empty(0, dtype=np.intp)
        for passes in range(BETTING_MAX_PASSES + 1):
            hedged = _hedged(self._log_up, self._log_down)
            surviving = np.flatnonzero(hedged < threshold)
            if passes == BETTING_MAX_PASSES:
                break
            if surviving.size:
                first, last = int(surviving[0]), int(surviving[-1])
                refined = self._refine(first - 1) or self._refine(last)
            else:
                # The confidence set lies inside a cell next to the candidate of least
                # capital.  This only chooses where to look: the endpoints are still
                # read from the survivors.  The cell above the candidate is refined
                # first, which leaves the indices below it unchanged.
                least = int(np.argmin(hedged))
                refined = (self._refine(least, search=True)
                           | self._refine(least - 1, search=True))
            if not refined:
                break
        if surviving.size == 0:
            return self._low, max(self._high, self._low)
        first, last = int(surviving[0]), int(surviving[-1])
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

    def _refine(self, cell: int, *, search: bool = False) -> bool:
        """Whether the cell starting at candidate `cell` was subdivided.

        A cell at the cell width `grain` is left alone unless `search` is set; the cell
        above zero, a cell no float lies inside, and every cell once the stream is
        sealed are always left alone.
        """
        if self._sealed:
            return False
        candidates = self._candidates
        if cell < 0 or cell + 1 >= candidates.size:
            return False
        a, b = candidates[cell], candidates[cell + 1]
        # The cell above zero has no relative width to refine to; it stays at the
        # grid's resolution, which is far below any floor a query can ask for.
        if a == 0.0 or (not search and b - a <= self.grain * b):
            return False
        fine = np.unique(np.linspace(a, b, BETTING_SPLIT + 1)[1:-1])
        fine = fine[(a < fine) & (fine < b)]
        if fine.size == 0:
            return False
        self._insert(fine)
        return True

    def seal_seconds(self, more: int) -> float:
        """The estimated seconds of the seal that `more` further samples would start, 0
        where they start none: the candidates the seal would lay over the range watched
        now, each replayed over `REPLAY_SAMPLES_MAX` samples at the speed the stream's
        own replays measured."""
        if self._sealed or self._stored + more < self.REPLAY_SAMPLES_MAX:
            return 0.0
        return (self._seal_count() * self.REPLAY_SAMPLES_MAX
                * self._replay_seconds)

    def _seal_count(self) -> int:
        """The number of candidates `_refine_inside` would lay now."""
        positive = self._candidates[self._candidates > 0.0]
        if positive.size < 2:
            return 0
        cells = math.ceil(math.log(float(positive[-1]) / float(positive[0]))
                          / math.log1p(self.grain))
        return max(0, min(cells - 1, self.SEAL_CANDIDATES_MAX))

    def _insert(self, fine: np.ndarray) -> None:
        """New candidates `fine`, none already watched, merged into the watched set with
        their capital computed from the stored samples."""
        stored = self._history[:, :self._stored]
        started = time.perf_counter()
        up, down = _capital_gains(stored[0], stored[1], fine)
        work = stored.shape[1] * fine.size
        if work >= 1 << 16:
            # Small replays are dominated by fixed costs and would overstate the speed's
            # inverse; they leave the measured speed as it is.
            self._replay_seconds = (time.perf_counter() - started) / work
        merged = np.concatenate([self._candidates, fine])
        order = np.argsort(merged, kind="stable")
        self._candidates = merged[order]
        self._log_up = np.concatenate([self._log_up, up])[order]
        self._log_down = np.concatenate([self._log_down, down])[order]

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0


# Entries of the sample-by-candidate arrays `_capital_gains` builds at once, which bounds
# its memory (8 MB per array) on a long replay.
_GAINS_CHUNK = 1 << 20


def _capital_gains(values: np.ndarray, stakes: np.ndarray, candidates: np.ndarray):
    """The two capital processes' log gains over `values`, per candidate mean, each value
    staked at most at its entry of `stakes`."""
    up = np.zeros(candidates.size)
    down = np.zeros(candidates.size)
    if values.size == 0 or candidates.size == 0:
        return up, down
    with np.errstate(divide="ignore"):
        cap_up = 0.5 / np.maximum(candidates, 1e-300)
        cap_down = 0.5 / np.maximum(1.0 - candidates, 1e-300)
    rows = max(1, _GAINS_CHUNK // candidates.size)
    for start in range(0, values.size, rows):
        chunk = values[start:start + rows, None]
        stake = stakes[start:start + rows, None]
        centred = chunk - candidates[None, :]
        up += np.log1p(np.minimum(stake, cap_up[None, :]) * centred).sum(axis=0)
        down += np.log1p(-np.minimum(stake, cap_down[None, :]) * centred).sum(axis=0)
    return up, down


def _hedged(log_up: np.ndarray, log_down: np.ndarray) -> np.ndarray:
    """The log of the hedged average capital, computed stably from the two logs."""
    top = np.maximum(log_up, log_down)
    return top + np.log1p(np.exp(-np.abs(log_up - log_down))) - math.log(2.0)


def met_relative(low: float, high: float, rel_err: float, origin: float = 0.0) -> bool:
    """Whether an interval certifies relative error `rel_err`, measured from `origin`.

    The interval must lie on one side of `origin`, and its width be at most `rel_err`
    times the distance of its nearer end from `origin`.  A query whose returned value is
    ``a * x + b`` of the sampled quantity `x`, with ``a > 0``, meets relative error on
    the returned value's scale exactly where `x`'s interval meets it from
    ``origin = -b / a``.

        >>> met_relative(0.99, 1.0, 0.02), met_relative(0.0, 1e-9, 0.02)
        (True, False)
        >>> met_relative(0.519, 0.521, 0.01, origin=0.5)
        False

    An inverted interval certifies nothing:

        >>> met_relative(2e-6, 1e-6, 0.01)
        False
    """
    if not low <= high:
        return False
    if low > origin:
        return high - low <= rel_err * (low - origin)
    if high < origin:
        return high - low <= rel_err * (origin - high)
    return False


def met_floor(high: float, floor: float | None) -> bool:
    """Whether an interval certifies the value to lie below `floor`.

        >>> met_floor(1e-9, 1e-6), met_floor(1e-9, None)
        (True, False)
    """
    return floor is not None and high <= floor
