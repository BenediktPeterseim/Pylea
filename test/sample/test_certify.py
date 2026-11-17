"""Coverage and stopping tests of the confidence sequences.

Coverage is tested at every accumulation step over seeded replications, for both stream
families by name: the Clopper-Pearson sequence of `BernoulliStream` and the
empirical-Bernstein sequences of `WeightedParticleStream` and of `SweepStream`, a
checkpointed sequence over sweeps defined here as the reference for the betting stream.
The bounds are conservative, so with the fixed seeds the expected violation count is zero;
the default suite runs a reduced replication count and the full grid runs under the `slow`
marker.  A fixed-seed regression pins the relative-error stopping rule's sample count to a
narrow window, so a certifier change that silently loosens or tightens stopping shows up
here.
"""
from __future__ import annotations

import numpy as np
import pytest

from pylea.sample.certify import (
    BernoulliStream, BettingStream, Budget, Certificate, WeightedParticleStream,
    cp_bounds, cp_interval, eb_interval, met_relative, ratio_interval,
)

DELTA = 0.05


class SweepStream:
    """I.i.d. sweep values in [0, 1], certified by empirical Bernstein per sweep: the
    checkpointed sequence the betting stream is compared against."""

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


def bernoulli_violations(mu: float, reps: int, steps: int, batch: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    violations = 0
    for _ in range(reps):
        stream = BernoulliStream(Budget(DELTA))
        for _ in range(steps):
            stream.add(batch, int((rng.random(batch) < mu).sum()))
            low, high = stream.interval()
            if not low <= mu <= high:
                violations += 1
                break
    return violations


def weighted_violations(draw, mu: float, reps: int, steps: int, batch: int,
                        seed: int) -> int:
    rng = np.random.default_rng(seed)
    violations = 0
    for _ in range(reps):
        stream = WeightedParticleStream(Budget(DELTA))
        for _ in range(steps):
            values = draw(rng, batch)
            stream.add(batch, float(values.sum()), float((values * values).sum()))
            low, high = stream.interval()
            if not low <= mu <= high:
                violations += 1
                break
    return violations


def sweep_violations(draw, mu: float, reps: int, sweeps: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    violations = 0
    for _ in range(reps):
        stream = SweepStream(Budget(DELTA))
        for value in draw(rng, sweeps):
            stream.add(float(value))
            low, high = stream.interval()
            if not low <= mu <= high:
                violations += 1
                break
    return violations


# A valid certificate at delta 0.05 allows a 5% violation rate per replication, so the
# thresholds below are set where the null's binomial tail falls under 1e-6, per the
# statistical-test policy; the empirical-Bernstein streams are conservative enough that
# their observed counts are zero, while exact Clopper-Pearson uses more of its budget.
def _allowed(reps: int) -> int:
    return {100: 20, 200: 30, 1000: 90}[reps]


def test_the_clopper_pearson_sequence_covers_at_every_step():
    assert bernoulli_violations(0.3, reps=200, steps=12, batch=64, seed=0) <= _allowed(200)
    assert bernoulli_violations(0.02, reps=200, steps=12, batch=64, seed=1) <= _allowed(200)


def test_the_particle_bernstein_sequence_covers_at_every_step():
    uniform = lambda rng, n: 0.5 + 0.02 * (2.0 * rng.random(n) - 1.0)  # noqa: E731
    bernoulli = lambda rng, n: (rng.random(n) < 0.3).astype(float)     # noqa: E731
    assert weighted_violations(uniform, 0.5, reps=200, steps=12, batch=64,
                               seed=2) <= _allowed(200)
    assert weighted_violations(bernoulli, 0.3, reps=200, steps=12, batch=64,
                               seed=3) <= _allowed(200)


def test_the_sweep_bernstein_sequence_covers_at_every_step():
    beta_like = lambda rng, n: rng.beta(2.0, 5.0, n)                   # noqa: E731
    assert sweep_violations(beta_like, 2.0 / 7.0, reps=200, sweeps=256,
                            seed=4) <= _allowed(200)


def betting_violations(draw, mu: float, reps: int, steps: int, batch: int,
                       seed: int) -> int:
    rng = np.random.default_rng(seed)
    violations = 0
    for _ in range(reps):
        stream = BettingStream(Budget(DELTA), rel_err=0.01)
        for _ in range(steps):
            stream.add_many(draw(rng, batch))
            low, high = stream.interval()
            if not low <= mu <= high:
                violations += 1
                break
    return violations


def test_the_betting_sequence_covers_at_every_step():
    beta_like = lambda rng, n: rng.beta(2.0, 5.0, n)                   # noqa: E731
    uniform = lambda rng, n: 0.5 + 0.02 * (2.0 * rng.random(n) - 1.0)  # noqa: E731
    bernoulli = lambda rng, n: (rng.random(n) < 0.02).astype(float)    # noqa: E731
    assert betting_violations(beta_like, 2.0 / 7.0, reps=100, steps=8, batch=64,
                              seed=5) <= _allowed(100)
    assert betting_violations(uniform, 0.5, reps=100, steps=8, batch=64,
                              seed=6) <= _allowed(100)
    assert betting_violations(bernoulli, 0.02, reps=100, steps=8, batch=64,
                              seed=7) <= _allowed(100)


def test_the_betting_sequence_is_tighter_than_the_checkpointed_bernstein():
    rng = np.random.default_rng(8)
    betting = BettingStream(Budget(DELTA), rel_err=0.01)
    bernstein = SweepStream(Budget(DELTA))
    for _ in range(16):
        values = rng.beta(2.0, 5.0, 256)
        betting.add_many(values)
        for value in values:
            bernstein.add(float(value))
    b_low, b_high = betting.interval()
    e_low, e_high = bernstein.interval()
    assert b_high - b_low < e_high - e_low
    assert b_low <= 2.0 / 7.0 <= b_high


class EarlySeal(BettingStream):
    """A betting stream that seals after 4096 stored samples."""

    __slots__ = ()
    REPLAY_SAMPLES_MAX = 4096


def test_the_betting_stream_drops_its_replay_history_at_the_cap():
    rng = np.random.default_rng(9)
    stream = EarlySeal(Budget(DELTA), rel_err=0.01)
    for _ in range(4096 // 8 + 8):
        stream.add_many(rng.beta(2.0, 5.0, 8))
    low, high = stream.interval()
    assert stream._sealed and stream._history.size == 0
    assert low <= 2.0 / 7.0 <= high
    assert high - low < 0.05


def test_the_stake_falls_with_the_sample_index_inside_a_batch():
    rng = np.random.default_rng(16)
    stream = BettingStream(Budget(DELTA), rel_err=0.01)
    stream.add_many(rng.beta(2.0, 5.0, 100))
    stakes = stream._stakes(4)
    assert np.all(np.diff(stakes) < 0.0)
    mean = stream.total / stream.n
    variance = (0.25 + stream.total_sq - 2 * mean * stream.total
                + stream.n * mean * mean) / (stream.n + 1)
    t = stream.n + 1
    first = np.sqrt(2.0 * np.log(2.0 / DELTA) / (variance * t * np.log1p(t)))
    assert stakes[0] == pytest.approx(first, rel=1e-12)


def watched_cells(stream: BettingStream) -> np.ndarray:
    """The relative widths of the watched cells above zero."""
    candidates = stream._candidates[stream._candidates > 0.0]
    return np.diff(candidates) / candidates[:-1]


def test_sealing_resolves_every_cell_inside_the_interval():
    rng = np.random.default_rng(17)
    stream = EarlySeal(Budget(DELTA), rel_err=0.01)
    while not stream._sealed:
        stream.add_many(rng.beta(2.0, 5.0, 256))
        stream.interval()
    assert watched_cells(stream).max() <= stream.grain * (1 + 1e-9)

    class FewCandidates(EarlySeal):
        __slots__ = ()
        SEAL_CANDIDATES_MAX = 16

    stream = FewCandidates(Budget(DELTA), rel_err=0.01)
    stream.add_many(rng.beta(2.0, 5.0, 256))
    low, high = stream.interval()
    while not stream._sealed:
        stream.add_many(rng.beta(2.0, 5.0, 256))
    # Where the cap binds, the candidates are spread geometrically over the range the
    # stream watched, at most `SEAL_CANDIDATES_MAX` of them.
    positive = stream._candidates[stream._candidates > 0.0]
    spread = (positive[-1] / positive[0]) ** (1 / 17) - 1
    assert watched_cells(stream).max() <= spread * (1 + 1e-9)
    assert spread > stream.grain


def test_a_sealed_betting_stream_keeps_narrowing():
    # The mean sits in the middle of a cell of the coarse grid, so an interval whose
    # ends could only fall on that grid would stay at least 0.68% wide.
    grid = BettingStream._GRID
    above = int(np.searchsorted(grid, 1.0 / 3.0))
    mean = float(np.sqrt(grid[above - 1] * grid[above]))

    class SealAfterEightBatches(BettingStream):
        __slots__ = ()
        REPLAY_SAMPLES_MAX = 8 * 1024

    rng = np.random.default_rng(3)
    stream = SealAfterEightBatches(Budget(DELTA), rel_err=0.01)
    low, high = 0.0, 1.0
    while stream.n < 1 << 18:
        stream.add_many(rng.uniform(mean - 0.1, mean + 0.1, 1024))
        low, high = stream.interval()
    assert stream._sealed
    assert low <= mean <= high
    assert (high - low) / low < 0.003


def test_the_interval_does_not_depend_on_how_often_it_is_read():
    # A stream of small variance: between two rare readings its confidence set shrinks
    # to inside one cell of the grid, where no candidate survives.
    def final_interval(read_after) -> tuple[float, float]:
        rng = np.random.default_rng(0)
        stream = BettingStream(Budget(DELTA / 2), rel_err=0.01)
        for batch in range(1, 401):
            stream.add_many(rng.uniform(0.2999, 0.3001, 4096))
            if read_after(batch):
                stream.interval()
        return stream.interval()

    always = final_interval(lambda batch: True)
    rarely = final_interval(lambda batch: batch in (1, 4, 16, 64, 256))
    assert always == rarely
    low, high = rarely
    assert low <= 0.3 <= high and (high - low) / low < 1e-4


def test_the_cell_width_follows_the_relative_target():
    for rel_err in (1e-2, 1e-3):
        rng = np.random.default_rng(18)
        stream = BettingStream(Budget(DELTA), rel_err=rel_err)
        for _ in range(16):
            stream.add_many(rng.beta(2.0, 5.0, 1024))
            stream.interval()
        assert surviving_candidates(stream).size >= 2
        # The cells holding the two ends of the interval are refined to the cell width.
        cells = watched_cells(stream)
        assert stream.grain == rel_err / 50
        assert cells[0] <= stream.grain and cells[-1] <= stream.grain


def test_the_betting_relative_error_stopping_count_is_pinned():
    rng = np.random.default_rng(6)
    stream = BettingStream(Budget(DELTA), rel_err=0.05)
    batch = 1024
    while True:
        stream.add_many(rng.beta(2.0, 5.0, batch))
        low, high = stream.interval()
        if met_relative(low, high, 0.05):
            break
    assert low <= 2.0 / 7.0 <= high
    # The window is the regression, as for the Clopper-Pearson stream.
    assert 6 * batch <= stream.n <= 9 * batch


def test_a_sealed_stream_with_one_surviving_candidate_is_exhausted():
    class Tiny(BettingStream):
        __slots__ = ()
        REPLAY_SAMPLES_MAX = 1024
        SEAL_CANDIDATES_MAX = 4

    rng = np.random.default_rng(19)
    stream = Tiny(Budget(DELTA), rel_err=0.01)
    assert not stream.exhausted
    while not stream.exhausted and stream.n < 1 << 22:
        stream.add_many(rng.beta(2.0, 5.0, 1024))
        stream.interval()
    assert stream.exhausted and stream._sealed
    low, high = stream.interval()
    stream.add_many(rng.beta(2.0, 5.0, 1 << 16))
    assert stream.interval() == (low, high)
    assert low <= 2.0 / 7.0 <= high


@pytest.mark.slow
def test_a_stream_fed_one_value_at_a_time_seals_on_samples_and_keeps_narrowing():
    # Relative variance about 12; `run_sweeps` feeds one value per call where a run has
    # more than 4096 particles.
    p, value = 0.0787, 0.01
    rng = np.random.default_rng(0)
    stream = BettingStream(Budget(DELTA / 2), rel_err=0.01)
    for _ in range(1025):
        stream.add_many((rng.random(1) < p) * value)
        stream.interval()
    assert not stream._sealed
    while not stream._sealed:
        stream.add_many((rng.random(1024) < p) * value)
        if not stream._sealed:
            stream.interval()
    assert watched_cells(stream).max() <= stream.grain * (1 + 1e-9)
    while stream.n < 1.7e7:
        stream.add_many((rng.random(1 << 16) < p) * value)
    low, high = stream.interval()
    assert low <= p * value <= high
    assert (high - low) / low < 0.01


@pytest.mark.slow
def test_the_betting_sequence_covers_at_every_step_after_sealing():
    class SealAfterEightBatches(BettingStream):
        __slots__ = ()
        REPLAY_SAMPLES_MAX = 8 * 64

    def violations(draw, mu: float, seed: int) -> int:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(1000):
            stream = SealAfterEightBatches(Budget(DELTA), rel_err=0.01)
            for _ in range(24):
                stream.add_many(draw(rng, 64))
                low, high = stream.interval()
                if not low <= mu <= high:
                    count += 1
                    break
        return count

    beta_like = lambda rng, n: rng.beta(2.0, 5.0, n)                   # noqa: E731
    uniform = lambda rng, n: 0.5 + 0.02 * (2.0 * rng.random(n) - 1.0)  # noqa: E731
    bernoulli = lambda rng, n: (rng.random(n) < 0.02).astype(float)    # noqa: E731
    assert violations(beta_like, 2.0 / 7.0, seed=20) <= _allowed(1000)
    assert violations(uniform, 0.5, seed=21) <= _allowed(1000)
    assert violations(bernoulli, 0.02, seed=22) <= _allowed(1000)


@pytest.mark.slow
def test_every_stream_family_covers_on_the_full_replication_grid():
    assert bernoulli_violations(0.3, reps=1000, steps=14, batch=64,
                                seed=10) <= _allowed(1000)
    assert bernoulli_violations(0.02, reps=1000, steps=14, batch=64,
                                seed=11) <= _allowed(1000)
    uniform = lambda rng, n: 0.5 + 0.02 * (2.0 * rng.random(n) - 1.0)  # noqa: E731
    assert weighted_violations(uniform, 0.5, reps=1000, steps=14, batch=64,
                               seed=12) <= _allowed(1000)
    beta_like = lambda rng, n: rng.beta(2.0, 5.0, n)                   # noqa: E731
    assert sweep_violations(beta_like, 2.0 / 7.0, reps=1000, sweeps=512,
                            seed=13) <= _allowed(1000)


def test_bernstein_width_adapts_to_a_low_variance_stream():
    rng = np.random.default_rng(5)
    stream = WeightedParticleStream(Budget(DELTA))
    values = 0.5 + 0.02 * (2.0 * rng.random(8192) - 1.0)
    stream.add(8192, float(values.sum()), float((values * values).sum()))
    low, high = stream.interval()
    hoeffding = 2.0 * np.sqrt(np.log(2.0 / (DELTA / 2)) / (2 * 8192))
    assert high - low < hoeffding / 3


def test_zero_hits_yield_an_upper_bound_scaling_like_log_over_n():
    for n in (10_000, 100_000, 1_000_000):
        low, high = cp_interval(n, 0, DELTA)
        assert low == 0.0
        assert 2.0 / n < high < 6.0 / n


def test_the_relative_error_stopping_count_is_pinned():
    rng = np.random.default_rng(6)
    stream = BernoulliStream(Budget(DELTA))
    batch = 4096
    while True:
        stream.add(batch, int((rng.random(batch) < 0.3).sum()))
        low, high = stream.interval()
        if met_relative(low, high, 0.05):
            break
    assert abs(stream.mean - 0.3) <= 0.05 * 0.3
    # The window is the regression: a certifier change that moves stopping by more than
    # one batch either loosened the bound or pays extra samples, and must be deliberate.
    assert 11 * batch <= stream.n <= 16 * batch


def test_the_ledger_records_splits_and_refuses_overspending():
    ledger = Certificate(0.9)
    ledger.share("numerator", 0.5)
    ledger.share("denominator", 0.5)
    assert len(ledger.splits) == 2
    with pytest.raises(ValueError):
        ledger.share("extra", 0.1)


def test_a_zero_denominator_bound_yields_the_vacuous_ratio():
    assert ratio_interval((0.1, 0.2), (0.0, 0.5)) == (0.0, 1.0)
    assert ratio_interval((0.1, 0.2), (0.4, 0.5)) == (0.2, 0.5)


def surviving_candidates(stream: BettingStream) -> np.ndarray:
    from pylea.sample.certify import _hedged

    hedged = _hedged(stream._log_up, stream._log_down)
    return stream._candidates[hedged < np.log(1.0 / DELTA)]


def test_the_betting_interval_contains_every_surviving_candidate_after_exhausting_refinement():
    # With no cell width to stop at every pass refines, so the pass budget runs out;
    # the endpoints must still be read from the grid as refined, not from a stale scan.
    rng = np.random.default_rng(14)
    stream = BettingStream(Budget(DELTA), rel_err=0.01)
    stream.grain = 0.0
    for _ in range(6):
        stream.add_many(rng.beta(2.0, 5.0, 256))
        low, high = stream.interval()
        alive = surviving_candidates(stream)
        assert alive.size and low <= alive.min() and alive.max() <= high


def test_the_betting_grid_keeps_only_the_running_intersection():
    rng = np.random.default_rng(15)
    stream = BettingStream(Budget(DELTA), rel_err=0.01)
    low, high = 0.0, 1.0
    for _ in range(4):
        stream.add_many(rng.beta(2.0, 5.0, 1024))
        low, high = stream.interval()
    assert stream._candidates.size < 100
    assert stream._candidates[0] <= low and high <= stream._candidates[-1]
    assert low <= 2.0 / 7.0 <= high


def test_the_cell_above_zero_is_never_refined():
    stream = BettingStream(Budget(DELTA), rel_err=0.01)
    low = 1.0
    for _ in range(8):
        stream.add_many(np.zeros(1024))
        low, _high = stream.interval()
    assert low == 0.0 and stream._candidates[0] == 0.0
    assert stream._candidates[1] == BettingStream._GRID[1]


@pytest.mark.parametrize("n, k, delta_j", [
    (968359936, 999, 1.46e-8), (968359936, 1000, 1.46e-8), (968359936, 1001, 1.46e-8),
    (10_000_004_946, 4947, 0.025), (1_000_000_002_213, 2214, 2e-9),
    (1_000_000_002_144, 2145, 0.025),
])
def test_clopper_pearson_bounds_hold_where_the_beta_quantile_is_wrong(n, k, delta_j):
    """scipy 1.18.1 returns a wrong quantile at the first three shapes, scipy 1.16.0 at
    the last three."""
    from scipy.special import betainc, betaincc

    low, high = cp_interval(n, k, delta_j)
    assert low < k / n < high
    assert betainc(k, n - k + 1, low) == pytest.approx(delta_j / 2, rel=1e-5)
    assert betainc(k, n - k + 1, low) <= delta_j / 2
    assert betaincc(k + 1, n - k, high) <= delta_j / 2


def test_every_clopper_pearson_bound_brackets_the_hit_rate():
    from scipy.special import betainc, betaincc

    for n in (10_000, 1_000_000, 150_000_000, 1_000_000_000, 4_000_000_000):
        hits = np.array([0, 1, 10, 999, 1000, 1001, 10_000, n - 1, n])
        hits = hits[hits <= n]
        for delta_j in (0.05, 1e-6, 1e-12):
            low, high = cp_bounds(n, hits, delta_j)
            assert np.all(low <= hits / n) and np.all(hits / n <= high)
            inner = (hits > 0)
            assert np.all(betainc(hits[inner], n - hits[inner] + 1, low[inner])
                          <= delta_j / 2)
            inner = (hits < n)
            assert np.all(betaincc(hits[inner] + 1, n - hits[inner], high[inner])
                          <= delta_j / 2)


def test_a_wrong_library_quantile_is_replaced_by_bisection(monkeypatch):
    import scipy.special
    from scipy.special import betainc, betaincc

    n, delta_j = 1_000_000, 1e-6
    hits = np.array([3, 30, 300, 3000, 30_000, n - 1])
    want_low, want_high = cp_bounds(n, hits, delta_j)
    inverse, complement = scipy.special.betaincinv, scipy.special.betainccinv

    def spoil(values):
        values = np.array(values, dtype=float, copy=True)
        values.flat[::3] *= 2.0
        values.flat[1::3] = np.nan
        return values

    monkeypatch.setattr(scipy.special, "betaincinv", lambda *a: spoil(inverse(*a)))
    monkeypatch.setattr(scipy.special, "betainccinv", lambda *a: spoil(complement(*a)))
    low, high = cp_bounds(n, hits, delta_j)
    np.testing.assert_allclose(low, want_low, rtol=1e-5)
    np.testing.assert_allclose(high, want_high, rtol=1e-5)
    assert np.all(betainc(hits, n - hits + 1, low) <= delta_j / 2)
    assert np.all(betaincc(hits + 1, n - hits, high) <= delta_j / 2)


def test_an_inverted_interval_never_meets_the_relative_target():
    assert not met_relative(2e-6, 1e-6, 0.01)


def test_combining_an_inverted_interval_raises():
    from pylea.sample.run import _combine

    class Inverted:
        mean = 0.5

        def interval(self):
            return 0.6, 0.4

    with pytest.raises(RuntimeError, match="invariant"):
        _combine(None, Inverted(), None)
    with pytest.raises(RuntimeError, match="invariant"):
        _combine(None, Inverted(), Inverted())


def test_a_tail_below_the_accuracy_of_the_beta_function_is_refused():
    from pylea.sample.certify import _beta_quantile

    with pytest.raises(ValueError, match="not accurate"):
        _beta_quantile(np.array([10]), np.array([9991]), 1e-300, upper=True)


def test_the_seal_estimate_is_zero_until_a_batch_would_start_the_seal():
    from pylea.sample.certify import BettingStream, Budget

    class SmallCap(BettingStream):
        __slots__ = ()
        REPLAY_SAMPLES_MAX = 1 << 14

    stream = SmallCap(Budget(0.05), 0.01)
    rng = np.random.default_rng(0)
    for _ in range(15):
        stream.add_many(rng.random(1 << 10) * 0.6)
        stream.interval()
    assert stream.seal_seconds(1 << 9) == 0.0          # 15 * 2**10 + 2**9 < 2**14
    assert stream.seal_seconds(1 << 10) > 0.0
    stream.add_many(rng.random(1 << 10) * 0.6)
    assert stream.seal_seconds(1 << 10) == 0.0         # sealed
