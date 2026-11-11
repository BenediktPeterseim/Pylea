"""Coverage and stopping tests of the confidence sequences.

Coverage is tested at every accumulation step over seeded replications, for both stream
families by name: the Clopper-Pearson sequence of `BernoulliStream` and the
empirical-Bernstein sequences of `WeightedParticleStream` and `SweepStream`.  The bounds
are conservative, so with the fixed seeds the expected violation count is zero; the
default suite runs a reduced replication count and the full grid runs under the `slow`
marker.  A fixed-seed regression pins the relative-error stopping rule's sample count to
a narrow window, so a certifier change that silently loosens or tightens stopping shows
up here.
"""
from __future__ import annotations

import numpy as np
import pytest

from pylea.sample.certify import (
    BernoulliStream, BettingStream, Budget, Certificate, SweepStream,
    WeightedParticleStream, cp_interval, met_relative, ratio_interval,
)

DELTA = 0.05


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
        stream = BettingStream(Budget(DELTA))
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
    betting = BettingStream(Budget(DELTA))
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


def test_the_betting_stream_drops_its_replay_history_at_the_cap():
    rng = np.random.default_rng(9)
    stream = BettingStream(Budget(DELTA))
    for _ in range(BettingStream._REPLAY_BATCHES + 8):
        stream.add_many(rng.beta(2.0, 5.0, 8))
    low, high = stream.interval()
    assert not stream._batches          # sealed: memory no longer grows with the run
    assert low <= 2.0 / 7.0 <= high
    assert high - low < 0.05            # the boundary was refined before sealing


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


def test_the_betting_interval_contains_every_surviving_candidate_after_exhausting_refinement(
        monkeypatch):
    # With no granularity floor every pass refines, so the pass budget runs out; the
    # endpoints must still be read from the grid as refined, not from a stale scan.
    monkeypatch.setattr(BettingStream, "_GRAIN", 0.0)
    rng = np.random.default_rng(14)
    stream = BettingStream(Budget(DELTA))
    for _ in range(6):
        stream.add_many(rng.beta(2.0, 5.0, 256))
        low, high = stream.interval()
        alive = surviving_candidates(stream)
        assert alive.size and low <= alive.min() and alive.max() <= high


def test_the_betting_grid_keeps_only_the_running_intersection():
    rng = np.random.default_rng(15)
    stream = BettingStream(Budget(DELTA))
    low, high = 0.0, 1.0
    for _ in range(4):
        stream.add_many(rng.beta(2.0, 5.0, 1024))
        low, high = stream.interval()
    assert stream._candidates.size < 100
    assert stream._candidates[0] <= low and high <= stream._candidates[-1]
    assert low <= 2.0 / 7.0 <= high


def test_the_cell_above_zero_is_never_refined():
    stream = BettingStream(Budget(DELTA))
    low = 1.0
    for _ in range(8):
        stream.add_many(np.zeros(1024))
        low, _high = stream.interval()
    assert low == 0.0 and stream._candidates[0] == 0.0
    assert stream._candidates[1] == BettingStream._GRID[1]
