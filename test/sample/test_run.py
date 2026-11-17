"""End-to-end certified runs on small models with exact anchors.

Each test runs a query to certification and checks the interval against the exact
engine's answer.  Anchors use a small containment slack for floating-point round-off
only; the statistical margin is the certificate's own.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from test.sample.grid import grid_model
from pylea import DPP, Bool, flip, observe
from pylea.sample.cost_model import SWEEP_BATCH_TARGET
from pylea.sample.run import Deadline, RunOutcome, run_iid, run_sweeps
from pylea.sample.sweep import SweepProgram, unfold

EVENT = np.array([0.0, 1.0])


def test_certified_prob_on_the_grid_family_contains_the_exact_answer():
    model = grid_model(8)
    exact = float(model.compiled().distribution()[1])
    outcome = run_iid(SweepProgram(unfold(model.program())), EVENT, rel_err=0.05,
                      confidence=0.95, batch=1 << 14,
                      seed_sequence=np.random.SeedSequence(0))
    assert outcome.target_met
    assert outcome.low - 1e-12 <= exact <= outcome.high + 1e-12
    assert outcome.high - outcome.low <= 0.05 * outcome.low


@DPP
def either() -> Bool:
    a, b = flip(0.5), flip(0.5)
    observe(a | b)
    return a


def test_a_conditioned_query_certifies_the_quotient_in_iid_mode():
    outcome = run_iid(SweepProgram(unfold(either.program())), EVENT, rel_err=0.05,
                      confidence=0.95, batch=1 << 14,
                      seed_sequence=np.random.SeedSequence(1))
    assert outcome.target_met
    assert outcome.low - 1e-12 <= 2.0 / 3.0 <= outcome.high + 1e-12


def test_a_conditioned_query_certifies_the_quotient_in_sweep_mode():
    outcome = run_sweeps(SweepProgram(unfold(either.program())), EVENT, rel_err=0.3,
                         confidence=0.95, n_particles=1024,
                         seed_sequence=np.random.SeedSequence(2))
    assert outcome.target_met
    assert outcome.low - 1e-12 <= 2.0 / 3.0 <= outcome.high + 1e-12


def test_an_expired_deadline_returns_the_standing_interval_unmet():
    model = grid_model(6)
    outcome = run_iid(SweepProgram(unfold(model.program())), EVENT, rel_err=1e-4,
                      confidence=0.95, batch=1 << 12,
                      seed_sequence=np.random.SeedSequence(3),
                      deadline=Deadline.after(0.0))
    assert isinstance(outcome, RunOutcome)
    assert not outcome.target_met
    assert 0.0 <= outcome.low <= outcome.high <= 1.0
    exact = float(model.compiled().distribution()[1])
    assert outcome.low - 1e-12 <= exact <= outcome.high + 1e-12


def test_the_floor_contract_answers_a_threshold_question_on_a_rare_event():
    @DPP
    def rare() -> Bool:
        return flip(1e-6) & flip(0.5)

    outcome = run_iid(SweepProgram(unfold(rare.program())), EVENT, rel_err=0.05,
                      confidence=0.95, batch=1 << 14, floor=1e-3,
                      seed_sequence=np.random.SeedSequence(4))
    assert outcome.target_met
    assert outcome.low == 0.0 and outcome.high <= 1e-3


def test_a_soft_weight_model_certifies_through_the_weighted_stream():
    from pylea import LikelihoodTable

    channel = LikelihoodTable((Bool,), (Bool,), np.array([[0.7, 0.4], [0.1, 0.4]]))

    @DPP
    def soft() -> Bool:
        return channel(flip(0.5))

    program = SweepProgram(unfold(soft.program()))
    assert not program.binary_weights and program.conditions
    outcome = run_iid(program, EVENT, rel_err=0.05, confidence=0.95, batch=1 << 14,
                      seed_sequence=np.random.SeedSequence(5))
    exact_table = soft.evaluated().matrix[:, 0]
    exact = exact_table[1] / exact_table.sum()
    assert outcome.target_met
    assert outcome.low - 1e-12 <= exact <= outcome.high + 1e-12


def test_a_run_with_no_samples_reports_the_middle_of_the_vacuous_interval():
    outcome = run_iid(SweepProgram(unfold(either.program())), EVENT, rel_err=0.05,
                      confidence=0.95, batch=1 << 14,
                      seed_sequence=np.random.SeedSequence(0),
                      deadline=Deadline.after(0.0))
    assert (outcome.point, outcome.low, outcome.high) == (0.5, 0.0, 1.0)
    assert not outcome.target_met


@DPP
def soft() -> Bool:
    a = flip(0.3)
    observe(a, if_true=0.9, if_false=0.2)
    return a


@DPP
def rare_either() -> Bool:
    a, b = flip(0.05), flip(0.1)
    observe(a | b)
    return a


def test_the_point_estimate_always_lies_in_the_interval():
    from pylea.sample.run import run_table

    for dpp in (either, soft, rare_either):
        program = SweepProgram(unfold(dpp.program()))
        for seed in range(20):
            for outcome in (
                    run_iid(program, EVENT, rel_err=0.1, confidence=0.95, batch=256,
                            seed_sequence=np.random.SeedSequence(seed)),
                    run_sweeps(program, EVENT, rel_err=0.1, confidence=0.95,
                               n_particles=64, seed_sequence=np.random.SeedSequence(seed),
                               deadline=Deadline.after(0.05))):
                assert outcome.low <= outcome.point <= outcome.high
            table = run_table(program, rel_err=0.1, confidence=0.95, floor=0.01,
                              seed_sequence=np.random.SeedSequence(seed), batch=256,
                              deadline=Deadline.after(0.05))
            assert np.all((table.lower <= table.point) & (table.point <= table.upper))
            assert table.total_low <= table.total_point <= table.total_high


def test_monty_hall_in_sweep_mode_reaches_one_percent():
    from pylea.examples.monty_hall import car_behind_door_2
    from pylea.sample.query import prepared

    program, _ops = prepared(car_behind_door_2)
    outcome = run_sweeps(program, EVENT, rel_err=0.01, confidence=0.95, n_particles=8,
                         seed_sequence=np.random.SeedSequence(0))
    assert outcome.target_met and outcome.n_samples <= 3e7
    assert outcome.low - 1e-12 <= 2.0 / 3.0 <= outcome.high + 1e-12


def test_an_exhausted_stream_ends_the_run_with_a_missed_target(monkeypatch):
    from pylea.sample.certify import BettingStream

    monkeypatch.setattr(BettingStream, "SEAL_CANDIDATES_MAX", 4)
    monkeypatch.setattr(BettingStream, "REPLAY_SAMPLES_MAX", 4096)
    outcome = run_sweeps(SweepProgram(unfold(either.program())), EVENT, rel_err=0.01,
                         confidence=0.95, n_particles=8,
                         seed_sequence=np.random.SeedSequence(0))
    assert not outcome.target_met
    assert outcome.low - 1e-12 <= 2.0 / 3.0 <= outcome.high + 1e-12
    assert outcome.trace[0].endswith("target not met")
    assert outcome.trace[1].startswith(
        "sampling: stopped; this run cannot narrow the interval to the target of 0.01, "
        "which would need more than about ")


class FixedStream:
    """A stand-in stream with a fixed interval."""

    def __init__(self, low: float, high: float, exhausted: bool):
        self.low, self.high, self.exhausted = low, high, exhausted

    def interval(self) -> tuple[float, float]:
        return self.low, self.high


def test_a_target_is_out_of_reach_only_where_an_exhausted_stream_rules_it_out():
    from pylea.sample.run import _unreachable

    wide = FixedStream(0.30, 0.31, exhausted=True)            # 3.3% wide, for good
    narrow = FixedStream(0.300, 0.3001, exhausted=True)
    free = FixedStream(0.49, 0.51, exhausted=False)
    assert _unreachable(wide, None, 0.01, None)
    assert _unreachable(wide, free, 0.01, None)
    assert not _unreachable(narrow, free, 0.01, None)
    assert not _unreachable(free, None, 0.01, None)
    assert not _unreachable(free, FixedStream(0.49, 0.51, exhausted=False), 0.01, None)
    # The floor stays reachable while the free denominator may still rise to 0.51.
    assert not _unreachable(wide, free, 0.01, 0.61)
    assert _unreachable(wide, free, 0.01, 0.60)
    # An exhausted denominator 4% wide rules out a quotient within 1%.
    assert _unreachable(FixedStream(0.3, 0.31, exhausted=False),
                        FixedStream(0.50, 0.52, exhausted=True), 0.01, None)
    assert not _unreachable(FixedStream(0.3, 0.31, exhausted=False),
                            FixedStream(0.500, 0.5001, exhausted=True), 0.01, None)


@pytest.mark.slow
def test_every_seed_of_monty_hall_terminates_by_sampling():
    from pylea import prob
    from pylea.examples.monty_hall import car_behind_door_2

    for seed in range(40):
        p = prob(car_behind_door_2, method="smc", seed=seed, max_time=30)
        assert p.target_met, seed
        assert p.low - 1e-12 <= 2.0 / 3.0 <= p.high + 1e-12


@pytest.mark.slow
def test_a_prepared_large_dpp_respects_a_one_second_deadline_within_half_a_second():
    import time
    import warnings

    from pylea import prob
    from pylea.sample.query import prepared

    @DPP
    def chain() -> Bool:
        acc = flip(0.5)
        for _ in range(30000):
            acc = acc & (flip(0.999) | flip(0.5))
        return acc

    prepared(chain)
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = prob(chain, method="smc", max_time=1.0, seed=0)
    took = time.monotonic() - started
    assert took < 1.5, (took, p.n_samples, p.trace)
    assert p.low <= float(p) <= p.high



def test_a_run_with_resampling_reads_its_interval_a_logarithmic_number_of_times(
        monkeypatch):
    import pylea.sample.run as run_module

    readings = []
    combine = run_module._combine

    def counted(*args):
        readings.append(1)
        return combine(*args)

    monkeypatch.setattr(run_module, "_combine", counted)
    outcome = run_sweeps(SweepProgram(unfold(soft.program())), EVENT, rel_err=0.02,
                         confidence=0.95, n_particles=8,
                         seed_sequence=np.random.SeedSequence(0))
    assert outcome.target_met
    steps = outcome.n_samples / (8 * max(1, SWEEP_BATCH_TARGET // 8))
    # One reading per growth of an eighth: about log(steps) / log(9 / 8), plus the first.
    assert len(readings) <= math.log(max(steps, 1.0)) / math.log(9 / 8) + 2
