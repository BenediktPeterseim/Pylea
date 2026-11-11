"""End-to-end certified runs on small models with exact anchors.

Each test runs a query to certification and checks the interval against the exact
engine's answer.  Anchors use a small containment slack for floating-point round-off
only; the statistical margin is the certificate's own.
"""
from __future__ import annotations

import numpy as np

from test.sample.grid import grid_model
from pylea import DPP, Bool, flip, observe
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
