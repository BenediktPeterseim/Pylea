"""Mode selection and sizing decisions of the pilot."""
from __future__ import annotations

import math

import numpy as np

from pylea import DPP, Bool, flip, observe
from pylea.sample.cost_model import SWEEP_N_MIN
from pylea.sample.pilot import run_pilot
from pylea.sample.schedule import schedule
from pylea.sample.sweep import SweepProgram, unfold

EVENT = np.array([0.0, 1.0])


def report_for(model, rel_err: float = 0.01, seed: int = 0):
    body = schedule(unfold(model.program()))
    return run_pilot(SweepProgram(body), EVENT, rel_err=rel_err, confidence=0.95,
                     unfolded_ops=len(body.ops),
                     seed_sequence=np.random.SeedSequence(seed))


def test_the_mode_follows_the_smaller_predicted_particle_count():
    @DPP
    def staged() -> Bool:
        ok = Bool.const(True)
        for _ in range(6):
            block = flip(0.9) | flip(0.5)
            observe(block | flip(0.3))
            ok = ok & block
        return ok

    report = report_for(staged)
    assert not report.zero_hits
    assert report.predicted_iid < math.inf
    # The model is packable, so the i.i.d. count is charged the executor gap first.
    from pylea.sample.cost_model import PACKED_SPEEDUP

    cheaper = ("iid" if report.predicted_iid / PACKED_SPEEDUP <= report.predicted_sweep
               else "sweep")
    assert report.mode == cheaper


def test_an_unconditioned_query_routes_iid():
    @DPP
    def plain() -> Bool:
        return flip(0.3) & flip(0.6)

    report = report_for(plain)
    assert report.mode == "iid" and report.predicted_sweep == math.inf


def test_an_all_zero_pilot_reports_infinite_cost_not_failure():
    @DPP
    def hopeless() -> Bool:
        x = flip(1e-9)
        observe(x | flip(1e-9))
        return x

    report = report_for(hopeless)
    assert report.zero_hits
    assert report.predicted_iid == math.inf
    assert report.predicted_particles == math.inf


def test_a_mass_near_the_bottom_of_the_float_range_predicts_a_finite_or_infinite_cost():
    from pylea.sample.cost_model import iid_particles

    for mass in (1e-300, 1e-310):
        assert iid_particles(mass, 0.05, 0.05, 2**16) > 1e300
        assert iid_particles(mass, 0.05, 0.05, 2**16, relvar=0.5) > 1e300


def test_the_sweep_size_follows_four_r_hat_within_its_bounds():
    @DPP
    def rare_stages() -> Bool:
        x = flip(0.5)
        for _ in range(100):
            observe(flip(0.2) | flip(0.05))
        return x

    report = report_for(rare_stages)
    assert report.r_hat > 100.0
    expected = int(np.clip(4 * report.r_hat, SWEEP_N_MIN, None))
    assert report.n_particles == expected
    # Deep staged evidence starves the i.i.d. mode outright, so only sweeps survive.
    assert report.mode == "sweep" and report.predicted_iid == math.inf


def test_the_sweep_prediction_prices_the_binding_stream():
    # Common evidence, rare event: the denominator's per-sweep relative variance is two
    # orders below the numerator's, so a denominator-only price would miss the cost.
    @DPP
    def rare_under_common_evidence() -> Bool:
        x = flip(0.01)
        observe(flip(0.9) | x)
        return x

    report = report_for(rare_under_common_evidence)
    # The numerator's relative variance at 512 particles per sweep is about
    # (1/p - 1) / 512, an order of 0.2; the denominator's is under 0.001.
    assert report.relvar > 0.05


def test_pilot_choices_are_reproducible_per_seed():
    @DPP
    def staged() -> Bool:
        x = flip(0.4)
        observe(x | flip(0.5))
        return x

    first = report_for(staged, seed=7)
    second = report_for(staged, seed=7)
    assert first.mass == second.mass
    assert first.predicted_iid == second.predicted_iid
    assert first.n_particles == second.n_particles
