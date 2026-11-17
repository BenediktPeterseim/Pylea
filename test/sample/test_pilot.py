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
    assert report.predicted_particles < math.inf
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
    assert report.predicted_iid == math.inf
    assert report.predicted_particles == math.inf


def test_a_mass_near_the_bottom_of_the_float_range_predicts_a_finite_or_infinite_cost():
    from pylea.sample.cost_model import iid_particles

    for mass in (1e-300, 1e-310):
        assert iid_particles([(mass, None)], 0.05, 0.05, 2**16) > 1e300
        assert iid_particles([(mass, 0.5)], 0.05, 0.05, 2**16) > 1e300


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
    assert report.relvar_numerator > 0.05
    assert report.relvar_denominator < 0.01


def test_a_conditioned_query_is_priced_for_both_streams():
    from pylea.examples.monty_hall import car_behind_door_2
    from pylea.sample.cost_model import iid_particles

    report = report_for(car_behind_door_2)
    numerator_only = iid_particles([(report.mass, None)], 0.01, 0.05, report.batch)
    assert report.predicted_iid >= 3 * numerator_only


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


def test_a_pilot_whose_deadline_has_passed_draws_nothing():
    from pylea.sample.run import Deadline

    @DPP
    def coin() -> Bool:
        return flip(0.3)

    body = schedule(unfold(coin.program()))
    report = run_pilot(SweepProgram(body), EVENT, rel_err=0.01, confidence=0.95,
                       unfolded_ops=len(body.ops), seed_sequence=np.random.SeedSequence(0),
                       deadline=Deadline.after(0.0))
    assert report.drawn == 0
    assert report.predicted_particles == math.inf


def test_a_pilot_under_a_short_deadline_runs_only_its_calibration_batch():
    from pylea.sample.run import Deadline

    # A run of this DPP costs milliseconds whatever its size, so the deadline has
    # passed once the calibration batch is done.
    @DPP
    def chain() -> Bool:
        acc = flip(0.5)
        for _ in range(3000):
            acc = acc & (flip(0.999) | flip(0.5))
        return acc

    body = schedule(unfold(chain.program()))
    program = SweepProgram(body)
    report = run_pilot(program, EVENT, rel_err=0.01, confidence=0.95,
                       unfolded_ops=len(body.ops), seed_sequence=np.random.SeedSequence(0),
                       deadline=Deadline.after(0.002))
    assert report.drawn == 64
    unbounded = run_pilot(program, EVENT, rel_err=0.01, confidence=0.95,
                          unfolded_ops=len(body.ops),
                          seed_sequence=np.random.SeedSequence(0))
    assert unbounded.drawn == 2 * unbounded.batch


def test_the_table_price_grows_with_the_entry_count_and_falls_with_the_binding_mass():
    from pylea.sample.cost_model import table_particles

    def price(mass, entries):
        return table_particles((mass, None), (1.0, None), 0.05, 0.05, entries, 2**16,
                               normalised=True)

    assert price(0.1, 4) < price(0.1, 64) < price(0.1, 1024)
    assert price(0.1, 4) < price(0.01, 4) < price(0.001, 4)
    assert price(0.0, 4) == math.inf


def test_a_pilot_under_a_deadline_leaves_the_certified_run_its_share_of_the_time(
        monkeypatch):
    import time

    import pylea.sample.run as run_module
    from pylea.sample.cost_model import PILOT_DEADLINE_SHARE
    from pylea.sample.run import Deadline

    # Every batch costs 50 ms whatever its size, as on a large DPP, where the work per
    # assignment dominates; a pilot filling the whole deadline would run a second batch.
    class Slow:
        def __init__(self, executor):
            self.executor = executor

        def run(self, *args, **kwargs):
            time.sleep(0.05)
            return self.executor.run(*args, **kwargs)

    packed_or_byte = run_module.packed_or_byte
    monkeypatch.setattr(run_module, "packed_or_byte", lambda p: Slow(packed_or_byte(p)))

    @DPP
    def coin() -> Bool:
        return flip(0.3)

    body = schedule(unfold(coin.program()))
    limit = 0.12
    started = time.perf_counter()
    report = run_pilot(SweepProgram(body), EVENT, rel_err=0.01, confidence=0.95,
                       unfolded_ops=len(body.ops), seed_sequence=np.random.SeedSequence(0),
                       deadline=Deadline.after(limit))
    assert report.drawn == 64
    assert time.perf_counter() - started < PILOT_DEADLINE_SHARE * limit + 0.03
