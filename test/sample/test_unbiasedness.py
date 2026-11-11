"""Differential tests of the particle interpreter against the reference semantics.

Sweep means over many replications must match `pylea.ir.reference.semantics` entry for
entry, within a tolerance of five standard errors plus a floor absorbing entries whose
standard-error estimate is itself noisy at the replication count.  Every stochastic
assertion is seed-fixed, so a failure is reproducible.

The corpus is the hand models below plus the random-program generator, which covers
MATRIX, EQUATE, ITE and UNIFORM through the generic column path, hierarchy through
unfolding, multi-bit input wires, repeated and dangling outputs, and caller tables whose
column masses exceed one, which exercises the weight rescaling.
"""
from __future__ import annotations

import numpy as np

from pylea import DPP, Bool, Enum, flip, observe
from pylea.ir.reference import semantics
from pylea.sample.sweep import SweepProgram, unfold, unfolded_op_count
from test.compile.random_programs import random_program


def estimate_matrix(prog, n: int, sweeps: int, seed: int, max_columns: int | None = None):
    """Mean and standard error of sweep estimates of the program's matrix."""
    body = unfold(prog)
    n_columns = 1 << body.in_bits
    if max_columns is not None:
        n_columns = min(n_columns, max_columns)
    rng = np.random.default_rng(seed)
    means, errors = [], []
    for column in range(n_columns):
        program = SweepProgram(body, input_pattern=column)
        scale = float(np.exp(program.log_scale))
        draws = []
        for _ in range(sweeps):
            result = program.run(n, rng, resample=True)
            draws.append(result.counts * (np.exp(result.log_z) * scale / n))
        draws = np.stack(draws)
        means.append(draws.mean(axis=0))
        errors.append(draws.std(axis=0, ddof=1) / np.sqrt(sweeps))
    return np.stack(means, axis=1), np.stack(errors, axis=1)


def assert_matches_reference(prog, n: int, sweeps: int, seed: int, floor: float,
                             max_columns: int | None = 8) -> None:
    exact = semantics(prog)
    mean, error = estimate_matrix(prog, n, sweeps, seed, max_columns)
    exact = exact[:, :mean.shape[1]]
    gap = np.abs(mean - exact)
    tolerance = 5.0 * error + floor
    assert (gap <= tolerance).all(), \
        f"worst violation {float((gap - tolerance).max()):.3e}"


@DPP
def state_dependent_chain() -> Bool:
    x = flip(0.5)
    for _ in range(6):
        observe(x | flip(0.5))
    return x


class Colour(Enum):
    elements = ["R", "G", "B", "Y"]


@DPP
def conditioned_enum() -> Colour:
    c = Colour.random({"R": 0.4, "G": 0.3, "B": 0.2, "Y": 0.1})
    observe(~(c == "Y"))
    return c


def test_sweeps_match_the_reference_on_the_disease_example():
    from pylea.examples.disease import has_disease

    assert_matches_reference(has_disease.program(), n=128, sweeps=2000, seed=0, floor=2e-3)


def test_sweeps_match_the_reference_on_a_state_dependent_observe_chain():
    assert_matches_reference(state_dependent_chain.program(), n=64, sweeps=2000, seed=1,
                             floor=3e-3)


def test_sweeps_match_the_reference_on_an_enum_model_with_conditioning():
    assert_matches_reference(conditioned_enum.program(), n=64, sweeps=2000, seed=2,
                             floor=3e-3)


def test_sweeps_match_the_reference_on_random_programs():
    for seed in range(25):
        prog = random_program(np.random.default_rng(seed))
        assert_matches_reference(prog, n=64, sweeps=250, seed=1000 + seed, floor=8e-3,
                                 max_columns=4)


def test_iid_batches_match_the_reference_without_resampling():
    for seed in (3, 7, 11, 19):
        prog = random_program(np.random.default_rng(seed))
        body = unfold(prog)
        program = SweepProgram(body, input_pattern=0)
        exact = semantics(prog)[:, 0]
        rng = np.random.default_rng(500 + seed)
        n, reps = 4096, 8
        draws = np.stack([program.run(n, rng, squares=True).counts / n
                          for _ in range(reps)])
        scale = float(np.exp(program.log_scale))
        mean = draws.mean(axis=0) * scale
        error = draws.std(axis=0, ddof=1) * scale / np.sqrt(reps)
        assert (np.abs(mean - exact) <= 5.0 * error + 8e-3).all()


def test_squared_weight_sums_equal_plain_counts_when_weights_are_binary():
    @DPP
    def conditioned() -> Bool:
        a, b = flip(0.5), flip(0.5)
        observe(a | b)
        return a

    program = SweepProgram(unfold(conditioned.program()))
    assert program.binary_weights
    result = program.run(2048, np.random.default_rng(0), squares=True)
    assert result.counts_sq is not None
    assert np.array_equal(result.counts, result.counts_sq)
    assert result.total == result.total_sq


def test_a_sweep_that_cannot_survive_returns_zero_mass():
    @DPP
    def impossible() -> Bool:
        x = flip(0.5)
        observe(x & ~x)
        return x

    result = SweepProgram(unfold(impossible.program())).run(
        256, np.random.default_rng(0), resample=True)
    assert result.total == 0.0 and not result.counts.any()


def test_deferral_keeps_the_live_frontier_below_the_flip_count():
    @DPP
    def wide_conjunction() -> Bool:
        acc = flip(0.9)
        for _ in range(63):
            acc = acc & flip(0.9)
        return acc

    program = SweepProgram(unfold(wide_conjunction.program()))
    # All 64 flips would be live at once if materialised up front; deferral and frees
    # keep only a flip, the accumulator and the fresh conjunction alive.
    assert program.peak_planes <= 4
    result = program.run(4096, np.random.default_rng(0))
    assert abs(result.counts[1] / 4096 - 0.9 ** 64) < 0.01


def test_stage_fractions_report_per_observe_survival():
    @DPP
    def two_stages() -> Bool:
        a, b = flip(0.8), flip(0.5)
        observe(a)
        observe(b)
        return b

    program = SweepProgram(unfold(two_stages.program()))
    result = program.run(1 << 14, np.random.default_rng(0), record_stages=True)
    fractions = result.stage_fractions
    assert fractions is not None and fractions.shape == (2,)
    assert abs(fractions[0] - 0.8) < 0.02
    assert abs(fractions[1] - 0.5) < 0.02


def test_stage_fractions_stay_on_one_scale_across_renormalisation():
    # 1100 soft rows of 0.8 drive every weight below the renormalisation threshold
    # mid-run; the recorded fractions must not pick up the factor moved into the log.
    from pylea.ir.body import BodyBuilder
    from pylea.ir.prim import Prim

    builder = BodyBuilder()
    (x,) = builder.emit(Prim.flip(0.0), ())
    for _ in range(1100):
        builder.emit(Prim.from_matrix(np.array([[0.8, 1.0]])), (x,))
    body = builder.finish(inputs=(), outputs=(x,))
    result = SweepProgram(body).run(8, np.random.default_rng(0), resample=True,
                                    record_stages=True)
    assert result.stage_fractions is not None
    assert np.allclose(result.stage_fractions, 0.8)


def test_the_unfolded_op_count_is_computed_without_unfolding():
    @DPP
    def leaf() -> Bool:
        return flip(0.5) & flip(0.5)

    level = leaf
    for _ in range(30):
        previous = level

        @DPP
        def level() -> Bool:
            return previous() & previous()      # noqa: B023, rebinding is the recursion

    prog = level.program()
    assert unfolded_op_count(prog) == 4 * 2 ** 30 - 1
