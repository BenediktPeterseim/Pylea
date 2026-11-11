"""Exact sub-tables spliced into the sampler.

Semantics preservation runs against the enumeration oracle; the behavioural tests pin
the two effects the pass exists for, soft internal evidence and depth scaling, and the
refusals that keep it sound.
"""
from __future__ import annotations

import numpy as np

from pylea import DPP, Bool, flip, observe, prob
from pylea.ir.program import op_count
from pylea.ir.reference import semantics
from pylea.sample.collapse import collapse
from pylea.sample.schedule import schedule
from pylea.sample.sweep import SweepProgram, unfold
from pylea.syntax.estimate import Estimate
from test.compile.random_programs import random_program


def test_collapse_preserves_the_semantics_on_random_programs():
    for seed in range(20):
        prog = random_program(np.random.default_rng(seed))
        collapsed = collapse(prog)
        assert np.allclose(semantics(prog), semantics(collapsed)), f"seed {seed}"


def make_noisy_cell():
    @DPP
    def noisy_cell() -> Bool:
        x = flip(0.5)
        for _ in range(6):
            observe(x | flip(0.05))     # deep internal evidence favouring x
        return x

    return noisy_cell


def test_internal_observes_become_a_static_scale_on_the_collapsed_table():
    cell = make_noisy_cell()

    @DPP
    def pair() -> Bool:
        return cell() & cell()

    collapsed = collapse(pair.program())
    assert len(collapsed.bodies) == 1
    assert np.allclose(semantics(pair.program(), max_wires=20),
                       semantics(collapsed))
    # The cell's internal evidence is state-independent once collapsed, so the
    # interpreter moves it into the log factor: the particles condition on nothing and
    # certify through the Bernoulli lane.
    program = SweepProgram(unfold(collapsed))
    assert program.binary_weights and not program.conditions
    assert program.log_scale < 0.0
    exact_mass = float(semantics(collapsed).sum())
    result = program.run(1 << 14, np.random.default_rng(0))
    assert result.total == float(1 << 14)
    assert abs(np.exp(program.log_scale) - exact_mass) < 1e-12


def test_a_deep_narrow_call_tree_samples_as_a_single_table():
    # No conditioning inside the family: two calls per level square any evidence mass,
    # which would leave float64 long before depth 24 and rightly stop the collapse.
    @DPP
    def leaf() -> Bool:
        return flip(0.6) & flip(0.9)

    level = leaf
    for _ in range(24):
        previous = level

        @DPP
        def level() -> Bool:
            return (previous() & flip(0.9)) | (previous() & flip(0.9))

    collapsed = collapse(level.program())
    # Everything below the root is one table; the root itself is never collapsed.
    assert len(collapsed.bodies) == 1
    assert op_count(collapsed) <= 7
    p = prob(level, method="smc", rel_err=0.05, seed=0)
    assert isinstance(p, Estimate) and p.target_met
    exact = float(level.compiled().distribution()[1])
    assert p.low - 1e-9 <= exact <= p.high + 1e-9


def test_the_root_and_wide_bodies_are_never_collapsed():
    width = 10

    @DPP
    def wide(x, types=(Bool,) * width):
        return tuple(bit ^ flip(0.1) for bit in x)

    @DPP
    def uses() -> Bool:
        ys = wide(*[flip(0.5) for _ in range(width)])
        acc = ys[0]
        for y in ys[1:]:
            acc = acc | y
        return acc

    collapsed = collapse(uses.program())
    assert any(op.is_call for op in collapsed.main.ops)
    assert collapsed.main is not None


def test_an_underflowing_table_is_declined_rather_than_spliced_as_zero():
    @DPP
    def rare() -> Bool:
        x = flip(1e-300)
        observe(x & flip(1e-300))
        return x

    @DPP
    def outer() -> Bool:
        return rare() | flip(0.5)

    collapsed = collapse(outer.program())
    # The callee's mass is e**-1381, below float64; splicing would silently zero it.
    assert any(op.is_call for op in collapsed.main.ops)


def test_the_scheduled_pipeline_runs_on_a_collapsed_program():
    cell = make_noisy_cell()

    @DPP
    def strip() -> Bool:
        cells = [cell() for _ in range(12)]
        ok = Bool.const(True)
        for a, b in zip(cells, cells[1:]):
            ok = ok & (a | b)
        return ok

    body = schedule(unfold(collapse(strip.program())))
    program = SweepProgram(body)
    result = program.run(4096, np.random.default_rng(0), squares=True)
    assert result.total > 0.0
    exact = float(strip.compiled().distribution()[1])
    estimate = float(result.counts[1] / result.total)
    assert abs(estimate - exact) < 0.05
