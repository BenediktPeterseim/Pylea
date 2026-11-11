"""Semantics preservation and effect of the schedule passes.

The differential tests run the full pass stack against the enumeration oracle; the
structural tests pin what each pass does to the op list, including the recognition
regression for pushed evidence: if the comparator's emitted circuit changes shape, the
pass silently stops firing, and only the regression test notices.
"""
from __future__ import annotations

import math

import numpy as np

from pylea import DPP, Bool, Enum, flip, observe
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program
from pylea.ir.reference import semantics
from pylea.sample.schedule import asap_observes, fold, push_evidence, schedule, split_observes
from pylea.sample.sweep import SweepProgram, unfold
from test.compile.random_programs import random_program


def observes_in(body) -> int:
    return sum(1 for op in body.ops
               if isinstance(op.callee, Prim) and op.callee.tag is Tag.OBSERVE)


def test_the_schedule_passes_preserve_the_semantics_on_random_programs():
    for seed in range(30):
        prog = random_program(np.random.default_rng(seed))
        body = unfold(prog)
        before = semantics(Program((body,), 0))
        after = semantics(Program((schedule(body),), 0))
        assert np.allclose(before, after), f"seed {seed} changed the semantics"


@DPP
def terminal_conjunction() -> Bool:
    x = flip(0.5)
    ok = Bool.const(True)
    for _ in range(20):
        ok = ok & (flip(0.5) | flip(0.5))
    observe(ok)
    return x


def test_folding_deletes_a_conjunction_of_a_wire_with_itself():
    @DPP
    def doubled() -> Bool:
        x = flip(0.3)
        observe(x & x)
        return flip(0.5)

    body = fold(unfold(doubled.program()))
    assert [op.callee for op in body.ops] == [Prim.flip(0.3), Prim.observe(), Prim.flip(0.5)]


def test_splitting_turns_a_terminal_conjunction_into_one_observe_per_conjunct():
    # Semantics preservation is covered by the random-program differential; this body is
    # too wide for the enumeration oracle, so the assertion here is structural.
    body = split_observes(fold(unfold(terminal_conjunction.program())))
    assert observes_in(body) == 20


def test_splitting_cuts_the_per_sweep_variance_of_a_terminal_conjunction():
    plain = unfold(terminal_conjunction.program())
    scheduled = schedule(plain)

    def relvar(body, seed: int) -> float:
        program = SweepProgram(body)
        rng = np.random.default_rng(seed)
        draws = np.array([
            float(r.total) * math.exp(r.log_z) / 64
            for r in (program.run(64, rng, resample=True) for _ in range(400))])
        return float(draws.var(ddof=1) / draws.mean() ** 2)

    assert relvar(scheduled, 1) < relvar(plain, 0) / 3


class Sky(Enum):
    elements = ["clear", "overcast"]


class Ground(Enum):
    elements = ["dry", "wet"]


def test_push_evidence_fires_on_a_traced_equality_observation():
    wet_given_sky = Ground.cpt([Sky], {("clear",): [0.9, 0.1], ("overcast",): [0.2, 0.8]})

    @DPP
    def rained() -> Sky:
        sky = Sky.random({"clear": 0.7, "overcast": 0.3})
        observe(wet_given_sky(sky) == "wet")
        return sky

    plain = unfold(rained.program())
    scheduled = schedule(plain)
    # The regression: the hard observe is gone, replaced by a single-row soft table.
    assert observes_in(scheduled) == 0
    assert any(isinstance(op.callee, Prim) and op.callee.tag is Tag.MATRIX
               and op.callee.out_bits == 0 for op in scheduled.ops)
    assert np.allclose(semantics(Program((scheduled,), 0)),
                       semantics(Program((plain,), 0)))
    assert not SweepProgram(scheduled).binary_weights


def test_push_evidence_leaves_contradictory_evidence_alone():
    @DPP
    def impossible() -> Sky:
        sky = Sky.random({"clear": 0.7, "overcast": 0.3})
        observe(sky == "clear")
        observe(sky == "overcast")
        return sky

    plain = unfold(impossible.program())
    scheduled = schedule(plain)
    assert observes_in(scheduled) >= 1
    assert not semantics(Program((scheduled,), 0)).any()


def test_scheduled_deep_evidence_sweeps_survive_where_plain_observes_die():
    # Each stage's conditional probability sits far below 1 / n, so a hard-observe sweep
    # loses its whole population at the first stage almost surely.
    rare = {("dry",): [0.9999, 0.0001], ("wet",): [0.9998, 0.0002]}
    step = Ground.cpt([Ground], rare)

    @DPP
    def soaked_chain() -> Ground:
        g = Ground.random({"dry": 0.999, "wet": 0.001})
        for _ in range(8):
            g = step(g)
            observe(g == "wet")
        return g

    exact_p_acc = soaked_chain.compiled().evidence()
    assert exact_p_acc < 1e-12

    plain = SweepProgram(unfold(soaked_chain.program()))
    scheduled = SweepProgram(schedule(unfold(soaked_chain.program())))
    rng = np.random.default_rng(0)
    plain_alive = sum(plain.run(256, rng, resample=True).total > 0 for _ in range(20))
    # Sweep values live on the rescaled scale; the static log factor maps them back.
    values = math.exp(scheduled.log_scale) * np.array([
        r.total * math.exp(r.log_z) / 256
        for r in (scheduled.run(256, rng, resample=True) for _ in range(20))])
    assert plain_alive == 0
    assert (values > 0.0).all()
    error = 5 * values.std(ddof=1) / np.sqrt(len(values))
    assert abs(values.mean() - exact_p_acc) <= error + 1e-15


def test_asap_pulls_an_observe_ahead_of_unrelated_sampling():
    @DPP
    def late() -> Bool:
        x = flip(0.5)
        rest = [flip(0.5) for _ in range(10)]
        acc = x
        for r in rest:
            acc = acc & r
        observe(x | flip(0.5))
        return acc

    body = asap_observes(unfold(late.program()))
    first_observe = next(i for i, op in enumerate(body.ops)
                         if isinstance(op.callee, Prim) and op.callee.tag is Tag.OBSERVE)
    # Its cone is three ops: the two flips and the disjunction.
    assert first_observe == 3
