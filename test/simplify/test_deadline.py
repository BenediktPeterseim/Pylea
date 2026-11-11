"""The rules taking a deadline, stopped at every reading of the clock they make.

The clock is replaced by one that reads 0 a given number of times and 1 from then on,
against a deadline of 0.5, so that a run can be stopped at each point where a rule tests
its deadline, in turn.
"""
import sys
from inspect import signature

import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern, validate
from pylea.ir.reference import semantics
from pylea.simplify import RULES, simplify, size
from test.simplify.programs import Clock, circuits, mixed_corpus, prog, same_matrix

BOUNDED = [rule for rule in RULES if "deadline" in signature(rule).parameters]


def _two_chains() -> Program:
    """Two chains of gates over private flips, each chain a block of its own."""
    b = BodyBuilder()
    tops = []
    for first in (0.1, 0.6):
        (w,) = b.emit(Prim.flip(first), ())
        for k, gate in enumerate((Prim.and_(), Prim.or_(), Prim.and_())):
            (x,) = b.emit(Prim.flip(first + 0.1 * (k + 1)), ())
            (w,) = b.emit(gate, (w, x))
        tops.append(w)
    return prog(b.finish((), tuple(tops)))


def _two_pairs() -> Program:
    """Two pairs of equal gates over private flips."""
    b = BodyBuilder()
    x, y, u, v = (b.emit(Prim.flip(q), ())[0] for q in (0.1, 0.2, 0.3, 0.4))
    outs = [b.emit(gate, args)[0] for gate, args in
            ((Prim.and_(), (x, y)), (Prim.and_(), (x, y)), (Prim.or_(), (u, v)), (Prim.or_(), (u, v)))]
    return prog(b.finish((), tuple(outs)))


CORPUS = circuits(15) + circuits(15, observe=True) + mixed_corpus(n=30) + [_two_chains(), _two_pairs()]


def _stopped(rule, p: Program, allowed: float, monkeypatch) -> tuple[Program, int]:
    """`rule` applied to `p` with the deadline passing after `allowed` readings, and the
    number of readings it made."""
    clock = Clock(allowed)
    with monkeypatch.context() as m:
        for name, module in list(sys.modules.items()):
            if name.startswith("pylea.simplify.") and hasattr(module, "perf_counter"):
                m.setattr(module, "perf_counter", clock)
        out = rule(p, deadline=0.5)
    return out, clock.readings


def test_the_rules_taking_a_deadline_are_those_whose_cost_grows_faster_than_a_body():
    assert {rule.__name__ for rule in BOUNDED} == {
        "block_fold", "bottom_cse", "mask", "absorb", "lift"}


@pytest.mark.parametrize("rule", BOUNDED, ids=lambda rule: rule.__name__)
def test_a_rule_stopped_at_any_reading_of_the_clock_keeps_the_matrix(rule, monkeypatch):
    for p in CORPUS:
        want = semantics(p)
        _, readings = _stopped(rule, p, float("inf"), monkeypatch)
        checked = set()
        for allowed in range(readings + 1):
            out, _ = _stopped(rule, p, allowed, monkeypatch)
            if out in checked:
                continue
            checked.add(out)
            validate(out)
            assert same_matrix(semantics(out), want), (allowed, p)
            assert out is p or intern(out) == p or size(intern(out)) < size(p)


@pytest.mark.parametrize("rule", BOUNDED, ids=lambda rule: rule.__name__)
def test_a_deadline_passing_inside_one_body_stops_the_rule_there(rule, monkeypatch):
    """Some run stops with part of the rewrites of a one-body program applied."""
    partial = 0
    for p in CORPUS:
        assert len(p.bodies) == 1
        full, readings = _stopped(rule, p, float("inf"), monkeypatch)
        outcomes = {intern(_stopped(rule, p, allowed, monkeypatch)[0])
                    for allowed in range(readings + 1)}
        partial += bool(outcomes - {intern(p), intern(full)})
    assert partial > 0


def test_simplify_gives_its_deadline_to_every_rule_taking_one():
    given = []

    def bounded(prog: Program, deadline: float | None = None) -> Program:
        given.append(deadline)
        return prog

    def unbounded(prog: Program) -> Program:
        return prog

    simplify(CORPUS[0], rules=(bounded, unbounded), budget_ms=1e6)
    assert len(given) == 1 and given[0] is not None
