"""Flip lifting, and the conditions it declines under."""
import sys
from time import perf_counter

import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, consumer_counts, intern, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import lift, size
from pylea.simplify.common import Circuit, leaving
from pylea.simplify.lift import ROOT, _idom, _sites
from test.simplify.programs import circuits, prog, same_matrix


def _or_tree() -> Program:
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (top,) = b.emit(Prim.or_(), (a, c))
    return prog(b.finish((), (top,)))


def _check(p: Program) -> Program:
    out = lift(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


def test_the_lifted_flip_ends_with_one_reader():
    p = _or_tree()
    assert consumer_counts(p.main)[0] == 2
    out = _check(p)
    assert consumer_counts(out.main)[0] == 1
    assert size(out) < size(p)


def test_an_and_tree_lifts_the_dual_way():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.and_(), (x, p))
    (c,) = b.emit(Prim.and_(), (x, q))
    (top,) = b.emit(Prim.and_(), (a, c))
    p_ = prog(b.finish((), (top,)))
    assert list(_sites(p_.main, Circuit(p_.main))) == [(0, 5, "and")]
    assert consumer_counts(_check(p_).main)[0] == 1


def test_a_conjunction_of_the_flip_below_the_dominator_becomes_false():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.and_(), (x, q))
    (e,) = b.emit(Prim.or_(), (x, c))
    (top,) = b.emit(Prim.or_(), (a, e))
    out = _check(prog(b.finish((), (top,))))
    assert consumer_counts(out.main)[0] == 1


def test_a_flip_whose_dominator_reads_it_is_lifted():
    """The dominator itself is a disjunction of the flip, and takes the new gate's place."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (q,) = b.emit(Prim.flip(0.5), ())
    (c,) = b.emit(Prim.or_(), (a, q))
    (top,) = b.emit(Prim.or_(), (x, c))
    out = _check(prog(b.finish((), (top,))))
    assert consumer_counts(out.main)[0] == 1


def test_a_flip_the_body_returns_is_not_lifted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, a))
    p_ = prog(b.finish((), (c, x)))
    assert x in leaving(p_.main) and _idom(p_.main, leaving(p_.main))[x] == ROOT
    assert lift(p_) == p_


def test_a_flip_read_by_any_op_but_a_gate_is_not_lifted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (x,))
    (top,) = b.emit(Prim.or_(), (a, m))
    p_ = prog(b.finish((), (top,)))
    assert lift(p_) == p_


def test_a_flip_read_by_a_negation_is_not_lifted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, n))
    (top,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (top,)))
    assert lift(p_) == p_


def test_a_shared_gate_output_whose_dominator_is_higher_up_is_not_lifted():
    b = BodyBuilder()
    (u,) = b.emit(Prim.flip(0.3), ())
    (v,) = b.emit(Prim.flip(0.6), ())
    (x,) = b.emit(Prim.and_(), (u, v))
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (top,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (top,)))
    assert lift(p_) == p_


def test_a_shared_gate_output_its_dominator_reads_is_left_to_masking():
    b = BodyBuilder()
    (u,) = b.emit(Prim.flip(0.3), ())
    (v,) = b.emit(Prim.flip(0.6), ())
    (x,) = b.emit(Prim.and_(), (u, v))
    (p,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (q,) = b.emit(Prim.flip(0.5), ())
    (c,) = b.emit(Prim.or_(), (a, q))
    (top,) = b.emit(Prim.or_(), (x, c))
    p_ = prog(b.finish((), (top,)))
    assert lift(p_) == p_


def test_a_table_reading_nothing_is_lifted_like_a_flip():
    b = BodyBuilder()
    (x,) = b.emit(Prim.from_matrix([[0.7], [0.3]]), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (top,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (top,)))
    assert consumer_counts(_check(p_).main)[x] == 1


def test_a_flip_in_one_disjunction_is_not_lifted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.and_(), (x, q))
    (top,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (top,)))
    assert lift(p_) == p_


def test_a_flip_with_no_implication_is_not_lifted():
    """`top = NOT(x) AND (x OR q)`: `x` makes `top` false, so neither implication holds."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (r,) = b.emit(Prim.flip(0.6), ())
    (c,) = b.emit(Prim.or_(), (x, q))
    (d,) = b.emit(Prim.or_(), (x, r))
    (e,) = b.emit(Prim.and_(), (c, d))
    (n,) = b.emit(Prim.not_(), (e,))
    (top,) = b.emit(Prim.and_(), (n, c))
    p_ = prog(b.finish((), (top,)))
    assert not [site for site in _sites(p_.main, Circuit(p_.main)) if site[0] == x]


def test_one_result_of_a_source_with_two_is_not_lifted():
    """The source cannot move to the new gate, as its other result is read where it is.

    Lifting `e` here regardless, with `d` at the top of a ladder of height 3 over `g1` and
    `g2`, raises the exact contraction cost from 198 to 206 and the treewidth from 2 to 3.
    """
    b = BodyBuilder()
    e, f = b.emit(Prim.from_matrix([[0.1], [0.2], [0.3], [0.4]]), ())
    (s_,) = b.emit(Prim.flip(0.5), ())
    (t,) = b.emit(Prim.flip(0.6), ())
    (low,) = b.emit(Prim.and_(), (f, s_))
    (g1,) = b.emit(Prim.or_(), (e, low))
    (g2,) = b.emit(Prim.or_(), (e, t))
    h, x = g1, g2
    for i in range(3):
        (r,) = b.emit(Prim.flip(0.2 + 0.05 * i), ())
        (x,) = b.emit(Prim.or_(), (x, r))
        (h,) = b.emit(Prim.or_(), (h, x))
    p_ = prog(b.finish((), (h,)))
    assert lift(p_) == p_


def _reads_of_each_flip(p: Program, probability: float) -> list[int]:
    """How often each wire a `flip` of `probability` produces is read in `p`."""
    reads = consumer_counts(p.main)
    return [reads[op.outs[0]] for op in p.main.ops if op.callee == Prim.flip(probability)]


def _two_trees():
    """Two copies of `_or_tree` under one disjunction, sharing nothing."""
    b = BodyBuilder()
    tops = []
    for _ in range(2):
        (x,) = b.emit(Prim.flip(0.3), ())
        (p,) = b.emit(Prim.flip(0.4), ())
        (q,) = b.emit(Prim.flip(0.5), ())
        (a,) = b.emit(Prim.or_(), (x, p))
        (c,) = b.emit(Prim.or_(), (x, q))
        tops.append(b.emit(Prim.or_(), (a, c))[0])
    (top,) = b.emit(Prim.or_(), tuple(tops))
    return prog(b.finish((), (top,)))


def _calls(monkeypatch, name: str) -> list:
    """A list gaining an entry at each call of the function `name` of the lift module."""
    calls: list = []
    module = sys.modules["pylea.simplify.lift"]
    original = getattr(module, name)

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_lifts_with_disjoint_regions_are_applied_in_one_sweep(monkeypatch):
    applied = _calls(monkeypatch, "_apply")
    out = _check(_two_trees())
    assert [len(sites) for _, sites in applied] == [2]
    assert _reads_of_each_flip(out, 0.3) == [1, 1]


def _meeting_regions() -> Program:
    """`x` is read by `a` and `c`, `y` by `c` and `g`, so the two regions share `c`."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.6), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, y))
    (g,) = b.emit(Prim.or_(), (y, q))
    (t,) = b.emit(Prim.or_(), (a, c))
    (top,) = b.emit(Prim.or_(), (t, g))
    return prog(b.finish((), (top,)))


def test_a_lift_whose_region_meets_an_earlier_one_is_checked_again_in_the_same_round(
        monkeypatch):
    p_ = _meeting_regions()
    assert len(list(_sites(p_.main, Circuit(p_.main)))) == 2
    checked = _calls(monkeypatch, "_still_a_site")
    applied = _calls(monkeypatch, "_apply")
    out = _check(p_)
    assert len(checked) == 1 and [len(sites) for _, sites in applied] == [1, 1]
    assert _reads_of_each_flip(out, 0.3) == [1] and _reads_of_each_flip(out, 0.6) == [1]


def test_the_lifted_flip_moves_to_just_before_its_one_reader():
    out = _check(_or_tree())
    ops = out.main.ops
    (k,) = [k for k, op in enumerate(ops) if op.callee == Prim.flip(0.3)]
    (j,) = [j for j, op in enumerate(ops) if ops[k].outs[0] in op.args]
    assert j == k + 1


def test_independent_lifts_take_one_sweep_however_many_there_are(monkeypatch):
    applied = _calls(monkeypatch, "_apply")
    checked = _calls(monkeypatch, "_still_a_site")
    b = BodyBuilder()
    tops = []
    for _ in range(50):
        (x,) = b.emit(Prim.flip(0.3), ())
        (p,) = b.emit(Prim.flip(0.4), ())
        (q,) = b.emit(Prim.flip(0.5), ())
        (a,) = b.emit(Prim.or_(), (x, p))
        (c,) = b.emit(Prim.or_(), (x, q))
        tops.append(b.emit(Prim.and_(), (a, c))[0])
    while len(tops) > 1:
        tops.append(b.emit(Prim.or_(), (tops.pop(0), tops.pop(0)))[0])
    p_ = prog(b.finish((), tuple(tops)))
    out = lift(p_)
    validate(out)
    assert [len(sites) for _, sites in applied] == [50] and not checked
    assert op_count(out) == op_count(p_) - 50


def test_the_lifts_found_before_the_deadline_are_applied_and_no_more(monkeypatch):
    """The clock passes the deadline right after the second site is found, whose region
    meets the first's: the first is applied, and the second is not checked again."""
    readings = iter([False, False])
    module = sys.modules["pylea.simplify.lift"]
    monkeypatch.setattr(module, "expired", lambda deadline: next(readings, True))
    checked = _calls(monkeypatch, "_still_a_site")
    p_ = _meeting_regions()
    out = lift(p_, deadline=perf_counter() + 3600.0)
    validate(out)
    assert same_matrix(semantics(out), semantics(p_))
    assert _reads_of_each_flip(out, 0.3) == [1] and _reads_of_each_flip(out, 0.6) == [2]
    assert not checked


CORPUS = circuits(60)


@pytest.mark.parametrize("p", CORPUS, ids=range(len(CORPUS)))
def test_lifting_a_random_circuit_preserves_its_matrix(p):
    _check(p)


def test_the_random_circuits_exercise_the_lift():
    assert sum(1 for p in CORPUS if lift(p) != p) > len(CORPUS) // 4


class _Ticks:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.mark.parametrize("checks", [0, 1, 2, 3])
def test_a_lift_stopped_partway_still_computes_the_right_matrix(checks, monkeypatch):
    monkeypatch.setattr(sys.modules["pylea.simplify.common"], "perf_counter", _Ticks())
    for p in CORPUS[:20]:
        out = lift(p, deadline=float(checks))
        validate(out)
        assert same_matrix(semantics(out), semantics(p))


def test_a_budget_that_is_never_reached_lifts_as_if_it_were_absent():
    p = intern(_or_tree())
    assert lift(p, deadline=perf_counter() + 3600.0) == lift(p)
