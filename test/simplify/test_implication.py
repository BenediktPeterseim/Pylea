"""Syntactic implication, against a brute-force closure of its defining rules."""
import sys

import numpy as np
import pytest

from pylea.ir.body import Body, BodyBuilder
from pylea.ir.prim import Prim, Tag
from pylea.simplify.common import gate_tag, neg, pos
from pylea.simplify.implication import Implications
from test.simplify.programs import Clock


def _reference(body: Body) -> list[set[int]]:
    """The relation by iterating the rules of the definition until nothing changes."""
    n = 2 * len(body.bits)
    rel = [{lit} for lit in range(n)]
    ands, ors, nots = [], [], []
    for op in body.ops:
        tag = gate_tag(op)
        if tag is None:
            continue
        (y,) = op.outs
        if tag is Tag.NOT:
            nots.append((y, op.args[0]))
        elif tag is Tag.AND:
            ands.append((y, *op.args))
        else:
            ors.append((y, *op.args))
    changed = True
    while changed:
        changed = False

        def add(i, j):
            nonlocal changed
            if j not in rel[i]:
                rel[i].add(j)
                changed = True

        for y, a, b in ands:
            add(pos(y), pos(a))
            add(pos(y), pos(b))
            for lit in range(n):
                if pos(a) in rel[lit] and pos(b) in rel[lit]:
                    add(lit, pos(y))
        for y, a, b in ors:
            add(pos(a), pos(y))
            add(pos(b), pos(y))
            for lit in range(n):
                if lit in rel[pos(a)] and lit in rel[pos(b)]:
                    add(pos(y), lit)
        for y, x in nots:
            add(pos(y), neg(x))
            add(neg(x), pos(y))
        for i in range(n):
            for j in list(rel[i]):
                add(j ^ 1, i ^ 1)
                for k in list(rel[j]):
                    add(i, k)
    return rel


def _random_body(rng, with_not: bool) -> Body:
    b = BodyBuilder()
    pool = [b.fresh() for _ in range(int(rng.integers(2, 5)))]
    n_leaves = len(pool)
    for _ in range(int(rng.integers(2, 10))):
        kind = rng.integers(3 if with_not else 2)
        if kind == 2:
            (w,) = b.emit(Prim.not_(), (pool[int(rng.integers(len(pool)))],))
        else:
            gate = Prim.and_() if kind == 0 else Prim.or_()
            (w,) = b.emit(gate, (pool[int(rng.integers(len(pool)))],
                                 pool[int(rng.integers(len(pool)))]))
        pool.append(w)
    return b.finish(inputs=tuple(pool[:n_leaves]), outputs=(pool[-1],))


@pytest.mark.parametrize("seed", range(120))
def test_the_relation_agrees_with_the_brute_force_closure(seed):
    body = _random_body(np.random.default_rng(seed), with_not=seed % 2 == 0)
    rel, want = Implications(body), _reference(body)
    n = 2 * len(body.bits)
    for i in range(n):
        for j in range(n):
            assert rel.implies(i, j) == (j in want[i]), (i, j)


def test_a_case_split_over_a_disjunction_is_found():
    """`(m AND p) OR (m AND q)` implies `m`, which no single-literal propagation gives."""
    b = BodyBuilder()
    m, p, q = b.fresh(), b.fresh(), b.fresh()
    (a,) = b.emit(Prim.and_(), (m, p))
    (c,) = b.emit(Prim.and_(), (m, q))
    (o,) = b.emit(Prim.or_(), (a, c))
    rel = Implications(b.finish((m, p, q), (o,)))
    assert rel.implies(pos(o), pos(m)) and not rel.implies(pos(m), pos(o))


def test_a_deadline_already_passed_leaves_the_reflexive_relation():
    b = BodyBuilder()
    x, y = b.fresh(), b.fresh()
    (o,) = b.emit(Prim.or_(), (x, y))
    rel = Implications(b.finish((x, y), (o,)), deadline=-1.0)
    assert not rel.complete
    assert rel.implies(pos(x), pos(x)) and not rel.implies(pos(x), pos(o))


def test_a_deadline_already_passed_allocates_no_relation():
    b = BodyBuilder()
    pool = [b.fresh() for _ in range(4)]
    for k in range(200):
        pool.append(b.emit(Prim.and_() if k % 2 else Prim.or_(), (pool[-1], pool[k % 4]))[0])
    rel = Implications(b.finish(tuple(pool[:4]), (pool[-1],)), deadline=-1.0)
    assert rel.rows.size == 0


@pytest.mark.parametrize("seed", range(40))
def test_a_relation_stopped_at_any_reading_of_the_clock_holds_only_implications(
        seed, monkeypatch):
    body = _random_body(np.random.default_rng(seed), with_not=seed % 2 == 0)
    want = _reference(body)
    module = sys.modules["pylea.simplify.common"]
    full = Clock(float("inf"))
    monkeypatch.setattr(module, "perf_counter", full)
    Implications(body, deadline=0.5)
    n = 2 * len(body.bits)
    for allowed in range(full.readings):
        monkeypatch.setattr(module, "perf_counter", Clock(allowed))
        rel = Implications(body, deadline=0.5)
        assert not rel.complete
        for i in range(n):
            for j in range(n):
                assert not rel.implies(i, j) or j in want[i], (allowed, i, j)


def test_only_the_literals_of_wires_a_gate_touches_are_held():
    b = BodyBuilder()
    x, y = b.fresh(), b.fresh()
    (t,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (x,))
    (o,) = b.emit(Prim.or_(), (x, y))
    rel = Implications(b.finish((x, y), (o, t)))
    assert rel.n == 3
    assert rel.implies(pos(t), pos(t)) and not rel.implies(pos(t), pos(o))
    assert rel.implies(pos(x), pos(o)) and rel.implies(neg(o), neg(x))
