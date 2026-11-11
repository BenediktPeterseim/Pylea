"""Common subexpression elimination over a private flip."""
import numpy as np
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, consumer_counts, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import flip_cse, size
from test.simplify.programs import circuits, prog, same_matrix


def _check(p: Program) -> Program:
    out = flip_cse(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


def _duplicates(private: bool, det: bool = True):
    """Two equal gates over `x` and a coin `y`; `y` is read elsewhere unless `private`."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (z,) = b.emit(Prim.flip(0.5), ())
    gate = Prim.and_() if det else Prim.from_matrix(np.full((2, 4), 0.5).tolist())
    (a,) = b.emit(gate, (x, y))
    (c,) = b.emit(gate, (x, y))
    (d,) = b.emit(Prim.or_(), (a, z))
    (e,) = b.emit(Prim.or_(), (c, x))
    outs = (d, e) if private else (d, e, b.emit(Prim.or_(), (y, z))[0])
    return prog(b.finish((), outs)), a, c


def test_duplicates_over_a_private_coin_are_merged():
    p, a, c = _duplicates(private=True)
    out = _check(p)
    assert op_count(out) == op_count(p) - 1 and size(out) < size(p)
    assert consumer_counts(out.main)[a] == 2


def test_duplicates_whose_coin_is_read_elsewhere_are_left():
    p, _, _ = _duplicates(private=False)
    assert flip_cse(p) == p


def test_duplicates_of_a_stochastic_table_are_left():
    p, _, _ = _duplicates(private=True, det=False)
    assert flip_cse(p) == p


def test_duplicates_over_shared_gates_alone_are_left():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (g,) = b.emit(Prim.or_(), (x, y))
    (h,) = b.emit(Prim.and_(), (x, y))
    (a,) = b.emit(Prim.and_(), (g, h))
    (c,) = b.emit(Prim.and_(), (g, h))
    (d,) = b.emit(Prim.or_(), (a, x))
    (e,) = b.emit(Prim.or_(), (c, y))
    p = prog(b.finish((), (d, e)))
    assert flip_cse(p) == p


def test_three_copies_collapse_to_one_in_one_pass():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    outs = []
    for _ in range(3):
        (a,) = b.emit(Prim.or_(), (x, y))
        (z,) = b.emit(Prim.flip(0.5), ())
        outs.append(b.emit(Prim.and_(), (a, z))[0])
    p = prog(b.finish((), tuple(outs)))
    out = _check(p)
    assert op_count(out) == op_count(p) - 2


def test_a_merged_result_that_is_a_body_output_stays_an_output():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, y))
    p = prog(b.finish((), (a, c)))
    out = _check(p)
    assert out.main.outputs[0] == out.main.outputs[1]


def test_a_coin_a_copy_reads_twice_does_not_count():
    b = BodyBuilder()
    (y,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.and_(), (y, y))
    (c,) = b.emit(Prim.and_(), (y, y))
    p = prog(b.finish((), (a, c)))
    assert flip_cse(p) == p


def test_duplicate_calls_of_a_deterministic_body_are_merged():
    inner = BodyBuilder()
    i1, i2 = inner.fresh(), inner.fresh()
    (o,) = inner.emit(Prim.and_(), (i1, i2))
    gate_body = inner.finish((i1, i2), (o,))
    outer = BodyBuilder(bodies=(gate_body,))
    (x,) = outer.emit(Prim.flip(0.3), ())
    (y,) = outer.emit(Prim.flip(0.4), ())
    (a,) = outer.emit(0, (x, y))
    (c,) = outer.emit(0, (x, y))
    (z,) = outer.emit(Prim.flip(0.5), ())
    (d,) = outer.emit(Prim.or_(), (a, z))
    (e,) = outer.emit(Prim.or_(), (c, x))
    p = Program((gate_body, outer.finish((), (d, e))), 1)
    out = _check(p)
    assert op_count(out) == op_count(p) - 1


def test_duplicate_calls_of_a_body_with_two_results_are_left():
    """Both results would have to run where the flip alone ran; the rule declines."""
    inner = BodyBuilder()
    i1, i2 = inner.fresh(), inner.fresh()
    (o1,) = inner.emit(Prim.and_(), (i1, i2))
    (o2,) = inner.emit(Prim.or_(), (i1, i2))
    pair = inner.finish((i1, i2), (o1, o2))
    outer = BodyBuilder(bodies=(pair,))
    (x,) = outer.emit(Prim.flip(0.3), ())
    (y,) = outer.emit(Prim.flip(0.4), ())
    (a1, a2) = outer.emit(0, (x, y))
    (c1, c2) = outer.emit(0, (x, y))
    (z,) = outer.emit(Prim.flip(0.5), ())
    (d,) = outer.emit(Prim.or_(), (a1, z))
    (e,) = outer.emit(Prim.and_(), (c2, x))
    p = Program((pair, outer.finish((), (d, e, a2, c1))), 1)
    assert flip_cse(p) == p


def test_duplicates_over_one_result_of_a_source_with_two_are_left():
    """The source cannot be taken into the surviving copy for nothing, as its other result
    is read where it is."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    s_, t = b.emit(Prim.from_matrix([[0.1], [0.2], [0.3], [0.4]]), ())
    (a,) = b.emit(Prim.and_(), (s_, x))
    (c,) = b.emit(Prim.and_(), (s_, x))
    (z,) = b.emit(Prim.or_(), (t, x))
    p = prog(b.finish((), (a, c, z)))
    assert flip_cse(p) == p


def test_duplicates_over_a_private_one_bit_table_are_merged_like_a_flip():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.from_matrix([[0.2], [0.5]]), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, y))
    p = prog(b.finish((), (a, c)))
    assert op_count(_check(p)) == op_count(p) - 1


def test_calls_of_a_body_that_draws_are_not_merged_over_a_private_flip():
    inner = BodyBuilder()
    i = inner.fresh(1)
    (coin,) = inner.emit(Prim.flip(0.5), ())
    (o,) = inner.emit(Prim.and_(), (i, coin))
    drawing = inner.finish((i,), (o,))
    outer = BodyBuilder(bodies=(drawing,))
    (u,) = outer.emit(Prim.flip(0.4), ())
    (a,) = outer.emit(0, (u,))
    (c,) = outer.emit(0, (u,))
    (r,) = outer.emit(Prim.and_(), (a, c))
    p = Program((drawing, outer.finish((), (r,))), 1)
    assert flip_cse(p) == p


@pytest.mark.parametrize("source", [Prim.const(1, 1), Prim.flip(0.5)])
def test_two_equal_sources_are_not_merged(source):
    """A source reads nothing, so there is no private flip to merge it over; two flips are
    moreover two draws."""
    b = BodyBuilder()
    (x,) = b.emit(source, ())
    (y,) = b.emit(source, ())
    (r,) = b.emit(Prim.and_(), (x, y))
    p = prog(b.finish((), (r,)))
    assert flip_cse(p) == p


CORPUS = circuits(60)


@pytest.mark.parametrize("p", CORPUS, ids=range(len(CORPUS)))
def test_a_random_circuit_keeps_its_matrix(p):
    _check(p)
