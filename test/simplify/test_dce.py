"""Dead code elimination and specialisation, and the discardability deciding what may go."""
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import op_count, reachable, validate
from pylea.ir.reference import semantics
from pylea.simplify import dce
from test.simplify.programs import prog, same_matrix


def test_an_unread_flip_goes():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (_dead,) = b.emit(Prim.flip(0.25), ())
    assert op_count(dce(prog(b.finish((), (x,))))) == 1


def test_an_unread_observe_stays():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    b.emit(Prim.observe(), (y,))
    p = prog(b.finish((), (x,)))
    out = dce(p)
    assert op_count(out) == 3
    assert same_matrix(semantics(out), semantics(p))


def test_an_unread_table_whose_columns_do_not_sum_to_one_stays():
    leaky = Prim.from_matrix([[1.0, 1.0], [0.0, 1.0]])
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (_y,) = b.emit(leaky, (x,))
    p = prog(b.finish((), (x,)))
    out = dce(p)
    assert op_count(out) == 2
    assert same_matrix(semantics(out), semantics(p))


def test_a_chain_of_dead_ops_goes_in_one_pass():
    b = BodyBuilder()
    (keep,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.flip(0.25), ())
    (c,) = b.emit(Prim.not_(), (a,))
    (_d,) = b.emit(Prim.and_(), (c, a))
    assert op_count(dce(prog(b.finish((), (keep,))))) == 1


def _two_output_callee():
    inner = BodyBuilder()
    (p,) = inner.emit(Prim.flip(0.5), ())
    (q,) = inner.emit(Prim.flip(0.25), ())
    callee = inner.finish((), (p, q))
    outer = BodyBuilder([callee])
    a, _b = outer.emit(0, ())
    return callee, outer.finish((), (a,))


def test_a_callee_is_narrowed_to_the_outputs_its_caller_reads():
    callee, caller = _two_output_callee()
    p = prog(callee, caller)
    out = dce(p)
    callee = out.main.ops[0].callee
    assert isinstance(callee, int)
    kept = out.bodies[callee]
    assert len(kept.outputs) == 1 and len(out.main.ops[0].outs) == 1
    assert op_count(out) == 2
    assert same_matrix(semantics(out), semantics(p))


def test_a_call_whose_results_are_all_dead_goes():
    inner = BodyBuilder()
    (coin,) = inner.emit(Prim.flip(0.5), ())
    callee = inner.finish((), (coin,))
    outer = BodyBuilder([callee])
    (x,) = outer.emit(Prim.flip(0.3), ())
    (_unused,) = outer.emit(0, ())
    p = prog(callee, outer.finish((), (x,)))
    out = dce(p)
    assert len(reachable(out)) == 1 and op_count(out) == 1
    assert same_matrix(semantics(out), semantics(p))


def test_a_callee_is_narrowed_once_to_the_union_over_its_sites():
    inner = BodyBuilder()
    (p,) = inner.emit(Prim.flip(0.5), ())
    (q,) = inner.emit(Prim.flip(0.25), ())
    (r,) = inner.emit(Prim.flip(0.75), ())
    callee = inner.finish((), (p, q, r))
    outer = BodyBuilder([callee])
    a1, _b1, _c1 = outer.emit(0, ())
    _a2, _b2, c2 = outer.emit(0, ())
    (both,) = outer.emit(Prim.and_(), (a1, c2))
    p = prog(callee, outer.finish((), (both,)))
    out = dce(p)
    assert len(out.bodies) == 2
    callee = out.main.ops[0].callee
    assert isinstance(callee, int) and len(out.bodies[callee].outputs) == 2
    assert same_matrix(semantics(out), semantics(p))


def test_a_call_to_a_body_that_conditions_stays():
    inner = BodyBuilder()
    (p,) = inner.emit(Prim.flip(0.5), ())
    inner.emit(Prim.observe(), (p,))
    callee = inner.finish((), ())
    outer = BodyBuilder([callee])
    (x,) = outer.emit(Prim.flip(0.3), ())
    outer.emit(0, ())
    p = prog(callee, outer.finish((), (x,)))
    out = dce(p)
    assert len(reachable(out)) == 2
    assert same_matrix(semantics(out), semantics(p))


def test_the_root_keeps_all_its_outputs():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    p = prog(b.finish((), (x, y)))
    assert dce(p).main.outputs == p.main.outputs


@pytest.mark.parametrize("prim,expected", [
    (Prim.flip(0.5), True), (Prim.and_(), True), (Prim.uniform(2), True),
    (Prim.from_matrix([[0.3, 0.6], [0.7, 0.4]]), True),
    (Prim.observe(), False), (Prim.equate(1), False),
])
def test_an_unread_op_goes_exactly_when_its_columns_sum_to_one(prim, expected):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    b.emit(prim, tuple(x for _ in range(prim.in_bits)))
    p = prog(b.finish((), (x,)))
    out = dce(p)
    validate(out)
    assert (op_count(out) == 1) is expected
    assert same_matrix(semantics(out), semantics(p))


@pytest.mark.parametrize("off", [1e-6, -1e-6])
def test_an_unread_table_summing_to_one_only_nearly_is_kept(off):
    """Its column departs from 1 by more than round-off, so it is a weight as written."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    b.emit(Prim.from_matrix([[0.4 + off], [0.6]]), ())
    p = prog(b.finish((), (x,)))
    out = dce(p)
    assert op_count(out) == 2
    assert same_matrix(semantics(out), semantics(p))


def test_an_unread_table_summing_to_one_up_to_round_off_is_deleted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    b.emit(Prim.from_matrix([[0.0], [1.0 + 2.0 ** -52]]), ())
    assert op_count(dce(prog(b.finish((), (x,))))) == 1
