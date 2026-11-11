"""Observation splitting and propagation, and repeated observations."""
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import observe_rules
from test.simplify.programs import prog, same_matrix


def _tags(p: Program) -> list:
    return [op.callee.tag for op in p.main.ops if isinstance(op.callee, Prim)]


def _check(p: Program) -> Program:
    out = observe_rules(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


def test_an_observed_conjunction_becomes_two_observations():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    b.emit(Prim.observe(), (a,))
    out = _check(prog(b.finish((), (x,))))
    assert _tags(out).count(Tag.OBSERVE) == 2 and Tag.AND not in _tags(out)


def test_an_observed_conjunction_read_elsewhere_is_not_split():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    b.emit(Prim.observe(), (a,))
    out = _check(prog(b.finish((), (a,))))
    assert Tag.AND in _tags(out)


def test_an_observed_disjunction_is_left_alone():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (o,) = b.emit(Prim.or_(), (x, y))
    b.emit(Prim.observe(), (o,))
    p = prog(b.finish((), (x,)))
    assert observe_rules(p) == p


def test_an_observed_negation_becomes_an_effect_admitting_zero():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (n,) = b.emit(Prim.not_(), (x,))
    b.emit(Prim.observe(), (n,))
    out = _check(prog(b.finish((), (x,))))
    assert Tag.NOT not in _tags(out)
    effect = [op.callee for op in out.main.ops if not op.outs][0]
    assert isinstance(effect, Prim) and effect.matrix.tolist() == [[1.0, 0.0]]


def test_an_observed_table_output_becomes_the_observed_row():
    b = BodyBuilder()
    a = b.fresh(1)
    (x,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (a,))
    b.emit(Prim.observe(), (x,))
    out = _check(prog(b.finish((a,), ())))
    (op,) = out.main.ops
    assert isinstance(op.callee, Prim) and op.callee.matrix.tolist() == [[0.8, 0.3]]


def test_an_effect_is_absorbed_down_a_chain_of_tables_in_one_sweep():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (a,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (x,))
    (c,) = b.emit(Prim.from_matrix([[0.6, 0.1], [0.4, 0.9]]), (a,))
    b.emit(Prim.observe(), (c,))
    out = _check(prog(b.finish((), (x,))))
    assert _tags(out) == [Tag.FLIP, Tag.MATRIX] and out.main.ops[1].args == (x,)


def test_absorption_down_a_chain_stops_at_a_table_read_elsewhere():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (a,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (x,))
    (c,) = b.emit(Prim.from_matrix([[0.6, 0.1], [0.4, 0.9]]), (a,))
    b.emit(Prim.observe(), (c,))
    out = _check(prog(b.finish((), (a,))))
    assert _tags(out) == [Tag.FLIP, Tag.MATRIX, Tag.MATRIX]
    assert out.main.ops[2].args == out.main.ops[1].outs


def test_absorption_into_a_producer_takes_only_its_whole_output():
    """An effect over the result of one op and something else is left as it is."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (x,))
    b.emit(Prim.from_matrix([[0.1, 0.2, 0.3, 0.4]]), (n, y))
    p = prog(b.finish((), ()))
    assert observe_rules(p) == p


@pytest.mark.parametrize("gate,kept", [(Prim.and_(), "argument"), (Prim.or_(), "constant")])
def test_a_gate_reading_an_observed_wire_is_evaluated_at_true(gate, kept):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (z,) = b.emit(Prim.flip(0.4), ())
    b.emit(Prim.observe(), (x,))
    (r,) = b.emit(gate, (x, z))
    out = _check(prog(b.finish((), (r,))))
    assert gate.tag not in _tags(out)
    (producer,) = [op.callee for op in out.main.ops if op.outs == out.main.outputs]
    assert producer == (Prim.flip(0.4) if kept == "argument" else Prim.const(1, 1))


def test_a_negation_of_an_observed_wire_is_false():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    b.emit(Prim.observe(), (x,))
    (r,) = b.emit(Prim.not_(), (x,))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP, Tag.OBSERVE, Tag.CONST]


def test_a_selection_on_an_observed_condition_is_the_branch():
    b = BodyBuilder()
    (c,) = b.emit(Prim.flip(0.3), ())
    (t,) = b.emit(Prim.flip(0.5), ())
    (e,) = b.emit(Prim.flip(0.6), ())
    b.emit(Prim.observe(), (c,))
    (r,) = b.emit(Prim.ite(1), (c, t, e))
    out = _check(prog(b.finish((), (r,))))
    assert Tag.ITE not in _tags(out) and out.main.outputs == (t,)


def test_a_selection_of_two_results_on_an_observed_condition_keeps_each_selected_producer():
    b = BodyBuilder()
    (c,) = b.emit(Prim.flip(0.5), ())
    b.emit(Prim.observe(), (c,))
    t0, t1, e0, e1 = (b.emit(Prim.flip(q), ())[0] for q in (0.1, 0.2, 0.3, 0.4))
    r0, r1 = b.emit(Prim.ite(2), (c, t0, t1, e0, e1))
    out = _check(prog(b.finish((), (r0, r1))))
    assert Tag.ITE not in _tags(out) and out.main.outputs == (t0, t1)


def test_a_table_reading_an_observed_wire_loses_that_argument():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (z,) = b.emit(Prim.flip(0.4), ())
    b.emit(Prim.observe(), (x,))
    (r,) = b.emit(Prim.from_matrix([[0.2, 0.7, 0.1, 0.9], [0.8, 0.3, 0.9, 0.1]]), (x, z))
    out = _check(prog(b.finish((), (r,))))
    table = [op for op in out.main.ops if op.outs == (r,)][0]
    assert table.args == (z,)


def test_a_table_reading_an_observed_wire_through_an_alias_is_evaluated_at_its_value():
    """The selection on the observed `x` is `x` itself, so the table reads `x` twice once
    the selection is gone, and is restricted at the observed value in both places."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.6), ())
    b.emit(Prim.observe(), (x,))
    (a,) = b.emit(Prim.ite(1), (x, x, y))
    (r,) = b.emit(Prim.from_matrix([[0.2, 0.7, 0.1, 0.9], [0.8, 0.3, 0.9, 0.1]]), (x, a))
    out = _check(prog(b.finish((), (r, y))))
    table = [op for op in out.main.ops if op.outs == out.main.outputs[:1]][0]
    assert table.args == ()


def test_a_soft_observation_of_an_observed_wire_loses_that_argument():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (v,) = b.emit(Prim.flip(0.4), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.from_matrix([[0.2, 0.5, 0.7, 0.9]]), (x, v))
    out = _check(prog(b.finish((), (x, v))))
    soft = [op for op in out.main.ops if not op.outs and op.callee != Prim.observe()]
    assert [op.args for op in soft] == [(v,)]


def test_a_soft_observation_of_observed_wires_alone_becomes_a_constant_weight():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.from_matrix([[0.2, 0.9]]), (x,))
    out = _check(prog(b.finish((), (x,))))
    weights = [op for op in out.main.ops if not op.outs and op.callee != Prim.observe()]
    assert [op.args for op in weights] == [()]


def test_the_observing_effect_keeps_its_read_and_a_pinning_effect_keeps_its_own():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.from_matrix([[0.0, 0.5]]), (x,))          # also pins `x`, with a weight
    out = _check(prog(b.finish((), (x,))))
    assert all(op.args == (x,) for op in out.main.ops if not op.outs)


def test_a_wire_pinned_to_two_values_is_not_propagated():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (z,) = b.emit(Prim.flip(0.4), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.from_matrix([[1.0, 0.0]]), (x,))
    (r,) = b.emit(Prim.and_(), (x, z))
    p = prog(b.finish((), (r,)))
    assert Tag.AND in _tags(observe_rules(p))


def test_a_wire_pinned_to_two_values_leaves_the_other_pinned_wires_propagated():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.6), ())
    (z,) = b.emit(Prim.flip(0.4), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.from_matrix([[1.0, 0.0]]), (x,))
    b.emit(Prim.observe(), (y,))
    (r,) = b.emit(Prim.and_(), (y, z))
    out = _check(prog(b.finish((), (r,))))
    assert Tag.AND not in _tags(out)


def test_a_repeated_observation_is_deleted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.observe(), (x,))
    out = _check(prog(b.finish((), (x,))))
    assert _tags(out).count(Tag.OBSERVE) == 1


def test_a_repeated_call_of_a_deterministic_body_without_results_is_deleted():
    inner = BodyBuilder()
    i = inner.fresh(1)
    (n,) = inner.emit(Prim.not_(), (i,))
    inner.emit(Prim.observe(), (n,))
    callee = inner.finish((i,), ())
    outer = BodyBuilder([callee])
    (x,) = outer.emit(Prim.flip(0.3), ())
    outer.emit(0, (x,))
    outer.emit(0, (x,))
    p = prog(callee, outer.finish((), (x,)))
    out = _check(p)
    assert sum(1 for op in out.main.ops if op.is_call) == 1


def test_a_repeated_draw_is_not_deleted():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    soft = Prim.from_matrix([[0.5, 0.2]])
    b.emit(soft, (x,))
    b.emit(soft, (x,))
    p = prog(b.finish((), (x,)))
    out = _check(p)
    assert op_count(out) == op_count(p)
