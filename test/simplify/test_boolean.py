"""The equational Boolean identities, one test per identity plus what they decline."""
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import bool_rules
from test.simplify.programs import prog, same_matrix


def _tags(p: Program) -> list:
    return [op.callee.tag for op in p.main.ops if isinstance(op.callee, Prim)]


def _check(p: Program) -> Program:
    out = bool_rules(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


@pytest.mark.parametrize("gate,value,collapses", [
    (Prim.and_(), 1, "argument"), (Prim.and_(), 0, "constant"),
    (Prim.or_(), 0, "argument"), (Prim.or_(), 1, "constant"),
])
def test_a_gate_on_a_constant_is_the_other_argument_or_a_constant(gate, value, collapses):
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(value, 1), ())
    (x,) = b.emit(Prim.flip(0.3), ())
    (r,) = b.emit(gate, (c, x))
    out = _check(prog(b.finish((), (r,))))
    if collapses == "argument":
        assert _tags(out) == [Tag.FLIP]            # the gate and the constant are gone
    else:
        assert _tags(out) == [Tag.CONST]           # the gate became one; the coin is dead


def test_a_shared_constant_stays_for_its_other_readers():
    b = BodyBuilder()
    (one,) = b.emit(Prim.const(1, 1), ())
    (x,) = b.emit(Prim.flip(0.3), ())
    (r,) = b.emit(Prim.and_(), (one, x))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (one,))
    (t,) = b.emit(Prim.or_(), (r, m))
    out = _check(prog(b.finish((), (t,))))
    assert Tag.AND not in _tags(out) and Tag.CONST in _tags(out)


def test_a_flip_of_zero_counts_as_a_constant():
    b = BodyBuilder()
    (never,) = b.emit(Prim.flip(0.0), ())
    (x,) = b.emit(Prim.flip(0.3), ())
    (r,) = b.emit(Prim.and_(), (never, x))
    assert Tag.AND not in _tags(_check(prog(b.finish((), (r,)))))


def test_the_negation_of_a_constant_is_the_other_constant():
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(0, 1), ())
    (r,) = b.emit(Prim.not_(), (c,))
    assert _tags(_check(prog(b.finish((), (r,))))) == [Tag.CONST]


def test_double_negation_takes_the_inner_not_when_nothing_else_reads_it():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (r,) = b.emit(Prim.not_(), (n,))
    assert op_count(_check(prog(b.finish((), (r,))))) == 1


def test_double_negation_keeps_the_inner_not_when_it_is_read_elsewhere():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (r,) = b.emit(Prim.not_(), (n,))
    (keep,) = b.emit(Prim.and_(), (n, y))
    out = _check(prog(b.finish((), (keep, r))))
    assert _tags(out).count(Tag.NOT) == 1
    assert out.main.outputs[1] == out.main.ops[0].outs[0]          # `r` is `x` itself


def test_the_inner_not_stays_when_its_readers_arrive_through_an_alias():
    """The reads of `n` here come from ops whose own arguments alias to it: an AND with a
    constant identity resolves onto `n`, its readers become `n`'s, and only then does a
    double negation consume `n`.  The `NOT` producing `n` has to stay for those readers."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (one,) = b.emit(Prim.const(1, 1), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (u,) = b.emit(Prim.and_(), (one, n))           # aliases onto n
    (y,) = b.emit(Prim.flip(0.6), ())
    (r,) = b.emit(Prim.and_(), (u, y))             # reads n through the alias, and is kept
    (z,) = b.emit(Prim.not_(), (u,))               # the double negation consuming n
    (keep,) = b.emit(Prim.or_(), (r, z))
    _check(prog(b.finish((), (keep,))))


@pytest.mark.parametrize("gate,value", [(Prim.and_(), 0), (Prim.or_(), 1)])
def test_a_gate_over_a_wire_and_its_negation_is_a_constant(gate, value):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (r,) = b.emit(gate, (x, n))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.CONST]


def test_a_gate_over_a_wire_and_a_negation_read_elsewhere_stays():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (r,) = b.emit(Prim.and_(), (x, n))
    out = _check(prog(b.finish((), (r, n))))
    assert Tag.AND in _tags(out)


@pytest.mark.parametrize("outer,inner", [(Prim.and_(), Prim.or_()), (Prim.or_(), Prim.and_())])
def test_the_negation_of_the_other_argument_inside_a_dual_gate_is_absorbed(outer, inner):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (h,) = b.emit(inner, (y, n))
    (r,) = b.emit(outer, (h, x))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP, Tag.FLIP, outer.tag]
    assert set(out.main.ops[-1].args) == {0, 1}


@pytest.mark.parametrize("outer,inner", [(Prim.and_(), Prim.or_()), (Prim.or_(), Prim.and_())])
def test_a_wire_inside_a_dual_gate_under_its_own_negation_is_absorbed(outer, inner):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (h,) = b.emit(inner, (x, y))
    (r,) = b.emit(outer, (n, h))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP, Tag.FLIP, Tag.NOT, outer.tag]


@pytest.mark.parametrize("outer,inner", [(Prim.or_(), Prim.and_()), (Prim.and_(), Prim.or_())])
def test_the_consensus_term_of_a_selection_in_gates_is_absorbed(outer, inner):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (z,) = b.emit(Prim.flip(0.5), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (m,) = b.emit(inner, (x, y))
    (m_,) = b.emit(inner, (n, z))
    (p,) = b.emit(outer, (m, m_))
    (q,) = b.emit(inner, (z, y))
    (r,) = b.emit(outer, (q, p))
    out = _check(prog(b.finish((), (r,))))
    assert op_count(out) == op_count(prog(b.finish((), (r,)))) - 2


def test_a_selection_in_gates_between_equal_branches_is_the_branch():
    b = BodyBuilder()
    (c,) = b.emit(Prim.flip(0.3), ())
    (a,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (c,))
    (m,) = b.emit(Prim.and_(), (c, a))
    (m_,) = b.emit(Prim.and_(), (a, n))
    (r,) = b.emit(Prim.or_(), (m, m_))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP]


def test_constant_condition_follows_from_constant_propagation():
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(1, 1), ())
    (a,) = b.emit(Prim.flip(0.4), ())
    (d,) = b.emit(Prim.flip(0.6), ())
    (n,) = b.emit(Prim.not_(), (c,))
    (m,) = b.emit(Prim.and_(), (c, a))
    (m_,) = b.emit(Prim.and_(), (n, d))
    (r,) = b.emit(Prim.or_(), (m, m_))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP] and out.main.ops[0].callee == Prim.flip(0.4)


def test_a_selection_in_gates_whose_other_branch_is_true_becomes_a_disjunction():
    b = BodyBuilder()
    (c,) = b.emit(Prim.flip(0.3), ())
    (a,) = b.emit(Prim.flip(0.4), ())
    (n,) = b.emit(Prim.not_(), (c,))
    (m,) = b.emit(Prim.and_(), (c, a))
    (m_,) = b.emit(Prim.and_(), (n, n))
    (r,) = b.emit(Prim.or_(), (m, m_))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP, Tag.FLIP, Tag.NOT, Tag.OR]


def test_a_selection_on_a_constant_condition_is_the_branch():
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(1, 1), ())
    t0, t1 = b.emit(Prim.uniform(2), ())
    e0, e1 = b.emit(Prim.uniform(2), ())
    r0, r1 = b.emit(Prim.ite(2), (c, t0, t1, e0, e1))
    out = _check(prog(b.finish((), (r0, r1))))
    assert _tags(out) == [Tag.UNIFORM] and out.main.outputs == out.main.ops[0].outs


@pytest.mark.parametrize("value", [0, 1])
def test_a_selection_of_two_results_on_a_constant_condition_keeps_each_selected_producer(value):
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(value, 1), ())
    t0, t1, e0, e1 = (b.emit(Prim.flip(q), ())[0] for q in (0.1, 0.2, 0.3, 0.4))
    r0, r1 = b.emit(Prim.ite(2), (c, t0, t1, e0, e1))
    out = _check(prog(b.finish((), (r0, r1))))
    kept = [op.callee for op in out.main.ops]
    assert kept == ([Prim.flip(0.1), Prim.flip(0.2)] if value else
                    [Prim.flip(0.3), Prim.flip(0.4)])


def test_a_selection_of_two_results_between_equal_branches_is_the_branches():
    b = BodyBuilder()
    (c,) = b.emit(Prim.flip(0.5), ())
    (t0,) = b.emit(Prim.flip(0.1), ())
    (t1,) = b.emit(Prim.flip(0.2), ())
    r0, r1 = b.emit(Prim.ite(2), (c, t0, t1, t0, t1))
    out = _check(prog(b.finish((), (r0, r1))))
    assert [op.callee for op in out.main.ops] == [Prim.flip(0.1), Prim.flip(0.2)]


def test_a_selection_between_equal_branches_is_the_branch():
    b = BodyBuilder()
    (c,) = b.emit(Prim.flip(0.3), ())
    (t,) = b.emit(Prim.flip(0.5), ())
    (r,) = b.emit(Prim.ite(1), (c, t, t))
    out = _check(prog(b.finish((), (r,))))
    assert _tags(out) == [Tag.FLIP]


def test_an_opaque_table_and_a_call_on_a_constant_are_left_alone():
    inner = BodyBuilder()
    i = inner.fresh(1)
    (n,) = inner.emit(Prim.not_(), (i,))
    callee = inner.finish((i,), (n,))
    outer = BodyBuilder([callee])
    (one,) = outer.emit(Prim.const(1, 1), ())
    (a,) = outer.emit(0, (one,))
    (c,) = outer.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (one,))
    (r,) = outer.emit(Prim.or_(), (a, c))
    p = prog(callee, outer.finish((), (r,)))
    assert bool_rules(p) == p
