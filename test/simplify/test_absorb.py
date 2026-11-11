"""Absorption by implication."""
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import absorb
from pylea.simplify.absorb import idempotence
from test.simplify.programs import circuits, prog, same_matrix


def _tags(p: Program) -> list:
    return [op.callee.tag for op in p.main.ops if isinstance(op.callee, Prim)]


def _check(p: Program) -> Program:
    out = absorb(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


def test_a_disjunction_absorbs_an_argument_that_implies_the_other():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (o,) = b.emit(Prim.or_(), (a, x))
    out = _check(prog(b.finish((), (o,))))
    assert _tags(out) == [Tag.FLIP] and out.main.outputs == (x,)


def test_a_conjunction_is_the_argument_that_implies_the_other():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (o,) = b.emit(Prim.or_(), (x, y))
    (a,) = b.emit(Prim.and_(), (o, x))
    out = _check(prog(b.finish((), (a,))))
    assert Tag.AND not in _tags(out) and out.main.outputs == (x,)


@pytest.mark.parametrize("gate", [Prim.and_(), Prim.or_()])
def test_a_gate_on_one_wire_twice_is_that_wire(gate):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (r,) = b.emit(gate, (x, x))
    out = _check(prog(b.finish((), (r,))))
    assert op_count(out) == 1


def test_absorption_through_a_case_split():
    b = BodyBuilder()
    (m,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.and_(), (m, p))
    (c,) = b.emit(Prim.and_(), (m, q))
    (o,) = b.emit(Prim.or_(), (a, c))
    (r,) = b.emit(Prim.or_(), (o, m))            # o implies m
    out = _check(prog(b.finish((), (r,))))
    assert out.main.outputs == (m,)


def test_gates_whose_arguments_do_not_imply_each_other_stay():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (o,) = b.emit(Prim.or_(), (x, y))
    p = prog(b.finish((), (o,)))
    assert absorb(p) == p


def test_absorbed_gates_in_one_sweep_agree_with_enumeration():
    for p in circuits(40):
        _check(p)


def test_a_deadline_already_passed_leaves_the_program_alone():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (r,) = b.emit(Prim.and_(), (x, x))
    p = prog(b.finish((), (r,)))
    assert absorb(p, deadline=-1.0) == p


@pytest.mark.parametrize("gate", [Prim.and_(), Prim.or_()])
def test_idempotence_deletes_a_gate_on_one_wire_twice_and_absorbs_nothing_else(gate):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.4), ())
    (r,) = b.emit(gate, (x, x))
    (a,) = b.emit(Prim.and_(), (r, y))
    (o,) = b.emit(Prim.or_(), (a, x))                    # a implies x
    p = prog(b.finish((), (o,)))
    out = idempotence(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    assert _tags(out) == [Tag.FLIP, Tag.FLIP, Tag.AND, Tag.OR]
    assert out.main.ops[2].args == (x, y)
