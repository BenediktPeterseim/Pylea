"""Constant folding: sources into readers, foldable blocks, and bottom CSE."""
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program, intern, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import block_fold, bottom_cse, const_fold
from test.simplify.programs import prog, same_matrix


def _tags(p: Program) -> list:
    return [op.callee.tag for op in p.main.ops if isinstance(op.callee, Prim)]


def _check(rule, p: Program) -> Program:
    out = rule(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


# ------------------------------------------------------------- sources into readers

def test_a_flip_feeding_one_table_becomes_its_column():
    b = BodyBuilder()
    a = b.fresh(1)
    (x,) = b.emit(Prim.flip(0.25), ())
    (y,) = b.emit(Prim.from_matrix([[0.2, 0.7, 0.1, 0.9], [0.8, 0.3, 0.9, 0.1]]), (x, a))
    out = _check(const_fold, intern(prog(b.finish((a,), (y,)))))
    assert len(out.main.ops) == 1 and out.main.ops[0].args == (a,)


def test_a_flip_read_twice_is_not_folded():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.25), ())
    (u,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (x,))
    (v,) = b.emit(Prim.from_matrix([[0.4, 0.1], [0.6, 0.9]]), (x,))
    (t,) = b.emit(Prim.and_(), (u, v))
    p = intern(prog(b.finish((), (t,))))
    assert const_fold(p) == p


def test_a_constant_read_twice_is_folded_into_both_readers():
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(1, 1), ())
    (u,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (c,))
    (v,) = b.emit(Prim.from_matrix([[0.4, 0.1], [0.6, 0.9]]), (c,))
    (t,) = b.emit(Prim.and_(), (u, v))
    out = _check(const_fold, intern(prog(b.finish((), (t,)))))
    assert Tag.CONST not in _tags(out) and all(not op.args for op in out.main.ops[:2])


def test_a_constant_feeding_a_gate_is_left_to_the_boolean_rules():
    b = BodyBuilder()
    (c,) = b.emit(Prim.const(1, 1), ())
    (x,) = b.emit(Prim.flip(0.5), ())
    (t,) = b.emit(Prim.and_(), (c, x))
    p = intern(prog(b.finish((), (t,))))
    assert const_fold(p) == p


def test_a_gate_over_flips_alone_becomes_a_flip():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.25), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    (t,) = b.emit(Prim.and_(), (x, y))
    out = _check(const_fold, intern(prog(b.finish((), (t,)))))
    assert out.main.ops == (type(out.main.ops[0])(Prim.flip(0.125), (), (t,)),) or \
        [op.callee for op in out.main.ops] == [Prim.flip(0.125)]


def test_a_gate_over_a_flip_and_something_else_keeps_its_tag():
    b = BodyBuilder()
    a = b.fresh(1)
    (x,) = b.emit(Prim.flip(0.25), ())
    (t,) = b.emit(Prim.and_(), (x, a))
    p = intern(prog(b.finish((a,), (t,))))
    assert const_fold(p) == p


def test_a_source_summing_to_one_only_nearly_folds_with_its_entries_kept():
    """The column sums to 1 within the tolerance of `is_stochastic`, but is no `flip`."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.from_matrix([[0.3], [0.70001]]), ())
    (n,) = b.emit(Prim.not_(), (x,))
    p = prog(b.finish((), (n,)))
    out = _check(const_fold, p)
    (folded,) = (op.callee for op in out.main.ops)
    assert isinstance(folded, Prim) and folded.matrix.tolist() == [[0.70001], [0.3]]


def test_the_negation_of_a_flip_is_the_opposite_flip():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.25), ())
    (n,) = b.emit(Prim.not_(), (x,))
    out = _check(const_fold, intern(prog(b.finish((), (n,)))))
    assert [op.callee for op in out.main.ops] == [Prim.flip(0.75)]


def test_an_observed_flip_becomes_a_scalar_weight():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.25), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    b.emit(Prim.observe(), (x,))
    out = _check(const_fold, intern(prog(b.finish((), (y,)))))
    assert any(isinstance(op.callee, Prim) and op.callee.in_bits == 0 and op.callee.out_bits == 0
               for op in out.main.ops)


# ---------------------------------------------------------------------- blocks

def test_a_closed_block_over_private_flips_becomes_one_flip():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    (z,) = b.emit(Prim.flip(0.5), ())
    (n,) = b.emit(Prim.not_(), (y,))
    (a,) = b.emit(Prim.and_(), (x, n))
    (o,) = b.emit(Prim.or_(), (a, z))
    out = _check(block_fold, prog(b.finish((), (o,))))
    assert [op.callee for op in out.main.ops] == [Prim.flip(0.625)]


def test_a_block_whose_inner_wire_is_read_outside_is_not_folded():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (o,) = b.emit(Prim.or_(), (a, y))
    p = prog(b.finish((), (o, a)))
    out = _check(block_fold, p)
    assert Tag.OR in _tags(out)


def test_a_conjunction_of_dependent_wires_is_not_foldable():
    """`x AND (x OR y)` depends on `x` twice, so no weight is computed bottom up."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    (o,) = b.emit(Prim.or_(), (x, y))
    (a,) = b.emit(Prim.and_(), (x, o))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (a,))
    p = prog(b.finish((), (m,)))
    assert block_fold(p) == p


def test_a_disjunction_of_dependent_wires_that_conflict_is_foldable():
    """`x AND y` and `x AND NOT y` are never both true, so the weights add."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (n,) = b.emit(Prim.not_(), (y,))
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, n))
    (o,) = b.emit(Prim.or_(), (a, c))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (o,))
    out = _check(block_fold, prog(b.finish((), (m,))))
    assert [op.callee for op in out.main.ops][0] == Prim.flip(0.5)


def test_a_shared_flip_blocks_the_fold_unless_guarded():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (a,))
    (o,) = b.emit(Prim.or_(), (m, x))
    p = prog(b.finish((), (o,)))
    assert block_fold(p) == p


def test_a_guarded_block_is_evaluated_on_a_private_copy_of_the_shared_flip():
    """`a = x AND y` matters only while `NOT x`; `x` matters elsewhere only while `x`."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (z,) = b.emit(Prim.flip(0.75), ())
    (n,) = b.emit(Prim.not_(), (x,))
    (a,) = b.emit(Prim.and_(), (y, z))
    (g,) = b.emit(Prim.and_(), (a, n))          # the block's output matters while NOT x
    (h,) = b.emit(Prim.and_(), (x, z))          # z elsewhere matters while x
    (o,) = b.emit(Prim.or_(), (g, h))
    p = prog(b.finish((), (o,)))
    out = _check(block_fold, p)
    assert op_count(out) < op_count(p)


def test_an_unnormalised_source_folds_exactly_when_private():
    b = BodyBuilder()
    (x,) = b.emit(Prim.from_matrix([[0.5], [0.2]]), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (a,) = b.emit(Prim.or_(), (x, y))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (a,))
    out = _check(block_fold, prog(b.finish((), (m,))))
    assert len(out.main.ops) == 2


@pytest.mark.parametrize("gate", [Prim.and_(), Prim.or_()])
def test_each_rule_of_the_weight_is_exact_over_unnormalised_sources(gate):
    """A negation, a disjunction of conflicting wires, and `gate` over disjoint ones."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.from_matrix([[0.5], [0.2]]), ())
    (y,) = b.emit(Prim.from_matrix([[0.3], [0.6]]), ())
    (z,) = b.emit(Prim.flip(0.25), ())
    (n,) = b.emit(Prim.not_(), (y,))
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, n))
    (o,) = b.emit(Prim.or_(), (a, c))
    (r,) = b.emit(gate, (o, z))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (r,))
    out = _check(block_fold, prog(b.finish((), (m,))))
    assert len(out.main.ops) == 2


def test_a_block_whose_disjuncts_never_both_true_sum_to_one_folds_to_no_negative_entry():
    """The four disjuncts cover every assignment, so their probabilities sum to 1, and to
    slightly more in floating point."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.8259915815030567), ())
    (y,) = b.emit(Prim.flip(0.9164608489206628), ())
    (z,) = b.emit(Prim.flip(0.7781581586734887), ())
    (ny,) = b.emit(Prim.not_(), (y,))
    (nz,) = b.emit(Prim.not_(), (z,))
    (nx,) = b.emit(Prim.not_(), (x,))
    (xy,) = b.emit(Prim.and_(), (x, y))
    (xyz,) = b.emit(Prim.and_(), (xy, z))
    (x_ny,) = b.emit(Prim.and_(), (x, ny))
    (xy_nz,) = b.emit(Prim.and_(), (xy, nz))
    (o1,) = b.emit(Prim.or_(), (xyz, x_ny))
    (o2,) = b.emit(Prim.or_(), (o1, xy_nz))
    (top,) = b.emit(Prim.or_(), (o2, nx))
    p = prog(b.finish((), (top,)))
    out = _check(block_fold, p)
    (folded,) = (op.callee for op in out.main.ops)
    assert isinstance(folded, Prim) and (folded.matrix >= 0.0).all()


def test_a_folded_block_keeps_the_relative_precision_of_a_small_probability():
    """`NOT(x1 OR x2 OR x3 OR x4)`, each `x` true with probability ``1 - 1e-5``, is true
    with probability about ``1e-20``, which ``1 - P(x1 OR ... OR x4)`` would round to 0."""
    b = BodyBuilder()
    (w,) = b.emit(Prim.flip(1 - 1e-5), ())
    for _ in range(3):
        (x,) = b.emit(Prim.flip(1 - 1e-5), ())
        (w,) = b.emit(Prim.or_(), (w, x))
    (n,) = b.emit(Prim.not_(), (w,))
    p = prog(b.finish((), (n,)))
    out = _check(block_fold, p)
    assert len(out.main.ops) == 1
    want, got = semantics(p)[1, 0], semantics(out)[1, 0]
    assert 0.0 < want < 1e-19
    assert abs(got / want - 1.0) < 1e-12


def test_a_block_over_a_source_of_no_weight_folds_to_a_source_of_no_weight():
    b = BodyBuilder()
    (x,) = b.emit(Prim.from_matrix([[0.0], [0.0]]), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (o,) = b.emit(Prim.or_(), (x, y))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (o,))
    out = _check(block_fold, prog(b.finish((), (m,))))
    folded = out.main.ops[0].callee
    assert isinstance(folded, Prim) and folded.matrix.tolist() == [[0.0], [0.0]]


# ------------------------------------------------------------------ bottom CSE

def test_equal_gates_over_private_flips_merge_into_one_flip():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, y))
    (m,) = b.emit(Prim.from_matrix([[0.2, 0.7, 0.1, 0.9], [0.8, 0.3, 0.9, 0.1]]), (a, c))
    out = _check(bottom_cse, prog(b.finish((), (m,))))
    assert op_count(out) == 2


def test_equal_gates_over_a_flip_read_outside_are_not_merged():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, y))
    p = prog(b.finish((), (a, c, x)))
    assert bottom_cse(p) == p


@pytest.mark.parametrize("rule", [const_fold, block_fold, bottom_cse])
def test_a_program_without_sources_is_untouched(rule):
    b = BodyBuilder()
    i, j = b.fresh(1), b.fresh(1)
    (a,) = b.emit(Prim.and_(), (i, j))
    p = prog(b.finish((i, j), (a,)))
    assert rule(p) == p
