"""Rebuilding a body op by op, the canonical form of a source, and propagation."""
import numpy as np
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import validate
from pylea.ir.reference import semantics
from pylea.simplify import const_fold, simplify
from pylea.simplify.common import Circuit, Propagation, Sweep, canonical
from test.simplify.programs import prog, random_circuit, same_matrix


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_aliasing_every_result_of_one_op_keeps_the_producer_of_every_target(order):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (y,) = b.emit(Prim.flip(0.6), ())
    pair = b.emit(Prim.from_matrix(np.eye(4)), (x, y))
    body = b.finish((), tuple(pair))
    s = Sweep(body)
    for op in body.ops:
        args = s.begin(op)
        if op.outs == tuple(pair):
            for k in order:
                s.alias(op.outs[k], args[k])
        else:
            s.keep(op, args)
    out = s.finish()
    validate(prog(out))
    assert [op.callee for op in out.ops] == [Prim.flip(0.3), Prim.flip(0.6)]
    assert out.outputs == tuple(op.outs[0] for op in out.ops)


def test_a_dropped_effect_takes_the_ops_only_it_read_and_leaves_one_that_conditions():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (w,) = b.emit(Prim.from_matrix([[0.5], [0.2]]), ())
    b.emit(Prim.observe(), (x,))
    b.emit(Prim.observe(), (w,))
    body = b.finish((), ())
    s = Sweep(body)
    for op in body.ops:
        args = s.begin(op)
        if not op.outs:
            s.drop()
        else:
            s.keep(op, args)
    out = s.finish()
    assert [op.callee for op in out.ops] == [Prim.from_matrix([[0.5], [0.2]])]


# ----------------------------------------------------------------- canonical form

@pytest.mark.parametrize("column", [
    [[0.75], [0.25]],
    [[0.0], [1.0]],
])
def test_a_column_that_is_a_flip_or_a_constant_becomes_one(column):
    assert canonical(Prim.from_matrix(column)).tag is not Tag.MATRIX


@pytest.mark.parametrize("column", [
    [[0.3], [0.7]],                     # sums to 1, but 1 - 0.7 is not 0.3 in floating point
    [[0.3], [0.70001]],                 # sums to 1 within the tolerance of `is_stochastic`
    [[0.0], [1.0000000000000002]],      # 1 exceeded by round-off
    [[0.5], [0.6]],
])
def test_a_column_that_is_no_flip_s_matrix_stays_a_table_with_its_entries(column):
    out = canonical(Prim.from_matrix(column))
    assert out.tag is Tag.MATRIX
    assert np.array_equal(out.matrix, np.asarray(column))


def test_a_source_whose_mass_exceeds_one_by_round_off_folds_into_a_certain_reader():
    """Four outcomes whose sum rounds to 1 + 2**-52, read by a table true for each."""
    b = BodyBuilder()
    (u, v) = b.emit(Prim.from_matrix([[0.56], [0.02], [0.32], [0.1]]), ())
    (seen,) = b.emit(Prim.from_matrix([[0.0] * 4, [1.0] * 4]), (u, v))
    p = prog(b.finish((), (seen,)))
    folded = const_fold(p)
    (source,) = (op.callee for op in folded.main.ops)
    assert isinstance(source, Prim) and source.matrix[1, 0] > 1.0
    for out in (folded, simplify(p)):
        validate(out)
        assert same_matrix(semantics(out), semantics(p))


# ------------------------------------------------------------------- propagation

def _circuit_with_other_readers():
    """`x` read by a gate, a selection of two results, an observation and a table."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    pair = b.emit(Prim.ite(2), (x, a, p, p, a))
    b.emit(Prim.observe(), (x,))
    (t,) = b.emit(Prim.from_matrix([[0.2, 0.7], [0.8, 0.3]]), (x,))
    (n,) = b.emit(Prim.not_(), (a,))
    return b.finish((), (n, t, *pair)), x, a, n


def test_a_propagation_passes_over_readers_that_are_not_gates():
    body, x, a, n = _circuit_with_other_readers()
    propagation = Propagation(Circuit(body), x, 1)
    propagation.advance(len(body.ops))
    assert propagation.known == {x: 1, a: 1, n: 0}
    assert propagation.done


@pytest.mark.parametrize("seed", range(150))
def test_excluding_a_gate_gives_the_propagation_that_left_it_out_from_the_start(seed):
    rng = np.random.default_rng(seed)
    body = random_circuit(rng, int(rng.integers(2, 6)), int(rng.integers(3, 25))).main
    circuit = Circuit(body)
    shared = [w for w, js in circuit.readers.items() if len(js) >= 2]
    if not shared:
        return
    x = shared[int(rng.integers(len(shared)))]
    js = circuit.readers[x]
    bit = {j: 1 << n for n, j in enumerate(js)}
    value = int(rng.integers(2))
    reached = int(rng.integers(len(body.ops)))
    excluded: set[int] = set()
    propagation = Propagation(circuit, x, value, excluded, bit)
    propagation.advance(reached)
    for j in rng.permutation(js)[:int(rng.integers(1, len(js) + 1))]:
        propagation.exclude(int(j))
        fresh = Propagation(circuit, x, value, set(excluded), bit)
        fresh.advance(reached)
        assert (propagation.known, propagation.needs) == (fresh.known, fresh.needs)
    propagation.advance(len(body.ops))
    whole = Propagation(circuit, x, value, set(excluded), bit)
    whole.advance(len(body.ops))
    assert (propagation.known, propagation.needs) == (whole.known, whole.needs)
