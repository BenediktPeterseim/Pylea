"""Masked reads, and the conditions under which a read stays."""
import importlib

import numpy as np
import pytest

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, consumer_counts, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import mask, size
from pylea.simplify.common import Circuit, Propagation, constant_bits, leaving
from test.simplify.programs import circuits, prog, random_circuit, same_matrix


def _check(p: Program) -> Program:
    out = mask(p)
    validate(out)
    assert same_matrix(semantics(out), semantics(p))
    return out


def _two_disjunctions(top):
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (d,) = b.emit(top, (a, c))
    return prog(b.finish((), (d,))), x


def test_a_disjunction_under_a_disjunction_is_masked():
    p, x = _two_disjunctions(Prim.or_())
    out = _check(p)
    assert consumer_counts(out.main)[x] == 1 and size(out) == size(p) - 4


def test_two_disjunctions_under_a_conjunction_both_keep_their_read():
    """Deciding the top needs both, so neither is masked: a site for lifting."""
    p, _ = _two_disjunctions(Prim.and_())
    assert mask(p) == p


def test_a_conjunction_under_a_forced_disjunction_becomes_false():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.and_(), (x, q))
    (d,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (d,)))
    out = _check(p_)
    assert consumer_counts(out.main)[x] == 1 and op_count(out) < op_count(p_)


def test_a_disjunction_that_leaves_the_circuit_keeps_its_read():
    """`c` is an output, so its read stays; `a`'s read goes, `c` forcing the top."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (d,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (d, c)))
    out = _check(p_)
    assert consumer_counts(out.main)[x] == 1
    assert any(x in op.args and op.outs[0] in out.main.outputs for op in out.main.ops)


def test_a_gate_an_observation_reads_leaves_the_circuit():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    b.emit(Prim.observe(), (c,))
    (d,) = b.emit(Prim.or_(), (a, c))
    p_ = prog(b.finish((), (d,)))
    out = _check(p_)
    assert consumer_counts(out.main)[x] == 1
    observed = next(op.args[0] for op in out.main.ops if op.callee == Prim.observe())
    assert x in next(op.args for op in out.main.ops if observed in op.outs)


def test_a_cut_need_not_be_a_dominator():
    """`x` decides neither top, yet while `x` is 1 each disjunction masks the other at the
    first top, and `n` is 0 so the conjunction is 0 whatever `c` is; one of the two reads
    goes, and the other, which then does the masking, stays."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (r,) = b.emit(Prim.flip(0.6), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (n,) = b.emit(Prim.not_(), (x,))
    (t1,) = b.emit(Prim.or_(), (a, c))
    (k,) = b.emit(Prim.and_(), (c, n))
    (t2,) = b.emit(Prim.and_(), (k, r))
    p_ = prog(b.finish((), (t1, t2)))
    out = _check(p_)
    assert consumer_counts(out.main)[x] == 2 and op_count(out) == op_count(p_) - 1


def test_the_dual_drops_a_conjunction_read_under_a_conjunction():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.and_(), (x, p))
    (c,) = b.emit(Prim.and_(), (x, q))
    (d,) = b.emit(Prim.and_(), (a, c))
    p_ = prog(b.finish((), (d,)))
    assert consumer_counts(_check(p_).main)[x] == 1


def test_rewrites_in_one_pass_do_not_rely_on_each_other():
    """Two reads of `x` under one disjunction: one may go, the other must stay."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.3), ())
    (p,) = b.emit(Prim.flip(0.4), ())
    (q,) = b.emit(Prim.flip(0.5), ())
    (a,) = b.emit(Prim.or_(), (x, p))
    (c,) = b.emit(Prim.or_(), (x, q))
    (d,) = b.emit(Prim.or_(), (a, c))
    (e,) = b.emit(Prim.and_(), (d, p))
    p_ = prog(b.finish((), (e,)))
    out = _check(p_)
    assert consumer_counts(out.main)[x] == 1


CORPUS = circuits(60) + circuits(30, observe=True)


@pytest.mark.parametrize("p", CORPUS, ids=range(len(CORPUS)))
def test_a_random_circuit_keeps_its_matrix(p):
    out = _check(p)
    if out != p:
        assert size(out) < size(p)


def test_the_random_circuits_exercise_the_rule():
    assert sum(1 for p in CORPUS if mask(p) != p) > len(CORPUS) // 2


def _reference(body):
    """The rewrites `mask` makes in one pass, found from the definition: each reader is
    tried against the wires a propagation leaving its gate out decides, and the search
    for an open path goes all the way."""
    circuit = Circuit(body)
    out = leaving(body)
    constant = {w for op in body.ops if constant_bits(op.callee) is not None for w in op.outs}
    rewrites, excluded = {}, set()

    def decided(x, value, left_out):
        propagation = Propagation(circuit, x, value, excluded | left_out)
        propagation.advance(len(body.ops))
        return propagation.known

    for x, js in circuit.readers.items():
        if len(js) < 2 or x in constant or body.bits[x] != 1:
            continue
        eligible = [j for j in js if circuit.tags[j] is not None and len(circuit.args[j]) == 2
                    and body.ops[j].args.count(x) == 1]
        for value in (1, 0):
            if not eligible or len(decided(x, value, set())) == 1:
                continue
            for j in eligible:
                if j in excluded:
                    continue
                blocked = decided(x, value, {j})
                stack, seen, open_ = [circuit.outs[j]], set(), False
                while stack and not open_:
                    w = stack.pop()
                    if w in seen or w in blocked:
                        continue
                    seen.add(w)
                    open_ = w in out
                    stack.extend(circuit.outs[k] for k in circuit.readers.get(w, ()))
                if open_:
                    continue
                decides = (body.ops[j].callee == Prim.or_()) == (value == 0)
                (c,) = [w for w in body.ops[j].args if w != x]
                rewrites[j] = None if decides else c
                excluded.add(j)
    return rewrites


@pytest.mark.parametrize("seed", range(300))
def test_the_rewrites_of_one_pass_are_those_the_definition_gives(seed):
    rng = np.random.default_rng(seed)
    body = random_circuit(rng, int(rng.integers(2, 7)), int(rng.integers(3, 30)),
                          observe=seed % 3 == 0).main
    module = importlib.import_module("pylea.simplify.mask")
    assert module._masked(body) == _reference(body)
