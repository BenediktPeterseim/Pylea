"""Inlining preserves a body's matrix.

Splicing a callee in is what the lowerer does to remove a callee it cannot hold as one
matrix, and what the simplifier does under a guard.  All of its content is the aliasing of
the callee's interface wires to the call's arguments and results, so what is pinned here is
the cases where that aliasing is not a renaming: a callee returning an argument, one
returning the same wire twice, and a call whose two arguments are the same wire.
"""
from __future__ import annotations

import numpy as np
import pytest

import pylea.ir.inline as ir_inline
from pylea.ir.body import ArityError, Body, BodyBuilder, prune_wires
from pylea.ir.inline import flatten_wide_calls, inline, inline_all, inline_calls
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern, op_count, reachable, validate
from pylea.ir.reference import semantics

from test.compile.random_programs import random_program


def _program(*bodies: Body) -> Program:
    return Program(tuple(bodies), len(bodies) - 1)


def _call_sites(prog: Program, body: int):
    return [i for i, op in enumerate(prog.bodies[body].ops) if not isinstance(op.callee, Prim)]


def test_inlining_preserves_the_matrix_on_random_programs():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(120):
        prog = random_program(rng)
        for site in _call_sites(prog, prog.root):
            before = semantics(prog)
            spliced = list(prog.bodies)
            spliced[prog.root] = inline(prog, prog.root, site)
            after_prog = Program(tuple(spliced), prog.root)
            validate(after_prog)
            after = semantics(after_prog)
            assert after.shape == before.shape
            assert np.allclose(after, before, atol=1e-12)
            checked += 1
    assert checked > 50, f"only {checked} call sites exercised"


def test_inlining_a_callee_that_returns_its_argument_needs_no_extra_op():
    inner = BodyBuilder()
    x = inner.fresh(1)
    identity = inner.finish([x], [x])

    outer = BodyBuilder([identity])
    (a,) = outer.emit(Prim.flip(0.25), [])
    (b,) = outer.emit(0, [a])
    (c,) = outer.emit(Prim.not_(), [b])
    prog = _program(identity, outer.finish([], [b, c]))

    before = semantics(prog)
    spliced = inline(prog, prog.root, 1)
    after = semantics(Program((identity, spliced), 1))
    assert not spliced.callees()
    assert np.allclose(after, before)


def test_inlining_a_callee_that_returns_one_wire_twice_needs_no_extra_op():
    inner = BodyBuilder()
    (y,) = inner.emit(Prim.flip(0.4), [])
    twice = inner.finish([], [y, y])

    outer = BodyBuilder([twice])
    p, q = outer.emit(0, [])
    (r,) = outer.emit(Prim.and_(), [p, q])
    prog = _program(twice, outer.finish([], [r]))

    before = semantics(prog)
    after = semantics(Program((twice, inline(prog, prog.root, 0)), 1))
    assert np.allclose(after, before)
    # p and q are the same draw, so the conjunction is that draw, not its square.
    assert np.allclose(before.reshape(-1) / before.sum(), [0.6, 0.4])


def test_inlining_a_call_whose_two_arguments_are_one_wire_needs_no_extra_op():
    inner = BodyBuilder()
    u = inner.fresh(1)
    v = inner.fresh(1)
    (w,) = inner.emit(Prim.and_(), [u, v])
    conj = inner.finish([u, v], [w])

    outer = BodyBuilder([conj])
    (a,) = outer.emit(Prim.flip(0.3), [])
    (b,) = outer.emit(0, [a, a])
    prog = _program(conj, outer.finish([], [b]))

    before = semantics(prog)
    after = semantics(Program((conj, inline(prog, prog.root, 1)), 1))
    assert np.allclose(after, before)


def test_a_body_repeating_an_input_wire_is_rejected():
    builder = BodyBuilder()
    x = builder.fresh(1)
    with pytest.raises(ArityError, match="EQUATE"):
        builder.finish([x, x], [x])


def test_flattening_removes_the_wide_callees_and_leaves_the_rest():
    inner = BodyBuilder()
    ws = [inner.fresh(1) for _ in range(4)]
    (a,) = inner.emit(Prim.and_(), [ws[0], ws[1]])
    wide = inner.finish(ws, [a] + ws)          # 4 in, 5 out: 9 interface bits

    narrow = BodyBuilder()
    (n,) = narrow.emit(Prim.flip(0.5), [])
    small = narrow.finish([], [n])             # 1 interface bit

    outer = BodyBuilder([wide, small])
    xs = [outer.emit(1, [])[0] for _ in range(4)]
    outs = outer.emit(0, xs)
    prog = _program(wide, small, outer.finish([], [outs[0]]))

    before = semantics(prog)
    flat = flatten_wide_calls(prog, max_interface_bits=4)
    validate(flat)
    assert np.allclose(semantics(flat), before)
    for i in reachable(flat):
        for callee in flat.bodies[i].callees():
            assert flat.bodies[callee].interface_bits <= 4


def test_flatten_wide_calls_is_a_no_op_when_everything_fits():
    prog = random_program(np.random.default_rng(3))
    flat = flatten_wide_calls(prog, max_interface_bits=64)
    assert np.allclose(semantics(flat), semantics(prog))


def _count_rebuilds(monkeypatch) -> list[int]:
    """A list that grows by one entry per body `pylea.ir.inline` builds."""
    built: list[int] = []
    compact = ir_inline._compact

    def counting(*args):
        built.append(1)
        return compact(*args)

    monkeypatch.setattr(ir_inline, "_compact", counting)
    return built


def _wide(k: int = 3) -> Body:
    """``k`` inputs and ``k`` outputs, each output the conjunction of an input and a coin
    flip: ``2 * k`` interface bits."""
    b = BodyBuilder()
    xs = [b.fresh() for _ in range(k)]
    outs = []
    for x in xs:
        (f,) = b.emit(Prim.flip(0.5), ())
        (o,) = b.emit(Prim.and_(), (x, f))
        outs.append(o)
    return b.finish(xs, outs)


def _nested(depth: int) -> Program:
    """A chain of `depth` bodies of one input and one output, each calling the one below
    it once after taking the disjunction of its input with a coin flip."""
    bodies: list[Body] = []
    for i in range(depth):
        b = BodyBuilder(bodies)
        x = b.fresh()
        (f,) = b.emit(Prim.flip(0.5 + 1e-6 * i), ())
        (y,) = b.emit(Prim.or_(), (x, f))
        if i:
            (y,) = b.emit(i - 1, (y,))
        bodies.append(b.finish((x,), (y,)))
    root = BodyBuilder(bodies)
    (x,) = root.emit(Prim.flip(0.5), ())
    (y,) = root.emit(depth - 1, (x,))
    return _program(*bodies, root.finish((), (y,)))


def _flatten_one_site_at_a_time(prog: Program, max_interface_bits: int) -> Program:
    """`flatten_wide_calls` by the single-site `inline`, rescanning after every splice."""
    bodies = list(prog.bodies)
    too_wide = set()
    for i in reachable(prog):                   # callees first
        while True:
            site = next((s for s in _call_sites(Program(tuple(bodies), prog.root), i)
                         if bodies[i].ops[s].callee in too_wide), None)
            if site is None:
                break
            bodies[i] = inline(Program(tuple(bodies), prog.root), i, site)
        if bodies[i].interface_bits > max_interface_bits:
            too_wide.add(i)
    return Program(tuple(bodies), prog.root)


def _pruned(prog: Program) -> Program:
    return intern(Program(tuple(prune_wires(b) for b in prog.bodies), prog.root))


def test_flattening_rebuilds_each_body_at_most_once_however_many_sites_call_it(monkeypatch):
    wide = _wide()
    root = BodyBuilder([wide])
    ws = [root.emit(Prim.flip(0.5), ())[0] for _ in range(3)]
    for _ in range(300):
        ws = list(root.emit(0, ws))
    prog = _program(wide, root.finish((), ws))
    built = _count_rebuilds(monkeypatch)
    flat = flatten_wide_calls(prog, max_interface_bits=4)
    assert len(built) <= len(reachable(prog))
    assert len(reachable(flat)) == 1
    assert op_count(flat) == 3 + 300 * len(wide.ops)


def test_inlining_a_chain_of_nested_bodies_rebuilds_only_the_outermost(monkeypatch):
    prog = _nested(300)
    built = _count_rebuilds(monkeypatch)
    flat = flatten_wide_calls(prog, max_interface_bits=1)
    assert len(built) == 1
    validate(flat)
    assert len(reachable(flat)) == 1
    assert op_count(flat) == op_count(prog) - 300


def test_inline_all_still_splices_the_calls_a_splice_exposes():
    prog = _nested(3)
    body = inline_all(prog, prog.root, lambda _: True)
    assert not body.callees()
    spliced = Program(prog.bodies[:-1] + (body,), prog.root)
    validate(spliced)
    assert np.allclose(semantics(spliced), semantics(prog))


def test_inline_all_returns_the_caller_itself_where_no_call_qualifies():
    prog = _nested(3)
    assert inline_all(prog, prog.root, lambda _: False) is prog.main


def test_splicing_leaves_every_interface_width_unchanged():
    rng = np.random.default_rng(11)
    for _ in range(150):
        prog = random_program(rng)
        chosen = {i for i in reachable(prog) if rng.random() < 0.5}
        spliced = inline_calls(prog, chosen.__contains__)
        validate(intern(spliced))
        for i in reachable(prog):
            assert spliced.bodies[i].interface_bits == prog.bodies[i].interface_bits
            assert spliced.bodies[i].in_bits == prog.bodies[i].in_bits


def test_flattening_agrees_with_splicing_one_site_at_a_time_on_random_programs():
    rng = np.random.default_rng(5)
    for _ in range(150):
        prog = random_program(rng, n_bodies=4)
        for limit in (0, 2, 4, 8):
            assert (_pruned(flatten_wide_calls(prog, limit))
                    == _pruned(_flatten_one_site_at_a_time(prog, limit)))
