"""Inlining of a body called once, and the refusal of every other body."""
import numpy as np

import pylea.ir.inline as ir_inline
from pylea.ir.body import Body, BodyBuilder, prune_wires
from pylea.ir.inline import inline_all
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern, op_count, reachable, use_counts, validate
from pylea.ir.reference import semantics
from pylea.simplify import inline, size
from test.simplify.programs import prog, same_matrix


def _coin_pair() -> Body:
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (r,) = b.emit(Prim.and_(), (x, y))
    return b.finish((), (r,))


def test_a_body_with_one_call_site_is_spliced_in():
    callee = _coin_pair()
    outer = BodyBuilder([callee])
    (a,) = outer.emit(0, ())
    (n,) = outer.emit(Prim.not_(), (a,))
    p = prog(callee, outer.finish((), (n,)))
    out = inline(p)
    assert len(reachable(out)) == 1
    assert op_count(out) == op_count(p) - 1
    assert same_matrix(semantics(out), semantics(p))


def test_splicing_keeps_a_callee_that_returns_its_own_argument():
    inner = BodyBuilder()
    i = inner.fresh(1)
    callee = inner.finish((i,), (i, i))
    outer = BodyBuilder([callee])
    (x,) = outer.emit(Prim.flip(0.3), ())
    a, c = outer.emit(0, (x,))
    (r,) = outer.emit(Prim.and_(), (a, c))
    p = prog(callee, outer.finish((), (r,)))
    out = inline(p)
    validate(out)
    assert len(reachable(out)) == 1
    assert same_matrix(semantics(out), semantics(p))


def test_a_body_with_several_sites_is_refused_whatever_its_size():
    inner = BodyBuilder()
    i = inner.fresh(1)
    (n,) = inner.emit(Prim.not_(), (i,))
    callee = inner.finish((i,), (n,))
    outer = BodyBuilder([callee])
    (x,) = outer.emit(Prim.flip(0.3), ())
    (a,) = outer.emit(0, (x,))
    (c,) = outer.emit(0, (x,))
    (r,) = outer.emit(Prim.and_(), (a, c))
    p = prog(callee, outer.finish((), (r,)))
    assert inline(p) == p


def test_splicing_a_one_op_body_at_every_site_can_raise_the_size():
    """The reason a one-op body with several sites is refused: here the op has a result
    the body does not return, so each copy is larger than the call it replaces."""
    inner = BodyBuilder()
    i = inner.fresh(1)
    kept, _unread = inner.emit(Prim.from_matrix(np.full((4, 2), 0.25)), (i,))
    callee = inner.finish((i,), (kept,))
    outer = BodyBuilder([callee])
    (x,) = outer.emit(Prim.flip(0.3), ())
    results = [outer.emit(0, (x,))[0] for _ in range(8)]
    p = prog(callee, outer.finish((), tuple(results)))
    spliced = intern(Program((callee, prune_wires(inline_all(p, 1, lambda i: i == 0))), 1))
    assert len(reachable(spliced)) == 1
    assert size(spliced) > size(p)
    assert inline(p) == p


def test_sites_are_counted_over_the_whole_call_graph():
    h = _coin_pair()
    wrappers = [h]
    for _ in range(2):
        w = BodyBuilder(wrappers)
        (a,) = w.emit(0, ())
        (n,) = w.emit(Prim.not_(), (a,))
        wrappers.append(w.finish((), (n,)))
    top = BodyBuilder(wrappers)
    (p,) = top.emit(1, ())
    (q,) = top.emit(2, ())
    (r,) = top.emit(Prim.and_(), (p, q))
    program = Program(tuple(wrappers) + (top.finish((), (r,)),), 3)
    assert use_counts(program)[0] == 2
    out = inline(program)
    assert len(reachable(out)) == 2             # the wrappers go, `h` stays
    assert same_matrix(semantics(out), semantics(program))


def test_the_depth_family_is_refused_at_every_level():
    bodies = []
    leaf = BodyBuilder()
    (x,) = leaf.emit(Prim.flip(0.5), ())
    bodies.append(leaf.finish((), (x,)))
    for k in range(1, 7):
        b = BodyBuilder(bodies)
        (p,) = b.emit(k - 1, ())
        (q,) = b.emit(k - 1, ())
        (r,) = b.emit(Prim.and_(), (p, q))
        bodies.append(b.finish((), (r,)))
    program = Program(tuple(bodies), len(bodies) - 1)
    assert inline(program) == program


def _count_rebuilds(monkeypatch) -> list[int]:
    """A list that grows by one entry per body `pylea.ir.inline` builds."""
    built: list[int] = []
    compact = ir_inline._compact

    def counting(*args):
        built.append(1)
        return compact(*args)

    monkeypatch.setattr(ir_inline, "_compact", counting)
    return built


def test_inlining_rebuilds_each_caller_once_however_many_callees_it_splices(monkeypatch):
    callees = []
    for i in range(300):
        b = BodyBuilder()
        x = b.fresh()
        (f,) = b.emit(Prim.flip(0.5 + 1e-6 * i), ())
        (y,) = b.emit(Prim.and_(), (x, f))
        callees.append(b.finish((x,), (y,)))
    root = BodyBuilder(callees)
    (w,) = root.emit(Prim.flip(0.5), ())
    for i in range(300):
        (w,) = root.emit(i, (w,))
    p = prog(*callees, root.finish((), (w,)))
    built = _count_rebuilds(monkeypatch)
    out = inline(p)
    assert len(built) == 1
    validate(out)
    assert len(reachable(out)) == 1
    assert op_count(out) == 1 + 2 * 300


def test_inlining_a_chain_of_nested_bodies_rebuilds_only_the_outermost(monkeypatch):
    bodies: list[Body] = []
    for i in range(300):
        b = BodyBuilder(bodies)
        x = b.fresh()
        (f,) = b.emit(Prim.flip(0.5 + 1e-6 * i), ())
        (y,) = b.emit(Prim.or_(), (x, f))
        if i:
            (y,) = b.emit(i - 1, (y,))
        bodies.append(b.finish((x,), (y,)))
    root = BodyBuilder(bodies)
    (x,) = root.emit(Prim.flip(0.5), ())
    (y,) = root.emit(len(bodies) - 1, (x,))
    p = prog(*bodies, root.finish((), (y,)))
    built = _count_rebuilds(monkeypatch)
    out = inline(p)
    assert len(built) == 1
    validate(out)
    assert len(reachable(out)) == 1
    assert op_count(out) == 1 + 2 * 300


def test_inlining_keeps_a_body_called_twice_and_splices_its_single_use_callee_into_it():
    h = _coin_pair()
    mid = BodyBuilder([h])
    (a,) = mid.emit(0, ())
    (n,) = mid.emit(Prim.not_(), (a,))
    twice = mid.finish((), (n,))
    top = BodyBuilder([h, twice])
    (p,) = top.emit(1, ())
    (q,) = top.emit(1, ())
    (r,) = top.emit(Prim.or_(), (p, q))
    program = Program((h, twice, top.finish((), (r,))), 2)
    out = inline(program)
    validate(out)
    assert len(reachable(out)) == 2                 # `h` goes into `twice`, which stays
    assert use_counts(out)[out.main.callees()[0]] == 2
    assert same_matrix(semantics(out), semantics(program))
