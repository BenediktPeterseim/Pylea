"""How a DPP keeps the DPPs it calls, and how its program is made from them.

Resolving a callee must not depend on an identity that can be reused.  `at_least` builds a
fresh `@DPP` on every call, and that DPP becomes unreachable as soon as the call returns.  A
cache keyed on the address would therefore answer for an address since handed to an
unrelated callee, silently confusing two callees of the same arity.

A DPP keeps its callees and makes its program from them the first time it is asked for, so
defining a chain of nested DPPs takes time in proportion to its length.  The program made
must hold each callee's program as it is, numbered and labelled by the stated rules.
"""
import gc
import weakref

import numpy as np
import pytest

from test.syntax.helpers import matrix_of

import pylea.syntax.dpp as dpp_module
from pylea.ir.program import Program, intern, reachable
from pylea.syntax.dpp import DPP, Bool, at_least
from pylea.syntax.flip import flip


def _p_true(dpp) -> float:
    v = matrix_of(dpp).ravel()
    return float(v[1] / v.sum())


def _at_least_exactly(k: int, ps: list[float]) -> float:
    """P(at least `k` of the coins land true), by enumeration."""
    total = 0.0
    for mask in range(1 << len(ps)):
        if bin(mask).count("1") < k:
            continue
        w = 1.0
        for i, p in enumerate(ps):
            w *= p if (mask >> i) & 1 else 1.0 - p
        total += w
    return total


def test_gates_of_different_arity_in_one_body_are_not_confused():
    ps = [0.1, 0.2, 0.3, 0.4, 0.5]

    @DPP
    def mixed() -> Bool:
        xs = [flip(p) for p in ps]
        a = at_least(2, *xs[:3])
        b = at_least(3, *xs)
        c = at_least(1, *xs[:4])
        return a & b & c

    assert _p_true(mixed) == pytest.approx(_joint(ps), rel=1e-9)


def _joint(ps: list[float]) -> float:
    """P(a & b & c) for the three gates above, by enumeration over all assignments."""
    total = 0.0
    for mask in range(1 << len(ps)):
        w = 1.0
        for i, p in enumerate(ps):
            w *= p if (mask >> i) & 1 else 1.0 - p
        bits = [(mask >> i) & 1 for i in range(len(ps))]
        a = sum(bits[:3]) >= 2
        b = sum(bits) >= 3
        c = sum(bits[:4]) >= 1
        if a and b and c:
            total += w
    return total


@pytest.mark.parametrize("k,n", [(1, 2), (2, 3), (3, 5), (1, 4), (4, 4)])
def test_a_voting_gate_agrees_with_enumeration_over_all_assignments(k, n):
    ps = [0.1 * (i + 1) for i in range(n)]

    @DPP
    def gate() -> Bool:
        return at_least(k, *[flip(p) for p in ps])

    assert _p_true(gate) == pytest.approx(_at_least_exactly(k, ps), rel=1e-9)


def test_the_callee_cache_is_keyed_on_the_callee_object_not_its_address():
    @DPP
    def inner() -> Bool:
        return flip(0.5)

    @DPP
    def outer() -> Bool:
        return inner()

    cache = getattr(outer, "_DPP__callees")
    assert cache, "nothing was absorbed, so this test is not checking anything"
    assert all(isinstance(k, DPP) for k in cache), f"keys are {[type(k) for k in cache]}"


def _chain(n: int) -> list[DPP]:
    """`n` DPPs, each calling the one before it."""
    levels: list[DPP] = []
    for _ in range(n):
        if not levels:
            def level() -> Bool:
                return flip(0.5)
        else:
            below = levels[-1]

            def level() -> Bool:
                return below() | flip(0.1)
        levels.append(DPP(level))
    return levels


def test_defining_nested_dpps_interns_nothing_and_their_program_interns_each_body_once(
        monkeypatch):
    interned: list[int] = []
    original = dpp_module._intern_with_remap

    def counting(program):
        interned.append(len(program.bodies))
        return original(program)

    monkeypatch.setattr(dpp_module, "_intern_with_remap", counting)
    levels = _chain(400)
    assert interned == []
    assert len(levels[-1].program().bodies) == 400
    assert interned == [400]
    levels[-1].program()
    assert interned == [400]


def test_bodies_are_numbered_callees_first_in_the_order_of_their_calls():
    @DPP
    def a() -> Bool:
        return flip(0.1)

    @DPP
    def b(x: Bool) -> Bool:
        return ~x & flip(0.2)

    @DPP
    def c() -> Bool:
        return b(a())

    @DPP
    def top() -> Bool:
        return c() | a()

    program = top.program()
    assert program.bodies[0] == a.program().main
    assert program.bodies[1] == b.program().main
    assert program.bodies[2].callees() == (0, 1)
    assert program.root == 3 and program.main.callees() == (2, 0)


def _examples_and_their_callees() -> list[DPP]:
    """Every DPP defined at the top level of an example module, and every DPP it calls."""
    import importlib
    import pkgutil

    import pylea.examples

    found: dict[DPP, None] = {}
    for info in pkgutil.iter_modules(pylea.examples.__path__):
        module = importlib.import_module(f"pylea.examples.{info.name}")
        found.update((value, None) for value in vars(module).values()
                     if isinstance(value, DPP))
    pending = list(found)
    while pending:
        for callee in getattr(pending.pop(), "_DPP__callees"):
            if callee not in found:
                found[callee] = None
                pending.append(callee)
    return list(found)


def test_a_callers_program_holds_each_callees_program_as_it_is():
    dpps = _examples_and_their_callees()
    assert len(dpps) > 20, "too few DPPs found for this test to check anything"
    for caller in dpps:
        program = caller.program()
        assert reachable(program) == list(range(len(program.bodies))), caller.name()
        called = set(program.main.callees())
        held = {intern(Program(program.bodies, i)) for i in called}
        for callee in getattr(caller, "_DPP__callees"):
            assert callee.program() in held, (caller.name(), callee.name())


def test_a_body_reached_through_two_names_is_labelled_with_the_one_reached_first():
    @DPP
    def first() -> Bool:
        return flip(0.5)

    @DPP
    def second() -> Bool:
        return flip(0.5)

    @DPP
    def inner() -> Bool:
        return second()

    @DPP
    def top() -> Bool:
        return inner() & first()

    # Without a name per call, a call box is labelled with the name of its body.
    svg = top.diagram(op_names={}).svg()
    assert ">second<" in svg and ">first<" not in svg


def test_a_chain_of_nested_dpps_is_released_once_unreachable():
    from pylea import prob
    from pylea.sample.query import collapsed

    def define_query_and_drop():
        levels = _chain(30)
        prob(levels[-1], method="exact")
        collapsed(levels[-1])
        return [weakref.ref(level) for level in levels]

    refs = define_query_and_drop()
    gc.collect()
    assert all(r() is None for r in refs)
