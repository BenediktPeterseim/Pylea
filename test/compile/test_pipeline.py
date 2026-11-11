"""Compiling and running a `Program`.

The test carrying the most weight is differential: random programs are contracted both by
the plan pipeline and by `pylea.ir.reference.semantics`, which enumerates every wire
assignment, and the two matrices must agree entry for entry.  The generator covers wide
wires, several inputs and outputs, repeated arguments, repeated outputs, outputs naming
inputs, dead results, unread inputs, conditioning and shared callees, and one test pins that
coverage so the sweep cannot quietly stop exercising it.

The remaining tests fix the structure the differential ones cannot see: which tensors a
network has, that a shared body is contracted once, and that slicing splits a sum.
"""
from __future__ import annotations

import numpy as np
import pytest

from pylea.compile.lower import MAX_LEAF_BITS, PlanTooWide, compile_program, lower_body
from pylea.compile.network import Tensor, Ones, build_network
from pylea.compile.order import AdaptiveOrderFinder, GreedyOrderFinder
from pylea.eval.interp import evaluate
from pylea.ir.body import BodyBuilder
from pylea.ir.inline import flatten_wide_calls
from pylea.ir.prim import Prim
from pylea.ir.program import Program
from pylea.ir.reference import TooLarge, semantics

from test.compile.random_programs import random_program

KAPPA_FORCING_SLICES = 3


def matrix_of(prog: Program, **kwargs) -> np.ndarray:
    plans, root = compile_program(prog, **kwargs)
    array, scale = evaluate(plans, root)
    return array * np.exp(scale)


def _program(*bodies) -> Program:
    return Program(tuple(bodies), len(bodies) - 1)


@pytest.mark.parametrize("kappa", [None, KAPPA_FORCING_SLICES])
def test_plan_agrees_with_exact_enumeration(kappa):
    rng = np.random.default_rng(0)
    compared = 0
    for _ in range(60):
        prog = random_program(rng)
        try:
            expected = semantics(prog, max_wires=16)
        except TooLarge:
            continue
        got = matrix_of(prog, kappa=kappa)
        assert got.shape == expected.shape
        assert np.allclose(got, expected, atol=1e-12, rtol=1e-9)
        compared += 1
    assert compared > 40, f"only {compared} programs were small enough to enumerate"


def test_random_programs_cover_the_hard_cases():
    """The differential sweep is only worth its runtime if it hits these."""
    rng = np.random.default_rng(0)
    seen = {k: 0 for k in ("repeated output", "output is input", "repeated argument",
                           "several outputs", "call", "conditioning", "wide wire",
                           "diagonal leaf", "spread output", "ones tensor", "dead result",
                           "slicing")}
    for _ in range(60):
        prog = random_program(rng)
        body = prog.main
        if len(set(body.outputs)) != len(body.outputs):
            seen["repeated output"] += 1
        if set(body.outputs) & set(body.inputs):
            seen["output is input"] += 1
        if any(len(set(op.args)) != len(op.args) for op in body.ops):
            seen["repeated argument"] += 1
        if len(body.outputs) > 1:
            seen["several outputs"] += 1
        if body.callees():
            seen["call"] += 1
        if any(isinstance(op.callee, Prim) and not op.callee.is_discardable
               for op in body.ops):
            seen["conditioning"] += 1
        if any(w > 1 for w in body.bits):
            seen["wide wire"] += 1
        flat = flatten_wide_calls(prog, MAX_LEAF_BITS)   # interning renumbers the bodies
        net = build_network(flat, flat.root)
        if any(len(set(a)) != len(a) for a in net.axes):
            seen["diagonal leaf"] += 1
        if len(net.output_axes) != len(net.output):
            seen["spread output"] += 1
        if any(isinstance(leaf, Ones) for leaf in net.leaves):
            seen["ones tensor"] += 1
        if any(isinstance(leaf, Tensor) and leaf.reduce for leaf in net.leaves):
            seen["dead result"] += 1
        plans, _ = compile_program(prog, kappa=KAPPA_FORCING_SLICES)
        if any(plan.sliced_dims for plan in plans.values()):
            seen["slicing"] += 1
    missing = [k for k, n in seen.items() if n == 0]
    assert not missing, f"the generator no longer produces: {missing}"


def test_slicing_changes_nothing_but_the_peak():
    rng = np.random.default_rng(11)
    for _ in range(20):
        prog = random_program(rng, n_bodies=2, n_ops=6)
        whole, _ = compile_program(prog, kappa=None)
        sliced, root = compile_program(prog, kappa=KAPPA_FORCING_SLICES)
        assert np.allclose(matrix_of(prog, kappa=None),
                           matrix_of(prog, kappa=KAPPA_FORCING_SLICES), atol=1e-12)
        if any(plan.sliced_dims for plan in sliced.values()):
            assert (max(p.peak_entries() for p in sliced.values())
                    <= max(p.peak_entries() for p in whole.values()))


# -- network structure ---------------------------------------------------------------

def test_fan_out_is_one_index_on_several_tensors():
    builder = BodyBuilder()
    (x,) = builder.emit(Prim.flip(0.3), [])
    (y,) = builder.emit(Prim.not_(), [x])
    (z,) = builder.emit(Prim.and_(), [x, y])
    prog = _program(builder.finish([], [z]))

    net = build_network(prog, prog.root)
    # One index on three tensors, which the order handles.  Nothing is tied,
    # nothing is copied, and no tensor is added.
    assert net.n_tensors == 3
    assert sum(1 for t in net.axes if x in t) == 3
    assert all(len(set(a)) == len(a) for a in net.axes)


def test_repeated_argument_reads_a_diagonal():
    builder = BodyBuilder()
    (x,) = builder.emit(Prim.flip(0.3), [])
    (y,) = builder.emit(Prim.and_(), [x, x])
    prog = _program(builder.finish([], [y]))

    net = build_network(prog, prog.root)
    conjunction = net.leaves.index(next(l for l in net.leaves
                                        if isinstance(l, Tensor) and l.callee == Prim.and_()))
    # The repeat goes to the order finder as it stands; only the plan resolves it.
    assert net.axes[conjunction] == (y, x, x)
    # `x & x` is `x`, so the answer must be the coin itself, not its square.
    assert np.allclose(matrix_of(prog).reshape(-1), [0.7, 0.3])


def test_repeated_output_is_spread_after_the_contraction():
    builder = BodyBuilder()
    (x,) = builder.emit(Prim.flip(0.25), [])
    prog = _program(builder.finish([], [x, x]))

    net = build_network(prog, prog.root)
    assert net.output_axes == (x, x) and net.output == (x,)
    plan = lower_body(prog, prog.root, GreedyOrderFinder())
    assert plan.shapes[plan.contracted] == (2,)        # contracted over the distinct wire
    assert plan.shapes[plan.result] == (2, 2)
    assert np.allclose(matrix_of(prog).reshape(-1), [0.75, 0.0, 0.0, 0.25])


def test_unread_input_becomes_a_ones_tensor():
    builder = BodyBuilder()
    x = builder.fresh(1)
    (y,) = builder.emit(Prim.flip(0.25), [])
    prog = _program(builder.finish([x], [y]))

    net = build_network(prog, prog.root)
    assert sum(1 for leaf in net.leaves if isinstance(leaf, Ones)) == 1
    assert np.allclose(matrix_of(prog), [[0.75, 0.75], [0.25, 0.25]])


def test_dead_result_is_summed_as_the_leaf_is_formed():
    builder = BodyBuilder()
    (x,) = builder.emit(Prim.flip(0.3), [])
    (kept,) = builder.emit(Prim.flip(0.4), [])
    (dead,) = builder.emit(Prim.not_(), [x])    # its result is never read
    prog = _program(builder.finish([], [kept]))

    net = build_network(prog, prog.root)
    reduced = [leaf for leaf in net.leaves if isinstance(leaf, Tensor) and leaf.reduce]
    assert len(reduced) == 1
    # Summed as the leaf is built, so the order finder never sees an index only it carries.
    assert dead not in {ix for t in net.axes for ix in t}
    assert np.allclose(matrix_of(prog).reshape(-1), [0.6, 0.4])


def test_every_network_index_is_free_or_shared():
    """The invariant the contraction order relies on: no index only one tensor can sum."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        prog = flatten_wide_calls(random_program(rng), MAX_LEAF_BITS)
        for body in range(len(prog.bodies)):
            net = build_network(prog, body)
            counts: dict[int, int] = {}
            for t in net.axes:
                for ix in set(t):
                    counts[ix] = counts.get(ix, 0) + 1
            for ix, n in counts.items():
                assert n >= 2 or ix in net.output


# -- sharing and materialisation ------------------------------------------------------

def test_a_shared_body_is_contracted_once():
    inner = BodyBuilder()
    (c,) = inner.emit(Prim.flip(0.5), [])
    coin = inner.finish([], [c])

    outer = BodyBuilder([coin])
    (a,) = outer.emit(0, [])
    (b,) = outer.emit(0, [])
    (both,) = outer.emit(Prim.and_(), [a, b])
    prog = _program(coin, outer.finish([], [both]))

    plans, root = compile_program(prog)
    assert set(plans) == {0, 1}
    assert plans[root].subplans() == (0,)
    # Two calls are two independent draws, so the conjunction is 0.25, not 0.5.
    assert np.allclose(matrix_of(prog).reshape(-1), [0.75, 0.25])


def test_depth_family_stays_linear():
    """`f[k] = f[k-1]() & f[k-1]()` has 2**k unfolded draws but one body per level."""
    bodies = []
    base = BodyBuilder()
    (c,) = base.emit(Prim.flip(0.5), [])
    bodies.append(base.finish([], [c]))
    for level in range(1, 41):
        builder = BodyBuilder(bodies)
        (x,) = builder.emit(level - 1, [])
        (y,) = builder.emit(level - 1, [])
        (z,) = builder.emit(Prim.and_(), [x, y])
        bodies.append(builder.finish([], [z]))
    prog = Program(tuple(bodies), len(bodies) - 1)

    plans, root = compile_program(prog)
    assert len(plans) == 41
    assert sum(len(p.steps) for p in plans.values()) < 41 * 12
    array, scale = evaluate(plans, root)
    # P(all 2**40 draws true) = 0.5 ** (2**40) is unrepresentable beside P(false) = 1, and
    # comes out 0.  The scale is one number for the whole array and cannot separate them;
    # what is linear here is the work, which is what this test is about.
    assert np.isclose(array[0, 0] * np.exp(scale), 1.0)


def test_a_wide_callee_is_inlined_rather_than_materialised():
    inner = BodyBuilder()
    ws = [inner.emit(Prim.flip(0.5), [])[0] for _ in range(6)]
    wide = inner.finish([], ws)                 # 6 output bits

    outer = BodyBuilder([wide])
    outs = outer.emit(0, [])
    (a,) = outer.emit(Prim.and_(), [outs[0], outs[1]])
    prog = _program(wide, outer.finish([], [a]))

    plans, root = compile_program(prog, max_leaf_bits=4)
    assert plans[root].subplans() == ()
    assert np.allclose(matrix_of(prog, max_leaf_bits=4).reshape(-1), [0.75, 0.25])
    assert np.allclose(matrix_of(prog, max_leaf_bits=64).reshape(-1), [0.75, 0.25])


def test_peak_budget_is_enforced():
    builder = BodyBuilder()
    ws = [builder.emit(Prim.flip(0.5), [])[0] for _ in range(12)]
    prog = _program(builder.finish([], ws))
    with pytest.raises(PlanTooWide):
        compile_program(prog, max_entries=64)


# -- the log scale -------------------------------------------------------------------

def test_a_probability_below_the_float_floor_survives_in_the_scale():
    """The case the scale exists for: every entry of the matrix underflows.

    Twenty coins of 1e-30, all observed true, leave a body with no outputs whose value is
    1e-600.  A float64 holds nothing below about 1e-308, so without the scale this is 0.
    """
    builder = BodyBuilder()
    for _ in range(20):
        (x,) = builder.emit(Prim.flip(1e-30), [])
        builder.emit(Prim.observe(), [x])
    prog = _program(builder.finish([], []))

    plans, root = compile_program(prog)
    array, scale = evaluate(plans, root)
    assert array.shape == (1, 1)
    assert np.isclose(np.log(array[0, 0]) + scale, 20 * np.log(1e-30), rtol=1e-12)


def test_the_scale_does_not_separate_entries_of_one_array():
    """One scale per array, so an entry underflowing beside a larger one is still lost.

    This is the same behaviour as plain floating point and is not a case the scale covers;
    it is recorded because the difference from the test above is easy to misread.
    """
    builder = BodyBuilder()
    conj, = builder.emit(Prim.flip(1e-30), [])
    for _ in range(19):
        (x,) = builder.emit(Prim.flip(1e-30), [])
        (conj,) = builder.emit(Prim.and_(), [conj, x])
    prog = _program(builder.finish([], [conj]))

    array, scale = evaluate(*compile_program(prog))
    assert array[1, 0] * np.exp(scale) == 0.0
    assert np.isclose(array[0, 0] * np.exp(scale), 1.0)


# -- order finders ---------------------------------------------------------------------

@pytest.mark.parametrize("finder", [GreedyOrderFinder(), AdaptiveOrderFinder()])
def test_order_finders_agree(finder):
    rng = np.random.default_rng(2)
    for _ in range(15):
        prog = random_program(rng)
        try:
            expected = semantics(prog, max_wires=16)
        except TooLarge:
            continue
        plans, root = compile_program(prog, finder, kappa=6)
        array, scale = evaluate(plans, root)
        assert np.allclose(array * np.exp(scale), expected, atol=1e-12)


def test_a_target_below_the_interface_relaxes_instead_of_failing():
    """Only summed indices are sliced, so a body cannot be sliced below its own interface."""
    builder = BodyBuilder()
    ws = [builder.emit(Prim.flip(0.5), [])[0] for _ in range(6)]
    prog = _program(builder.finish([], ws))
    plan = lower_body(prog, prog.root, AdaptiveOrderFinder(), kappa=2)
    assert plan.peak_entries() >= 2 ** 6
