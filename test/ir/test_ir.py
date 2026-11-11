"""The IR's invariants, and the properties the simplifier's arguments rest on.

Two of these carry weight beyond the IR itself.  The inlining arithmetic bounds the work on
a deep call graph, so a fault in it restores exponential behaviour and shows up as a
timeout rather than as a wrong answer.  `is_deterministic` on an opaque matrix
decides whether CSE may merge two calls, so over-claiming it merges two independent draws.
"""
import numpy as np
import pytest

from pylea.ir import (
    ArityError, Body, BodyBuilder, Op, Prim, Program, Tag,
    intern, is_indicator, is_stochastic, multi_parent_wires, reachable, op_count, shared_reads,
    use_counts, validate,
)
from pylea.semantics import matrices


# --- primitives ---

def test_every_matrix_has_the_declared_shape_and_is_non_negative():
    prims = [Prim.and_(), Prim.or_(), Prim.not_(), Prim.observe(), Prim.flip(0.25),
             Prim.uniform(3), Prim.const(2, 3), Prim.equate(2), Prim.ite(2),
             Prim.from_matrix(np.full((4, 2), 0.5))]
    for p in prims:
        f = p.matrix
        assert f.shape == (1 << p.out_bits, 1 << p.in_bits), p
        assert f.min() >= 0.0, p


def test_derived_matrices_are_shared_rather_than_copied():
    """All ANDs point at one table; materialising a copy per op would be wasteful."""
    assert Prim.and_().matrix is Prim.and_().matrix


def test_a_primitives_matrix_is_read_only():
    with pytest.raises(ValueError):
        Prim.and_().matrix[0, 0] = 7.0


@pytest.mark.parametrize("arr,det", [
    ([[0.5], [0.5]], False),                    # a coin
    ([[0.0, 1.0], [1.0, 0.0]], True),           # a permutation
    ([[1.0, 0.0], [0.0, 1.0]], True),           # identity
    ([[1.0, 1.0], [0.0, 1.0]], False),          # column sums to 2: not a function
    ([[0.0, 0.0], [0.0, 1.0]], True),           # a zero column excludes that input
    ([[0.0, 0.5], [0.0, 0.5]], False),          # a coin on the inputs it admits
])
def test_determinism_of_an_opaque_matrix_is_decided_from_its_entries(arr, det):
    assert Prim.from_matrix(arr).is_deterministic is det
    assert is_indicator(np.asarray(arr, float)) is det


@pytest.mark.parametrize("arr,disc", [
    ([[0.5], [0.5]], True),                     # a coin is a distribution
    ([[0.0, 1.0], [1.0, 0.0]], True),           # so is a permutation
    ([[0.0, 0.0], [0.0, 1.0]], False),          # a zero column conditions
    ([[1.0, 1.0], [0.0, 1.0]], False),          # column sums to 2
])
def test_discardability_of_an_opaque_matrix_is_decided_from_its_entries(arr, disc):
    assert Prim.from_matrix(arr).is_discardable is disc
    assert is_stochastic(np.asarray(arr, float)) is disc


def test_conditioning_primitives_are_not_discardable():
    assert Prim.observe().is_discardable is False
    assert Prim.equate(1).is_discardable is False
    assert Prim.and_().is_discardable is True


def _every_builtin() -> list[Prim]:
    return [Prim.and_(), Prim.or_(), Prim.not_(), Prim.observe(), Prim.uniform(0),
            Prim.uniform(2), Prim.const(0, 1), Prim.const(3, 2), Prim.flip(0.0),
            Prim.flip(0.25), Prim.flip(1.0), Prim.equate(1), Prim.equate(2), Prim.ite(1)]


@pytest.mark.parametrize("prim", _every_builtin(), ids=lambda p: str(p.tag) + str(p.params))
def test_the_tag_shortcuts_never_over_claim_against_the_array(prim):
    """Every tag but MATRIX answers from its tag; the answer must not exceed the matrix's.

    Under-claiming is allowed and happens: `flip(0.0)` is a constant, so its array is a
    function, but FLIP answers "not deterministic" for every probability.
    """
    array = prim.matrix
    assert prim.is_deterministic <= is_indicator(array)
    assert prim.is_discardable == is_stochastic(array)


def test_identical_matrices_intern_and_different_ones_do_not():
    a = Prim.from_matrix([[0.5], [0.5]])
    b = Prim.from_matrix([[0.5], [0.5]])
    c = Prim.from_matrix([[0.25], [0.75]])
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert len({a, b, c}) == 2


def test_matrix_rejects_a_non_power_of_two_shape():
    with pytest.raises(ValueError, match="power of two"):
        Prim.from_matrix(np.ones((3, 2)))


def test_matrix_rejects_negative_entries():
    with pytest.raises(ValueError, match="non-negative"):
        Prim.from_matrix([[1.0], [-0.5]])


# --- the builder ---

def test_builder_rejects_wrong_arity_and_wrong_width():
    b = BodyBuilder()
    x = b.fresh(1)
    with pytest.raises(ArityError, match="argument"):
        b.emit(Prim.and_(), (x,))                       # AND takes 2
    wide = b.fresh(3)
    with pytest.raises(ArityError, match="bits"):
        b.emit(Prim.not_(), (wide,))                    # NOT takes a 1-bit wire


def test_builder_rejects_a_wire_from_another_body():
    b = BodyBuilder()
    with pytest.raises(ArityError, match="not a wire"):
        b.emit(Prim.not_(), (99,))


def test_builder_resolves_a_called_body_signature():
    inner = BodyBuilder()
    i = inner.fresh(2)
    inner_body = inner.finish((i,), (i,))               # 2 bits -> 2 bits
    bodies = [inner_body]
    outer = BodyBuilder(bodies)
    w = outer.fresh(2)
    (out,) = outer.emit(0, (w,))
    assert outer.bits[out] == 2
    narrow = outer.fresh(1)
    with pytest.raises(ArityError, match="bits"):
        outer.emit(0, (narrow,))


# --- body structure ---

def test_fanout_and_discard_need_no_primitive():
    """A wire used twice is fan-out; a wire absent from the outputs is discarded."""
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.and_(), (x, x))                  # fan-out
    (_unused,) = b.emit(Prim.flip(0.1), ())             # discarded
    body = b.finish((), (y,))
    validate(Program((body,), 0))
    assert multi_parent_wires(body) == 1                # x is read twice


def test_a_body_may_return_an_input_unchanged():
    b = BodyBuilder()
    i = b.fresh(1)
    body = b.finish((i,), (i,))
    validate(Program((body,), 0))
    assert body.in_bits == body.out_bits == 1


def test_validate_catches_a_non_ssa_body():
    prim = Prim.flip(0.5)
    bad = Body(bits=(1,), inputs=(), outputs=(0,),
               ops=(Op(prim, (), (0,)), Op(prim, (), (0,))))
    with pytest.raises(ValueError, match="not SSA"):
        validate(Program((bad,), 0))


# --- the measures of a body ---

def _depth(n: int) -> Program:
    """`f[k] = f[k-1]() & f[k-1]()`: unfolded size 2^k, but n+1 bodies."""
    bodies: list[Body] = []
    b = BodyBuilder(bodies)
    (x,) = b.emit(Prim.flip(0.5), ())
    bodies.append(b.finish((), (x,)))
    for k in range(1, n + 1):
        b = BodyBuilder(bodies)
        (u,) = b.emit(k - 1, ())
        (v,) = b.emit(k - 1, ())
        (w,) = b.emit(Prim.and_(), (u, v))
        bodies.append(b.finish((), (w,)))
    return Program(tuple(bodies), n)


@pytest.mark.parametrize("n", [5, 10, 20, 40, 80])
def test_deep_call_graph_stays_linear(n):
    p = _depth(n)
    validate(p)
    assert len(reachable(p)) == n + 1
    assert op_count(p) == 3 * n + 1                        # linear in n, not 2**n


@pytest.mark.parametrize("n", [5, 20, 80])
def test_only_the_leaf_and_the_root_are_inlinable_at_any_depth(n):
    """The guard `s <= 1 or n_ops <= 1` must refuse every intermediate level.

    Inlining one would duplicate it at both call sites, and iterating that is exactly the
    2^N blowup the hierarchy exists to avoid.
    """
    p = _depth(n)
    s = use_counts(p)
    inlinable = [i for i in reachable(p)
                 if s.get(i, 0) <= 1 or len(p.bodies[i].ops) <= 1]
    assert sorted(inlinable) == sorted({0, p.root})


def test_the_op_count_after_inlining_matches_the_closed_form():
    p = _depth(20)
    s = use_counts(p)
    for i in reachable(p):
        n_ops, sites = len(p.bodies[i].ops), s.get(i, 0)
        predicted = (sites - 1) * (n_ops - 1) - 1
        assert (predicted <= -1) == (sites <= 1 or n_ops <= 1)


def test_use_counts_are_global_not_per_caller():
    """Two wrappers each calling `h` once still gives `h` a count of 2.

    Counting per caller would see 1 in each and licence an inline that duplicates `h`.
    """
    bodies: list[Body] = []
    h = BodyBuilder(bodies)
    (hx,) = h.emit(Prim.flip(0.5), ())
    (hy,) = h.emit(Prim.not_(), (hx,))
    bodies.append(h.finish((), (hy,)))                  # body 0 = h, 2 ops

    for _ in range(2):                                  # two distinct wrappers
        w = BodyBuilder(bodies)
        (a,) = w.emit(0, ())
        bodies.append(w.finish((), (a,)))               # bodies 1, 2

    top = BodyBuilder(bodies)
    (p1,) = top.emit(1, ())
    (p2,) = top.emit(2, ())
    (r,) = top.emit(Prim.and_(), (p1, p2))
    bodies.append(top.finish((), (r,)))
    prog = Program(tuple(bodies), 3)

    assert use_counts(prog)[0] == 2                     # not 1
    assert (2 - 1) * (2 - 1) - 1 == 0                   # so inlining h is refused


def test_shared_reads_counts_the_reads_beyond_the_first():
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.and_(), (x, x))
    shared = Program((b.finish((), (y,)),), 0)
    assert (shared_reads(shared.main), op_count(shared)) == (1, 2)


# --- interning ---

def test_interning_merges_structurally_identical_bodies():
    p = _depth(3)
    dup = Program(p.bodies + (p.bodies[0],), p.root)
    assert len(intern(dup).bodies) == len(p.bodies)


def test_interning_reports_where_each_body_went():
    """A caller holding a table keyed by body index needs the map to carry it across."""
    from pylea.ir import intern_with_remap

    def coin_body():
        b = BodyBuilder()
        (x,) = b.emit(Prim.flip(0.5), ())
        return b.finish(inputs=(), outputs=(x,))

    caller = BodyBuilder(bodies=[coin_body(), coin_body()])
    (a,) = caller.emit(0, ())
    (c,) = caller.emit(1, ())
    (out,) = caller.emit(Prim.and_(), (a, c))
    prog = Program((coin_body(), coin_body(),
                    caller.finish(inputs=(), outputs=(out,))), root=2)

    merged, remap = intern_with_remap(prog)
    assert remap[0] == remap[1], "the two equal bodies should share an index"
    assert merged.bodies[remap[2]] is merged.main
    assert intern(prog) == merged


def test_interning_drops_unreachable_bodies():
    p = _depth(3)
    orphan = BodyBuilder()
    (z,) = orphan.emit(Prim.flip(0.9), ())
    with_orphan = Program(p.bodies + (orphan.finish((), (z,)),), p.root)
    assert len(intern(with_orphan).bodies) == len(reachable(p))


def test_interning_is_bottom_up():
    """Callers only become identical once their callees have been merged."""
    bodies: list[Body] = []
    for _ in range(2):                                  # two identical leaves
        b = BodyBuilder(bodies)
        (x,) = b.emit(Prim.flip(0.5), ())
        bodies.append(b.finish((), (x,)))
    for leaf in (0, 1):                                 # two callers, one per leaf
        b = BodyBuilder(bodies)
        (u,) = b.emit(leaf, ())
        (v,) = b.emit(Prim.not_(), (u,))
        bodies.append(b.finish((), (v,)))
    top = BodyBuilder(bodies)
    (a,) = top.emit(2, ())
    (c,) = top.emit(3, ())
    (r,) = top.emit(Prim.and_(), (a, c))
    bodies.append(top.finish((), (r,)))

    merged = intern(Program(tuple(bodies), 4))
    assert len(merged.bodies) == 3                      # leaf, caller, top


def test_interning_never_adds_ops_or_sharing():
    p = _depth(6)
    dup = Program(p.bodies + (p.bodies[2],), p.root)
    assert op_count(intern(dup)) <= op_count(dup)
    assert (sum(shared_reads(intern(dup).bodies[i]) for i in reachable(intern(dup)))
            <= sum(shared_reads(dup.bodies[i]) for i in reachable(dup)))
