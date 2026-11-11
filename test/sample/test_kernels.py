"""Exact equality of the byte and packed executors.

The two consume the identical RNG byte stream by construction, so their results are
compared bit for bit rather than statistically; a divergence means one lane's draws or
semantics changed.  The corpus covers every packable op, including the deterministic
EQUATE and ITE, multi-bit UNIFORM and CONST, and conditioning through OBSERVE.
"""
from __future__ import annotations

import numpy as np
import pytest

from pylea import DPP, Bool, Enum, flip, observe
from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.sample.packed import NotPackable, PackedProgram
from pylea.sample.sweep import SweepProgram, unfold


def assert_lanes_agree(program: SweepProgram, n: int = 4096, seeds=(0, 1, 2)) -> None:
    packed = PackedProgram(program)
    for seed in seeds:
        byte = program.run(n, np.random.default_rng(seed), squares=True)
        word = packed.run(n, np.random.default_rng(seed), squares=True)
        assert np.array_equal(byte.counts, word.counts), f"seed {seed} diverged"
        assert byte.total == word.total
        assert word.counts_sq is not None and byte.counts_sq is not None
        assert np.array_equal(byte.counts_sq, word.counts_sq)


def test_the_lanes_agree_on_a_fault_tree_shape():
    @DPP
    def tree() -> Bool:
        cells = [flip(0.01 * (k + 1)) for k in range(24)]
        layer = [a | b for a, b in zip(cells[::2], cells[1::2])]
        while len(layer) > 1:
            layer = [a & b for a, b in zip(layer[::2], layer[1::2])] \
                + ([layer[-1]] if len(layer) % 2 else [])
        return layer[0]

    assert_lanes_agree(SweepProgram(unfold(tree.program())))


def test_the_lanes_agree_under_conditioning():
    @DPP
    def conditioned() -> Bool:
        a, b, c = flip(0.5), flip(0.25), flip(0.75)
        observe(a | b)
        observe(b | c)
        return a & ~c

    assert_lanes_agree(SweepProgram(unfold(conditioned.program())))


def test_the_lanes_agree_on_equate_ite_uniform_and_const():
    builder = BodyBuilder()
    (a,) = builder.emit(Prim.flip(0.3), ())
    (b,) = builder.emit(Prim.flip(0.7), ())
    (c,) = builder.emit(Prim.flip(0.5), ())
    u1, u2 = builder.emit(Prim.uniform(2), ())
    (one,) = builder.emit(Prim.const(1, 1), ())
    (chosen,) = builder.emit(Prim.ite(1), (c, a, u1))
    (equal,) = builder.emit(Prim.equate(1), (chosen, b))
    (out,) = builder.emit(Prim.and_(), (equal, one))
    (out2,) = builder.emit(Prim.or_(), (out, u2))
    body = builder.finish(inputs=(), outputs=(out2,))
    assert_lanes_agree(SweepProgram(body))


def test_a_soft_table_refuses_the_packed_kernel():
    from pylea import LikelihoodTable

    soft = LikelihoodTable((Bool,), (Bool,), np.array([[0.7, 0.4], [0.1, 0.4]]))

    @DPP
    def weighted() -> Bool:
        return soft(flip(0.5))

    with pytest.raises(NotPackable):
        PackedProgram(SweepProgram(unfold(weighted.program())))


def test_the_lanes_agree_across_a_table_boundary():
    class Colour(Enum):
        elements = ["r", "g", "b"]

    @DPP
    def drawn() -> Bool:
        first = Colour.random({"r": 0.5, "g": 0.3, "b": 0.2})
        second = Colour.random({"r": 0.2, "g": 0.2, "b": 0.6})
        observe(~(first == second))
        return (first == "r") | (second == "b")

    assert_lanes_agree(SweepProgram(unfold(drawn.program())))


def test_a_packed_batch_must_be_a_multiple_of_64():
    @DPP
    def coin() -> Bool:
        return flip(0.5)

    packed = PackedProgram(SweepProgram(unfold(coin.program())))
    with pytest.raises(ValueError):
        packed.run(100, np.random.default_rng(0))
    with pytest.raises(ValueError):
        packed.run(128, np.random.default_rng(0), resample=True)


def test_a_body_with_inputs_needs_an_explicit_pattern_and_refuses_the_packed_kernel():
    @DPP
    def channel(x: Bool) -> Bool:
        return x & flip(0.9) | ~x & flip(0.1)

    body = unfold(channel.program())
    with pytest.raises(ValueError, match="input_pattern"):
        SweepProgram(body)
    byte_lane = SweepProgram(body, input_pattern=1)
    assert abs(byte_lane.run(4096, np.random.default_rng(0)).counts[1] / 4096 - 0.9) < 0.03
    with pytest.raises(NotPackable):
        PackedProgram(byte_lane)
