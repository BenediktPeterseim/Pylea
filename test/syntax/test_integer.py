"""`UInt`: the binary encoding, the adder circuit, `count`, and their guards."""
import math

import numpy as np
import pytest

from pylea import DPP, Bool, UInt, count, expectation, flip, prob


class Nibble(UInt):
    width = 4


class Byte(UInt):
    width = 8


def test_the_encoding_decodes_every_pattern_to_its_integer():
    encoding = Nibble.encoding()
    assert encoding.decoded() == {i: i for i in range(16)}
    assert encoding(11).to_bits() == (True, False, True, True)  # type: ignore[call-arg]
    assert encoding.from_bits(True, False, True, True).value == 11


def test_addition_of_constants_matches_python_addition_modulo_the_width():
    for a in (0, 1, 7, 12, 15):
        for b in (0, 3, 9, 15):
            @DPP
            def summed() -> Nibble:
                return Nibble.const(a) + b

            distribution = summed.distribution()
            assert distribution[(a + b) % 16] == 1.0, (a, b)
            assert distribution.sum() == 1.0


def test_addition_of_a_uniform_variable_and_a_constant_permutes_the_distribution():
    @DPP
    def shifted() -> Nibble:
        return Nibble.uniform_bits() + 5

    assert np.allclose(shifted.distribution(), np.full(16, 1 / 16))


def test_addition_of_two_random_variables_convolves_their_distributions():
    from typing import cast

    @DPP
    def two_bits_plus_two_bits() -> Nibble:
        def two_bits() -> Nibble:
            return cast(Nibble, Nibble.make(Bool.const(False), Bool.const(False),
                                            flip(0.5), flip(0.5)))
        return two_bits() + two_bits()

    # Two independent uniforms on {0..3}: the triangle distribution on {0..6}.
    expected = np.convolve(np.full(4, 0.25), np.full(4, 0.25))
    assert np.allclose(two_bits_plus_two_bits.distribution()[:7], expected)


def test_comparisons_against_plain_ints_are_numeric():
    @DPP
    def below_ten() -> Bool:
        return Nibble.uniform_bits() < 10

    assert prob(below_ten) == 10 / 16


def test_reflected_addition_gives_the_same_sum():
    @DPP
    def reflected() -> Nibble:
        return 3 + Nibble.const(4)

    assert reflected.distribution()[7] == 1.0


def test_count_of_fair_flips_is_binomial():
    n, p = 5, 0.3

    @DPP
    def hits() -> UInt:
        return count(flip(p) for _ in range(n))

    distribution = hits.distribution()
    for k in range(n + 1):
        assert distribution[k] == pytest.approx(math.comb(n, k) * p**k * (1 - p)**(n - k))
    assert expectation(hits) == pytest.approx(n * p)


def test_count_into_a_declared_type_keeps_that_type():
    @DPP
    def hits() -> Byte:
        return count((flip(0.5) for _ in range(3)), data_type=Byte)

    assert hits.output_types() == (Byte,)
    assert hits.distribution()[3] == pytest.approx(1 / 8)


def test_count_of_nothing_is_the_constant_zero():
    @DPP
    def none_at_all() -> UInt:
        return count([])

    assert none_at_all.distribution()[0] == 1.0


def test_count_refuses_a_type_too_narrow_for_the_maximum():
    class Crumb(UInt):
        width = 1

    with pytest.raises(ValueError, match="too narrow"):
        @DPP
        def overflowing() -> Crumb:
            return count((flip(0.5) for _ in range(2)), data_type=Crumb)


def test_a_width_free_subclass_cannot_be_used():
    with pytest.raises(NotImplementedError):
        UInt.encoding()


def test_a_constant_outside_the_range_is_refused():
    with pytest.raises(ValueError, match="outside"):
        @DPP
        def too_big() -> Nibble:
            return Nibble.const(16)


def test_a_subclass_of_a_subclass_builds_its_own_encoding():
    class Narrow(UInt):
        width = 4

    assert Narrow.nbits() == 4                  # materialises Narrow's encoding

    class Wide(Narrow):
        width = 8

    assert Wide.nbits() == 8 and Narrow.nbits() == 4
    assert Wide.encoding()(200).value == 200
