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


# --- values, widths and operations ---

def test_a_float_is_refused_as_a_uint_value_even_where_it_is_whole():
    for value in (2.5, 3.0):
        with pytest.raises(TypeError, match="Nibble holds integers"):
            @DPP
            def fractional() -> Nibble:
                return Nibble.const(value)


def test_a_map_into_a_uint_returning_a_fraction_names_the_value():
    from pylea import Enum

    class Die(Enum):
        elements = [1, 2, 3, 4, 5, 6]

    with pytest.raises(ValueError, match="f returned 0.5, which Nibble cannot hold"):
        @DPP
        def halved() -> Nibble:
            return Die.uniform().map(lambda x: x / 2, data_type=Nibble)


def test_a_numpy_integer_is_a_uint_value():
    @DPP
    def from_numpy() -> Nibble:
        return Nibble.const(np.int64(11)) + np.uint8(2)

    assert from_numpy.distribution()[13] == 1.0


@pytest.mark.parametrize("width, error", [
    (0, ValueError), (-1, ValueError), (4.0, TypeError), (True, TypeError), ("4", TypeError)])
def test_a_uint_with_a_width_below_one_or_not_an_int_is_refused_when_the_class_is_created(
        width, error):
    with pytest.raises(error, match="width"):
        type("Bad", (UInt,), {"width": width})


def test_a_comparison_with_an_int_beyond_the_range_is_a_constant():
    for relation, expected in [(lambda x: x < 100, 1.0), (lambda x: x == 100, 0.0),
                               (lambda x: x >= -1, 1.0), (lambda x: x > 15, 0.0),
                               (lambda x: x != -3, 1.0), (lambda x: x <= -1, 0.0)]:
        @DPP
        def compared() -> Bool:
            return relation(Nibble.uniform_bits())

        assert prob(compared) == expected


def test_adding_an_int_beyond_the_range_wraps_modulo_the_width():
    for a, b in [(3, 9 + 16), (0, -1), (5, -21)]:
        @DPP
        def summed() -> Nibble:
            return Nibble.const(a) + b

        assert summed.distribution()[(a + b) % 16] == 1.0, (a, b)


def test_the_adder_records_no_unused_assignment():
    from pylea.ir.program import op_count
    from pylea.syntax.integer import make_adder

    assert op_count(make_adder(4).program()) == 53


# --- `count` ---

def test_count_accepts_python_bools_as_constants():
    @DPP
    def mixed() -> UInt:
        return count([flip(0.5), True, np.True_, False])

    distribution = mixed.distribution()
    assert distribution[2] == pytest.approx(0.5) and distribution[3] == pytest.approx(0.5)


def test_count_refuses_an_item_that_is_not_a_bool():
    from pylea import Enum

    class Colour(Enum):
        elements = ["R", "G", "B"]

    with pytest.raises(TypeError, match="item 1 is a Colour"):
        @DPP
        def wrong() -> UInt:
            return count([flip(0.5), Colour.uniform()])
    with pytest.raises(TypeError, match="item 0 is a int"):
        @DPP
        def number() -> UInt:
            return count([1])


def test_two_counts_of_one_width_compare_without_a_declared_type():
    @DPP
    def tie() -> Bool:
        return count(flip(0.5) for _ in range(3)) == count(flip(0.5) for _ in range(3))

    assert prob(tie) == pytest.approx(20 / 64)


def test_two_counts_of_different_widths_point_to_the_data_type_of_count():
    with pytest.raises(TypeError, match="data_type argument of count"):
        @DPP
        def mismatched() -> Bool:
            return count(flip(0.5) for _ in range(3)) == count(flip(0.5) for _ in range(4))


def test_a_32_bit_uint_holds_constants_and_compares_with_ints():
    class Word(UInt):
        width = 32

    @DPP
    def compared() -> Bool:
        x = Word.const(3_000_000_000) + count((flip(0.5) for _ in range(2)), data_type=Word)
        return (x > 3_000_000_000) & (x <= 3_000_000_002) & (x != 1 << 40)

    assert prob(compared) == pytest.approx(0.75)
