"""`Enum` encodings, and the matrices that `random`, `match` and the two combined build.

A model with `k` input bits and `m` output bits has a ``(2**m, 2**k)`` matrix whose
column `j` is the output distribution given input state `j`.
"""
import pytest
import numpy as np
from pylea.syntax.enum import Enum
from pylea.examples.simple import (
    Coin, Color,
    coin_random,
    color_random,
    color_random_missing_element,
    color_random_unnormalised,
    coin_match_swap,
    color_match_to_bool,
    coin_match_with_variable,
    color_random_then_match,
)
from test.syntax.helpers import _assert_proportional, matrix_of


class TwoElements(Enum):
    elements = [False, True]                # 1 bit

class ThreeElements(Enum):
    elements = ["Cat", "Cow", "Dog"]        # 2 bits, one pattern spare

class FourElements(Enum):
    elements = range(4)                     # 2 bits, no pattern spare

class LargeEnum(Enum):
    elements = range(37)                    # 6 bits, 27 patterns spare

@pytest.mark.parametrize("cls, expected_nbits", [
    (TwoElements, 1),
    (ThreeElements, 2),
    (FourElements, 2),
    (LargeEnum, 6),
])
def test_an_enum_uses_the_fewest_bits_holding_all_its_elements(cls, expected_nbits):
    assert cls.encoding().nbits() == expected_nbits
    assert cls.nbits() == expected_nbits


def test_one_element_past_a_power_of_two_needs_another_bit():
    class E64(Enum):
        elements = range(64)
    class E65(Enum):
        elements = range(65)
    assert E64.nbits() == 6
    assert E65.nbits() == 7


@pytest.mark.parametrize("cls", [TwoElements, ThreeElements, FourElements, LargeEnum])
def test_every_element_survives_encoding_and_decoding(cls):
    enc = cls.encoding()
    for element in cls.elements:
        encoded = enc(element)
        bits = encoded.to_bits()
        assert len(bits) == enc.nbits()
        assert enc.from_bits(*bits).value == element


def test_every_valid_bit_pattern_decodes_to_the_element_at_its_index():
    enc = FourElements.encoding()
    assert enc.from_bits(False, False).to_bits() == (False, False)
    assert enc.from_bits(False, True).to_bits() == (False, True)
    assert enc.from_bits(True, False).to_bits() == (True, False)
    assert enc.from_bits(True, True).to_bits() == (True, True)


def test_a_bit_pattern_past_the_last_element_is_refused():
    enc = ThreeElements.encoding()  # 3 elements, 2 bits; index 3 is out of range
    with pytest.raises(ValueError):
        enc.from_bits(True, True)  # binary 11 = index 3



def _eval(dpp):
    """The matrix of a DPP."""
    return matrix_of(dpp)


#
# For a 0-input DPP returning an n-bit enum, M has shape (2^n, 1).
# Row i = P(output bit-pattern = i).
# Coin states:  H=0, T=1
# Color states: R=0, G=1, B=2, invalid=3

def test_random_puts_each_elements_weight_on_its_own_state():
    M = _eval(coin_random)
    expected = [0.3, 0.7]
    _assert_proportional(M.flatten(), np.array(expected))


def test_random_over_three_elements_lands_on_the_right_states():
    M = _eval(color_random)
    # R=0, G=1, B=2, invalid(11)=3 must have probability 0
    expected = [0.2, 0.5, 0.3, 0.0]
    _assert_proportional(M.flatten(), np.array(expected))


def test_a_bit_pattern_encoding_no_element_gets_probability_zero():
    M = _eval(color_random)
    assert M[3, 0] == pytest.approx(0.0, abs=1e-10)


def test_an_element_absent_from_the_distribution_gets_probability_zero():
    M = _eval(color_random_missing_element)
    expected = [0.6, 0.0, 0.4, 0.0]  # R=0.6, G=0.0, B=0.4, invalid=0.0
    _assert_proportional(M.flatten(), np.array(expected))


def test_weights_that_do_not_sum_to_one_are_normalised():
    M = _eval(color_random_unnormalised)
    expected = [0.5, 0.25, 0.25, 0.0]
    _assert_proportional(M.flatten(), np.array(expected))


def test_the_output_probabilities_of_a_random_draw_sum_to_one():
    for dpp in [coin_random, color_random, color_random_missing_element, color_random_unnormalised]:
        M = _eval(dpp)
        assert M.sum() == pytest.approx(1.0, abs=1e-10), dpp


def test_matching_a_coin_onto_its_opposite_is_a_permutation_matrix():
    M = _eval(coin_match_swap)
    expected = np.array([[0, 1], [1, 0]], dtype=float)
    _assert_proportional(M.flatten(), expected.flatten())


def test_a_deterministic_match_on_a_coin_scales_every_column_equally():
    M = _eval(coin_match_swap)
    col_sums = M.sum(axis=0)
    np.testing.assert_allclose(col_sums, col_sums[0] * np.ones(2), atol=1e-10)


def test_matching_onto_constants_selects_them_per_input_state():
    """The fourth input state encodes no element and takes the first value, True."""
    M = _eval(color_match_to_bool)
    expected = np.array([[0, 1, 0, 0],
                         [1, 0, 1, 1]], dtype=float)
    _assert_proportional(M.flatten(), expected.flatten())


def test_a_deterministic_match_onto_constants_scales_every_column_equally():
    M = _eval(color_match_to_bool)
    col_sums = M.sum(axis=0)
    np.testing.assert_allclose(col_sums, col_sums[0] * np.ones(4), atol=1e-10)


def test_matching_onto_variables_selects_their_distributions():
    M = _eval(coin_match_with_variable)
    expected = np.array([[0.6, 0.4],
                         [0.4, 0.6]])
    _assert_proportional(M.flatten(), expected.flatten())


def test_a_match_onto_variables_stays_non_negative():
    M = _eval(coin_match_with_variable)
    assert np.all(M >= -1e-10)


def test_matching_on_a_random_draw_sums_the_weights_of_each_branch():
    M = _eval(color_random_then_match)
    expected = [0.5, 0.5]
    _assert_proportional(M.flatten(), np.array(expected))


def test_a_draw_followed_by_a_match_stays_non_negative():
    M = _eval(color_random_then_match)
    assert np.all(M >= -1e-10)


