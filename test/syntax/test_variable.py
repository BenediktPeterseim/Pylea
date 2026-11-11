"""Uniform draws over a type, and the comparison operators on one.

A model's matrix has shape ``(2**out_bits, 2**in_bits)``.  Column `j` is the input bit
pattern `j`, MSB first across all input bits, and row `i` is the weight of output pattern
`i`.  A model with no inputs therefore has a single column.
"""
import pytest

from test.syntax.helpers import matrix_of
import numpy as np
from pylea.syntax.dpp import DPP, Bool
from pylea.syntax.enum import Enum


class Coin(Enum):
    elements = ["H", "T"]                       # 1 bit, no pattern spare


class Color(Enum):
    elements = ["R", "G", "B"]                  # 2 bits, one pattern spare


class Direction(Enum):
    elements = ["N", "E", "S", "W"]             # 2 bits, no pattern spare


class Season(Enum):
    # 3 bits, three patterns spare
    elements = ["Spring", "Summer", "Autumn", "Winter", "Monsoon"]


def _eval(dpp) -> np.ndarray:
    """The matrix of a DPP."""
    return matrix_of(dpp)


def _eval_normalized(dpp) -> np.ndarray:
    """The matrix of a DPP, column-normalised.

    A comparison against a type with spare bit patterns carries an overall scale, so
    normalising is what lets it be checked against the 0/1 matrix it means.
    """
    m = _eval(dpp)
    col_sums = m.sum(axis=0)
    return np.where(col_sums > 0, m / col_sums, m)


@DPP
def bool_uniform_bits() -> Bool:
    return Bool.uniform_bits()


@DPP
def coin_uniform_bits() -> Coin:
    return Coin.uniform_bits()


@DPP
def color_uniform_bits() -> Color:
    return Color.uniform_bits()


@DPP
def direction_uniform_bits() -> Direction:
    return Direction.uniform_bits()


def test_uniform_bits_over_a_bool_is_even():
    m = _eval(bool_uniform_bits).flatten()
    np.testing.assert_allclose(m, [0.5, 0.5], atol=1e-10)


def test_uniform_bits_over_a_two_element_type_is_even():
    m = _eval(coin_uniform_bits).flatten()
    np.testing.assert_allclose(m, [0.5, 0.5], atol=1e-10)


def test_uniform_bits_over_a_four_element_type_is_even():
    m = _eval(direction_uniform_bits).flatten()
    np.testing.assert_allclose(m, [0.25, 0.25, 0.25, 0.25], atol=1e-10)


def test_uniform_bits_covers_the_patterns_encoding_no_element_too():
    m = _eval(color_uniform_bits).flatten()
    np.testing.assert_allclose(m, [0.25, 0.25, 0.25, 0.25], atol=1e-10)


@DPP
def bool_uniform() -> Bool:
    return Bool.uniform_bits()


@DPP
def direction_uniform() -> Direction:
    return Direction.uniform()


@DPP
def color_uniform() -> Color:
    return Color.uniform()


@DPP
def season_uniform() -> Season:
    return Season.uniform()


def test_a_uniform_bool_is_even():
    m = _eval(bool_uniform).flatten()
    np.testing.assert_allclose(m, [0.5, 0.5], atol=1e-10)


def test_uniform_and_uniform_bits_agree_where_no_pattern_is_spare():
    m = _eval(direction_uniform).flatten()
    np.testing.assert_allclose(m, [0.25, 0.25, 0.25, 0.25], atol=1e-10)


def test_uniform_puts_equal_weight_on_every_element():
    m = _eval(color_uniform).flatten()
    np.testing.assert_allclose(m[:3], [1/3, 1/3, 1/3], atol=1e-10)


def test_uniform_gives_a_pattern_encoding_no_element_probability_zero():
    m = _eval(color_uniform).flatten()
    assert m[3] == pytest.approx(0.0, abs=1e-10)


def test_a_uniform_draw_over_three_elements_sums_to_one():
    m = _eval(color_uniform).flatten()
    assert m.sum() == pytest.approx(1.0, abs=1e-10)


def test_uniform_over_five_elements_in_three_bits_is_even_across_them():
    m = _eval(season_uniform).flatten()
    np.testing.assert_allclose(m[:5], [1/5] * 5, atol=1e-10)


def test_uniform_over_five_elements_gives_the_three_spare_patterns_zero():
    m = _eval(season_uniform).flatten()
    np.testing.assert_allclose(m[5:], [0.0, 0.0, 0.0], atol=1e-10)


@DPP
def bool_le(x: Bool, y: Bool) -> Bool:
    return x <= y


@DPP
def color_le_const_B(x: Color) -> Bool:
    return x <= "B"


def test_at_most_on_booleans_matches_its_truth_table():
    m = _eval(bool_le)
    expected = np.array([[0, 0, 1, 0],
                         [1, 1, 0, 1]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_at_most_against_a_constant_holds_on_the_right_states():
    m = _eval_normalized(color_le_const_B)
    expected = np.array([[0, 0, 0, 1],
                         [1, 1, 1, 0]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


@DPP
def bool_eq(x: Bool, y: Bool) -> Bool:
    return x == y


@DPP
def color_eq_const_R(x: Color) -> Bool:
    return x == "R"


@DPP
def color_eq(x: Color, y: Color) -> Bool:
    return x == y


def test_equality_on_booleans_matches_its_truth_table():
    m = _eval(bool_eq)
    expected = np.array([[0, 1, 1, 0],
                         [1, 0, 0, 1]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_equality_against_a_constant_holds_on_the_right_states():
    m = _eval(color_eq_const_R)
    expected = np.array([[0, 1, 1, 1],
                         [1, 0, 0, 0]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_equality_of_two_variables_is_the_identity_matrix_on_its_diagonal():
    m = _eval(color_eq)
    # Reshape to (n_out, n_x, n_y) = (2, 4, 4) for readability.
    # Column j = 4*i_x + i_y (MSB = x bits, LSB = y bits)
    # → m[1, 4*i+i] == 1 for i in 0..3, all off-diagonals == 0 in row 1.
    for i in range(4):
        for j in range(4):
            col = 4 * i + j
            if i == j:
                assert m[1, col] == pytest.approx(1.0, abs=1e-10), f"Expected True at ({i},{j})"
                assert m[0, col] == pytest.approx(0.0, abs=1e-10)
            else:
                assert m[0, col] == pytest.approx(1.0, abs=1e-10), f"Expected False at ({i},{j})"
                assert m[1, col] == pytest.approx(0.0, abs=1e-10)


@DPP
def bool_ne(x: Bool, y: Bool) -> Bool:
    return x != y


@DPP
def color_ne_const_R(x: Color) -> Bool:
    return x != "R"


def test_inequality_on_booleans_matches_its_truth_table():
    m = _eval(bool_ne)
    expected = np.array([[1, 0, 0, 1],
                         [0, 1, 1, 0]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_inequality_against_a_constant_holds_on_the_right_states():
    m = _eval(color_ne_const_R)
    expected = np.array([[1, 0, 0, 0],
                         [0, 1, 1, 1]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


@DPP
def bool_lt(x: Bool, y: Bool) -> Bool:
    return x < y


@DPP
def color_lt_const_B(x: Color) -> Bool:
    return x < "B"


def test_below_on_booleans_matches_its_truth_table():
    m = _eval(bool_lt)
    expected = np.array([[1, 0, 1, 1],
                         [0, 1, 0, 0]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_below_against_a_constant_holds_on_the_right_states():
    m = _eval_normalized(color_lt_const_B)
    expected = np.array([[0, 0, 1, 1],
                         [1, 1, 0, 0]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


@DPP
def bool_gt(x: Bool, y: Bool) -> Bool:
    return x > y


@DPP
def color_gt_const_G(x: Color) -> Bool:
    return x > "G"


def test_above_on_booleans_matches_its_truth_table():
    m = _eval(bool_gt)
    expected = np.array([[1, 1, 0, 1],
                         [0, 0, 1, 0]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_above_against_a_constant_holds_on_the_right_states():
    m = _eval_normalized(color_gt_const_G)
    expected = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


@DPP
def bool_ge(x: Bool, y: Bool) -> Bool:
    return x >= y


@DPP
def color_ge_const_B(x: Color) -> Bool:
    return x >= "B"


def test_at_least_on_booleans_matches_its_truth_table():
    m = _eval(bool_ge)
    expected = np.array([[0, 1, 0, 0],
                         [1, 0, 1, 1]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)


def test_at_least_against_a_constant_holds_on_the_right_states():
    m = _eval_normalized(color_ge_const_B)
    expected = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-10)
