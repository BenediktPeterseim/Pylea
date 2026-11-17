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


# --- values are checked against the type's values ---

def test_an_enum_of_mixed_types_compares_with_each_of_its_elements():
    from pylea import prob

    class Mixed(Enum):
        elements = [0, "high"]

    @DPP
    def is_high() -> Bool:
        return Mixed.uniform() == "high"

    @DPP
    def is_zero() -> Bool:
        return Mixed.uniform() == 0

    assert prob(is_high) == pytest.approx(0.5)
    assert prob(is_zero) == pytest.approx(0.5)


def test_an_enum_of_numbers_compares_with_an_int_and_a_float_alike():
    from pylea import prob

    class Levels(Enum):
        elements = [0.5, 1, 2]

    @DPP
    def at_one_as_int() -> Bool:
        return Levels.uniform() == 1

    @DPP
    def at_one_as_float() -> Bool:
        return Levels.uniform() == 1.0

    assert prob(at_one_as_int) == pytest.approx(1 / 3)
    assert prob(at_one_as_float) == pytest.approx(1 / 3)


def test_a_numpy_scalar_is_accepted_wherever_its_python_value_is():
    from pylea import UInt, prob

    class Nibble(UInt):
        width = 4

    @DPP
    def both() -> Bool:
        return (Direction.uniform() == np.str_("N")) | (Nibble.const(np.int64(3)) == np.int8(3))

    assert prob(both) == 1.0


def test_a_value_that_is_not_an_element_is_refused_with_the_type_named():
    with pytest.raises(ValueError, match="'X' is not an element of Color"):
        @DPP
        def typo() -> Bool:
            return Color.uniform() == "X"
    with pytest.raises(ValueError, match="'X' is not an element of Color"):
        @DPP
        def constant() -> Color:
            return Color.const("X")


def test_adding_variables_of_two_types_names_the_operation_and_both_types():
    from pylea import UInt

    class U4(UInt):
        width = 4

    class V4(UInt):
        width = 4

    with pytest.raises(TypeError, match="Cannot add a U4 variable and a V4 variable"):
        @DPP
        def mixed() -> U4:
            return U4.const(1) + V4.const(2)
    with pytest.raises(TypeError, match="Cannot compare a U4 variable and a V4 variable"):
        @DPP
        def compared() -> Bool:
            return U4.const(1) < V4.const(2)


def test_a_python_type_given_as_data_type_is_refused_with_bool_suggested():
    with pytest.raises(TypeError, match="use Bool"):
        @DPP
        def as_python_bool() -> Bool:
            return Color.uniform().match({"R": True, "G": False, "B": False}, data_type=bool)


def test_a_subclass_holding_the_same_values_compares_with_its_base():
    from pylea import prob

    class Renamed(Color):
        pass

    @DPP
    def same() -> Bool:
        return Color.const("G") == Renamed.const("G")

    assert prob(same) == 1.0


def test_a_subclass_with_reordered_elements_is_refused_in_a_comparison():
    class Reordered(Color):
        elements = ["B", "G", "R"]

    with pytest.raises(TypeError, match="Cannot compare a (Color|Reordered) variable and a (Color|Reordered)"):
        @DPP
        def mixed() -> Bool:
            return Color.const("R") == Reordered.const("R")


# --- one comparator per relation, shared by width ---

def test_each_ordering_comparison_records_a_single_comparator():
    from pylea import UInt
    from pylea.ir.program import op_count

    class Nibble(UInt):
        width = 4

    counts = {}
    for name, relation in [("<", lambda x: x < 5), ("<=", lambda x: x <= 5),
                           (">", lambda x: x > 5), (">=", lambda x: x >= 5)]:
        @DPP
        def compared(x: Nibble) -> Bool:
            return relation(x)
        counts[name] = op_count(compared.program())
    for name in ("<=", ">", ">="):
        assert counts["<"] <= counts[name] <= counts["<"] + 1, counts


def test_every_ordering_agrees_with_python_on_constants_and_variables():
    from pylea import UInt

    class Three(UInt):
        width = 3

    for a in range(8):
        for b in range(-1, 10):
            @DPP
            def relations() -> tuple[Bool, Bool, Bool, Bool, Bool, Bool]:
                x = Three.const(a)
                y = Three.const(b) if 0 <= b < 8 else b
                return x < y, x <= y, x > y, x >= y, x == y, x != y
            pattern = int(np.argmax(relations.distribution()))
            got = [bool((pattern >> (5 - i)) & 1) for i in range(6)]
            assert got == [a < b, a <= b, a > b, a >= b, a == b, a != b], (a, b)


def test_tracing_keeps_no_type_alive():
    import gc
    import weakref
    from pylea import UInt, count, flip, pick, prob

    def define_and_drop():
        class Temporary(UInt):
            width = 3

        class Cards(Enum):
            elements = ["a", "b", "c"]

        @DPP
        def uses_everything() -> Bool:
            x = Temporary.uniform_bits() + 1
            c = count(flip(0.5) for _ in range(3))
            (p, q), _ = pick([Cards.const(e) for e in Cards], 2)
            return (x < 3) & (c >= 2) & (p == q) & (Cards.uniform() <= "b")

        prob(uses_everything)
        return weakref.ref(Temporary), weakref.ref(Cards)

    refs = define_and_drop()
    gc.collect()
    assert all(r() is None for r in refs)


def test_types_of_one_width_share_one_comparator_circuit():
    from pylea.syntax.variable import _comparator_circuit

    class First(Enum):
        elements = ["a", "b", "c"]

    class Second(Enum):
        elements = [10, 20, 30, 40]

    @DPP
    def compared() -> Bool:
        return (First.uniform() < "b") & (Second.uniform() < 30)

    assert _comparator_circuit("LT", 2) is _comparator_circuit("LT", 2)
    root = compared.program().bodies[compared.program().root]
    callees = {op.callee for op in root.ops if isinstance(op.callee, int)}
    assert len(callees) == 1


def test_a_base_type_stands_where_a_subclass_that_reads_the_same_is_expected():
    from pylea import DPP, Bool, Enum, flip

    class Base(Enum):
        elements = ["x", "y"]

    class Same(Base):
        pass

    @DPP
    def takes(v: Same) -> Bool:
        return v == "x"

    @DPP
    def returns() -> Same:
        return Base.uniform()

    @DPP
    def mixed() -> Bool:
        a = flip(0.5).match({True: Same.const("x"), False: Base.const("y")})
        b = Base.uniform().match({"x": Same.uniform(), "y": Base.uniform()})
        return takes(a) & (b == "y")

    assert returns.output_types() == (Base,)
    assert abs(float(mixed.compiled().distribution()[1]) - 0.25) < 1e-12
