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




# --- `elements`, checked when the class is created ---

def test_elements_with_a_repeated_value_are_refused_when_the_class_is_created():
    with pytest.raises(ValueError, match=r"elements\[0\] = 'a' and .*elements\[2\] = 'a'"):
        class Repeated(Enum):
            elements = ["a", "b", "a"]


@pytest.mark.parametrize("elements", [[1, True], [0, False], [1, 1.0]])
def test_elements_equal_in_python_such_as_1_and_true_are_refused(elements):
    with pytest.raises(ValueError, match="in Python"):
        type("Twice", (Enum,), {"elements": elements})


@pytest.mark.parametrize("make", [
    lambda: {"low", "mid", "high"},
    lambda: frozenset({"low", "mid", "high"}),
    lambda: {"low": 0, "mid": 1}.keys(),
    lambda: (x for x in ["low", "mid", "high"]),
])
def test_elements_given_as_a_set_or_a_generator_are_refused(make):
    with pytest.raises(TypeError, match="order"):
        class Unordered(Enum):
            elements = make()


def test_elements_given_as_a_string_are_refused():
    with pytest.raises(TypeError, match="string"):
        class Letters(Enum):
            elements = "RGB"


def test_empty_elements_are_refused_when_the_class_is_created():
    with pytest.raises(ValueError, match="empty"):
        class Empty(Enum):
            elements = []


def test_unhashable_elements_are_refused_when_the_class_is_created():
    with pytest.raises(TypeError, match=r"\[1, 2\]"):
        class Lists(Enum):
            elements = [[1, 2], [3]]


def test_an_element_not_equal_to_itself_is_refused():
    with pytest.raises(ValueError, match="nan"):
        class WithNan(Enum):
            elements = [0.0, float("nan")]


def test_elements_given_as_a_range_tuple_or_numpy_array_keep_their_order():
    class FromRange(Enum):
        elements = range(3, 0, -1)

    class FromTuple(Enum):
        elements = ("c", "a", "b")

    class FromArray(Enum):
        elements = np.array([2.5, 0.5, 1.5])

    class FromIntArray(Enum):
        elements = np.arange(3)

    assert FromRange.elements == (3, 2, 1)
    assert FromTuple.elements == ("c", "a", "b")
    assert FromArray.elements == (2.5, 0.5, 1.5)
    assert all(type(e) is float for e in FromArray.elements)
    assert all(type(e) is int for e in FromIntArray.elements)
    assert FromTuple.encoding().decoded() == {0: "c", 1: "a", 2: "b"}


def test_a_subclass_inherits_the_elements_of_its_parent():
    class Parent(Enum):
        elements = ["x", "y", "z"]

    class Child(Parent):
        pass

    assert Child.elements == ("x", "y", "z")
    assert Child.encoding().decoded() == Parent.encoding().decoded()


def test_an_abstract_enum_raises_not_implemented_error_wherever_it_is_used():
    from pylea import DPP, Bool

    class Abstract(Enum):
        pass

    with pytest.raises(NotImplementedError, match="declares no elements"):
        @DPP
        def drawn() -> Bool:
            Abstract.random({})
            return Bool.const(True)
    with pytest.raises(NotImplementedError, match="declares no elements"):
        @DPP
        def uniform() -> Bool:
            Abstract.uniform()
            return Bool.const(True)
    with pytest.raises(NotImplementedError, match="declares no elements"):
        Abstract.cpt([], {(): [1.0]})
    with pytest.raises(NotImplementedError, match="declares no elements"):
        Abstract.encoding()


# --- weights of `random` ---

def test_a_weight_for_a_value_that_is_not_an_element_is_refused():
    from pylea import DPP

    with pytest.raises(ValueError, match="'Purple' is not an element of Color"):
        @DPP
        def misspelt() -> Color:
            return Color.random({"R": 0.5, "Purple": 0.5})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_weight_is_refused(bad):
    from pylea import DPP

    with pytest.raises(ValueError, match="not finite"):
        @DPP
        def drawn() -> Color:
            return Color.random({"R": bad, "G": 0.5})
    with pytest.raises(ValueError, match="not finite"):
        Color.cpt([Coin], {("H",): [0.5, 0.5, 0.0], ("T",): [bad, 0.5, 0.0]})


def test_a_random_draw_given_a_list_reads_it_in_element_order():
    from pylea import DPP, probability_mass_function

    @DPP
    def listed() -> Color:
        return Color.random([2, 1, 1])

    assert probability_mass_function(listed) == pytest.approx({"R": 0.5, "G": 0.25, "B": 0.25})


def test_a_list_of_weights_of_the_wrong_length_is_refused_by_random():
    from pylea import DPP

    with pytest.raises(ValueError, match="needs 3 entries"):
        @DPP
        def short() -> Color:
            return Color.random([0.5, 0.5])


# --- `match` ---

class _Grass(Enum):
    elements = ["dry", "damp", "wet"]


class _Colour(Enum):
    elements = ["R", "G", "B", "Y"]


def test_a_match_value_of_another_type_is_refused_even_at_the_same_width():
    from pylea import DPP, UInt

    class U2(UInt):
        width = 2

    for other in (lambda: _Colour.const("B"), lambda: U2.const(1)):
        with pytest.raises(TypeError, match=r"value for 'B' must be a _Grass; got a"):
            @DPP
            def mixed() -> _Grass:
                return Color.uniform().match(
                    {"R": _Grass.const("dry"), "G": _Grass.uniform(), "B": other()},
                    data_type=_Grass)


def test_a_match_over_variables_of_several_types_asks_for_data_type():
    from pylea import DPP

    with pytest.raises(TypeError, match="several types"):
        @DPP
        def mixed() -> _Grass:
            return Color.uniform().match(
                {"R": _Grass.uniform(), "G": _Colour.uniform(), "B": _Grass.uniform()})


def test_a_match_key_that_is_not_an_element_is_refused():
    from pylea import DPP, Bool

    with pytest.raises(ValueError, match="'hail' is not an element of Color"):
        @DPP
        def extra() -> Bool:
            return Color.uniform().match({"R": True, "G": False, "B": False, "hail": True},
                                         data_type=Bool)


def test_a_match_missing_a_key_names_it_even_without_data_type():
    from pylea import DPP, Bool

    with pytest.raises(ValueError, match="no value for 'B'"):
        @DPP
        def missing() -> Bool:
            return Color.uniform().match({"R": True, "G": False})


def test_a_match_over_constants_is_recorded_as_one_table():
    from pylea import DPP, Bool
    from pylea.ir.program import op_count

    @DPP
    def tabulated() -> Bool:
        return Color.uniform().match({"R": True, "G": False, "B": True}, data_type=Bool)

    assert op_count(tabulated.program()) == 2


def test_a_match_over_constants_agrees_with_the_same_match_over_variables():
    from pylea import DPP

    rng = np.random.default_rng(0)
    for _ in range(5):
        weights = rng.random(5) + 0.1
        choice = [_Grass.elements[i] for i in rng.integers(0, 3, size=5)]
        mapping = dict(zip(Season.elements, choice))

        @DPP
        def constants() -> _Grass:
            return Season.random(list(weights)).match(mapping, data_type=_Grass)

        @DPP
        def variables() -> _Grass:
            return Season.random(list(weights)).match(
                {k: _Grass.const(v) for k, v in mapping.items()})

        np.testing.assert_allclose(constants.distribution(), variables.distribution())


class Season(Enum):
    elements = ["spring", "summer", "autumn", "winter", "monsoon"]


def test_a_match_over_constants_wider_than_the_dense_bound_uses_the_selection_tree():
    from pylea import DPP, UInt

    class Wide(UInt):
        width = 19                          # 2 + 19 bits, beyond a table's 20

    @DPP
    def chosen() -> Wide:
        return Color.random([0.5, 0.25, 0.25]).match(
            {"R": 5, "G": 300_000, "B": (1 << 19) - 1}, data_type=Wide)

    distribution = chosen.distribution()
    assert distribution[[5, 300_000, (1 << 19) - 1]] == pytest.approx([0.5, 0.25, 0.25])
    assert distribution.sum() == pytest.approx(1.0)
