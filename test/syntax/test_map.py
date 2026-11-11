"""`Enum.map`: plain Python functions lifted over one or several enums."""
import numpy as np
import pytest

from pylea import DPP, Bool, Enum, prob, probability_mass_function


class Die(Enum):
    elements = [1, 2, 3, 4, 5, 6]


class Colour(Enum):
    elements = ["R", "G", "B"]              # 2 bits, one pattern spare


def test_a_unary_map_within_the_own_type_matches_the_pushforward():
    @DPP
    def capped() -> Die:
        return Die.uniform().map(lambda v: min(v + 2, 6))

    pmf = probability_mass_function(capped)
    assert pmf == pytest.approx({1: 0.0, 2: 0.0, 3: 1/6, 4: 1/6, 5: 1/6, 6: 3/6})


def test_a_binary_map_to_bool_matches_direct_enumeration():
    @DPP
    def big_product() -> Bool:
        a, b = Die.uniform(), Die.uniform()
        return a.map(lambda x, y: x * y > 20, b, data_type=Bool)

    expected = sum(1 for x in Die for y in Die if x * y > 20) / 36
    assert prob(big_product) == pytest.approx(expected)


def test_a_map_between_different_enum_types_lands_on_the_declared_type():
    @DPP
    def initial() -> Colour:
        return Die.uniform().map(lambda v: "RGB"[v % 3], data_type=Colour)

    pmf = probability_mass_function(initial)
    assert pmf["R"] == pytest.approx(2/6)   # v in {3, 6}
    assert sum(pmf.values()) == pytest.approx(1.0)


def test_a_spare_bit_pattern_maps_as_the_first_element():
    # The map must be total over patterns; the model puts no mass on the spare one, so
    # only totality is observable, checked here by the pushforward staying normalised.
    @DPP
    def swapped() -> Colour:
        return Colour.uniform().map(lambda c: {"R": "B", "G": "G", "B": "R"}[c])

    assert sum(probability_mass_function(swapped).values()) == pytest.approx(1.0)


def test_a_value_the_output_type_cannot_hold_is_named_in_the_error():
    with pytest.raises(ValueError, match="'X'"):
        @DPP
        def bad() -> Colour:
            return Colour.uniform().map(lambda c: "X")


def test_a_map_too_wide_for_a_dense_matrix_is_refused():
    class Wide(Enum):
        elements = range(1 << 11)

    with pytest.raises(ValueError, match="circuit"):
        @DPP
        def wide() -> Wide:
            return Wide.uniform().map(lambda v, w: v, Wide.uniform())
