"""`Tuple`: the checks of `types`, and values written as plain Python tuples."""
import numpy as np
import pytest

from pylea import DPP, Bool, Enum, Tuple, prob


class Rank(Enum):
    elements = ["A", "K"]


class Colour(Enum):
    elements = ["red", "black", "green"]       # 2 bits, one pattern spare


class Card(Tuple):
    types = (Rank, Colour)


class Hand(Tuple):
    types = (Card, Card)


def test_a_tuple_with_no_types_or_a_non_variable_type_is_refused_when_the_class_is_created():
    with pytest.raises(ValueError, match="empty"):
        class Empty(Tuple):
            types = ()
    with pytest.raises(TypeError, match="use Bool"):
        class Plain(Tuple):
            types = (bool, Rank)
    with pytest.raises(TypeError, match=r"types = \(Rank,\)"):
        class Lone(Tuple):
            types = (Rank)


def test_using_a_tuple_without_types_raises_not_implemented_error():
    class Abstract(Tuple):
        pass

    with pytest.raises(NotImplementedError, match="declares no types"):
        Abstract.nbits()
    with pytest.raises(NotImplementedError, match="declares no types"):
        @DPP
        def built() -> Bool:
            Abstract.of(Rank.const("A"))
            return Bool.const(True)


def test_a_tuple_constant_is_written_as_a_plain_python_tuple():
    @DPP
    def fixed() -> Card:
        return Card.const(("A", "black"))

    assert fixed.distribution()[0b001] == 1.0


def test_a_tuple_variable_compares_with_a_plain_python_tuple():
    @DPP
    def red_ace() -> Bool:
        return Card.of(Rank.uniform(), Colour.uniform()) == ("A", "red")

    assert prob(red_ace) == pytest.approx(1 / 6)


def test_a_tuple_takes_plain_values_as_components():
    @DPP
    def built() -> Card:
        return Card.of("K", Colour.const("green"))

    assert built.distribution()[0b110] == 1.0


def test_a_tuple_decodes_to_plain_python_tuples():
    assert Card.encoding().decoded() == {
        0: ("A", "red"), 1: ("A", "black"), 2: ("A", "green"),
        4: ("K", "red"), 5: ("K", "black"), 6: ("K", "green")}


def test_nested_tuples_decode_to_nested_python_tuples():
    decoded = Hand.encoding().decoded()
    assert decoded[0b001_110] == (("A", "black"), ("K", "green"))
    assert len(decoded) == 36


def test_n_values_counts_only_the_patterns_that_encode_a_value():
    assert Card.encoding().n_values() == 6
    assert Hand.encoding().n_values() == 36


def test_a_table_over_a_tuple_type_prints_its_values():
    @DPP
    def dealt() -> Card:
        return Card.of(Rank.uniform(), Colour.uniform())

    text = str(dealt.evaluated())
    assert "object at" not in text
    assert "('A', 'red')" in text and "('K', 'green')" in text


def test_a_tuple_constant_with_a_component_outside_its_type_is_refused():
    with pytest.raises(ValueError, match="Card component 1: 'blue' is not an element of Colour"):
        @DPP
        def wrong() -> Card:
            return Card.const(("A", "blue"))
    with pytest.raises(ValueError, match="2 components"):
        @DPP
        def short() -> Card:
            return Card.const(("A",))


def test_a_tuple_refuses_a_component_variable_of_another_type():
    with pytest.raises(TypeError, match="component 0 must be a Rank; got a Colour"):
        @DPP
        def swapped() -> Card:
            return Card.of(Colour.uniform(), Rank.uniform())
