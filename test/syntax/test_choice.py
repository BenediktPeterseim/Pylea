"""`pick`: uniform draws without replacement, single and several at once."""
import math

import numpy as np
import pytest

from pylea import DPP, Bool, Enum, pick, prob


class Card(Enum):
    elements = ["A", "K", "Q", "J"]


def _deck():
    return [Card.const(e) for e in Card]


def test_every_ordered_pair_of_two_draws_is_equally_likely():
    for first, second in [("A", "K"), ("K", "A"), ("Q", "J")]:
        @DPP
        def this_pair() -> Bool:
            (a, b), _ = pick(_deck(), 2)
            return (a == first) & (b == second)

        assert prob(this_pair) == pytest.approx(1 / 12), (first, second)


def test_drawing_everything_is_a_uniform_ordering():
    @DPP
    def one_ordering() -> Bool:
        drawn, rest = pick(_deck(), 4)
        assert rest == []
        return Bool.all(c == e for c, e in zip(drawn, list(Card)))

    assert prob(one_ordering) == pytest.approx(1 / math.factorial(4))


def test_the_rest_never_contains_the_drawn_value():
    @DPP
    def collision() -> Bool:
        drawn, rest = pick(_deck())
        return Bool.any(drawn == r for r in rest)

    assert prob(collision) == 0.0


def test_drawing_none_returns_everything_untouched():
    @DPP
    def untouched() -> Bool:
        none, rest = pick(_deck(), 0)
        assert none == [] and len(rest) == 4
        return rest[0] == "A"

    assert prob(untouched) == 1.0


def test_drawing_more_than_there_is_or_from_nothing_is_refused():
    with pytest.raises(ValueError, match="cannot draw"):
        @DPP
        def overdrawn() -> Bool:
            return pick(_deck(), 5)[0][0] == "A"

    with pytest.raises(ValueError, match="non-empty"):
        @DPP
        def empty() -> Bool:
            return pick([])[0]


def test_mixed_types_are_refused():
    class Coin(Enum):
        elements = ["H", "T"]

    with pytest.raises(TypeError, match="same type"):
        @DPP
        def mixed() -> Bool:
            return pick([Card.const("A"), Coin.const("H")])[0] == "A"
