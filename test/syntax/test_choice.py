"""`pick`: uniform draws without replacement, single and several at once."""
import itertools
import math
import time

import numpy as np
import pytest

from pylea import DPP, Bool, Enum, Tuple, pick, prob
from pylea.ir.program import op_count


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
        assert len(rest) == 0 and list(rest) == []
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


def test_choices_that_are_not_variables_are_refused():
    with pytest.raises(TypeError, match="must be variables; got str at position 0"):
        @DPP
        def raw() -> Card:
            return pick(["A", "K"])[0]  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="list of variables; got a str"):
        @DPP
        def text() -> Card:
            return pick("AK")[0]  # type: ignore[arg-type]


def test_a_count_that_is_not_an_int_is_refused():
    with pytest.raises(TypeError, match="n must be an integer; got 1.0"):
        @DPP
        def fractional() -> Card:
            return pick(_deck(), 1.0)[0][0]  # type: ignore[arg-type]


def _joint_distribution(n: int, draws: list[int], slots=None) -> np.ndarray:
    """The distribution of the drawn values, in drawing order, followed by the rest, where
    `pick` is applied to `n` distinct values once per entry of `draws`, each time to the
    rest of the one before; of the rest only the `slots` given, where they are given."""
    Value = type("Value", (Enum,), {"elements": list(range(n))})
    width = sum(draws) + (n - sum(draws) if slots is None else len(slots))
    Joint = type("Joint", (Tuple,), {"types": (Value,) * width})

    @DPP
    def joint() -> Joint:
        rest = [Value.const(v) for v in range(n)]
        drawn = []
        for k in draws:
            more, rest = pick(rest, k)
            drawn += more
        kept = list(rest) if slots is None else [rest[m] for m in slots]
        return Joint.of(*drawn, *kept)

    return joint.distribution()


def _enumerated_distribution(n: int, k: int, slots=None) -> np.ndarray:
    """The distribution `_joint_distribution` must give for `k` draws in all, found by
    enumerating every ordered draw, each equally likely, with the rest in ascending
    order."""
    bits = max(1, math.ceil(math.log2(n)))
    width = n if slots is None else k + len(slots)
    expected = np.zeros(1 << (bits * width))
    for drawn in itertools.permutations(range(n), k):
        rest = sorted(set(range(n)) - set(drawn))
        kept = rest if slots is None else [rest[m] for m in slots]
        pattern = 0
        for v in (*drawn, *kept):
            pattern = (pattern << bits) | v
        expected[pattern] += 1.0
    return expected / expected.sum()


def test_every_draw_and_remainder_agree_with_enumerating_all_orderings():
    for n in range(1, 5):
        for k in range(min(n, 3) + 1):
            np.testing.assert_allclose(_joint_distribution(n, [k]),
                                       _enumerated_distribution(n, k),
                                       atol=1e-12, err_msg=f"{k} of {n}")
    for n in (5, 6):
        for k in (1, 2):
            for m in range(n - k):
                np.testing.assert_allclose(_joint_distribution(n, [k], [m]),
                                           _enumerated_distribution(n, k, [m]),
                                           atol=1e-12, err_msg=f"{k} of {n}, slot {m}")


def test_drawing_on_from_a_remainder_continues_without_replacement():
    for draws in ([1, 1], [1, 2], [2, 1], [0, 2], [1, 1, 1], [2, 0, 2]):
        np.testing.assert_allclose(_joint_distribution(4, draws),
                                   _enumerated_distribution(4, sum(draws)),
                                   atol=1e-12, err_msg=str(draws))


def test_a_remainder_builds_only_the_slots_that_are_read():
    def traced(n: int, read: int) -> int:
        Value = type("Value", (Enum,), {"elements": list(range(n))})

        @DPP
        def two_drawn() -> Bool:
            (a, b), rest = pick([Value.const(v) for v in range(n)], 2)
            assert rest[0] is rest[0]
            return Bool.any([a == b, *(r == 0 for r in rest[1:1 + read])])

        return op_count(two_drawn.program())

    # Both index types are 6 bits wide, so the draws record the same assignments, and
    # the two programs differ by the constants of the choices alone.
    assert traced(60, 0) - traced(33, 0) == 60 - 33
    assert traced(33, 0) < traced(33, 1) < traced(33, 2)


def test_a_remainder_read_outside_its_definition_is_refused():
    kept = []

    @DPP
    def first() -> Card:
        drawn, rest = pick(_deck())
        kept.append(rest)
        return drawn

    with pytest.raises(RuntimeError, match="rest of a pick was used outside"):
        kept[0][0]

    with pytest.raises(RuntimeError, match="rest of a pick was used outside"):
        @DPP
        def second() -> Card:
            return pick(kept[0])[0]


@pytest.mark.slow
def test_drawing_two_of_twenty_is_answered_exactly_in_under_a_second():
    Value = type("Value", (Enum,), {"elements": list(range(20))})

    @DPP
    def second_is_zero() -> Bool:
        (_, second), _ = pick([Value.const(v) for v in range(20)], 2)
        return second == 0

    start = time.perf_counter()
    assert prob(second_is_zero, method="exact") == pytest.approx(1 / 20)
    assert time.perf_counter() - start < 1.0
