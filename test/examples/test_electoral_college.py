"""The electoral-college example against independent enumeration and integer counting."""
from fractions import Fraction
from itertools import product

import pytest

from pylea import prob
from pylea.examples.electoral_college import ELECTORAL_VOTES, decisive, pivotal


def counted_power(member: str, weights: dict[str, int], quota: int) -> Fraction:
    """The decisive fraction by exact integer counting over running totals."""
    total_votes = sum(weights.values())
    counts = [0] * (total_votes + 1)
    counts[0] = 1
    for name, votes in weights.items():
        if name == member:
            continue
        for t in range(total_votes, votes - 1, -1):
            counts[t] += counts[t - votes]
    w = weights[member]
    hits = sum(counts[t] for t in range(max(0, quota - w), quota))
    return Fraction(hits, 2 ** (len(weights) - 1))


def test_a_small_game_matches_brute_force_enumeration_exactly():
    weights = {"A": 5, "B": 3, "C": 3, "D": 2, "E": 1}
    quota = 8
    for member in weights:
        others = [w for name, w in weights.items() if name != member]
        hits = sum(
            1 for sides in product([0, 1], repeat=len(others))
            if quota - weights[member] <= sum(w * s for w, s in zip(others, sides))
            <= quota - 1)
        assert prob(decisive(member, weights, quota), method="exact") == \
            pytest.approx(hits / 2 ** len(others)), member


def test_the_full_electoral_college_matches_exact_integer_counting():
    for state in ("Wyoming", "California"):
        expected = counted_power(state, ELECTORAL_VOTES, 270)
        assert prob(pivotal(state), method="exact") == pytest.approx(float(expected),
                                                                     rel=1e-12), state


def test_a_quota_outside_the_vote_range_is_refused():
    with pytest.raises(ValueError, match="quota"):
        decisive("A", {"A": 2, "B": 3}, quota=6)


@pytest.mark.slow
def test_every_state_matches_exact_integer_counting():
    for state in ELECTORAL_VOTES:
        expected = counted_power(state, ELECTORAL_VOTES, 270)
        assert prob(pivotal(state), method="exact") == pytest.approx(float(expected),
                                                                     rel=1e-12), state
