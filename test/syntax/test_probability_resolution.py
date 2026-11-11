"""Probabilities must survive the trip from Python float to evaluated answer.

Rare-event models are where this matters, an improbable event silently becoming an
impossible one being a wrong answer rather than an inaccurate one.  Three properties are
pinned here: a small probability round-trips, a conjunction of several of them still does,
and anything a float cannot express raises instead of collapsing to zero.
"""
from fractions import Fraction

import numpy as np
import pytest

from test.syntax.helpers import matrix_of

from pylea.syntax.dpp import DPP, Bool
from pylea.syntax.enum import Enum
from pylea.syntax.flip import flip


def _prob(dpp) -> float:
    """P(dpp() is true), for a DPP with no inputs and one `Bool` output."""
    v = matrix_of(dpp).ravel()
    return float(v[1] / v.sum())


SMALL = [1e-3, 1e-6, 1e-9, 1e-10, 7.635206808016444e-13, 1e-15, 1e-18]


@pytest.mark.parametrize("p", SMALL)
def test_a_small_probability_round_trips(p):
    @DPP
    def one() -> Bool:
        return flip(p)

    got = _prob(one)
    assert got > 0.0, f"flip({p!r}) collapsed to exactly zero"
    assert got == pytest.approx(p, rel=1e-6)


def test_folding_a_conjunction_of_rare_events_agrees_with_not_folding():
    """The property constant folding relies on.

    Folding replaces a conjunction of independent coins by one coin carrying their product,
    and the two forms must agree.  Only the folded form passes that product through `flip`,
    the unfolded one forming it in the contraction, so this has to be a differential test
    rather than a property of the conjunction alone.
    """
    ps = [0.004018670992852913, 0.007489573008041761, 0.0034273370498886235,
          0.001053254464333575, 0.007027343293247279]

    @DPP
    def unfolded() -> Bool:
        acc = flip(ps[0])
        for p in ps[1:]:
            acc = acc & flip(p)
        return acc

    @DPP
    def folded() -> Bool:                      # what constant folding produces
        return flip(float(np.prod(ps)))

    a, b = _prob(unfolded), _prob(folded)
    assert b > 0.0, "the folded form collapsed to exactly zero"
    assert a == pytest.approx(b, rel=1e-6)


@pytest.mark.parametrize("p", [1e-19, 1e-30, 1e-300])
def test_a_probability_a_float_can_hold_is_kept(p):
    """The limit on a probability is the float's own, and nothing narrower."""
    @DPP
    def tiny() -> Bool:
        return flip(p)
    assert _prob(tiny) == pytest.approx(p, rel=1e-9)


@pytest.mark.parametrize("p", [Fraction(1, 10 ** 400), Fraction(1, 2 ** 1200)])
def test_a_probability_below_the_float_floor_is_refused_not_rounded(p):
    with pytest.raises(ValueError, match="indistinguishable from an impossible event"):
        @DPP
        def tiny() -> Bool:
            return flip(p)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_exact_zero_and_one_are_accepted_as_constants(p):
    @DPP
    def certain() -> Bool:
        return flip(p)

    assert _prob(certain) == pytest.approx(p)


def test_a_rare_enum_state_is_not_silently_dropped():
    class Three(Enum):
        elements = ["common", "rare", "other"]

    @DPP
    def pick() -> Bool:
        x = Three.random({"common": 0.5, "rare": 1e-12, "other": 0.5 - 1e-12})
        return x == "rare"

    got = _prob(pick)
    assert got > 0.0, "rare enum state collapsed to exactly zero"
    assert got == pytest.approx(1e-12, rel=1e-5)


def test_an_enum_state_is_as_expressive_as_a_coin():
    """`Enum.random` and `flip` accept the same range of probabilities."""
    class Two(Enum):
        elements = ["a", "b"]

    @DPP
    def two() -> Two:
        return Two.random({"a": 1.0, "b": 1e-30})

    v = matrix_of(two).ravel()
    assert float(v[1] / v.sum()) == pytest.approx(1e-30, rel=1e-12)


def test_an_enum_refuses_a_state_that_normalising_would_round_to_zero():
    """5e-324 is the smallest positive double, so dividing it by 2 gives exactly 0."""
    class Two(Enum):
        elements = ["a", "b"]

    with pytest.raises(ValueError, match="underflows to 0"):
        Two.random({"a": 2.0, "b": 5e-324})
