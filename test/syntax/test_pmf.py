"""The probability mass function: exact at any acceptance probability, and refusing what
it cannot define."""
from __future__ import annotations

import pytest

from pylea import DPP, Enum, flip, observe, probability_mass_function


class Two(Enum):
    elements = ["a", "b"]


class Three(Enum):
    elements = [0, 1, 2]


def test_a_pmf_is_exact_however_small_the_acceptance_probability():
    @DPP
    def rare() -> Two:
        coins = [flip(1e-30) for _ in range(20)]
        for coin in coins:
            observe(coin)
        return Two.random({"a": 0.25, "b": 0.75})

    pmf = probability_mass_function(rare, method="exact")
    assert float(pmf["a"]) == pytest.approx(0.25, rel=1e-12)
    assert float(pmf["b"]) == pytest.approx(0.75, rel=1e-12)


@pytest.mark.parametrize("acceptance", [1e-318, 1e-321])
def test_a_pmf_keeps_full_precision_at_a_subnormal_acceptance_probability(acceptance):
    @DPP
    def subnormal() -> Two:
        observe(flip(acceptance))
        return Two.random({"a": 0.3, "b": 0.7})

    pmf = probability_mass_function(subnormal, method="exact")
    assert float(pmf["a"]) == pytest.approx(0.3, rel=1e-12)


def test_a_pmf_of_an_impossible_observation_raises_as_prob_does():
    @DPP
    def never() -> Two:
        observe(flip(0.0))
        return Two.uniform()

    with pytest.raises(ZeroDivisionError):
        probability_mass_function(never, method="exact")


@pytest.mark.parametrize("method", ["exact", "auto", "smc"])
def test_a_pmf_with_mass_on_unencoded_patterns_raises(method):
    @DPP
    def stray() -> Three:
        return Three.uniform_bits()

    options = {"seed": 0} if method == "smc" else {}
    with pytest.raises(ValueError, match="encode no value of Three"):
        probability_mass_function(stray, method=method, **options)
