"""`expectation` on models with a numeric output, and its refusal of anything else."""
import math

import pytest

from pylea import DPP, Bool, Enum, Tuple, expectation, flip, observe, prob
from pylea.examples.german_tank import fit
from pylea.examples.simple import Color, two_inputs_dpp, two_outputs_dpp


FACES = [1, 2, 3, 4, 5, 6]


class Die(Enum):
    elements = FACES


class DieSquared(Enum):
    elements = [n * n for n in FACES]


@DPP
def roll() -> Die:
    return Die.uniform()


@DPP
def high_roll() -> Die:
    d = Die.uniform()
    observe(d >= 4)
    return d


def test_the_expectation_of_a_uniform_die_is_its_mean():
    assert expectation(roll) == pytest.approx(3.5, rel=1e-12)


def test_conditioning_gives_the_mean_of_the_posterior():
    assert expectation(high_roll) == pytest.approx(5.0, rel=1e-12)


def test_the_expectation_of_a_boolean_output_is_its_probability():
    @DPP
    def event() -> Bool:
        return flip(0.3) | flip(0.4)

    assert expectation(event) == pytest.approx(prob(event), rel=1e-12)


def test_a_pushforward_gives_the_second_moment_and_with_it_the_variance():
    @DPP
    def squared() -> DieSquared:
        return roll().match({n: n * n for n in FACES}, data_type=DieSquared)

    second = expectation(squared)
    assert second == pytest.approx(91 / 6, rel=1e-12)
    assert second - expectation(roll) ** 2 == pytest.approx(35 / 12, rel=1e-12)


def test_keyword_arguments_reach_the_compiler_without_changing_the_answer():
    assert expectation(high_roll, simplify="always", kappa=4) == pytest.approx(
        expectation(high_roll), rel=1e-12)


def test_the_posterior_mean_of_the_german_tank_problem_matches_its_closed_form():
    # Three captures with 15 the largest, so the posterior is proportional to 1/n**3 on
    # the fleet sizes from 15 up.
    weights = {n: 1 / n ** 3 for n in range(15, 41)}
    total = math.fsum(weights.values())
    expected = math.fsum(n * w for n, w in weights.items()) / total
    assert expectation(fit([15, 14, 3])) == pytest.approx(expected, rel=1e-12)


def test_the_expectation_survives_evidence_too_small_for_a_likelihood_table():
    @DPP
    def against_the_odds() -> Die:
        d = Die.uniform()
        for _ in range(20):
            observe(flip(1e-30))
        return d

    with pytest.raises(ValueError, match="does not fit float64"):
        against_the_odds.evaluated()
    assert expectation(against_the_odds) == pytest.approx(3.5, rel=1e-12)


def test_expectation_rejects_a_dpp_with_inputs():
    with pytest.raises(ValueError, match="zero-input"):
        expectation(two_inputs_dpp)  # type: ignore[arg-type]


def test_expectation_rejects_a_dpp_with_several_outputs():
    with pytest.raises(ValueError, match="single output"):
        expectation(two_outputs_dpp)


def test_expectation_rejects_an_output_whose_values_are_not_numbers():
    @DPP
    def colour() -> Color:
        return Color.uniform()

    with pytest.raises(ValueError, match="are numbers"):
        expectation(colour)


def test_expectation_rejects_an_output_of_a_type_holding_several_values():
    class Pair(Tuple):
        types = (Die, Die)

    @DPP
    def both() -> Pair:
        return Pair(*Die.uniform().bits(), *Die.uniform().bits())

    with pytest.raises(ValueError, match="are numbers"):
        expectation(both)


def test_probability_on_a_pattern_encoding_no_value_is_refused():
    class Triple(Enum):
        elements = [1, 2, 3]

    @DPP
    def over_patterns() -> Triple:
        return Triple.uniform_bits()

    with pytest.raises(ValueError, match="encode no value"):
        expectation(over_patterns)


def test_evidence_of_probability_zero_raises_rather_than_dividing():
    @DPP
    def impossible() -> Die:
        d = Die.uniform()
        observe(d >= 4)
        observe(d <= 3)
        return d

    with pytest.raises(ZeroDivisionError):
        expectation(impossible)
