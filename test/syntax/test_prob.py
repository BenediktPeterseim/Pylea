"""`prob` on events, and its refusal of anything that is not one."""
import pytest
from pylea.syntax.dpp import DPP, Bool
from pylea.syntax.flip import flip
from pylea.syntax.observe import observe
from pylea.syntax.prob import prob
from pylea.examples.simple import (
    not_after_flip,
    fault_tree_and_only,
    obs_dpp,
    two_inputs_dpp,
    two_outputs_dpp,
)


def test_prob_returns_a_float_in_the_unit_interval():
    p = prob(not_after_flip)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_probability_of_a_single_flip_is_its_parameter(p):
    @DPP
    def e() -> Bool:
        return flip(p)
    pi = prob(e)
    assert pi == pytest.approx(p, abs=1e-9)
    assert pi == pytest.approx(p, abs=1e-9)


def test_negating_a_flip_gives_one_minus_its_probability():
    pi = prob(not_after_flip)
    assert pi == pytest.approx(0.7, abs=1e-9)
    assert pi == pytest.approx(0.7, abs=1e-9)


def test_two_independent_flips_conjoined_multiply():
    pi = prob(fault_tree_and_only)
    assert pi == pytest.approx(0.02, abs=1e-9)
    assert pi == pytest.approx(0.02, abs=1e-9)


def test_observing_the_returned_event_makes_it_certain():
    pi = prob(obs_dpp)
    assert pi == pytest.approx(1.0, abs=1e-9)
    assert pi == pytest.approx(1.0, abs=1e-9)


def test_a_probability_stays_within_the_unit_interval():
    pi = prob(fault_tree_and_only)
    assert 0.0 <= pi <= 1.0


def test_prob_rejects_a_dpp_with_inputs():
    with pytest.raises(ValueError, match="no input"):
        prob(two_inputs_dpp)  # type: ignore[arg-type]


def test_prob_rejects_a_dpp_with_several_outputs():
    with pytest.raises(ValueError, match="no input"):
        prob(two_outputs_dpp)  # type: ignore[arg-type]
