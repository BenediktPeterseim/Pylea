"""`observe` with likelihoods: the virtual-evidence generalisation of conditioning."""
import numpy as np
import pytest

from pylea import DPP, Bool, flip, observe, prob


def test_soft_observation_equals_the_match_of_flips_construction_exactly():
    @DPP
    def native() -> Bool:
        x = flip(0.2)
        observe(x, if_true=0.7, if_false=0.1)
        return x

    @DPP
    def sugared() -> Bool:
        x = flip(0.2)
        observe(x.match({True: flip(0.7), False: flip(0.1)}))
        return x

    assert prob(native) == prob(sugared)


def test_only_the_likelihood_ratio_reaches_a_conditional_query():
    def posterior(a: float, b: float) -> float:
        @DPP
        def reported() -> Bool:
            x = flip(0.2)
            observe(x, if_true=a, if_false=b)
            return x

        return prob(reported)

    assert posterior(0.7, 0.1) == pytest.approx(posterior(7.0, 1.0))


def test_the_defaults_are_hard_conditioning():
    @DPP
    def defaulted() -> Bool:
        x = flip(0.5)
        observe(x | flip(0.5))
        return x

    assert prob(defaulted) == pytest.approx(2 / 3)


def test_constant_strength_evidence_scales_the_mass_rather_than_disappearing():
    @DPP
    def massed() -> Bool:
        x = flip(0.5)
        observe(x, if_true=0.4, if_false=0.4)
        return x

    assert float(massed.evaluated().matrix.sum()) == pytest.approx(0.4)
    assert prob(massed) == pytest.approx(0.5)


def test_a_negative_or_all_zero_likelihood_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        @DPP
        def negative() -> Bool:
            x = flip(0.5)
            observe(x, if_true=-0.5, if_false=0.5)
            return x

    with pytest.raises(ValueError, match="both zero"):
        @DPP
        def annihilated() -> Bool:
            x = flip(0.5)
            observe(x, if_true=0.0, if_false=0.0)
            return x


def test_soft_observation_certifies_on_the_sampling_lane():
    @DPP
    def reported() -> Bool:
        x = flip(0.2)
        observe(x, if_true=0.7, if_false=0.1)
        return x

    from pylea.syntax.estimate import Estimate

    estimate = prob(reported, method="smc", rel_err=0.05, seed=0)
    assert isinstance(estimate, Estimate)
    assert estimate.low <= 0.14 / 0.22 <= estimate.high
