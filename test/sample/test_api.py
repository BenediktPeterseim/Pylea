"""The public query surface: method resolution, sampled results, guards, caching.

The first test pins, in a fresh interpreter, that exact queries never import
`pylea.sample`.
"""
from __future__ import annotations

import copy
import pickle
import subprocess
import sys

import pytest

from pylea import DPP, Bool, Enum, flip, observe, prob, probability_mass_function
from pylea.sample.estimate import EstimatedLikelihoodTable, TargetNotMet
from pylea.syntax.estimate import Estimate
from pylea.syntax.expectation import expectation


@DPP
def either() -> Bool:
    a, b = flip(0.5), flip(0.5)
    observe(a | b)
    return a


def test_exact_queries_never_import_the_sampling_package():
    # The compiled surface and explicitly exact queries must leave the sampling
    # machinery unimported; only a query that can sample loads it.
    code = (
        "import sys\n"
        "from pylea import DPP, Bool, flip, observe, prob, expectation\n"
        "from pylea import probability_mass_function, Enum\n"
        "@DPP\n"
        "def e() -> Bool:\n"
        "    x = flip(0.25)\n"
        "    observe(x | flip(0.5))\n"
        "    return x\n"
        "assert round(prob(e, method='exact'), 6) == round(1/2.5, 6)\n"
        "assert round(prob(e, rel_err=0.0), 6) == round(1/2.5, 6)\n"
        "assert e.evaluated(method='exact') is not None\n"
        "assert float(e.compiled().distribution()[1]) > 0\n"
        "assert not any(m.startswith('pylea.sample') for m in sys.modules), "
        "sorted(m for m in sys.modules if m.startswith('pylea.sample'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_a_sampled_probability_is_a_certified_float():
    p = prob(either, method="smc", rel_err=0.05, seed=0)
    assert isinstance(p, Estimate)
    assert p.engine == "smc" and p.target_met
    assert p.low - 1e-12 <= 2.0 / 3.0 <= p.high + 1e-12
    assert repr(p) == repr(float(p))
    assert isinstance(p + 0.0, float) and not isinstance(p + 0.0, Estimate)


def test_method_resolution_rejects_inconsistent_requests():
    with pytest.raises(ValueError):
        prob(either, method="wrong")
    with pytest.raises(ValueError):
        prob(either, method="exact", rel_err=0.1)   # a target needs a sampling lane
    assert prob(either, method="smc", rel_err=0.0) == prob(either, method="exact")


def test_an_exact_engine_option_with_no_method_pins_the_exact_engine():
    # An exact-engine option states how the answer is computed, so naming one without
    # a method means the exact engine, never a sampled answer.
    assert prob(either, kappa=20) == prob(either, method="exact")
    with pytest.raises(ValueError):
        prob(either, kappa=20, rel_err=0.1)     # the two ask for different engines


def test_a_sampled_estimate_survives_pickling_and_deepcopy():
    p = prob(either, method="smc", rel_err=0.1, seed=9)
    assert isinstance(p, Estimate)
    clone = pickle.loads(pickle.dumps(p))
    assert isinstance(clone, Estimate)
    assert float(clone) == float(p) and clone.low == p.low and clone.trace == p.trace
    again = copy.deepcopy(p)
    assert again.high == p.high and again.engine == p.engine


def test_an_estimated_table_records_the_floor_it_certified_against():
    table = either.evaluated(method="smc", rel_err=0.1, seed=5)
    assert isinstance(table, EstimatedLikelihoodTable)
    # Two entries, so the default floor is min(rel_err, 1 / (4 * 2)).
    assert table.floor == 0.1


def test_the_default_method_is_the_portfolio_and_answers_small_models_exactly():
    p = prob(either)
    assert isinstance(p, Estimate)
    assert p.engine == "exact" and p.low == p.high == float(p) == 2.0 / 3.0
    with_target = prob(either, rel_err=0.1, seed=0)
    assert isinstance(with_target, Estimate)
    assert with_target.low - 1e-12 <= 2.0 / 3.0 <= with_target.high + 1e-12


def test_exact_only_compile_options_raise_under_pure_sampling():
    with pytest.raises(ValueError):
        prob(either, method="smc", rel_err=0.1, kappa=20)
    with pytest.raises(ValueError):
        either.evaluated(method="smc", rel_err=0.1, simplify=False)


class Die(Enum):
    elements = [1, 2, 3, 4, 5, 6]


@DPP
def high_roll() -> Die:
    d = Die.uniform()
    observe(d >= 4)
    return d


def test_a_sampled_expectation_maps_its_interval_back_to_the_value_scale():
    m = expectation(high_roll, method="smc", rel_err=0.05, seed=0)
    assert isinstance(m, Estimate)
    assert m.low - 1e-9 <= 5.0 <= m.high + 1e-9
    assert 1.0 <= m.low <= m.high <= 6.0


@pytest.mark.parametrize("options", [{"method": "exact"}, {"rel_err": 0.0},
                                     {"method": "exact", "max_time": 60.0}])
def test_an_exact_answer_is_an_estimate_of_zero_width(options):
    p = prob(either, **options)
    m = expectation(high_roll, **options)
    pmf = probability_mass_function(high_roll, **options)
    for value, truth in [(p, 2.0 / 3.0), (m, 5.0), (pmf[4], 1.0 / 3.0), (pmf[1], 0.0)]:
        assert isinstance(value, Estimate)
        assert value.engine == "exact" and value.target_met
        assert value.low == value.high == float(value) == pytest.approx(truth)


class Digit(Enum):
    elements = [0, 1, 2, 3]


@DPP
def nonzero() -> Digit:
    d = Digit.uniform()
    observe(d != 0)
    return d


def test_a_sampled_pmf_brackets_every_posterior_entry():
    exact = probability_mass_function(nonzero)
    sampled = probability_mass_function(nonzero, method="smc", rel_err=0.1, seed=1)
    for element, estimate in sampled.items():
        assert isinstance(estimate, Estimate)
        assert estimate.low - 1e-12 <= exact[element] <= estimate.high + 1e-12


def test_a_sampled_table_brackets_the_exact_table_and_normalises():
    exact = either.evaluated().matrix
    table = either.evaluated(method="smc", rel_err=0.1, seed=2)
    assert isinstance(table, EstimatedLikelihoodTable)
    assert table.target_met
    assert (table.lower - 1e-12 <= exact).all() and (exact <= table.upper + 1e-12).all()
    posterior = table.normalised()
    exact_posterior = either.evaluated().probabilities
    assert (posterior.lower - 1e-12 <= exact_posterior).all()
    assert (exact_posterior <= posterior.upper + 1e-12).all()


def test_a_table_with_inputs_rejects_sampling_with_a_costed_message():
    @DPP
    def channel(x: Bool) -> Bool:
        return x & flip(0.9) | ~x & flip(0.1)

    with pytest.raises(ValueError, match="input columns"):
        channel.evaluated(method="smc", rel_err=0.1)


def test_a_missed_target_warns_by_default_and_raises_under_strict():
    @DPP
    def hard() -> Bool:
        return flip(0.3)

    with pytest.warns(TargetNotMet):
        p = prob(hard, method="smc", rel_err=1e-4, max_time=0.0, seed=0)
    assert isinstance(p, Estimate)
    assert not p.target_met
    assert p.low <= 0.3 <= p.high
    with pytest.raises(ValueError):
        prob(hard, method="smc", rel_err=1e-4, max_time=0.0, seed=0, strict=True)


def test_seeded_sampled_results_are_cached_and_reproducible():
    first = prob(either, method="smc", rel_err=0.1, seed=42)
    second = prob(either, method="smc", rel_err=0.1, seed=42)
    assert isinstance(first, Estimate) and first is second   # the cached object returns
    fresh = prob(either, method="smc", rel_err=0.1, seed=43)
    assert isinstance(fresh, Estimate)
    assert float(fresh) != float(first) or fresh.n_samples == first.n_samples


def test_the_expectation_floor_is_read_on_the_value_scale():
    class Payout(Enum):
        elements = [0.0, 1000.0]

    @DPP
    def long_shot() -> Payout:
        return flip(1e-6).match({True: 1000.0, False: 0.0}, data_type=Payout)

    m = expectation(long_shot, method="smc", rel_err=0.05, floor=0.5, seed=6)
    assert isinstance(m, Estimate)
    # Certifying "the mean lies below 0.5" needs real evidence on the value scale; a
    # floor mistaken for the rescaled one would fire immediately with a high bound in
    # the hundreds.
    assert m.target_met and m.high <= 0.5


def test_the_floor_contract_reaches_prob():
    @DPP
    def rare() -> Bool:
        return flip(1e-7)

    p = prob(rare, method="smc", rel_err=0.05, floor=1e-3, seed=3)
    assert isinstance(p, Estimate)
    assert p.target_met and p.high <= 1e-3


def test_the_unfolding_guard_refuses_an_exponential_program_for_pure_sampling():
    # The interfaces are too wide to collapse, so the exponential unfolding stands.
    width = 10

    @DPP
    def leaf(x, types=(Bool,) * width):
        return tuple(bit & flip(0.5) for bit in x)

    level = leaf
    for _ in range(20):
        previous = level

        @DPP
        def wider(x, types=(Bool,) * width):
            return previous(*previous(*x))

        level = wider

    @DPP
    def deep() -> Bool:
        ys = level(*[flip(0.5) for _ in range(width)])
        acc = ys[0]
        for y in ys[1:]:
            acc = acc & y
        return acc

    with pytest.raises(ValueError, match="feasibility cap"):
        prob(deep, method="smc", rel_err=0.1, seed=0)


def test_estimated_table_totals_bracket_the_evidence():
    table = either.evaluated(method="smc", rel_err=0.1, seed=4)
    assert isinstance(table, EstimatedLikelihoodTable)
    assert float(table.total_lower[0]) - 1e-12 <= 0.75 <= float(table.total_upper[0]) + 1e-12


def test_sampling_options_raise_under_the_exact_engine_instead_of_being_ignored():
    for options in ({"floor": 0.5}, {"seed": 3}, {"strict": True}, {"confidence": 0.9}):
        with pytest.raises(ValueError, match="sampling lane"):
            prob(either, method="exact", **options)     # type: ignore[arg-type]
        with pytest.raises(ValueError, match="sampling lane"):
            expectation(high_roll, method="exact", **options)   # type: ignore[arg-type]
        with pytest.raises(ValueError):                 # pinned exact, same contradiction
            prob(either, kappa=20, **options)           # type: ignore[arg-type]


def test_device_auto_leaves_the_method_open():
    from pylea.syntax.method import resolve

    assert resolve(None, 0.05, {"device": "auto"}).method == "auto"
    assert resolve(None, None, {"device": "mps"}).method == "exact"
    p = prob(either, device="auto", rel_err=0.05, seed=0)
    assert isinstance(p, Estimate)
    assert p.low - 1e-12 <= 2.0 / 3.0 <= p.high + 1e-12


def test_seeded_auto_results_are_cached_like_smc_ones():
    first = prob(either, rel_err=0.1, seed=11)
    assert isinstance(first, Estimate)
    assert prob(either, rel_err=0.1, seed=11) is first
    assert prob(either, rel_err=0.1, seed=11, max_time=60.0) is not first


def test_a_sampled_table_refuses_a_log_scale_outside_float64():
    @DPP
    def faint() -> Bool:
        x = flip(0.5)
        for _ in range(200):
            observe(x | ~x, if_true=1e-3, if_false=1e-3)
        return x

    with pytest.raises(ValueError, match="does not fit float64"):
        faint.evaluated(method="smc", rel_err=0.1, seed=0)
    with pytest.raises(ValueError, match="does not fit float64"):
        faint.evaluated(method="exact")
    # The scalar contract is normalised, so the same factor cancels there.
    p = prob(faint, method="smc", rel_err=0.1, seed=0)
    assert isinstance(p, Estimate)
    assert p.low - 1e-12 <= 0.5 <= p.high + 1e-12


def test_the_sampling_funnel_refuses_a_program_with_inputs():
    from pylea.sample.query import prepared

    @DPP
    def channel(x: Bool) -> Bool:
        return x & flip(0.9) | ~x & flip(0.1)

    with pytest.raises(ValueError, match="zero-input"):
        prepared(channel)
