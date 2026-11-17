"""Checking the options of a query before anything is computed."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pylea import DPP, Bool, Enum, expectation, flip, prob
from pylea.syntax.method import resolve


@DPP
def coin() -> Bool:
    return flip(0.3)


class Score(Enum):
    elements = [-1.0, 0.0, 2.0]


@DPP
def score() -> Score:
    return Score.uniform()


@pytest.mark.parametrize("options, error", [
    ({"floor": -1.0}, ValueError),
    ({"floor": 2.0}, ValueError),
    ({"floor": math.nan}, ValueError),
    ({"floor": "low"}, TypeError),
    ({"max_time": -1.0}, ValueError),
    ({"max_time": math.nan}, ValueError),
    ({"max_time": "5"}, TypeError),
    ({"seed": "x"}, TypeError),
    ({"seed": -1}, ValueError),
    ({"seed": [1, 2]}, TypeError),
    ({"seed": True}, TypeError),
    ({"rel_err": math.inf}, ValueError),
    ({"rel_err": 1.0}, ValueError),
    ({"rel_err": 1.5}, ValueError),
    ({"rel_err": -0.1}, ValueError),
    ({"rel_err": True}, TypeError),
    ({"confidence": True}, TypeError),
    ({"confidence": 1.0}, ValueError),
    ({"strict": "yes"}, TypeError),
    ({"method": "EXACT"}, ValueError),
    ({"kappa": -5}, ValueError),
    ({"kappa": 2.5}, ValueError),
    ({"kappa": "x"}, TypeError),
    ({"kappa": True}, TypeError),
    ({"simplify": "sometimes"}, ValueError),
    ({"simplify": 1}, TypeError),
    ({"recompile": "yes"}, TypeError),
    ({"device": 0}, TypeError),
    ({"order_finder": "greedy"}, TypeError),
])
def test_resolve_refuses_an_invalid_option(options, error):
    with pytest.raises(error):
        prob(coin, **options)


@pytest.mark.parametrize("method", ["exact", "auto", "smc"])
@pytest.mark.parametrize("max_time", [None, 30.0])
def test_a_misspelt_keyword_is_refused_under_every_method(method, max_time):
    with pytest.raises(TypeError, match="did you mean 'max_time'"):
        prob(coin, method=method, max_time=max_time, max_tme=5)
    with pytest.raises(TypeError, match="did you mean 'rel_err'"):
        coin.evaluated(method=method, max_time=max_time, rel_error=0.1)


@pytest.mark.parametrize("deadline", [math.inf, 1e10, np.float64(1e300)])
def test_an_infinite_deadline_means_no_deadline(deadline):
    assert resolve(None, None, {}, max_time=deadline).max_time is None
    assert float(prob(coin, method="exact", max_time=deadline)) == pytest.approx(0.3)


def test_compiled_refuses_an_invalid_option_when_called_directly():
    with pytest.raises(ValueError, match="kappa"):
        coin.compiled(kappa=-1)
    with pytest.raises(TypeError, match="GreedyOrderFinder"):
        coin.compiled(order_finder="greedy")
    with pytest.raises(ValueError, match="simplify"):
        coin.compiled(simplify="sometimes")


def test_the_message_for_rel_err_zero_with_a_seed_names_rel_err():
    with pytest.raises(ValueError, match="rel_err=0 asks for the exact value, which seed"):
        prob(coin, rel_err=0.0, seed=0)


def test_rel_err_zero_under_the_sampling_engine_is_refused():
    with pytest.raises(ValueError, match="rel_err=0"):
        prob(coin, method="smc", rel_err=0.0)


def test_an_expectation_floor_lies_on_the_value_scale():
    resolve("smc", None, {}, floor=1.5, floor_range=(-1.0, 2.0))
    with pytest.raises(ValueError, match=r"floor must lie in \[-1.0, 2.0\]"):
        expectation(score, method="smc", floor=3.0, seed=0)
    assert float(expectation(score, method="smc", floor=1.5, seed=0, rel_err=0.1)) < 2.0


def test_numpy_numbers_are_accepted_as_options():
    contract = resolve("auto", np.float32(0.05), {"kappa": np.int64(20)},
                       seed=np.int64(3), max_time=np.float64(10.0))
    assert (contract.rel_err, contract.seed, contract.max_time) == (
        pytest.approx(0.05), 3, 10.0)
    assert type(contract.seed) is int


def test_a_kappa_beyond_the_float_range_means_no_size_limit():
    from pylea.compile.order import slice_target

    assert slice_target(2000) is None and slice_target(1023) == 2.0 ** 1023
    assert float(prob(coin, kappa=2 ** 40)) == pytest.approx(0.3)


def test_an_integer_beyond_the_float_range_is_read_as_infinity():
    assert resolve(None, None, {}, max_time=10 ** 400).max_time is None
    with pytest.raises(ValueError, match="max_time"):
        resolve(None, None, {}, max_time=-(10 ** 400))
    with pytest.raises(ValueError, match="finite"):
        resolve("smc", None, {}, floor=10 ** 400)


def test_numpy_booleans_are_accepted_wherever_a_bool_is():
    assert coin.compiled(recompile=np.True_, simplify=np.False_) is coin.compiled(
        simplify=False)
    assert resolve("smc", None, {}, strict=np.True_).strict is True



def test_the_floor_range_in_a_message_is_printed_exactly():
    with pytest.raises(ValueError, match=r"\[0.0, 123456789.0\]"):
        resolve("smc", None, {}, floor=123456790, floor_range=(0.0, 123456789.0))


def test_a_torch_device_is_accepted_as_a_device():
    torch = pytest.importorskip("torch")
    assert float(prob(coin, device=torch.device("cpu"))) == pytest.approx(0.3)
