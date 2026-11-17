"""The `Estimate` result type: an immutable float with its interval."""
from __future__ import annotations

import copy
import pickle

import pytest

from pylea.syntax.estimate import Estimate


def _sampled(**changes) -> Estimate:
    fields = dict(low=0.24, high=0.26, confidence=0.95, engine="smc", n_samples=1000,
                  elapsed=0.1, target_met=True, trace=("a", "b"))
    value = changes.pop("value", 0.25)
    fields.update(changes)
    return Estimate(value, **fields)


def test_an_estimate_survives_every_pickle_protocol_and_copy():
    e = _sampled()
    copies = [pickle.loads(pickle.dumps(e, protocol=p))
              for p in range(pickle.HIGHEST_PROTOCOL + 1)]
    copies += [copy.copy(e), copy.deepcopy(e)]
    for clone in copies:
        assert type(clone) is Estimate and float(clone) == 0.25
        assert (clone.low, clone.high, clone.engine, clone.trace) == (
            0.24, 0.26, "smc", ("a", "b"))


def test_an_estimate_cannot_be_changed_in_place():
    e = _sampled()
    with pytest.raises(AttributeError, match="replace"):
        e.low = 0.0
    with pytest.raises(AttributeError):
        del e.high
    with pytest.raises(AttributeError):
        e.extra = 1                                     # type: ignore[attr-defined]
    assert e.low == 0.24


def test_replace_gives_a_new_estimate_with_the_given_fields():
    e = _sampled()
    later = e.replace(elapsed=0.0)
    assert later.elapsed == 0.0 and e.elapsed == 0.1
    assert later.replace(value=0.26) == 0.26
    with pytest.raises(TypeError):
        e.replace(lowest=0.0)
    with pytest.raises(ValueError):
        e.replace(value=0.3)


def test_an_estimate_outside_its_own_interval_is_refused():
    with pytest.raises(ValueError, match="interval"):
        _sampled(value=0.3)
    with pytest.raises(ValueError, match="interval"):
        _sampled(low=0.26, high=0.24)


def test_a_nan_estimate_is_refused():
    for field in ("value", "low", "high"):
        with pytest.raises(ValueError, match="NaN"):
            _sampled(**{field: float("nan")})
    with pytest.raises(ValueError, match="NaN"):
        Estimate.from_exact(float("nan"))


def test_the_exact_constructor_is_not_an_attribute_of_an_answer():
    e = Estimate.from_exact(0.3)
    assert not hasattr(e, "exact")
    assert (e.low, e.high, e.engine, e.target_met) == (0.3, 0.3, "exact", True)


def test_an_inverted_interval_is_named_as_such():
    with pytest.raises(ValueError, match="low <= high"):
        _sampled(low=0.6, high=0.4, value=0.5)


def test_a_trace_given_as_one_string_is_refused():
    with pytest.raises(TypeError, match="trace"):
        _sampled(trace="abc")


def test_copies_of_a_subclass_keep_their_class():
    class Labelled(Estimate):
        __slots__ = ()

    e = Labelled(0.25, low=0.24, high=0.26, confidence=0.95, engine="smc",
                 n_samples=1, elapsed=0.0, target_met=True, trace=())
    assert type(e.replace(elapsed=1.0)) is Labelled
    assert type(copy.deepcopy(e)) is Labelled


def test_an_estimate_of_a_string_is_refused():
    with pytest.raises(TypeError, match="number"):
        _sampled(value="0.25")


@pytest.mark.parametrize("field, value", [
    ("confidence", float("nan")), ("confidence", 7.0), ("confidence", 0.0),
    ("elapsed", float("nan")), ("elapsed", -1.0), ("n_samples", -3), ("n_samples", 2.5),
])
def test_every_field_of_an_estimate_is_checked(field, value):
    with pytest.raises(ValueError, match=field):
        _sampled(**{field: value})


@pytest.mark.parametrize("field, value, error", [
    ("confidence", True, TypeError), ("confidence", "0.9", TypeError),
    ("elapsed", "1", TypeError), ("value", 10 ** 400, ValueError),
    ("engine", 5, ValueError), ("engine", "mcmc", ValueError),
    ("target_met", "no", TypeError),
])
def test_a_field_of_the_wrong_type_or_value_is_refused_with_a_message(field, value, error):
    with pytest.raises(error, match="estimate"):
        _sampled(**{field: value})


def test_numpy_numbers_become_python_numbers_in_an_estimate():
    import numpy as np

    e = _sampled(n_samples=np.int64(5), target_met=np.True_)
    assert type(e.n_samples) is int and e.target_met is True
