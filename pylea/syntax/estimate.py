"""The result type of `pylea.syntax.prob.prob`, `pylea.syntax.expectation.expectation` and
`pylea.syntax.pmf.probability_mass_function`.

`Estimate` is a float subclass, so a result can be used wherever a plain probability or
mean is expected; the interval and the record of how the answer was computed are
attributes.  The exact engine's answer is an `Estimate` too, with an interval of zero
width, so `low` and `high` can be read whichever engine answered:

    >>> from pylea import DPP, Bool, flip, prob
    >>> @DPP
    ... def both() -> Bool:
    ...     return flip(0.5) & flip(0.5)
    >>> p = prob(both, method="exact")
    >>> p, p.low, p.high, p.engine
    (0.25, 0.25, 0.25, 'exact')

Two costs of the float subclass: any arithmetic on an `Estimate` returns a plain float,
discarding the interval, so a caller combining results must read `low` and `high` first;
and a result whose `target_met` is False looks like an ordinary number to a caller who
never reads the attribute.  For the second, every query emits a
`pylea.sample.estimate.TargetNotMet` warning on each missed target, and raises
`pylea.sample.estimate.TargetNotMetError`, which carries the result, instead under
`strict=True`.

An `Estimate` cannot be changed after it is made, since one result may be returned to
several callers, as the result cache of a seeded query does; `Estimate.replace` gives a
copy with some fields changed.
"""
from __future__ import annotations

import math
import numbers

import numpy as np

_FIELDS = ("low", "high", "confidence", "engine", "n_samples", "elapsed", "target_met",
           "trace")


def _rebuild(cls: type, value: float, fields: dict) -> Estimate:
    """The `Estimate` that `Estimate.__reduce__` describes."""
    return cls(value, **fields)


class Estimate(float):
    """A point estimate with an interval.

    The float value is the point estimate, which lies in ``[low, high]``; `low` and
    `high` bound the true value with probability at least `confidence`; `engine` names
    the engine that answered ("exact" or "smc"); `n_samples` is the number of samples
    drawn, 0 for the exact engine; `elapsed` is the query's wall-clock time in seconds;
    `target_met` says whether the requested accuracy was reached before any deadline;
    `trace` records how the answer was computed, so a surprising runtime is read back
    rather than guessed at.  A sampled query that drew no samples returns the middle of
    its interval.

        >>> e = Estimate(0.25, low=0.24, high=0.26, confidence=0.95, engine="smc",
        ...              n_samples=1000, elapsed=0.1, target_met=True, trace=())
        >>> e
        0.25
        >>> e.low, e.high, e + 0.75
        (0.24, 0.26, 1.0)

    A value outside its interval, NaN in the value or the interval, a `confidence` outside
    ``(0, 1]``, a negative or NaN `elapsed`, an `n_samples` that is not a non-negative
    integer or an `engine` other than "exact" and "smc" raises `ValueError`, and a field
    of the wrong type `TypeError`; assigning an attribute raises `AttributeError`.
    """

    __slots__ = _FIELDS

    low: float
    high: float
    confidence: float
    engine: str
    n_samples: int
    elapsed: float
    target_met: bool
    trace: tuple[str, ...]

    def __new__(cls, value: float, *, low: float, high: float, confidence: float,
                engine: str, n_samples: int, elapsed: float, target_met: bool,
                trace: tuple[str, ...]) -> Estimate:
        for name, number in (("value", value), ("low", low), ("high", high),
                             ("confidence", confidence), ("elapsed", elapsed)):
            if type(number) is not float and (isinstance(number, bool)
                                              or not isinstance(number, numbers.Real)):
                raise TypeError(f"an estimate's {name} must be a number; got {number!r}")
        try:
            value, low, high = float(value), float(low), float(high)
            confidence, elapsed = float(confidence), float(elapsed)
        except OverflowError:
            raise ValueError(f"an estimate's numbers must fit a float; got {value!r} in "
                             f"[{low!r}, {high!r}]") from None
        if math.isnan(value) or math.isnan(low) or math.isnan(high):
            raise ValueError(
                f"an estimate cannot be NaN; got {value!r} in [{low!r}, {high!r}]")
        if not low <= high:
            raise ValueError(f"an estimate's interval must have low <= high; got "
                             f"[{low!r}, {high!r}]")
        if isinstance(trace, str):
            raise TypeError("trace must be a sequence of lines, not one string")
        if not low <= value <= high:
            raise ValueError(
                f"an estimate must lie in its interval; got {value!r} outside "
                f"[{low!r}, {high!r}]")
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"an estimate's confidence must lie in (0, 1]; got "
                             f"{confidence!r}")
        if not elapsed >= 0.0:
            raise ValueError(f"an estimate's elapsed time must be at least 0; got "
                             f"{elapsed!r}")
        if isinstance(n_samples, bool) or not isinstance(n_samples, numbers.Integral) \
                or n_samples < 0:
            raise ValueError(f"an estimate's n_samples must be a non-negative integer; "
                             f"got {n_samples!r}")
        n_samples = int(n_samples)
        if engine not in ("exact", "smc"):
            raise ValueError(f"an estimate's engine must be 'exact' or 'smc'; got "
                             f"{engine!r}")
        if not isinstance(target_met, (bool, np.bool_)):
            raise TypeError(f"an estimate's target_met must be True or False; got "
                            f"{target_met!r}")
        target_met = bool(target_met)
        estimate = super().__new__(cls, value)
        for name, field in zip(_FIELDS, (low, high, confidence, engine, n_samples, elapsed,
                                         target_met, tuple(trace))):
            object.__setattr__(estimate, name, field)
        return estimate

    def __setattr__(self, name, value) -> None:
        raise AttributeError(f"an Estimate cannot be changed; use replace() to get one "
                             f"with another {name!r}")

    def __delattr__(self, name) -> None:
        raise AttributeError("an Estimate cannot be changed")

    def __reduce__(self):
        """What lets `pickle`, under every protocol, and `copy` rebuild an instance whose
        `__new__` takes keyword-only fields and whose attributes cannot be assigned."""
        return _rebuild, (type(self), float(self), self._fields())

    def _fields(self) -> dict:
        return {name: getattr(self, name) for name in _FIELDS}

    def replace(self, **changes) -> Estimate:
        """A copy of this estimate with the fields named in `changes` replaced; `value`
        names the point estimate.

            >>> e = Estimate.from_exact(0.5, elapsed=2.0)
            >>> e.replace(elapsed=0.0).elapsed, e.elapsed
            (0.0, 2.0)
        """
        fields = self._fields()
        value = changes.pop("value", float(self))
        unknown = set(changes) - set(_FIELDS)
        if unknown:
            raise TypeError(f"Estimate has no field {sorted(unknown)[0]!r}")
        fields.update(changes)
        return type(self)(value, **fields)

    @classmethod
    def from_exact(cls, value: float, trace: tuple[str, ...] = (),
                   elapsed: float = 0.0) -> Estimate:
        """The exact engine's answer `value`: an interval of zero width that met its
        target by definition.

            >>> Estimate.from_exact(0.3).low, Estimate.from_exact(0.3).engine
            (0.3, 'exact')

        Raises `ValueError` for NaN.
        """
        if math.isnan(value):
            raise ValueError("the exact engine's answer is NaN")
        return cls(value, low=value, high=value, confidence=1.0, engine="exact",
                   n_samples=0, elapsed=elapsed, target_met=True, trace=trace)
