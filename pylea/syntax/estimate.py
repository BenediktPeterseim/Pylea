"""The result type of `pylea.syntax.prob.prob`, `pylea.syntax.expectation.expectation` and
`pylea.syntax.pmf.probability_mass_function`.

`Estimate` is a float subclass, so every call site expecting a plain probability or mean
keeps working and printing unchanged; the interval and the record of how the answer was
computed ride along as attributes.  The exact engine's answer is an `Estimate` too, with an
interval of zero width, so `low` and `high` can be read whichever engine answered:

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
never reads the attribute.  For the second, every sampled query emits a
`pylea.sample.estimate.TargetNotMet` warning on a missed target, and raises instead under
`strict=True`.
"""
from __future__ import annotations


class Estimate(float):
    """A point estimate with an interval.

    The float value is the point estimate; `low` and `high` bound the true value with
    probability at least `confidence`; `engine` names the engine that answered ("exact" or
    "smc"); `n_samples` is the number of samples drawn, 0 for the exact engine; `elapsed`
    is the query's wall-clock time in seconds; `target_met` says whether the requested
    contract was reached before any deadline; `trace` records the portfolio algorithm's
    decisions, predictions and probe costs, so a surprising runtime is read back rather
    than guessed at.

        >>> e = Estimate(0.25, low=0.24, high=0.26, confidence=0.95, engine="smc",
        ...              n_samples=1000, elapsed=0.1, target_met=True, trace=())
        >>> e
        0.25
        >>> e.low, e.high, e + 0.75
        (0.24, 0.26, 1.0)
    """

    __slots__ = ("low", "high", "confidence", "engine", "n_samples", "elapsed",
                 "target_met", "trace")

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
        estimate = super().__new__(cls, value)
        estimate.low = low
        estimate.high = high
        estimate.confidence = confidence
        estimate.engine = engine
        estimate.n_samples = n_samples
        estimate.elapsed = elapsed
        estimate.target_met = target_met
        estimate.trace = trace
        return estimate

    def __getnewargs_ex__(self):
        """The constructor arguments, which is what lets pickle and deepcopy rebuild an
        instance whose `__new__` takes keyword-only certificate fields."""
        return (float(self),), {"low": self.low, "high": self.high,
                                "confidence": self.confidence, "engine": self.engine,
                                "n_samples": self.n_samples, "elapsed": self.elapsed,
                                "target_met": self.target_met, "trace": self.trace}

    @classmethod
    def exact(cls, value: float, trace: tuple[str, ...] = (),
              elapsed: float = 0.0) -> Estimate:
        """An exact engine's answer: a zero-width interval that met its target by
        definition."""
        return cls(value, low=value, high=value, confidence=1.0, engine="exact",
                   n_samples=0, elapsed=elapsed, target_met=True, trace=trace)
