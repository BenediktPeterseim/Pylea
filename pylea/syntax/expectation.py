"""The expected value of a DPP's output.

`expectation` takes a DPP with no input and one output whose values are numbers, and
returns the mean of that output under the distribution the DPP defines.

A function to apply to the output before averaging is not an argument here, because such a
function is a DPP of its own: the expected value of ``g(X)`` is the expectation of the
DPP that returns ``g(X)``.

    >>> from pylea import DPP, Enum, expectation
    >>> class Weather(Enum):
    ...     elements = ["dry", "rain", "snow"]
    >>> class Cost(Enum):
    ...     elements = [0.0, 200.0, 9000.0]
    >>> @DPP
    ... def tomorrow() -> Weather:
    ...     return Weather.random({"dry": 0.7, "rain": 0.25, "snow": 0.05})
    >>> @DPP
    ... def repair_cost() -> Cost:
    ...     return tomorrow().match({"dry": 0.0, "rain": 200.0, "snow": 9000.0},
    ...                             data_type=Cost)
    >>> expectation(repair_cost)
    500.0

The values are applied to the normalised distribution rather than to the table
`pylea.syntax.dpp.DPP.evaluated` returns.  Normalising cancels the magnitude the evaluator
carries in log scale, so a DPP whose acceptance probability lies far below the smallest
float still has an expected value, where that table has no float64 form.

The sampling engine estimates the mean of the output rescaled to [0, 1] by the declared
type's smallest and largest value, maps the interval back, and meets `rel_err` on the
returned value's scale: the interval's width is at most `rel_err` times the distance of
its nearer end from 0.  A mean near 0 therefore needs many samples, as a probability
near 0 does, and a `floor` or `max_time` bounds the cost.  The cost also grows with the
declared range divided by the mean, since the rescaled values are certified on [0, 1]:
a `pylea.syntax.integer.UInt` declared much wider than the values it holds makes the
query slower, and a type no wider than its values keeps it fast.  The number of samples
is proportional to the second moment of the rescaled value over its squared mean, which
is large where a rare large value dominates the mean.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import replace

import numpy as np

from pylea.semantics.likelihood_table import normalised_distribution
from pylea.syntax.dpp import DPP
from pylea.syntax.estimate import Estimate
from pylea.syntax.method import (query_target, refuse_stray_mass, refuse_wide_output,
                                 resolve, scalar_query)


def expectation(f: DPP, *, rel_err: float | None = None, floor: float | None = None,
                confidence: float | None = None, max_time: float | None = None,
                method: str | None = None, seed: int | None = None,
                strict: bool = False, **kwargs) -> Estimate:
    """The mean of `f`'s single output.

    `f` must be a DPP, or `TypeError` is raised, with no input and one output whose
    values are finite numbers, of at most `pylea.syntax.method.MAX_OUTPUT_BITS` bits,
    since the mean is taken over one value per bit pattern; another DPP raises
    `ValueError`.  Keyword arguments go to `pylea.syntax.dpp.DPP.compiled`, so the
    contraction tree finder and the size limit can be chosen.

        >>> from pylea import DPP, Enum, observe
        >>> class Die(Enum):
        ...     elements = [1, 2, 3, 4, 5, 6]
        >>> @DPP
        ... def roll() -> Die:
        ...     return Die.uniform()
        >>> round(expectation(roll), 12)
        3.5

    Where the DPP conditions, the mean is that of the posterior:

        >>> @DPP
        ... def high_roll() -> Die:
        ...     d = Die.uniform()
        ...     observe(d >= 4)
        ...     return d
        >>> round(expectation(high_roll), 12)
        5.0

    `method`, `rel_err`, `floor`, `confidence`, `max_time`, `seed` and `strict` behave as
    on `pylea.syntax.prob.prob`, relative error being met on the returned value's scale
    as the module docstring states.  `floor` is read on the value scale and must lie
    between the smallest and the largest declared value: ``floor=c`` asks whether the mean
    lies below `c`.  The result is an `pylea.syntax.estimate.Estimate`:

        >>> m = expectation(roll, method="smc", rel_err=0.05, seed=0)
        >>> bool(m.low <= 3.5 <= m.high)
        True

    Raises `ValueError` where positive probability lies on a bit pattern that encodes no
    value, which `pylea.syntax.variable.Variable.uniform_bits` puts there for a type whose
    element count is not a power of two.  The exact engine checks this exhaustively; the
    sampling engine raises when a drawn sample lands on such a pattern, so stray mass small
    enough never to be sampled can go undetected there.  Raises `ZeroDivisionError`
    where the acceptance probability is zero.
    """
    query_target(f, "expectation")
    if len(f.output_types()) != 1:
        raise ValueError("expectation needs a DPP with no inputs and a single output; "
                         f"{f.name()!r} returns {len(f.output_types())} outputs.")
    refuse_wide_output(f, "expectation")
    output = f.output_types()[0]

    decoded = output.encoding().decoded()
    patterns, declared = _declared_values(decoded, output.__name__)
    values = np.zeros(1 << output.nbits())
    values[patterns] = declared

    lowest = float(declared.min())
    highest = float(declared.max())
    contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                       max_time=max_time, seed=seed, strict=strict, what="expectation",
                       floor_range=(lowest, highest))
    floor = contract.floor

    span = highest - lowest
    rescaled = np.zeros_like(values)
    if span:
        rescaled[patterns] = (declared - lowest) / span
    forbidden = np.ones_like(values)
    forbidden[patterns] = 0.0
    if not forbidden.any():
        forbidden = None
    if floor is not None and span > 0.0:
        # The caller's floor is on the value scale; the run certifies on the rescaled
        # one, so the threshold maps through the same affine change.
        floor = (floor - lowest) / span
    if span == 0.0:
        # Every declared value is the same, so the rescaled mean is exactly zero and the
        # affine map collapses the interval onto that value; a floor of 1 lets the run
        # stop at its first checkpoint instead of chasing an unreachable relative target.
        floor = 1.0
    if floor != contract.floor:
        contract = replace(contract, floor=floor)

    return scalar_query(
        f, contract, kwargs, what="expectation", values=rescaled,
        exact_value=lambda: _exact_mean(values, decoded, output,
                                        f.compiled(**kwargs).distribution()),
        exact_from_matrix=lambda array, _scale: _exact_mean(
            values, decoded, output, normalised_distribution(array)),
        forbidden=forbidden, affine=(span, lowest))


def _declared_values(decoded: dict, type_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The bit patterns of `decoded` and the value of each as a float; `ValueError` where
    a value is not a real number or not a finite float64."""
    for kind in {type(value) for value in decoded.values()}:
        if not issubclass(kind, numbers.Real):
            value = next(v for v in decoded.values() if type(v) is kind)
            raise ValueError("expectation requires an output whose values are real "
                             f"numbers, such as int, float or Fraction; "
                             f"{type_name} has the value {value!r}.")
    patterns = np.fromiter(decoded.keys(), dtype=np.int64, count=len(decoded))
    try:
        declared = np.fromiter(decoded.values(), dtype=float, count=len(decoded))
    except OverflowError:                       # an integer beyond the float64 range
        declared = None
    if declared is None or not np.isfinite(declared).all():
        value = next(v for v in decoded.values() if not _finite(v))
        raise ValueError("expectation requires an output whose values are finite; "
                         f"{type_name} has the value {value!r}.")
    return patterns, declared


def _finite(value: numbers.Real) -> bool:
    """Whether `value` is a finite float64 number; an integer beyond its range is not."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _exact_mean(values: np.ndarray, decoded, output, distribution: np.ndarray) -> float:
    """The exact posterior mean, refusing mass on patterns that encode no value."""
    refuse_stray_mass(distribution, decoded, output.__name__, "expectation")
    return float(values @ distribution)
