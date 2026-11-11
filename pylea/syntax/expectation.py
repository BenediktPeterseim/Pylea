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

The sampled engines certify the mean on the output affinely rescaled to [0, 1] by the
declared type's smallest and largest value, and map the interval back.  Relative error is
invariant under scaling but not under shifting: for a value set with a negative minimum
and a mean near zero, `rel_err` on the rescaled mean can correspond to a large relative
error of the returned value, and a caller who needs the original-scale relative error
reads the returned interval.  A mean dominated by a rare large value is expensive to
certify; that is a property of the estimation problem, not of the implementation.
"""

from __future__ import annotations

import numbers
from dataclasses import replace

import numpy as np

from pylea.semantics.likelihood_table import normalised_distribution
from pylea.syntax.dpp import DPP
from pylea.syntax.estimate import Estimate
from pylea.syntax.method import resolve, scalar_query


def expectation(f: DPP, *, rel_err: float | None = None, floor: float | None = None,
                confidence: float | None = None, max_time: float | None = None,
                method: str | None = None, seed: int | None = None,
                strict: bool = False, **kwargs) -> Estimate:
    """The mean of `f`'s single output.

    `f` must have no input and one output whose values are numbers; anything else raises
    `ValueError`.  Keyword arguments go to `pylea.syntax.dpp.DPP.compiled`, so the
    contraction tree and the width bound can be chosen.

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
    on `pylea.syntax.prob.prob`; the module docstring states the sampled contract's
    scale, and `floor` is read on the value scale, so ``floor=0.5`` asks whether the
    mean lies below one half.  The result is an `pylea.syntax.estimate.Estimate`:

        >>> m = expectation(roll, method="smc", rel_err=0.05, seed=0)
        >>> bool(m.low <= 3.5 <= m.high)
        True

    Raises `ValueError` where positive probability lies on a bit pattern that encodes no
    value, which `pylea.syntax.variable.Variable.uniform_bits` puts there for a type whose
    element count is not a power of two.  The exact engine checks this exhaustively; a
    sampling lane raises when a drawn sample lands on such a pattern, so stray mass small
    enough never to be sampled can go undetected there.  Raises `ZeroDivisionError`
    where the acceptance probability is zero.
    """
    if f.input_types() != tuple():
        raise ValueError(
            f"expectation requires a zero-input DPP; got input types {f.input_types()}.")
    if len(f.output_types()) != 1:
        raise ValueError("expectation requires a DPP with a single output; got output "
                         f"types {f.output_types()}.")
    output = f.output_types()[0]

    decoded = output.encoding().decoded()
    values = np.zeros(1 << output.nbits())
    for pattern, value in decoded.items():
        if not isinstance(value, numbers.Real):
            raise ValueError("expectation requires an output whose values are numbers; "
                             f"{output.__name__} has the value {value!r}.")
        values[pattern] = float(value)

    contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                       max_time=max_time, seed=seed, strict=strict)
    floor = contract.floor

    lowest = min(values[pattern] for pattern in decoded)
    highest = max(values[pattern] for pattern in decoded)
    span = highest - lowest
    rescaled = np.zeros_like(values)
    forbidden = np.ones_like(values)
    for pattern in decoded:
        rescaled[pattern] = (values[pattern] - lowest) / span if span else 0.0
        forbidden[pattern] = 0.0
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


def _exact_mean(values: np.ndarray, decoded, output, distribution: np.ndarray) -> float:
    """The exact posterior mean, refusing mass on patterns that encode no value."""
    has_value = np.zeros(distribution.shape, dtype=bool)
    has_value[list(decoded)] = True
    # Summed over the patterns that encode nothing rather than taken as one minus the
    # rest, so that the exact zeros the DPP puts there stay exactly zero.
    stray = float(distribution[~has_value].sum())
    if stray > 0.0:
        raise ValueError(
            f"expectation is undefined: probability {stray:.4g} lies on bit patterns "
            f"that encode no value of {output.__name__}.")
    return float(values @ distribution)
