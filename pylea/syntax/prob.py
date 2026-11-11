"""The probability of an event."""

from __future__ import annotations

import numpy as np

from pylea.semantics.likelihood_table import normalised_distribution
from pylea.syntax.dpp import Bool, DPP
from pylea.syntax.estimate import Estimate
from pylea.syntax.method import resolve, scalar_query


def prob(e: DPP, *, rel_err: float | None = None, floor: float | None = None,
         confidence: float | None = None, max_time: float | None = None,
         method: str | None = None, seed: int | None = None, strict: bool = False,
         **kwargs) -> Estimate:
    """The probability that the event `e` is true.

    An event is a DPP with no inputs and a single `Bool` output; anything else raises
    `ValueError`.  Keyword arguments go to `pylea.syntax.dpp.DPP.compiled`, so the
    contraction tree and the width bound can be chosen.

        >>> from pylea import DPP, Bool, flip, observe
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.5)
        >>> prob(both)
        0.25

    Conditioning is exact however small the acceptance probability:

        >>> @DPP
        ... def rare() -> Bool:
        ...     x = flip(1e-300)
        ...     observe(x | flip(1e-300))
        ...     return x
        >>> round(prob(rare), 6)
        0.5

    `method` selects the engine: ``"exact"`` computes the value to floating-point
    precision, ``"smc"`` estimates it by certified Monte Carlo, and ``"auto"``, the
    default, runs whichever a cost comparison predicts to be faster, which on a DPP
    the exact engine finds trivial is the exact value, as above.  ``rel_err = 0.0`` also
    forces exactness, and so does an exact-engine option (`kappa`, `order_finder`,
    `simplify`, `recompile`, or an accelerator `device`) with no method named, since the
    sampling lane cannot honour it; a sampling option (`rel_err`, `floor`, `confidence`,
    `seed`, `strict`) combined with the exact engine raises rather than being ignored.
    The result is an `pylea.syntax.estimate.Estimate`, a float whose interval
    ``[low, high]`` contains the true probability with probability at least
    `confidence`, 0.95 by default; from the exact engine the interval has zero width:

        >>> p = prob(both, method="smc", rel_err=0.05, seed=0)
        >>> bool(p.low <= 0.25 <= p.high)
        True

    A sampled certificate's cost grows like ``1 / (rel_err**2 * p)``, so a probability
    far below `rel_err` is out of the sampling lane's reach; passing `floor` changes the
    contract to "relative error `rel_err`, or certified below `floor`", which answers a
    threshold question at bounded cost.  Without a floor and without `max_time`, a
    ``method="smc"`` query whose answer is out of reach runs indefinitely by contract.

    `max_time` (seconds, wall clock) bounds the whole query.  A sampled query then
    returns the interval standing at the deadline, with `target_met` False and a
    `pylea.sample.estimate.TargetNotMet` warning, or an exception under `strict`; an
    exact query that misses the deadline raises `TimeoutError` rather than falling back
    to sampling, since that would silently change the requested contract.  `seed` makes
    a sampled answer reproducible: without `max_time` a ``method="smc"`` answer is
    bit-reproducible, and under either sampling method the seeded answer is cached, so
    a repeated query returns the first call's certificate; under ``"auto"`` that
    certificate may come from whichever lane won a timing-dependent race.
    """
    if e.input_types() != tuple() or e.output_types() != (Bool,):
        raise ValueError(
            "Can only compute the probability of events, "
            "i.e. DPPs with no input and a single Boolean output."
        )
    contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                       max_time=max_time, seed=seed, strict=strict)
    return scalar_query(
        e, contract, kwargs, what="prob", values=np.array([0.0, 1.0]),
        exact_value=lambda: float(e.distribution(**kwargs)[1]),
        exact_from_matrix=lambda array, _scale: float(normalised_distribution(array)[1]))
