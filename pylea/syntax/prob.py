"""The probability of an event."""

from __future__ import annotations

import numpy as np

from pylea.semantics.likelihood_table import normalised_distribution
from pylea.syntax.dpp import Bool, DPP
from pylea.syntax.estimate import Estimate
from pylea.syntax.method import query_target, resolve, scalar_query


def prob(e: DPP, *, rel_err: float | None = None, floor: float | None = None,
         confidence: float | None = None, max_time: float | None = None,
         method: str | None = None, seed: int | None = None, strict: bool = False,
         **kwargs) -> Estimate:
    """The probability that the event `e` is true.

    An event is a DPP with no inputs and a single `Bool` output; another DPP raises
    `ValueError`, and anything that is not a DPP `TypeError`.  Keyword arguments go to
    `pylea.syntax.dpp.DPP.compiled`, so the contraction tree finder and the size limit can
    be chosen.

        >>> from pylea import DPP, Bool, flip, observe
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.5)
        >>> prob(both)
        0.25

    Exact answers keep their relative precision however small the probabilities involved:
    numbers too small for float64 add a relative error of at most ``2**-40``, about 1e-12,
    to the rounding of floating-point arithmetic.  Where an evaluation cannot vouch for
    that, it is repeated with an exponent per entry, which is slower.  A probability below
    the smallest positive float64, about 5e-324, is returned as 0.0:

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
    sampling engine cannot honour it; a sampling option (`rel_err`, `floor`, `confidence`,
    `seed`, `strict`) combined with the exact engine raises rather than being ignored.
    `rel_err` must lie in ``[0, 1)``, `floor` in ``[0, 1]`` and `confidence` in
    ``(0, 1)``; a misspelt keyword argument raises `TypeError` under every method.
    The result is an `pylea.syntax.estimate.Estimate`, a float whose interval
    ``[low, high]`` contains the true probability with probability at least
    `confidence`, 0.95 by default; from the exact engine the interval has zero width:

        >>> p = prob(both, method="smc", rel_err=0.05, seed=0)
        >>> bool(p.low <= 0.25 <= p.high)
        True

    A sampled certificate's cost grows like ``1 / (rel_err**2 * p)``, so a probability
    far below `rel_err` is out of the sampling engine's reach; passing `floor` changes the
    contract to "relative error `rel_err`, or certified below `floor`", which answers a
    threshold question at bounded cost.  Without a floor and without `max_time`, a
    ``method="smc"`` query whose answer is out of reach runs indefinitely by contract.

    The exact engine may refuse a DPP: `pylea.compile.lower.ExactInferenceTooLarge`
    where it needs a tensor with more than ``max(2**29, 2**kappa)`` entries that no
    slicing avoids (never with ``kappa=None``), and
    `pylea.compile.lower.ExactInferenceTooSlow` where evaluation is estimated to take
    longer than about 11.6 days.  Under ``"exact"`` the refusal is raised; under
    ``"auto"`` the sampling engine answers instead where it can.

    `max_time` (seconds, wall clock, at least 0) bounds sampling, the exact engine and the
    choice between them; infinity means no deadline.  The first sampled query of a DPP
    also prepares it for sampling, in time linear in its size once calls are expanded,
    and that preparation is not interrupted, so on a large DPP the first query can
    exceed `max_time`.  A sampled query then returns the interval standing at the deadline,
    with `target_met` False and a
    `pylea.sample.estimate.TargetNotMet` warning naming the line that made the query, or,
    under `strict`, a `pylea.sample.estimate.TargetNotMetError` (a `ValueError`) whose
    `result` is that interval; an
    exact query that misses the deadline raises `TimeoutError` rather than falling back
    to sampling, since that would silently change the requested contract.  `seed` makes
    a sampled answer reproducible: without `max_time` a ``method="smc"`` answer is
    bit-reproducible, and under either sampling method the seeded answer is cached, so
    a repeated query returns the first call's certificate; under ``"auto"`` that
    certificate may come from either engine, whichever answered first, which depends on
    timing.
    """
    query_target(e, "prob", outputs=(Bool,), output_kind=" and one Bool output")
    contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                       max_time=max_time, seed=seed, strict=strict, what="prob")
    return scalar_query(
        e, contract, kwargs, what="prob", values=np.array([0.0, 1.0]),
        exact_value=lambda: float(e.distribution(**kwargs)[1]),
        exact_from_matrix=lambda array, _scale: float(normalised_distribution(array)[1]))
