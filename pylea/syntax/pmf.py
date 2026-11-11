"""The posterior over a single finite-typed output."""

from __future__ import annotations

import time
from typing import Any

from pylea.syntax.dpp import DPP
from pylea.syntax.enum import Enum
from pylea.syntax.estimate import Estimate
from pylea.syntax.method import resolve


def probability_mass_function(f: DPP, *, rel_err: float | None = None,
                              floor: float | None = None, confidence: float | None = None,
                              max_time: float | None = None, method: str | None = None,
                              seed: int | None = None, strict: bool = False,
                              **kwargs) -> dict[Any, Estimate]:
    """Each element of `f`'s output type mapped to its posterior probability.

    A thin wrapper over `pylea.syntax.dpp.DPP.evaluated`, to which `kwargs` are passed on.
    Raises `ValueError` if `f` has inputs, or if its output is not a single `Enum` type.

        >>> from pylea import DPP, Enum, observe
        >>> class Digit(Enum):
        ...     elements = [0, 1, 2, 3]
        >>> @DPP
        ... def nonzero() -> Digit:
        ...     d = Digit.uniform()
        ...     observe(d != 0)
        ...     return d
        >>> {d: round(p, 4) for d, p in probability_mass_function(nonzero).items()}
        {0: 0.0, 1: 0.3333, 2: 0.3333, 3: 0.3333}

    `method`, `rel_err`, `confidence`, `max_time`, `seed` and `strict` behave as on
    `pylea.syntax.prob.prob`.  Each probability is an `pylea.syntax.estimate.Estimate`.
    Under sampling the intervals hold *jointly* at `confidence`, the failure budget having
    been split over the entries; each single interval's marginal confidence is therefore
    higher.  `floor` is the two-parameter table contract's
    threshold: an entry below it may be certified small instead of relatively accurate,
    the default being ``min(rel_err, 1 / (4 n))`` for `n` elements, since a uniform
    relative guarantee would be priced by the rarest entry.  Certification cost grows
    with the element count, every entry being certified separately.
    """
    if f.input_types() != tuple():
        raise ValueError(
            "probability_mass_function requires a zero-input DPP; "
            f"got input types {f.input_types()}.")
    out = f.output_types()
    if len(out) != 1 or not (isinstance(out[0], type) and issubclass(out[0], Enum)):
        raise ValueError(
            "probability_mass_function requires a DPP with a single Enum output; "
            f"got output types {out}.")

    contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                       max_time=max_time, seed=seed, strict=strict)
    if contract.method != "smc":
        # The scalar cost model does not price a table, so under "auto" `evaluated`
        # runs the exact lane; sampling remains reachable by explicit method="smc".
        started = time.monotonic()
        table = f.evaluated(method=contract.method, max_time=max_time, **kwargs)
        elapsed = time.monotonic() - started
        # The element at list index `i` is encoded as bit pattern `i`, so its row is
        # also `i`.
        column = table.probabilities[:, 0]
        return {element: Estimate.exact(float(column[i]), elapsed=elapsed)
                for i, element in enumerate(out[0])}

    from pylea.sample.estimate import EstimatedLikelihoodTable
    from pylea.sample.query import certified_table

    outcome = certified_table(f, rel_err=contract.rel_err, confidence=contract.confidence,
                              floor=contract.floor, max_time=max_time,
                              seed=contract.seed, strict=contract.strict,
                              normalised_contract=True, what="probability_mass_function")
    # The posterior quotients come from the one place that propagates table bounds.
    posterior = EstimatedLikelihoodTable.from_outcome(
        f.input_types(), out, outcome, confidence=contract.confidence).normalised()
    return {element: Estimate(float(posterior.matrix[i, 0]),
                              low=float(posterior.lower[i, 0]),
                              high=float(posterior.upper[i, 0]),
                              confidence=contract.confidence, engine="smc",
                              n_samples=outcome.n_samples, elapsed=outcome.elapsed,
                              target_met=outcome.target_met, trace=outcome.trace)
            for i, element in enumerate(out[0])}
