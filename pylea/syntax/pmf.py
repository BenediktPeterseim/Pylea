"""The posterior over a single finite-typed output."""

from __future__ import annotations

from typing import Any

from pylea.semantics.likelihood_table import normalised_distribution
from pylea.syntax.dpp import DPP
from pylea.syntax.enum import Enum
from pylea.syntax.estimate import Estimate
from pylea.syntax.method import (ExactTable, MAX_OUTPUT_BITS, query_target,
                                 refuse_stray_mass, refuse_wide_output,
                                 report_missed_target, resolve,
                                 table_query)


def probability_mass_function(f: DPP, *, rel_err: float | None = None,
                              floor: float | None = None, confidence: float | None = None,
                              max_time: float | None = None, method: str | None = None,
                              seed: int | None = None, strict: bool = False,
                              **kwargs) -> dict[Any, Estimate]:
    """Each element of `f`'s output type mapped to its posterior probability.

    Raises `TypeError` if `f` is not a DPP.  Raises `ValueError` if `f` has inputs, if
    its output is not a single `Enum` type or is wider than
    `pylea.syntax.method.MAX_OUTPUT_BITS` bits, or where positive probability lies on a
    bit pattern that encodes no element, which
    `pylea.syntax.variable.Variable.uniform_bits` puts there for a type whose element
    count is not a power of two.  Raises `ZeroDivisionError` where the acceptance
    probability is zero, as `pylea.syntax.prob.prob` does.

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
    `pylea.syntax.prob.prob`, and `kwargs` go to `pylea.syntax.dpp.DPP.compiled`.  The
    exact answer is accurate to floating-point rounding however small the acceptance
    probability, the distribution being normalised before any magnitude is folded in.
    Each probability is an `pylea.syntax.estimate.Estimate`.  Under sampling the
    intervals hold *jointly* at `confidence`, the failure budget having been split over
    the entries; each single interval's marginal confidence is therefore higher.  `floor`
    is the threshold below which an entry may be certified small instead of relatively
    accurate, the default being ``min(rel_err, 1 / (4 n))``
    for the ``n = 2**bits`` bit patterns of the output, since a uniform relative guarantee
    would be priced by the rarest entry.  Certification cost grows with that number, every
    entry being certified separately, and under ``"auto"`` the cost comparison between
    the engines prices it so.
    """
    query_target(f, "probability_mass_function")
    out = f.output_types()
    if len(out) != 1 or not (isinstance(out[0], type) and issubclass(out[0], Enum)):
        raise ValueError(
            "probability_mass_function needs a DPP with no inputs and a single Enum "
            f"output; {f.name()!r} returns "
            f"{', '.join(t.__name__ for t in out) or 'nothing'}.")
    refuse_wide_output(f, "probability_mass_function")
    element_type = out[0]
    patterns = range(len(element_type.elements))

    contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                       max_time=max_time, seed=seed, strict=strict,
                       what="probability_mass_function")
    answer = table_query(f, contract, kwargs, normalised_contract=True,
                         what="probability_mass_function")
    if isinstance(answer, ExactTable):
        distribution = normalised_distribution(answer.array)
        refuse_stray_mass(distribution, patterns, element_type.__name__,
                          "probability_mass_function")
        # The element at list index `i` is encoded as bit pattern `i`.
        return {element: Estimate.from_exact(float(distribution[i]), answer.trace,
                                             answer.elapsed)
                for i, element in enumerate(element_type)}

    from pylea.sample.estimate import EstimatedLikelihoodTable

    if answer.n_samples:
        # With no samples every entry's point is the middle of its interval, which says
        # nothing about where the mass lies.
        refuse_stray_mass(answer.point, patterns, element_type.__name__,
                          "probability_mass_function")
    # The posterior quotients come from the one place that propagates table bounds.
    posterior = EstimatedLikelihoodTable.from_outcome(
        f.input_types(), out, answer, confidence=contract.confidence).normalised()
    result = {element: Estimate(float(posterior.matrix[i, 0]),
                                low=float(posterior.lower[i, 0]),
                                high=float(posterior.upper[i, 0]),
                                confidence=contract.confidence, engine="smc",
                                n_samples=answer.n_samples, elapsed=answer.elapsed,
                                target_met=answer.target_met, trace=answer.trace)
              for i, element in enumerate(element_type)}
    report_missed_target(result, contract, "probability_mass_function",
                         answer.target_met, "an interval for each element")
    return result
