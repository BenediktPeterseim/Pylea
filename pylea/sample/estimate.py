"""The missed-target warning and the table result of sampled queries.

`TargetNotMet` is the warning a sampled query emits when it returns before reaching its
requested contract; `notify_missed_target`, which every sampled entry point calls, emits it,
or raises under `strict=True`, so a failed contract is never silent.

`EstimatedLikelihoodTable` is a `pylea.semantics.likelihood_table.LikelihoodTable` whose
matrix is the point estimate and which carries entrywise bound matrices.  All its
intervals hold jointly at `confidence`, the failure budget having been split over the
entries and the column totals; each single interval's marginal confidence is therefore
higher.  `normalised` propagates the bounds by interval quotients, which needs no
independence between numerator and total.
"""
from __future__ import annotations

import warnings

import numpy as np

from pylea.semantics.likelihood_table import LikelihoodTable


class TargetNotMet(UserWarning):
    """Warned when a sampled query returns before reaching its requested contract."""


def notify_missed_target(target_met: bool, strict: bool, what: str) -> None:
    """Nothing on a met target; otherwise a `TargetNotMet` warning, or `ValueError` under
    `strict`."""
    if target_met:
        return
    message = (f"{what} did not reach the requested accuracy before the deadline; "
               "the returned interval is still valid.  Read `low` and `high`, or pass "
               "a larger `max_time`.")
    if strict:
        raise ValueError(message)
    warnings.warn(message, TargetNotMet, stacklevel=3)


class EstimatedLikelihoodTable(LikelihoodTable):
    """A likelihood table whose entries carry jointly valid bounds.

    `matrix` is the entrywise point estimate; `lower` and `upper` bound each entry;
    `total_lower` and `total_upper` bound each column's total mass, certified by their
    own stream rather than by summing entry bounds.  `floor` is the threshold below
    which an entry may be certified small instead of relatively accurate, on the
    normalised scale; `target_met` says whether every entry reached one of the two.
    """

    def __init__(self, input_types, output_types, matrix: np.ndarray, *,
                 lower: np.ndarray, upper: np.ndarray, total_lower: np.ndarray,
                 total_upper: np.ndarray, floor: float | None, confidence: float,
                 target_met: bool, trace: tuple[str, ...]):
        super().__init__(input_types, output_types, matrix)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.total_lower = np.asarray(total_lower, dtype=float)
        self.total_upper = np.asarray(total_upper, dtype=float)
        self.floor = floor
        self.confidence = confidence
        self.target_met = target_met
        self.trace = trace

    @classmethod
    def from_outcome(cls, input_types, output_types, outcome, *,
                     confidence: float) -> EstimatedLikelihoodTable:
        """The table a `pylea.sample.run.TableOutcome` certifies, entries as the single
        column of a zero-input DPP, the floor read from the outcome."""
        shape = (outcome.point.size, 1)
        return cls(input_types, output_types, outcome.point.reshape(shape),
                   lower=outcome.lower.reshape(shape),
                   upper=outcome.upper.reshape(shape),
                   total_lower=np.array([outcome.total_low]),
                   total_upper=np.array([outcome.total_high]),
                   floor=outcome.floor, confidence=confidence,
                   target_met=outcome.target_met, trace=outcome.trace)

    def normalised(self) -> EstimatedLikelihoodTable:
        """The normalised table, bounds propagated by interval quotients per column.

        A column whose certified total can be zero gets the vacuous [0, 1] bounds.

            >>> from pylea import DPP, Bool, flip, observe
            >>> @DPP
            ... def either() -> Bool:
            ...     a, b = flip(0.5), flip(0.5)
            ...     observe(a | b)
            ...     return a
            >>> posterior = either.evaluated(method="smc", rel_err=0.1, seed=0).normalised()
            >>> bool(posterior.lower[1, 0] <= 2 / 3 <= posterior.upper[1, 0])
            True
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            lower = np.where(self.total_upper > 0.0, self.lower / self.total_upper, 0.0)
            upper = np.where(self.total_lower > 0.0,
                             np.minimum(self.upper / self.total_lower, 1.0), 1.0)
        return EstimatedLikelihoodTable(
            self.input_types(), self.output_types(), self.probabilities,
            lower=lower, upper=upper,
            total_lower=np.ones_like(self.total_lower),
            total_upper=np.ones_like(self.total_upper),
            floor=self.floor, confidence=self.confidence, target_met=self.target_met,
            trace=self.trace)
