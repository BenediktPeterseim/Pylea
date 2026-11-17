"""The missed-target warning and the table result of sampled queries.

`TargetNotMet` is the warning a query emits when it returns without reaching its
requested contract, and `TargetNotMetError` the exception it raises instead under
`strict=True`; `notify_missed_target`, which every public query function calls on the
result it returns, emits one or the other, so a missed target is never silent.  The
warning names the line of the caller's code that made the query, and Python's default
warning filter shows it every time, so a query in a loop warns on every miss; a filter
the caller sets, such as ``warnings.simplefilter("ignore", TargetNotMet)``, applies as
usual.

`EstimatedLikelihoodTable` is a `pylea.semantics.likelihood_table.LikelihoodTable` whose
matrix is the point estimate and which carries entrywise bound matrices.  All its
intervals hold jointly at `confidence`, the failure budget having been split over the
entries and the column totals; each single interval's marginal confidence is therefore
higher.  `normalised` propagates the bounds by interval quotients, which needs no
independence between numerator and total.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from pylea.semantics.likelihood_table import LikelihoodTable


class TargetNotMet(UserWarning):
    """Warned when a query returns without reaching its requested contract."""


class TargetNotMetError(ValueError):
    """Raised under `strict=True` when a query returns without reaching its requested
    contract; `result` is what the query would have returned."""

    def __init__(self, message: str, result):
        super().__init__(message)
        self.result = result


_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def notify_missed_target(result, *, target_met: bool, strict: bool, what: str,
                         contract, returned: str) -> None:
    """Nothing on a met target; otherwise a `TargetNotMet` warning at the caller's line,
    or `TargetNotMetError` carrying `result` under `strict`.

    `contract` is the query's `pylea.syntax.method.Contract`, and `returned` describes
    `result` in a few words, such as ``"0.5 in [0, 1]"``.
    """
    if target_met:
        return
    target = f"relative error {contract.rel_err:g}"
    if contract.floor is not None:
        target += f" or a bound below the floor {contract.floor:g}"
    reason = ("before the deadline" if contract.max_time is not None else
              "because no number of further samples could reach it")
    message = (f"{what} did not reach {target} at confidence {contract.confidence:g} "
               f"{reason}; it returned {returned}, and its interval still holds at that "
               "confidence.")
    if contract.max_time is not None:
        message += "  A larger max_time gives a narrower interval."
    if strict:
        raise TargetNotMetError(message, result)
    # The first frame outside the package is the caller's line.  `warn_explicit` with a
    # fresh registry shows the warning on every miss under the default filter, where
    # `warnings.warn` would show it once per line.
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back
    if frame is None:
        warnings.warn(message, TargetNotMet, stacklevel=2)
        return
    # The module's globals let the warning show its source line through the module's
    # loader; a script run as `__main__` has no loader, and its file is read directly.
    module_globals = frame.f_globals if frame.f_globals.get("__spec__") else None
    warnings.warn_explicit(message, TargetNotMet, frame.f_code.co_filename,
                           frame.f_lineno, module=frame.f_globals.get("__name__"),
                           registry=None, module_globals=module_globals)


class EstimatedLikelihoodTable(LikelihoodTable):
    """A likelihood table whose entries carry jointly valid bounds.

    `matrix` is the entrywise point estimate; `lower` and `upper` bound each entry;
    `total_lower` and `total_upper` bound each column's total mass, certified by their
    own stream rather than by summing entry bounds.  `floor` is the threshold below
    which an entry may be certified small instead of relatively accurate, on the
    normalised scale; `target_met` says whether every entry reached one of the two;
    `n_samples` is the number of samples drawn.
    """

    def __init__(self, input_types, output_types, matrix: np.ndarray, *,
                 lower: np.ndarray, upper: np.ndarray, total_lower: np.ndarray,
                 total_upper: np.ndarray, floor: float | None, confidence: float,
                 target_met: bool, trace: tuple[str, ...], n_samples: int):
        super().__init__(input_types, output_types, matrix)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.total_lower = np.asarray(total_lower, dtype=float)
        self.total_upper = np.asarray(total_upper, dtype=float)
        self.floor = floor
        self.confidence = confidence
        self.target_met = target_met
        self.trace = trace
        self.n_samples = n_samples

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
                   target_met=outcome.target_met, trace=outcome.trace,
                   n_samples=outcome.n_samples)

    def normalised(self) -> EstimatedLikelihoodTable:
        """The normalised table, bounds propagated by interval quotients per column.

        A column whose certified total can be zero gets the vacuous [0, 1] bounds.  Each
        point estimate is the entry's share of its column, moved into its bounds where it
        lies outside them, and the middle of its bounds where no sample was drawn or none
        was accepted.

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
        # A column without samples, or without accepted ones, carries no information on
        # how its mass is shared, so each entry's point is the middle of its interval.
        informed = (self.matrix.sum(axis=0) > 0.0) & (self.n_samples > 0)
        shares = np.where(informed, self.probabilities, (lower + upper) / 2)
        return EstimatedLikelihoodTable(
            self.input_types(), self.output_types(), np.clip(shares, lower, upper),
            lower=lower, upper=upper,
            total_lower=np.ones_like(self.total_lower),
            total_upper=np.ones_like(self.total_upper),
            floor=self.floor, confidence=self.confidence, target_met=self.target_met,
            trace=self.trace, n_samples=self.n_samples)
