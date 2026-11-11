"""A compiled DPP, and its answers.

`Compiled` holds `traced`, the DPP as it was written; `program`, what was actually lowered;
and `trace`, which of `pylea.compile.planner`'s stages ran and whether each was kept.
Supplying an `order_finder` leaves the choice of contraction tree to that finder, so the
planner lowers with it and simplifies without a budget.

`matrix` evaluates the plan and returns the array with the log of its magnitude taken out,
the two kept apart because a probability can lie far below the smallest float while the
ratio a caller wants is representable.  `evaluated` folds the magnitude back in and returns
a typed table, raising where float64 cannot hold the result.  `distribution` normalises, and
`evidence` gives the total mass, which for a DPP without inputs is its acceptance
probability.

The backend is chosen on the first evaluation rather than while compiling, since deciding
can cost a contraction and a `Compiled` that is never evaluated should not pay for it.
Passing `device` takes that decision instead.  A dtype too narrow for the result shows as an
exact zero, and the evaluation is then repeated in float64, with `underflowed` recording it.

`DEFAULT_KAPPA` bounds one intermediate to ``2**kappa`` entries; a contraction exceeding it
is sliced into pieces that do not.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from pylea.compile.cost import contraction_cost, step_count
from pylea.compile.lower import compile_program
from pylea.compile.planner import plan_compilation
from pylea.eval.backend import Backend, NumpyBackend, select
from pylea.eval.interp import evaluate, underflowed
from pylea.simplify import simplify as simplify_program

# Bound on a single intermediate, as a power of two entries.  ``2**28`` float64 entries is
# 2 GiB; a contraction exceeding it is sliced into pieces that do not.
DEFAULT_KAPPA = 28


def planned(program, order_finder, kappa: int | None,
            simplify: bool | Literal["always"]):
    """`program` lowered to plans: the lowered `Program`, its plans, the root, the trace.

    With no `order_finder` the planner's ladder decides everything; with one, the caller
    has taken the decision the ladder would take, so it steps aside rather than
    searching on top of the finder it was given.  `Compiled` and the exact-lane
    subprocess worker both compile through this, so the two cannot drift apart.
    """
    if order_finder is None:
        laddered = plan_compilation(program, kappa=kappa, simplify=simplify)
        return laddered.program, laddered.plans, laddered.root, laddered.trace
    lowered = simplify_program(program) if simplify else program
    plans, root = compile_program(lowered, order_finder, kappa=kappa)
    return lowered, plans, root, ("order finder supplied by the caller",)


def chosen_backend(plans, device: str | None) -> Backend:
    """The evaluation backend: the caller's `device`, else decided by the plans' shape."""
    if device is not None:
        return select(device)                       # the caller has taken the decision
    return select(steps=step_count(plans), ops=contraction_cost(plans))


def evaluated_matrix(plans, root, backend: Backend):
    """The plans evaluated on `backend`: array, log scale, and whether the evaluation
    was repeated in float64 because the backend's dtype underflowed.

    A zero from a narrow dtype is either an impossibility or an underflow, and only
    float64 tells the two apart; see `pylea.eval.interp.underflowed` for why the result
    is tested rather than its range predicted.
    """
    array, scale = evaluate(plans, root, backend=backend)
    if backend.exact or not underflowed(array):
        return array, scale, False
    array, scale = evaluate(plans, root, backend=NumpyBackend())
    return array, scale, True


class Compiled:
    """A `pylea.syntax.dpp.DPP` lowered to an executable plan.

    `trace` records what the planner did, which is how a slow or surprising compilation is
    read back rather than guessed at:

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.25)
        >>> coin.compiled().trace
        ('A greedy: cost 0.000e+00',)
    """

    def __init__(self, dpp, order_finder=None, kappa: int | None = DEFAULT_KAPPA,
                 simplify: bool | Literal["always"] = True, device: str | None = None):
        self._dpp = dpp
        self._device = device
        self._backend: Backend | None = None
        self._underflowed = False
        self.traced = dpp.program()
        self.program, self.plans, self.root, self.trace = planned(
            self.traced, order_finder, kappa, simplify)

    @property
    def underflowed(self) -> bool:
        """Whether the chosen backend's dtype proved too narrow, so float64 was used instead.

        False on the default backend, which is float64 already:

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.25)
            >>> compiled = coin.compiled()
            >>> _ = compiled.matrix()
            >>> compiled.underflowed
            False
        """
        return self._underflowed

    @property
    def backend(self) -> Backend:
        """The backend this evaluates on, decided once and then reused.

        The decision waits for the first evaluation rather than happening while compiling,
        since checking a narrow dtype can cost a contraction and a `Compiled` that is never
        evaluated should not pay for it.  A small DPP stays on the CPU:

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.25)
            >>> coin.compiled().backend.name
            'numpy'
        """
        if self._backend is None:
            self._backend = self._choose_backend()
        return self._backend

    def _choose_backend(self) -> Backend:
        return chosen_backend(self.plans, self._device)

    def matrix(self) -> tuple[np.ndarray, float]:
        """The ``(2**out_bits, 2**in_bits)`` matrix, and the log of the magnitude taken out.

        The true value is ``array * exp(log_scale)``.  The two are kept apart because a
        probability can be far below the smallest representable float while the ratio the
        caller wants is perfectly representable.

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.25)
            >>> array, log_scale = coin.compiled().matrix()
            >>> array.tolist(), log_scale
            ([[0.75], [0.25]], 0.0)
        """
        # The backend is left as it was on an underflow, so a caller can still see what
        # was tried.
        array, scale, underflowed_now = evaluated_matrix(self.plans, self.root,
                                                         self.backend)
        self._underflowed = self._underflowed or underflowed_now
        return array, scale

    def distribution(self) -> np.ndarray:
        """The normalised distribution over the outputs, flat, of ``2**out_bits`` entries.

        Index `i` is the outputs read as one bit pattern, the first output variable most
        significant, so two independent coin flips give the four joint outcomes in that
        order:

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def two_coins() -> tuple[Bool, Bool]:
            ...     return flip(0.5), flip(0.25)
            >>> two_coins.compiled().distribution().round(4).tolist()
            [0.375, 0.125, 0.375, 0.125]

        Raises `ValueError` where the DPP takes inputs, and `ZeroDivisionError` where its
        acceptance probability is zero.
        """
        if self.plans[self.root].in_bits:
            raise ValueError(
                "distribution is defined for a DPP with no inputs; this one takes "
                f"{self.plans[self.root].in_bits} input bits.  Use `matrix()` instead.")
        from pylea.semantics.likelihood_table import normalised_distribution

        array, _scale = self.matrix()
        return normalised_distribution(array)

    def evaluated(self):
        """The semantics of the DPP, as a
        `pylea.semantics.likelihood_table.LikelihoodTable` over its declared types.

        The evaluator's log scale is folded back into the matrix, so the table holds true
        magnitudes and carries the declared types with it:

            >>> from pylea import DPP, Enum
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> @DPP
            ... def draw() -> Colour:
            ...     return Colour.random({"R": 0.5, "G": 0.3, "B": 0.2})
            >>> print(draw.compiled().evaluated())
               |   R |   G |   B
            --------------------
            () | 0.5 | 0.3 | 0.2

        Raises `ValueError` where float64 cannot hold those magnitudes, in which case
        `matrix` returns the array and its scale apart.
        """
        from pylea.semantics.likelihood_table import scaled_table

        array, scale = self.matrix()
        return scaled_table(self._dpp.input_types(), self._dpp.output_types(), array,
                            scale)

    def evidence(self) -> float:
        """The total mass of the matrix, which for a DPP without inputs is its acceptance
        probability.

            >>> from pylea import DPP, Bool, flip, observe
            >>> @DPP
            ... def either() -> Bool:
            ...     a, b = flip(0.5), flip(0.5)
            ...     observe(a | b)
            ...     return a
            >>> either.compiled().evidence()
            0.75
        """
        array, scale = self.matrix()
        return float(np.asarray(array, dtype=float).sum() * np.exp(scale))

    def peak_entries(self) -> int:
        """The largest number of array entries that evaluating this holds at once.

        This is what `kappa` bounds, and reading it is how a DPP too wide to evaluate is
        recognised before it is run:

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def chain() -> Bool:
            ...     acc = flip(0.5)
            ...     for _ in range(5):
            ...         acc = acc & flip(0.5)
            ...     return acc
            >>> chain.compiled().peak_entries()
            16
        """
        return max(plan.peak_entries() for plan in self.plans.values())

    def __str__(self) -> str:
        return (f"<Compiled '{self._dpp.name()}': {len(self.plans)} plan(s), "
                f"peak 2**{np.log2(self.peak_entries()):.1f} entries>")
