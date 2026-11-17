"""A compiled DPP, and its answers.

`Compiled` holds `traced`, the DPP as it was written; `program`, the DPP that was
compiled, simplified or not; and `trace`, the steps compilation took: which contraction
tree finders ran, whether simplification was tried, and which result was kept.
Supplying an `order_finder` leaves the choice of contraction tree to that finder, and
simplification then runs without a time limit.

`matrix` evaluates the compiled DPP and returns the array with the log of its magnitude taken out,
the two kept apart because a probability can lie far below the smallest float while the
ratio a caller wants is representable.  `evaluated` folds the magnitude back in and returns
a typed table, raising where float64 cannot hold the result.  `distribution` normalises, and
`evidence` gives the total mass, which for a DPP without inputs is its acceptance
probability.

The backend is chosen on the first evaluation rather than while compiling, since deciding
can cost a contraction and a `Compiled` that is never evaluated should not pay for it.
Passing `device` takes that decision instead.  Every result keeps its relative precision
however small the probabilities involved: numbers too small for the dtype add a relative
error of at most ``2**-40``, about 1e-12, to its rounding.  Where the chosen backend cannot
vouch for that, the evaluation is repeated in float64 and, where float64 cannot vouch for it
either, with an exponent per entry, which is slower; `recomputed` records it.

`DEFAULT_KAPPA` is the size limit slicing aims for, ``2**kappa`` entries per intermediate
tensor; a contraction exceeding it is sliced into pieces that do not.

**Exact inference may be refused.**  Compiling raises
`pylea.compile.lower.ExactInferenceTooLarge` where the DPP needs a tensor with more than
``max(2**29, 2**kappa)`` entries that no slicing avoids, such as the joint values of its
output variables, and `pylea.compile.lower.ExactInferenceTooSlow` where, without an
`order_finder`, evaluation is estimated to take longer than about 11.6 days.  Both are
`pylea.compile.lower.ExactInferenceInfeasible`.  With `kappa=None` nothing is refused as
too large.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from pylea.compile.cost import contraction_cost, step_count
from pylea.compile.lower import compile_program, entry_limit
from pylea.compile.order import check_finder_and_kappa
from pylea.compile.planner import plan_compilation
from pylea.eval.backend import Backend, select
from pylea.eval.interp import evaluate_certified
from pylea.simplify import simplify as simplify_program

# The size limit slicing aims for, as a power of two entries.  ``2**28`` float64 entries
# is 2 GiB; several tensors of that size exist at once during a contraction.
DEFAULT_KAPPA = 28


# The keyword arguments of `pylea.syntax.dpp.DPP.compiled`, which every query passes on.
COMPILE_OPTIONS = ("order_finder", "kappa", "recompile", "simplify", "device")


def check_compile_options(order_finder=None, kappa=DEFAULT_KAPPA, recompile=False,
                          simplify=True, device=None):
    """``(kappa, recompile, simplify)`` as an `int` or `None`, a `bool`, and a `bool` or
    ``"always"``, after checking every compile option.

    `order_finder` and `kappa` are checked by `pylea.compile.order.check_finder_and_kappa`;
    `simplify` must be `True`, `False` or ``"always"``, `recompile` `True` or `False`
    (numpy's booleans count as such), and `device` `None`, a string or a
    `torch.device`.  A value of the
    wrong type raises `TypeError`, a value out of range `ValueError`:

        >>> check_compile_options(simplify="sometimes")
        Traceback (most recent call last):
          ...
        ValueError: simplify must be True, False or 'always'; got 'sometimes'
    """
    kappa = check_finder_and_kappa(order_finder, kappa)
    if simplify is True or simplify is False:
        pass
    elif isinstance(simplify, np.bool_):
        simplify = bool(simplify)
    elif not isinstance(simplify, str):
        raise TypeError(f"simplify must be True, False or 'always'; got {simplify!r}")
    elif simplify != "always":
        raise ValueError(f"simplify must be True, False or 'always'; got {simplify!r}")
    if not isinstance(recompile, (bool, np.bool_)):
        raise TypeError(f"recompile must be True or False; got {recompile!r}")
    recompile = bool(recompile)
    if device is not None and not isinstance(device, str) and not _is_torch_device(device):
        raise TypeError(f"device must be None, a device name such as 'cpu' or 'cuda', or a "
                        f"torch.device; got {device!r}")
    return kappa, recompile, simplify


def _is_torch_device(device) -> bool:
    """Whether `device` is a `torch.device`, decided without importing torch."""
    return type(device).__module__ == "torch" and type(device).__name__ == "device"


def planned(program, order_finder, kappa: int | None,
            simplify: bool | Literal["always"], name: str | None = None):
    """`program` compiled: the `Program` compiled, simplified or not; per definition, the
    steps contracting its tensor network (`pylea.compile.plan.Plan`); the index of the
    main function; and the trace.

    With no `order_finder`, `pylea.compile.planner` chooses the contraction tree and
    whether to simplify; with one, the caller has taken that decision, and no further
    search runs on top of the finder it gave, nor is the estimated time checked against
    its limit.  The size limit applies either way.  A refusal names the DPP by `name`.
    `Compiled` and the exact engine's subprocess worker both compile through this, so
    the two cannot drift apart.
    """
    if order_finder is None:
        laddered = plan_compilation(program, kappa=kappa, simplify=simplify, name=name)
        return laddered.program, laddered.plans, laddered.root, laddered.trace
    lowered = simplify_program(program) if simplify else program
    plans, root = compile_program(lowered, order_finder, kappa=kappa,
                                  max_entries=entry_limit(kappa), name=name)
    return lowered, plans, root, ("compilation: contraction tree from the order_finder "
                                  "given",)


def chosen_backend(plans, device: str | None) -> Backend:
    """The evaluation backend: the caller's `device`, else decided by the size of the
    contraction."""
    if device is not None:
        return select(device)                       # the caller has taken the decision
    return select(steps=step_count(plans), ops=contraction_cost(plans))


def evaluated_matrix(plans, root, backend: Backend):
    """The compiled DPP evaluated on `backend`: array, log scale, and which evaluation
    gave them, ``"backend"``, ``"float64"`` or ``"extended"``
    (`pylea.eval.interp.evaluate_certified`)."""
    result = evaluate_certified(plans, root, backend)
    return result.array, result.log_scale, result.path


class Compiled:
    """A `pylea.syntax.dpp.DPP` compiled for the exact engine.

    `trace` records the steps compilation took, which is how a slow or surprising one is
    read back rather than guessed at:

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.25)
        >>> coin.compiled().trace
        ('compilation: greedy contraction tree, cost 0.000e+00',)
    """

    def __init__(self, dpp, order_finder=None, kappa: int | None = DEFAULT_KAPPA,
                 simplify: bool | Literal["always"] = True, device: str | None = None):
        self._dpp = dpp
        self._device = device
        self._backend: Backend | None = None
        self._recomputed: str | None = None
        self.traced = dpp.program()
        self.program, self.plans, self.root, self.trace = planned(
            self.traced, order_finder, kappa, simplify, dpp.name())

    @property
    def recomputed(self) -> str | None:
        """How the last evaluation was repeated to stay accurate: `None` where it was not,
        ``"float64"`` where the chosen backend's dtype was too narrow, ``"extended"``
        where float64 was too, and the evaluation ran with an exponent per entry.

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.25)
            >>> compiled = coin.compiled()
            >>> _ = compiled.matrix()
            >>> compiled.recomputed is None
            True
            >>> @DPP
            ... def both_rare() -> Bool:
            ...     return flip(1e-200) & flip(1e-200)
            >>> compiled = both_rare.compiled(simplify=False)
            >>> _ = compiled.matrix()
            >>> compiled.recomputed
            'extended'
        """
        return self._recomputed

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
        # The backend is left as it was where the evaluation was repeated, so a caller can
        # still see what was tried.
        array, scale, path = evaluated_matrix(self.plans, self.root, self.backend)
        self._recomputed = None if path == "backend" else path
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

        Slicing aims to keep it at most ``2**kappa``, and compiling refuses a DPP for which
        it would exceed ``max(2**29, 2**kappa)``:

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def coins() -> tuple[Bool, Bool, Bool, Bool]:
            ...     return flip(0.5), flip(0.5), flip(0.5), flip(0.5)
            >>> coins.compiled().peak_entries()
            16
        """
        return max(plan.peak_entries() for plan in self.plans.values())

    def __str__(self) -> str:
        return (f"<Compiled '{self._dpp.name()}': {len(self.plans)} definition(s), "
                f"peak 2**{np.log2(self.peak_entries()):.1f} entries>")
