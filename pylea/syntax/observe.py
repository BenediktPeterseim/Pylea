"""Conditioning on an event, hard or by likelihood."""

import math

import numpy as np

from pylea.syntax.dpp import Primitive, Bool
from pylea.ir.prim import Prim

_OBSERVE_ = Primitive(
    input_types=(Bool,),
    output_types=tuple(),
    name="_OBSERVE_",
    prim=Prim.observe(),
)


def observe(x: Bool, *, if_true: float = 1.0, if_false: float = 0.0) -> None:
    """Nothing, having conditioned the enclosing DPP on the event `x`.

    With the defaults this is hard conditioning: combinations of values where `x` is false
    contribute no weight, so the DPP's total mass becomes its acceptance probability
    rather than 1.

        >>> from pylea import DPP, Bool, flip, observe, prob
        >>> @DPP
        ... def first_given_either() -> Bool:
        ...     a, b = flip(0.5), flip(0.5)
        ...     observe(a | b)
        ...     return a
        >>> round(prob(first_given_either), 6)
        0.666667

    `if_true` and `if_false` generalise this to virtual evidence, in Pearl's sense: an
    unreliable witness reports that `x` happened, and the two numbers are the
    likelihoods of that report, so the weight of each combination of values is multiplied
    by `if_true` where `x` is true and by `if_false` where it is false.  Only their ratio
    matters to any conditional query, since a common factor cancels when the posterior is
    normalised.  The construction is not a new primitive of the language: for
    likelihoods at most 1 it is exactly conditioning on a coin flip that is true with
    probability `if_true` or `if_false`, as ``observe(x.match({True: flip(if_true),
    False: flip(if_false)}))``, and the defaults ``(1.0, 0.0)`` are hard conditioning.

        >>> @DPP
        ... def reported() -> Bool:
        ...     x = flip(0.5)
        ...     observe(x, if_true=0.9, if_false=0.3)
        ...     return x
        >>> prob(reported)
        0.75

    Both likelihoods must be finite and non-negative, and at least one positive, since
    weight zero everywhere leaves nothing to condition on; `ValueError` says otherwise.
    Must be called inside an active DPP definition.
    """
    if if_true == 1.0 and if_false == 0.0:
        _OBSERVE_(x)
        return
    for name, value in (("if_true", if_true), ("if_false", if_false)):
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
            raise ValueError(
                f"observe: {name} must be a finite non-negative likelihood; "
                f"got {value!r}.")
    if if_true == 0.0 and if_false == 0.0:
        raise ValueError(
            "observe: if_true and if_false are both zero, which gives every "
            "combination of values weight zero and leaves nothing to condition on.")
    Primitive(
        input_types=(Bool,),
        output_types=tuple(),
        name=f"_OBSERVE_{float(if_true)!r}_{float(if_false)!r}_",
        prim=Prim.from_matrix(np.array([[float(if_false), float(if_true)]])),
    )(x)
