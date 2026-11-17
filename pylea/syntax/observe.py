"""Conditioning on an event, hard or by likelihood."""

import math
import numbers

import numpy as np

from pylea.syntax.dpp import Primitive, Bool
from pylea.syntax.variable import is_bool
from pylea.ir.prim import Prim

_OBSERVE_ = Primitive(
    input_types=(Bool,),
    output_types=tuple(),
    name="_OBSERVE_",
    prim=Prim.observe(),
    label="observe",
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

    `if_true` and `if_false` weigh the event instead: the weight of each combination of
    values is multiplied by `if_true` where `x` is true and by `if_false` where it is
    false.  This is how an uncertain observation of `x` is written, the two numbers being
    the likelihoods of that observation given `x` and given ``~x``.  Only their ratio
    matters to any conditional query, since a common factor cancels when the posterior is
    normalised.  For likelihoods at most 1 it is the same as conditioning on a coin flip
    that is true with probability `if_true` or `if_false`, as
    ``observe(x.match({True: flip(if_true), False: flip(if_false)}))``, and the defaults
    ``(1.0, 0.0)`` are hard conditioning.

        >>> @DPP
        ... def reported() -> Bool:
        ...     x = flip(0.5)
        ...     observe(x, if_true=0.9, if_false=0.3)
        ...     return x
        >>> prob(reported)
        0.75

    `x` is a `Bool`, or a Python `True` or `False`, which is a constant event.  Both
    likelihoods are real numbers, such as Python or numpy floats, finite and
    non-negative, and at least one positive, since weight zero everywhere leaves nothing
    to condition on; `ValueError` says otherwise, and `TypeError` for a likelihood that is
    not a number.  Must be called inside an active DPP definition.
    """
    likelihoods = []
    for name, value in (("if_true", if_true), ("if_false", if_false)):
        if is_bool(value) or not isinstance(value, numbers.Real):
            raise TypeError(f"observe: {name} must be a number; got {value!r}.")
        try:
            likelihood = float(value)
        except OverflowError:                   # an integer beyond the float range
            likelihood = math.inf
        if not (math.isfinite(likelihood) and likelihood >= 0):
            raise ValueError(
                f"observe: {name} must be a finite non-negative likelihood; "
                f"got {value!r}.")
        likelihoods.append(likelihood)
    if_true, if_false = likelihoods
    if if_true == 0.0 and if_false == 0.0:
        raise ValueError(
            "observe: if_true and if_false are both zero, which gives every "
            "combination of values weight zero and leaves nothing to condition on.")
    if if_true == 1.0 and if_false == 0.0:
        _OBSERVE_(x)
        return
    Primitive(
        input_types=(Bool,),
        output_types=tuple(),
        name=f"_OBSERVE_{if_true!r}_{if_false!r}_",
        prim=Prim.from_matrix(np.array([[if_false, if_true]])),
        label=f"observe(if_true={if_true!r}, if_false={if_false!r})",
    )(x)
