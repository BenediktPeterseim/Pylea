"""The coin flip primitive.

`flip(p)` is a Boolean variable true with probability `p`, held as a float so that any
probability a double represents survives.  A positive `p` that a double cannot tell from 0,
or one below 1 that it cannot tell from 1, is refused rather than rounded, since rounding
would turn an improbable event into an impossible or a certain one.
"""
from fractions import Fraction

from pylea.ir.prim import Prim
from pylea.syntax.dpp import Bool, Primitive

# The smallest positive probability a float tells from 0.  A positive `p` below it is
# rejected rather than rounded, since rounding an improbable event to an impossible one
# would give a wrong answer with no warning.
_MIN_P = 5e-324          # the smallest positive subnormal double


def flip(p: float | Fraction) -> Bool:
    """A coin flip: a Boolean variable that is true with probability `p`.

    `p` is held as a float, so any probability a double can represent is kept, and a
    probability of 1e-300 stays 1e-300.  Raises `ValueError` if `p` is outside [0, 1], if
    a positive `p` is too small for a double to tell from 0, or if a `p` below 1 is too
    close to 1 to tell from certainty.  Must be called inside an active DPP definition.

        >>> from pylea import DPP, Bool, prob
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.3)
        >>> prob(coin)
        0.3
    """
    if isinstance(p, Fraction):
        value = p.numerator / p.denominator
        if value == 0.0 and p > 0:
            raise ValueError(
                f"flip({p}): probability is below {_MIN_P:.3e}, the smallest a float "
                f"tells from 0, and would be indistinguishable from an impossible event.")
        if value == 1.0 and p < 1:
            raise ValueError(
                f"flip({p}): probability is within 2**-53 of 1 and would be "
                f"indistinguishable from a certain event.")
    elif isinstance(p, float) or isinstance(p, int):
        value = float(p)
    else:
        raise TypeError(f"Expected float or Fraction; got {type(p).__name__}")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Expected a probability between 0 and 1; got {p}.")

    return Primitive(
        input_types=tuple(),
        output_types=(Bool,),
        name=f"_FLIP__{value!r}_",
        prim=Prim.flip(value),
    )()
