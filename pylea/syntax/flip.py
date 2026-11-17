"""The coin flip primitive.

`flip(p)` is a Boolean variable true with probability `p`, held as a float so that any
probability a double represents survives.  A positive `p` that a double cannot tell from 0,
or one below 1 that it cannot tell from 1, is refused rather than rounded, since rounding
would turn an improbable event into an impossible or a certain one.
"""
import numbers
from fractions import Fraction

from pylea.ir.prim import Prim
from pylea.syntax.dpp import Bool, Primitive
from pylea.syntax.variable import Variable, is_bool

# The smallest positive probability a float tells from 0.  A positive `p` below it is
# rejected rather than rounded, since rounding an improbable event to an impossible one
# would give a wrong answer with no warning.
_MIN_P = 5e-324          # the smallest positive subnormal double


def flip(p: float | Fraction) -> Bool:
    """A coin flip: a Boolean variable that is true with probability `p`.

    `p` is a real number: a Python or numpy integer or float, or a `Fraction`.  It is held
    as a float, so any probability a double can represent is kept, and a probability of
    1e-300 stays 1e-300.  An integer or a `Fraction` is tested against [0, 1] before it
    is rounded, so a fraction just above 1 is refused rather than rounded to 1.

        >>> from pylea import DPP, Bool, prob
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.3)
        >>> prob(coin)
        0.3

    Raises `ValueError` if `p` is outside [0, 1], if a positive `p` is too small for a
    double to tell from 0, or if a `p` below 1 is too close to 1 to tell from certainty.
    Raises `TypeError` for a `bool`, which is a value rather than a probability, and for
    a variable: a probability that depends on a variable is written with
    `pylea.syntax.dpp.Bool.match`, as ``x.match({True: flip(0.9), False: flip(0.1)})``.
    Must be called inside an active DPP definition.
    """
    if is_bool(p):
        raise TypeError(f"flip({p!r}): a bool is not a probability; a constant is written "
                        f"Bool.const({bool(p)!r})")
    if isinstance(p, Variable):
        raise TypeError("flip: the probability must be a number; a probability that "
                        "depends on a variable is written with match, as "
                        "x.match({True: flip(0.9), False: flip(0.1)})")
    if not isinstance(p, numbers.Real):
        raise TypeError(f"flip: the probability must be a real number; got {p!r} "
                        f"({type(p).__name__})")
    if isinstance(p, numbers.Rational):
        exact = Fraction(int(p.numerator), int(p.denominator))
        if not 0 <= exact <= 1:
            raise ValueError(f"Expected a probability between 0 and 1; got {p}.")
        value = exact.numerator / exact.denominator
        if value == 0.0 and exact > 0:
            raise ValueError(
                f"flip({p}): probability is below {_MIN_P:.3e}, the smallest a float "
                f"tells from 0, and would be indistinguishable from an impossible event.")
        if value == 1.0 and exact < 1:
            raise ValueError(
                f"flip({p}): probability is within 2**-53 of 1 and would be "
                f"indistinguishable from a certain event.")
    else:
        value = float(p)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Expected a probability between 0 and 1; got {p}.")

    return Primitive(
        input_types=tuple(),
        output_types=(Bool,),
        name=f"_FLIP__{value!r}_",
        prim=Prim.flip(value),
        label=f"flip({value!r})",
    )()
