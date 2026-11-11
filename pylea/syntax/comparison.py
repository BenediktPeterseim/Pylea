"""Comparator circuits.

Each is an ordinary DPP over ``2n`` Boolean wires and is linear in `n`: equality conjoins
the bitwise agreements, and less-than accumulates the first position at which the two differ
in the right direction, guarded by agreement above it.

`pylea.syntax.variable` wraps one in a `pylea.syntax.dpp.Primitive` that retypes the
interface to two n-bit variables and keeps the circuit; use the operators on a variable
rather than calling these directly.
"""
from pylea.syntax.dpp import DPP, Bool


def make_equality_comparator(n: int):
    """The `DPP` comparing two n-bit values for equality.

    Use `==` on a `pylea.syntax.variable.Variable` instead; this is what that builds.

        >>> from pylea import DPP, Bool, Enum, prob
        >>> class Colour(Enum):
        ...     elements = ["R", "G", "B"]
        >>> @DPP
        ... def is_red() -> Bool:
        ...     return Colour.uniform() == "R"
        >>> round(prob(is_red), 6)
        0.333333
    """

    @DPP
    def _eq(x, types=(Bool,) * 2 * n) -> Bool:
        a = x[:n]
        b = x[n:]
        assert len(a) == n and len(b) == n

        result = Bool.const(True)
        for i in range(n):
            bit_eq = (a[i] & b[i]) | (~a[i] & ~b[i])
            result = result & bit_eq

        return result

    return _eq


def make_less_than_comparator(n: int):
    """The `DPP` deciding whether the first of two n-bit values is below the second.

    Use `<` on a `pylea.syntax.variable.Variable` instead; this is what that builds.
    """

    @DPP
    def _lt(x, types=(Bool,) * 2 * n) -> Bool:
        a = x[:n]
        b = x[n:]

        assert len(a) == n and len(b) == n

        eq_prefix = Bool.const(True)
        lt = Bool.const(False)

        for i in range(n):
            ai, bi = a[i], b[i]

            # `a` is below `b` at this bit where the two agree above it and `ai` is the
            # smaller of the pair.
            lt = lt | (eq_prefix & (~ai & bi))
            bit_eq = (ai & bi) | (~ai & ~bi)
            eq_prefix = eq_prefix & bit_eq

        return lt

    return _lt
