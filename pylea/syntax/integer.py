"""Fixed-width unsigned integers.

A `UInt` subclass declares its width in the class variable `width`.  Its values are the
integers ``0`` to ``2**width - 1``, encoded in binary, MSB first, so the comparisons every
`pylea.syntax.variable.Variable` carries order them numerically and compare against plain
ints.  `+` adds two variables of the same type, or a variable and a plain int, as a
ripple-carry adder circuit, linear in the width where a matrix over the same interface
would be exponential.  The sum is taken modulo ``2**width``, so a subclass is declared wide
enough for the largest total it must hold.

    >>> from pylea import DPP, Bool, flip, prob
    >>> class Total(UInt):
    ...     width = 4
    >>> @DPP
    ... def lands_on_ten() -> Bool:
    ...     total = Total.const(3)
    ...     step = flip(0.5).match({True: 2, False: 7}, data_type=Total)
    ...     return total + step == 10
    >>> prob(lands_on_ten)
    0.5

`count` sums an iterable of `Bool`s into a `UInt`, which is how "how many of these events
happened" is written; the result compares against thresholds and, its values being
numbers, has an expectation.

    >>> from pylea import expectation
    >>> @DPP
    ... def heads() -> UInt:
    ...     return count(flip(0.5) for _ in range(3))
    >>> round(expectation(heads), 12)
    1.5
"""
from __future__ import annotations

from typing import ClassVar, Iterable, Self, TypeVar, cast, overload

from pylea.syntax.dpp import DPP, Bool, Primitive
from pylea.syntax.encoding import BitEncoded, Encoding
from pylea.syntax.variable import Variable

_U = TypeVar('_U', bound='UInt')


def make_adder(n: int):
    """The `DPP` adding two n-bit unsigned integers modulo ``2**n``.

    Use `+` on a `UInt` instead; this is what that builds.  A ripple-carry adder from the
    least significant bit up, linear in `n`; the carry out of the top bit is dropped,
    which is the modulo.
    """

    @DPP
    def _add(x, types=(Bool,) * (2 * n)):
        a = x[:n]
        b = x[n:]
        assert len(a) == n and len(b) == n

        carry = Bool.const(False)
        total: list[Bool] = [Bool.const(False)] * n
        for i in reversed(range(n)):             # bits are MSB first, so the last is lowest
            partial = a[i] ^ b[i]
            total[i] = partial ^ carry
            carry = (a[i] & b[i]) | (carry & partial)
        return tuple(total)

    return _add


class UInt(Variable):
    """An unsigned integer variable of a fixed width.

    A subclass declares its width by setting the class variable `width`; its values are
    ``0`` to ``2**width - 1``.  Every bit pattern encodes a value, so
    `pylea.syntax.variable.Variable.uniform_bits` draws uniformly over the whole range.

        >>> class Nibble(UInt):
        ...     width = 4
        >>> Nibble.nbits(), Nibble.encoding().n_values()
        (4, 16)

    Instantiating a subclass that has not set `width` raises `NotImplementedError`.
    """

    width: ClassVar[int | None] = None
    # The built encoding, stored on the class that built it and read from its own
    # namespace only, so a subclass declaring its own parameters never inherits it.
    _encoding: ClassVar[Encoding | None] = None
    _adder_cache: ClassVar[dict] = {}

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        if type(self).width is None:
            raise NotImplementedError(
                f"Cannot instantiate subclass \"{type(self).__name__}\" of UInt; "
                "class variable `width` unspecified."
            )
        super().__init__(*xs, name=name)

    @classmethod
    def encoding(cls) -> Encoding:
        cached = vars(cls).get("_encoding")
        if cached is not None:
            return cached
        if cls.width is None:
            raise NotImplementedError(
                f"Cannot get encoding for \"{cls.__name__}\"; "
                "class variable `width` unspecified."
            )
        n = cls.width
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"{cls.__name__}: `width` must be a positive int; got {n!r}.")

        class EncodedUInt(BitEncoded):

            def __init__(self, v: int) -> None:
                if not 0 <= v < (1 << n):
                    raise ValueError(
                        f"Value {v} is outside 0 to {(1 << n) - 1}, the range of a "
                        f"{n}-bit unsigned integer.")
                self._value = int(v)

            @property
            def value(self) -> int:
                return self._value

            @classmethod
            def nbits(cls) -> int:
                return n

            @classmethod
            def base_type(cls) -> type:
                return int

            @classmethod
            def decoded(cls) -> dict[int, object]:
                return {i: i for i in range(1 << n)}

            @classmethod
            def from_bits(cls, *bits: bool) -> "EncodedUInt":
                if len(bits) != n:
                    raise ValueError(f"Expected {n} bits; got {len(bits)}.")
                return EncodedUInt(sum(int(b) << (n - 1 - i) for i, b in enumerate(bits)))

            def to_bits(self) -> tuple[bool, ...]:
                return tuple(bool((self._value >> (n - 1 - i)) & 1) for i in range(n))

        cls._encoding = EncodedUInt
        return EncodedUInt

    def __add__(self, other: Self | int) -> Self:
        """The sum of `self` and `other`, modulo ``2**width``.

        `other` may be a variable of the same type or a plain int, which goes through
        `pylea.syntax.variable.Variable.const`.  Built once per type as an adder circuit.

            >>> from pylea import DPP, prob
            >>> class Twobit(UInt):
            ...     width = 2
            >>> @DPP
            ... def wraps() -> Bool:
            ...     return Twobit.const(3) + 2 == 1        # 5 modulo 4
            >>> prob(wraps)
            1.0
        """
        t = type(self)
        if t not in UInt._adder_cache:
            UInt._adder_cache[t] = Primitive(
                input_types=(t, t),
                output_types=(t,),
                name=f"_ADD_{t.__name__}_",
                source=make_adder(t.nbits()),
            )
        return cast(Self, UInt._adder_cache[t](self, self._coerced(other)))

    # A plain int does not know how to add a `UInt`, so `3 + x` reaches here through
    # Python's reflected-operator fallback; addition commutes, so it is the same circuit.
    __radd__ = __add__


@overload
def count(xs: Iterable[Bool]) -> UInt: ...
@overload
def count(xs: Iterable[Bool], data_type: type[_U]) -> _U: ...
def count(xs: Iterable[Bool], data_type: type[UInt] | None = None) -> UInt:
    """How many of the events `xs` happen, as a `UInt`.

    `data_type` names the result type, which must be wide enough to hold ``len(xs)``;
    left out, a fresh type of exactly that width is made.  Built as a chain of adders,
    each adding one event's bit, so the cost is linear in ``len(xs)`` times the width.
    An empty `xs` counts to the constant 0.  Must be called inside an active DPP
    definition.

        >>> from pylea import DPP, Bool, flip, prob
        >>> class Hits(UInt):
        ...     width = 3
        >>> @DPP
        ... def majority() -> Bool:
        ...     return count((flip(0.5) for _ in range(5)), data_type=Hits) >= 3
        >>> prob(majority)
        0.5
    """
    events = list(xs)
    n = len(events)
    if data_type is None:
        size = max(1, n.bit_length())

        class _Count(UInt):
            width = size

        data_type = _Count
    else:
        if not (isinstance(data_type, type) and issubclass(data_type, UInt)):
            raise TypeError(
                f"count: data_type must be a UInt subclass; got {data_type!r}.")
        if (1 << data_type.nbits()) <= n:
            raise ValueError(
                f"count: {data_type.__name__} holds 0 to {(1 << data_type.nbits()) - 1}, "
                f"too narrow for a count of up to {n}.")

    padding = data_type.nbits() - 1
    total = data_type.const(0)
    for x in events:
        one = cast(UInt, data_type.make(*([Bool.const(False)] * padding), x))
        total = total + one
    return cast(UInt, total)
