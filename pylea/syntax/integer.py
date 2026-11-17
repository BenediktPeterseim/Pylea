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

Any width holds constants, comparisons and `+`.  A query whose output is a `UInt` has
``2**width`` outputs, so `pylea.syntax.expectation.expectation` and
`pylea.syntax.pmf.probability_mass_function` refuse an output of more than
`pylea.syntax.method.MAX_OUTPUT_BITS` (20) bits; a wide `UInt` is compared or counted inside
the DPP instead, and the query asks for the probability of an event about it.
"""
from __future__ import annotations

import numbers
from functools import lru_cache
from typing import ClassVar, Iterable, Self, TypeVar, cast, overload

from pylea.syntax.dpp import DPP, Bool, Primitive
from pylea.syntax.encoding import BitEncoded, Encoding
from pylea.syntax.variable import Variable, is_bool

_U = TypeVar('_U', bound='UInt')


def make_adder(n: int):
    """The `DPP` adding two n-bit unsigned integers modulo ``2**n``.

    Use `+` on a `UInt` instead; this is what that builds.  A ripple-carry adder from the
    least significant bit up, linear in `n`; the carry out of the top bit is dropped,
    which is the modulo.

        >>> from pylea.ir.program import op_count
        >>> op_count(make_adder(4).program())
        53
    """

    @DPP
    def _add(x, types=(Bool,) * (2 * n)):
        a = x[:n]
        b = x[n:]
        assert len(a) == n and len(b) == n

        carry = Bool.const(False)
        total: list[Bool] = []
        for i in reversed(range(n)):             # bits are MSB first, so the last is lowest
            partial = a[i] ^ b[i]
            total.append(partial ^ carry)
            carry = (a[i] & b[i]) | (carry & partial)
        return tuple(reversed(total))

    return _add


@lru_cache(maxsize=None)
def _adder_circuit(nbits: int) -> DPP:
    """`make_adder(nbits)`, built once per width and shared by every type of that width."""
    return make_adder(nbits)


class UInt(Variable):
    """An unsigned integer variable of a fixed width.

    A subclass declares its width by setting the class variable `width` to a positive
    int; its values are the integers ``0`` to ``2**width - 1``, numpy integers included.
    A float is refused, even a whole one, as is an integer outside the range.  Every bit
    pattern encodes a value, so `pylea.syntax.variable.Variable.uniform_bits` draws
    uniformly over the whole range.

        >>> class Nibble(UInt):
        ...     width = 4
        >>> Nibble.nbits(), Nibble.encoding().n_values()
        (4, 16)

    Comparing with an integer outside the range gives a constant: ``x < 100`` holds and
    ``x == -1`` does not, for every `x` of `Nibble`.  Any width holds constants,
    comparisons and `+`; a query returning a `UInt` has ``2**width`` outputs, and
    `expectation` and `probability_mass_function` refuse one of more than 20 bits.  Using a
    subclass that has not set `width` raises `NotImplementedError`.
    """

    width: ClassVar[int | None] = None
    # The built encoding, stored on the class that built it and read from its own
    # namespace only, so a subclass declaring its own parameters never inherits it.
    _encoding: ClassVar[Encoding | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        width = vars(cls).get("width")
        if width is None:
            return
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError(f"{cls.__name__}.width must be an int; got {width!r}")
        if width < 1:
            raise ValueError(f"{cls.__name__}.width must be at least 1; got {width}")

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        _require_width(type(self), f"Cannot create a {type(self).__name__} variable")
        super().__init__(*xs, name=name)

    @classmethod
    def _value_table(cls) -> object:
        return cls.width

    @classmethod
    def encoding(cls) -> Encoding:
        cached = vars(cls).get("_encoding")
        if cached is not None:
            return cached
        _require_width(cls, f"{cls.__name__}.encoding")
        n: int = cls.width  # type: ignore[assignment]
        name = cls.__name__

        class EncodedUInt(BitEncoded):
            """An integer of a `UInt`, encoded in binary, MSB first."""

            def __init__(self, v: int) -> None:
                if not isinstance(v, numbers.Integral):
                    raise TypeError(
                        f"{name} holds integers; got {v!r} ({type(v).__name__})")
                if not 0 <= v < (1 << n):
                    raise ValueError(
                        f"{v} is outside 0 to {(1 << n) - 1}, the range of {name}")
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

    def _fixed_order(self, other) -> int | None:
        if not isinstance(other, numbers.Integral) or is_bool(other):
            return None
        if other < 0:
            return -1
        if other >= 1 << type(self).nbits():
            return 1
        return None

    def __add__(self, other: Self | int) -> Self:
        """The sum of `self` and `other`, modulo ``2**width``.

        `other` may be a variable of the same type or an integer, negative ones included,
        which is taken modulo ``2**width`` first, so ``x + (-1)`` subtracts one.

            >>> from pylea import DPP, prob
            >>> class Twobit(UInt):
            ...     width = 2
            >>> @DPP
            ... def wraps() -> Bool:
            ...     return Twobit.const(3) + 2 == 1        # 5 modulo 4
            >>> prob(wraps)
            1.0
            >>> @DPP
            ... def steps_back() -> Bool:
            ...     return Twobit.const(0) + (-1) == 3
            >>> prob(steps_back)
            1.0
        """
        t = type(self)
        if isinstance(other, numbers.Integral) and not is_bool(other):
            other = int(other) % (1 << t.nbits())
        adder = Primitive(
            input_types=(t, t),
            output_types=(t,),
            name=f"_ADD_{t.__name__}_",
            source=_adder_circuit(t.nbits()),
        )
        return cast(Self, adder(self, self._coerced(other, "add")))

    # A plain int does not know how to add a `UInt`, so `3 + x` reaches here through
    # Python's reflected-operator fallback; addition commutes, so it is the same circuit.
    __radd__ = __add__


def _require_width(cls, what: str) -> None:
    """Nothing, or `NotImplementedError` where `cls` declares no `width`."""
    if cls.width is None:
        raise NotImplementedError(
            f"{what}: {cls.__name__} declares no width; set its class variable `width` in "
            "a subclass")


@lru_cache(maxsize=None)
def _count_type(width: int) -> type[UInt]:
    """The `UInt` type of `width` bits that `count` returns without a `data_type`, one
    per width, so that two such counts of one width compare."""
    return cast(type[UInt], type(f"Count{width}", (UInt,), {
        "width": width, "_made_by_count": True, "__module__": __name__,
        "__doc__": f"The result type of `count` over at most {(1 << width) - 1} events."}))


@overload
def count(xs: Iterable[Bool]) -> UInt: ...
@overload
def count(xs: Iterable[Bool], data_type: type[_U]) -> _U: ...
def count(xs: Iterable[Bool], data_type: type[UInt] | None = None) -> UInt:
    """How many of the events `xs` happen, as a `UInt`.

    `xs` holds `Bool`s, and a Python or numpy Boolean counts as a constant.  `data_type`
    names the result type, which must be wide enough to hold ``len(xs)``; left out, the
    type is the one of exactly that width that every such `count` shares, so two counts
    of one width compare.  Built as a chain of adders, each adding one event's bit, so the
    cost is linear in ``len(xs)`` times the width.  An empty `xs` counts to the constant
    0.  Must be called inside an active DPP definition.

        >>> from pylea import DPP, Bool, flip, prob
        >>> class Hits(UInt):
        ...     width = 3
        >>> @DPP
        ... def majority() -> Bool:
        ...     return count((flip(0.5) for _ in range(5)), data_type=Hits) >= 3
        >>> prob(majority)
        0.5
        >>> @DPP
        ... def tie() -> Bool:
        ...     return count(flip(0.5) for _ in range(3)) == count(flip(0.5) for _ in range(3))
        >>> prob(tie)
        0.3125

    Raises `TypeError` for an item that is neither a `Bool` nor a Boolean.
    """
    events: list[Bool] = []
    for i, x in enumerate(xs):
        if isinstance(x, Bool):
            events.append(x)
        elif is_bool(x):
            events.append(Bool.const(bool(x)))
        else:
            raise TypeError(f"count: every item must be a Bool or a Boolean; item {i} is "
                            f"a {type(x).__name__}")
    n = len(events)
    if data_type is None:
        data_type = _count_type(max(1, n.bit_length()))
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
