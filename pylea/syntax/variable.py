"""Typed random variables.

A `Variable` subclass is a type; an instance holds a fixed number of `Bool`s, each naming
one wire of the enclosing definition's body, and the type's
`pylea.syntax.encoding.Encoding` says how a bit pattern decodes.

The base class supplies what every type shares.  `const` fixes a variable to a value of the
base type, `uniform_bits` draws every bit pattern equally often, and the six comparison
operators compare bit patterns as unsigned integers, MSB first.

A comparison records a `pylea.syntax.dpp.Primitive` whose `source` is a circuit over ``2n``
Boolean wires, built once per type and kind.  Only the interface is retyped, which at the
bit level changes nothing, so the circuit plugs straight in.  Holding the comparison as a
matrix instead would take ``2 x 2**(2n)`` entries where the circuit is linear in `n`.

    >>> from pylea import DPP, Bool, Enum, prob
    >>> class Digit(Enum):
    ...     elements = [0, 1, 2, 3]
    >>> @DPP
    ... def small() -> Bool:
    ...     return Digit.uniform() < 2
    >>> prob(small)
    0.5

Outside a definition `__eq__` and `__ne__` fall back to identity, so a variable can be put
in a set or a dict.  Inside one they are the event that two values agree or differ.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pylea.syntax.encoding import Encoding
from pylea.ir.prim import Prim
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pylea.syntax.dpp import DPP, Bool

class Variable(ABC):
    """A random variable of some `DataType`, held as a fixed-width tuple of `Bool`s."""

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        n = type(self).nbits()
        if len(xs) == n and n != 0:
            dpp = xs[0].to_be_used_within()
            self.__bits: tuple[Bool, ...] = xs
            self.__to_be_used_within: DPP = dpp
            self.name = name
        elif n == 0:
            raise ValueError(f"Cannot create variable of type {type(self)} of size zero.")
        else:
            raise ValueError(f"Expected {n} bits; got {len(xs)}.")

    @classmethod
    def make(cls: DataType, *xs: Bool, name: str | None = None) -> Variable:
        return cls(*xs, name=name)

    @classmethod
    @abstractmethod
    def encoding(cls: DataType) -> Encoding: ...

    @classmethod
    def nbits(cls: DataType) -> int:
        return cls.encoding().nbits()

    @classmethod
    def base_type(cls: DataType) -> type:
        return cls.encoding().base_type()

    def to_be_used_within(self) -> DPP:
        return self.__to_be_used_within

    def bits(self) -> tuple[Bool, ...]:
        return self.__bits

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        return "_".join(f"w{b.wire}" for b in self.bits())

    @classmethod
    def const(cls, value) -> Self:
        """A variable of this type fixed to `value`, which must be of the base type.

        Comparisons coerce a bare base-type value through this, so it is needed explicitly
        only where a constant is passed on as a variable, as a branch of a `match`:

            >>> from pylea import DPP, Bool, Enum, prob
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> @DPP
            ... def stays_red() -> Bool:
            ...     drawn = Colour.random({"R": 0.5, "G": 0.3, "B": 0.2})
            ...     kept = drawn.match({"R": drawn, "G": Colour.const("R"),
            ...                         "B": Colour.const("R")})
            ...     return kept == "R"
            >>> prob(stays_red)
            1.0

        Must be called inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        if not isinstance(value, cls.base_type()):
            raise ValueError(
                f"Expected value of base type {cls.base_type().__name__} "
                f"to create constant variable of type {cls.__name__}; "
                f"got {type(value).__name__}."
            )
        bits = cls.encoding()(value).to_bits()
        if not bits:
            raise ValueError(f"Cannot create constant for zero-bit type {cls.__name__}.")
        # `to_bits` is MSB-first, the same packing the IR matrices use.
        index = sum(int(b) << (len(bits) - 1 - i) for i, b in enumerate(bits))
        return Primitive(
            input_types=tuple(),
            output_types=(cls,),
            name=f"_CONST_{cls.__name__}_",
            prim=Prim.const(index, cls.nbits()),
            label=str(value),
        )()

    _uniform_bits_cache: dict = {}

    @classmethod
    def uniform_bits(cls) -> Self:
        """A variable of this type whose bit pattern is uniformly distributed.

        Every one of the ``2**nbits()`` patterns is equally likely, including any decoding
        to no value.  `pylea.syntax.enum.Enum.uniform` draws over the elements instead, and
        the two differ exactly where the element count is not a power of two:

            >>> from pylea import DPP, Enum
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> @DPP
            ... def over_patterns() -> Colour:
            ...     return Colour.uniform_bits()
            >>> @DPP
            ... def over_elements() -> Colour:
            ...     return Colour.uniform()
            >>> over_patterns.distribution().round(4).tolist()
            [0.25, 0.25, 0.25, 0.25]
            >>> over_elements.distribution().round(4).tolist()
            [0.3333, 0.3333, 0.3333, 0.0]

        Must be called inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        if cls in Variable._uniform_bits_cache:
            return Variable._uniform_bits_cache[cls]()

        k = cls.nbits()
        _UNIF_BITS_ = Primitive(
            input_types=tuple(),
            output_types=(cls,),
            name="_UNIF_BITS_" + cls.__name__,
            prim=Prim.uniform(k),
        )
        Variable._uniform_bits_cache[cls] = _UNIF_BITS_
        return _UNIF_BITS_()

    __hash__ = object.__hash__  # type: ignore

    def _coerced(self, other):
        """`other` as a variable of this type, a raw base-type value going through `const`."""
        t = type(self)
        if isinstance(other, t):
            return other
        if isinstance(other, t.base_type()):
            return t.const(other)
        raise TypeError(
            f"Cannot compare variable of type {t.__name__} with value of type "
            f"{type(other).__name__}. Expected a {t.__name__} variable or a "
            f"{t.base_type().__name__} base-type value."
        )

    _comparator_cache: dict = {}

    def _compare(self, other, kind: str, make) -> Bool:
        """``comparator(self, other)``, the comparator being built once per type and kind.

        The comparator is a `pylea.syntax.dpp.Primitive` that only retypes the interface,
        turning 2n `Bool` wires into two n-bit variables.  It keeps the circuit as its
        `source`, since the matrix would be ``2 x 2**(2n)``, and at the IR's bit level the
        retyping changes nothing, so the circuit plugs straight in.
        """
        t = type(self)
        key = (t, kind)
        if key not in Variable._comparator_cache:
            from pylea.syntax.dpp import Primitive, Bool as BoolType
            Variable._comparator_cache[key] = Primitive(
                input_types=(t, t),
                output_types=(BoolType,),
                name=f"_{kind}_{t.__name__}_",
                source=make(t.nbits()),
            )
        return Variable._comparator_cache[key](self, self._coerced(other))

    def __eq__(self, other: object) -> Bool:  # type: ignore[override]
        """The event that `self` and `other` are equal, inside a DPP definition.

        `other` may be a variable of the same type or a raw base-type value, which goes
        through `const`.  Outside a definition this is ordinary identity, so that
        variables can be put in sets and dicts and tested for membership.
        """
        from pylea.syntax.dpp import DPP
        if DPP._DPP__currently_processing is None:  # type: ignore
            return self is other  # type: ignore
        from pylea.syntax.comparison import make_equality_comparator
        return self._compare(other, "EQ", make_equality_comparator)

    def __lt__(self, other) -> Bool:
        """The event that `self` is strictly below `other`, comparing bit patterns.

        Both bit patterns are read as unsigned integers, MSB first.  A subclass wanting a
        different ordering, such as a signed integer type, overrides this.
        """
        from pylea.syntax.comparison import make_less_than_comparator
        return self._compare(other, "LT", make_less_than_comparator)

    def __le__(self, other) -> Bool:
        """The event that `self` is at most `other`, comparing bit patterns."""
        return (self < other) | (self == other)

    def __ne__(self, other: object) -> Bool:  # type: ignore[override]
        """The event that `self` and `other` differ, inside a DPP definition.

        Outside a definition this is ordinary identity, as `__eq__` is.
        """
        from pylea.syntax.dpp import DPP
        if DPP._DPP__currently_processing is None:  # type: ignore
            return self is not other  # type: ignore
        return ~(self == other)

    def __gt__(self, other) -> Bool:
        """The event that `self` is strictly above `other`, comparing bit patterns."""
        return ~(self <= other)

    def __ge__(self, other) -> Bool:
        """The event that `self` is at least `other`, comparing bit patterns."""
        return (self > other) | (self == other)

type DataType = type[Variable]

