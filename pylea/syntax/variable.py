"""Typed random variables.

A `Variable` subclass is a type; an instance holds a fixed number of `Bool`s, one per bit,
each a variable of the enclosing definition, and the type's
`pylea.syntax.encoding.Encoding` says how a bit pattern decodes.

The base class supplies what every type shares.  `const` fixes a variable to a value of the
type, `uniform_bits` draws every bit pattern equally often, and the six comparison
operators compare bit patterns as unsigned integers, MSB first.  A comparison is recorded
as a call of a comparator circuit from `pylea.syntax.comparison`, which says why it is a
circuit rather than a matrix; one circuit serves every type of the same width.

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

Wherever a variable of a declared type is expected, `as_variable` decides what is accepted:
a variable of that type, a variable of a subclass that holds the same values in the same
bit patterns, or a value of the type, which becomes a constant.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from pylea.syntax.encoding import Encoding
from pylea.ir.prim import Prim
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pylea.syntax.dpp import DPP, Bool

def is_bool(value) -> bool:
    """Whether `value` is a Python or numpy Boolean, which a `Bool` accepts as a constant;
    the integers 0 and 1 are not.

        >>> import numpy as np
        >>> is_bool(True), is_bool(np.True_), is_bool(1)
        (True, True, False)
    """
    return isinstance(value, (bool, np.bool_))


def as_variable(value, data_type: DataType, what: str) -> Variable:
    """`value` as a variable of `data_type`, for a place where one is expected.

    A variable of `data_type` is taken as it is, and so is one of a subclass or a base
    class with the same bits and the same value per bit pattern, which reads the same;
    any other variable
    raises `TypeError`, since reading it bit for bit as `data_type` would silently mean
    something else.  A value that is not a variable becomes a constant of `data_type`
    through `Variable.const`.  Every message starts with `what`, which names the place.

        >>> from pylea import DPP, Bool, Enum
        >>> class Weather(Enum):
        ...     elements = ["dry", "rain", "snow"]
        >>> class Grass(Enum):
        ...     elements = ["dry", "damp", "wet"]
        >>> @DPP
        ... def mismatch() -> Bool:
        ...     as_variable(Weather.uniform(), Grass, "is_wet's argument g")
        ...     return Bool.const(True)
        Traceback (most recent call last):
          ...
        TypeError: is_wet's argument g must be a Grass; got a Weather
    """
    if isinstance(value, Variable):
        found = type(value)
        if reads_as(found, data_type):
            return value
        raise TypeError(f"{what} must be a {data_type.__name__}; got a {found.__name__}")
    try:
        data_type.encoding()(value)
    except (TypeError, ValueError) as exc:
        raise _prefixed(exc, what) from None
    return data_type.const(value)


def reads_as(found: DataType, declared: DataType) -> bool:
    """Whether a variable of type `found` may stand where `declared` is expected.

    True for `declared` itself, and for a subclass or a base class of it with the same
    bits and the same value per bit pattern, such as a subclass that only renames its
    base: the two read every bit pattern alike.  Where `declared` declares no values, as
    `pylea.syntax.integer.UInt` itself does, every subclass reads as it.

        >>> from pylea import Enum
        >>> class Weather(Enum):
        ...     elements = ["dry", "rain"]
        >>> class Renamed(Weather):
        ...     pass
        >>> class Reordered(Weather):
        ...     elements = ["rain", "dry"]
        >>> reads_as(Renamed, Weather), reads_as(Weather, Renamed)
        (True, True)
        >>> reads_as(Reordered, Weather)
        False
    """
    if found is declared:
        return True
    subclass = issubclass(found, declared)
    if not (subclass or issubclass(declared, found)):
        return False
    try:
        return (found.nbits() == declared.nbits()
                and found._value_table() == declared._value_table())
    except NotImplementedError:
        # A type declaring no values, such as `UInt` itself, stands for all its subclasses.
        return subclass


def require_variable_type(t, where: str) -> None:
    """Nothing, or `TypeError` unless `t` is a `Variable` subclass; `where` names the place.

        >>> require_variable_type(bool, "map: data_type")
        Traceback (most recent call last):
          ...
        TypeError: map: data_type must be a variable type such as Bool, an Enum or a UInt; got bool, for which use Bool
    """
    if isinstance(t, type) and issubclass(t, Variable):
        return
    shown = t.__name__ if isinstance(t, type) else repr(t)
    hint = ", for which use Bool" if t is bool else ""
    raise TypeError(f"{where} must be a variable type such as Bool, an Enum or a UInt; "
                    f"got {shown}{hint}")


def require_definition(what: str) -> None:
    """Nothing inside a DPP definition; `RuntimeError` naming `what` outside one.

    Called first by what builds a constant or a draw, so that the message names what the
    user wrote rather than the primitive it records.
    """
    from pylea.syntax.dpp import active_definition
    if active_definition() is None:
        raise RuntimeError(
            f"{what} was called outside a DPP definition: flip, observe, operations on "
            "variables and calls of DPPs work only inside a function decorated with @DPP.")


def _prefixed(exc: Exception, prefix: str) -> Exception:
    """A `TypeError` or `ValueError`, as `exc` is, whose message is `exc`'s after `prefix`."""
    kind = TypeError if isinstance(exc, TypeError) else ValueError
    return kind(f"{prefix}: {exc}")


@lru_cache(maxsize=None)
def _comparator_circuit(kind: str, nbits: int) -> DPP:
    """The comparator circuit `kind` ("EQ" or "LT") over two `nbits`-bit values.

    Held per width, not per type, so that types made on the fly, as `count` and user
    factories make them, share circuits and are not kept alive by them.
    """
    from pylea.syntax.comparison import make_equality_comparator, make_less_than_comparator
    make = make_equality_comparator if kind == "EQ" else make_less_than_comparator
    return make(nbits)


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
            raise ValueError(
                f"Cannot create a variable of type {type(self).__name__}, which has no bits.")
        else:
            raise ValueError(f"Expected {n} bits; got {len(xs)}.")

    @classmethod
    def make(cls: DataType, *xs: Bool, name: str | None = None) -> Variable:
        """A variable of this type made of the `Bool`s `xs`, MSB first, one per bit."""
        return cls(*xs, name=name)

    @classmethod
    @abstractmethod
    def encoding(cls: DataType) -> Encoding:
        """The `pylea.syntax.encoding.BitEncoded` subclass giving this type's bit patterns."""

    @classmethod
    def nbits(cls: DataType) -> int:
        """The number of bits a variable of this type holds."""
        return cls.encoding().nbits()

    @classmethod
    def base_type(cls: DataType) -> type:
        """The Python type of this type's values, `object` where they are of several.

        Informative only: which values a type holds is decided by its encoding.
        """
        return cls.encoding().base_type()

    @classmethod
    def _value_table(cls: DataType) -> object:
        """What decides the value of each bit pattern, equal for two types exactly where
        they decode every pattern alike; `reads_as` compares it.

        The default is the table of `decoded`; a type whose table follows from fewer
        parameters returns those, so that a wide type is never enumerated.
        """
        return cls.encoding().decoded()

    def to_be_used_within(self) -> DPP:
        """The DPP whose definition this variable belongs to."""
        return self.__to_be_used_within

    def bits(self) -> tuple[Bool, ...]:
        """The `Bool`s holding this variable's bits, MSB first."""
        return self.__bits

    def __str__(self) -> str:
        """The variable's name where it has one, and otherwise ``w`` followed by the
        number of each of its bits in the definition, joined by underscores."""
        if self.name is not None:
            return self.name
        return "_".join(f"w{b.wire}" for b in self.bits())

    @classmethod
    def const(cls, value) -> Self:
        """A variable of this type fixed to `value`, which must be one of its values.

        Comparisons, `+`, `match` and calls of a DPP accept a value where a variable is
        expected and build the constant themselves, so `const` is needed where a fixed
        value is itself the variable: returned from a DPP, given to `pylea.syntax.tuple.Tuple.of`,
        or one of the choices of `pylea.syntax.choice.pick`.

            >>> from pylea import DPP, Enum, probability_mass_function
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> @DPP
            ... def always_red() -> Colour:
            ...     return Colour.const("R")
            >>> probability_mass_function(always_red)
            {'R': 1.0, 'G': 0.0, 'B': 0.0}

        A value the type does not hold raises `ValueError`, or `TypeError` where it is not
        even of the right kind, such as a float for a `pylea.syntax.integer.UInt`.  Must be
        called inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        require_definition(f"{cls.__name__}.const({value!r})")
        encoded = cls.encoding()(value)
        return Primitive(
            input_types=tuple(),
            output_types=(cls,),
            name=f"_CONST_{cls.__name__}_",
            prim=Prim.const(cls._pattern(encoded), cls.nbits()),
            label=str(encoded.value),
        )()

    @staticmethod
    def _pattern(encoded) -> int:
        """The bit pattern of `encoded`, read as an unsigned integer, MSB first."""
        bits = encoded.to_bits()
        return sum(int(b) << (len(bits) - 1 - i) for i, b in enumerate(bits))

    @classmethod
    def uniform_bits(cls) -> Self:
        """A variable of this type whose bit pattern is uniformly distributed.

        Every one of the ``2**nbits()`` patterns is equally likely, including any decoding
        to no value.  `pylea.syntax.enum.Enum.uniform` draws over the elements instead, and
        the two differ exactly where the element count is not a power of two, or is 1:

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

        require_definition(f"{cls.__name__}.uniform_bits()")
        return Primitive(
            input_types=tuple(),
            output_types=(cls,),
            name="_UNIF_BITS_" + cls.__name__,
            prim=Prim.uniform(cls.nbits()),
        )()

    __hash__ = object.__hash__  # type: ignore

    def _coerced(self, other, operation: str = "combine") -> Self:
        """`other` as a variable of this type, for the operation named `operation`.

        A variable is accepted where one of the two types reads as the other, as
        `reads_as` decides: Python hands a comparison to the right operand first where its
        type is a subclass of the left one's, so the rule must not depend on the order.  A
        value becomes a constant through `const`.  Every message names the operation and
        both operands.
        """
        t = type(self)
        if isinstance(other, Variable):
            found = type(other)
            if found is t:
                return other  # type: ignore[return-value]
            if reads_as(found, t) or reads_as(t, found):
                # Read as this type, so that both operands of the operation agree.
                return t.make(*other.bits())  # type: ignore[return-value]
            hint = ""
            if getattr(t, "_made_by_count", False) or getattr(found, "_made_by_count", False):
                hint = "  The type of a count is set by the data_type argument of count."
            raise TypeError(
                f"Cannot {operation} a {t.__name__} variable and a {found.__name__} "
                f"variable; both must be {t.__name__} variables, or one a value of "
                f"{t.__name__}.{hint}")
        try:
            t.encoding()(other)
        except (TypeError, ValueError) as exc:
            raise _prefixed(
                exc, f"Cannot {operation} a {t.__name__} variable and {other!r}") from None
        return t.const(other)

    def _fixed_order(self, other) -> int | None:
        """-1 where `other` is a value below every value of this type, 1 where it is above
        every one, and `None` otherwise, in which case the comparison is recorded.

        Every type answers `None`; `pylea.syntax.integer.UInt` answers for integers
        outside its range, against which a comparison is a constant.
        """
        return None

    def _compare(self, kind: str, a: Variable, b: Variable) -> Bool:
        """``a == b`` or ``a < b`` (`kind` "EQ" or "LT"), both of this type, as recorded."""
        from pylea.syntax.dpp import Primitive, Bool as BoolType
        t = type(self)
        comparator = Primitive(
            input_types=(t, t),
            output_types=(BoolType,),
            name=f"_{kind}_{t.__name__}_",
            source=_comparator_circuit(kind, t.nbits()),
        )
        return comparator(a, b)  # type: ignore[return-value]

    def _ordered(self, other, holds_if_above: bool) -> Bool | None:
        """The constant answer where `other` lies outside this type's range, else `None`.

        `holds_if_above` is the answer where `other` is above every value; below every
        value the answer is the opposite.
        """
        fixed = self._fixed_order(other)
        if fixed is None:
            return None
        from pylea.syntax.dpp import Bool as BoolType
        return BoolType.const(holds_if_above if fixed > 0 else not holds_if_above)

    def __eq__(self, other: object) -> Bool:  # type: ignore[override]
        """The event that `self` and `other` are equal, inside a DPP definition.

        `other` may be a variable of the same type or one of its values, which goes
        through `const`.  Outside a definition this is ordinary identity, so that
        variables can be put in sets and dicts and tested for membership.

        Inside a definition, `in`, `list.index` and chained comparisons such as
        ``a <= x <= b`` ask Python for the truth value of a `Bool`, which has none while
        the DPP is defined, and raise `TypeError`.  Membership is written
        ``Bool.any(x == v for v in values)``, and a chained comparison as
        ``(a <= x) & (x <= b)``:

            >>> from pylea import DPP, Bool, Enum, prob
            >>> class Digit(Enum):
            ...     elements = [0, 1, 2, 3]
            >>> @DPP
            ... def odd() -> Bool:
            ...     d = Digit.uniform()
            ...     return Bool.any(d == v for v in (1, 3))
            >>> prob(odd)
            0.5
        """
        from pylea.syntax.dpp import active_definition, Bool as BoolType
        if active_definition() is None:
            return self is other  # type: ignore
        if self._fixed_order(other) is not None:
            return BoolType.const(False)
        return self._compare("EQ", self, self._coerced(other, "compare"))

    def __lt__(self, other) -> Bool:
        """The event that `self` is strictly below `other`, comparing bit patterns.

        Both bit patterns are read as unsigned integers, MSB first.  A subclass wanting a
        different ordering, such as a signed integer type, overrides this; `<=`, `>` and
        `>=` are derived from it.
        """
        fixed = self._ordered(other, holds_if_above=True)
        if fixed is not None:
            return fixed
        return self._compare("LT", self, self._coerced(other, "compare"))

    def __le__(self, other) -> Bool:
        """The event that `self` is at most `other`, that is, not ``other < self``."""
        fixed = self._ordered(other, holds_if_above=True)
        if fixed is not None:
            return fixed
        return ~(self._coerced(other, "compare") < self)

    def __ne__(self, other: object) -> Bool:  # type: ignore[override]
        """The event that `self` and `other` differ, inside a DPP definition.

        Outside a definition this is ordinary identity, as `__eq__` is.
        """
        from pylea.syntax.dpp import active_definition, Bool as BoolType
        if active_definition() is None:
            return self is not other  # type: ignore
        if self._fixed_order(other) is not None:
            return BoolType.const(True)
        return ~(self == other)

    def __gt__(self, other) -> Bool:
        """The event that `self` is strictly above `other`, that is, ``other < self``."""
        fixed = self._ordered(other, holds_if_above=False)
        if fixed is not None:
            return fixed
        return self._coerced(other, "compare") < self

    def __ge__(self, other) -> Bool:
        """The event that `self` is at least `other`, that is, not ``self < other``."""
        fixed = self._ordered(other, holds_if_above=False)
        if fixed is not None:
            return fixed
        return ~(self < other)

type DataType = type[Variable]
