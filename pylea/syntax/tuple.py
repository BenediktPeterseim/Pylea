"""Product types.

A `Tuple` subclass declares its component types in the class variable `types`.  The encoding
is the components' encodings concatenated in declaration order, so the width is their sum,
and a value is a plain Python tuple of one value per component.  `Tuple.of` builds a
variable from one variable or value per component, and `Tuple.components` takes one apart
again; neither needs the caller to know how the components are packed into bits.

    >>> from pylea import DPP, Enum
    >>> class Coin(Enum):
    ...     elements = ["H", "T"]
    >>> class Colour(Enum):
    ...     elements = ["R", "G", "B"]
    >>> class CoinAndColour(Tuple):
    ...     types = (Coin, Colour)
    >>> CoinAndColour.nbits()
    3
    >>> @DPP
    ... def colour_of(pair: CoinAndColour) -> Colour:
    ...     _coin, colour = pair.components()
    ...     return colour
    >>> @DPP
    ... def green() -> Colour:
    ...     return colour_of(CoinAndColour.of(Coin.const("H"), Colour.const("G")))
    >>> green.distribution().tolist()
    [0.0, 1.0, 0.0, 0.0]

A pattern encodes a value where each component's part encodes one, so `n_values` is the
product of the components' counts.
"""
from __future__ import annotations
import math
from pylea.syntax.dpp import Bool
from pylea.syntax.encoding import Encoding, BitEncoded
from pylea.syntax.variable import (
    Variable, DataType, as_variable, require_variable_type, _prefixed,
)
from typing import ClassVar, Self


def _require_types(cls, what: str) -> None:
    """Nothing, or `NotImplementedError` where `cls` declares no `types`."""
    if cls.types is None:
        raise NotImplementedError(
            f"{what}: {cls.__name__} declares no types; set its class variable `types` in "
            "a subclass")


class Tuple(Variable):
    """A fixed-length tuple of variables, each of its own type.

    A subclass declares its component types by setting the class variable `types` to a
    non-empty tuple or list of variable types, checked when the class is created.  The
    encoding is the components' encodings concatenated in order, so `nbits` is their sum,
    and a value is a plain tuple, which `const` and `==` take:

        >>> from pylea import DPP, Bool, Enum, prob
        >>> class Rank(Enum):
        ...     elements = ["A", "K"]
        >>> class Suit(Enum):
        ...     elements = ["red", "black", "green"]
        >>> class Card(Tuple):
        ...     types = (Rank, Suit)
        >>> @DPP
        ... def red_ace() -> Bool:
        ...     return Card.of(Rank.uniform(), Suit.uniform()) == ("A", "red")
        >>> round(prob(red_ace), 6)
        0.166667
        >>> @DPP
        ... def fixed() -> Card:
        ...     return Card.const(("K", "green"))
        >>> fixed.distribution().tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    Using a subclass that has not set `types` raises `NotImplementedError`.
    """

    types: ClassVar[tuple[DataType, ...] | None] = None
    # The built encoding, stored on the class that built it and read from its own
    # namespace only, so a subclass declaring its own parameters never inherits it.
    _encoding: ClassVar[Encoding | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        types = vars(cls).get("types")
        if types is None:
            return
        if isinstance(types, type):
            raise TypeError(f"{cls.__name__}.types must be a tuple of types; for one "
                            f"component write types = ({types.__name__},)")
        if not isinstance(types, (tuple, list)):
            raise TypeError(f"{cls.__name__}.types must be a tuple of types; got a "
                            f"{type(types).__name__}")
        if not types:
            raise ValueError(f"{cls.__name__}.types is empty; a Tuple needs at least one "
                             "component")
        for i, t in enumerate(types):
            require_variable_type(t, f"{cls.__name__}.types[{i}]")
        cls.types = tuple(types)

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        _require_types(type(self), f"Cannot create a {type(self).__name__} variable")
        super().__init__(*xs, name=name)

    @classmethod
    def _value_table(cls) -> object:
        _require_types(cls, cls.__name__)
        return tuple((t.nbits(), t._value_table()) for t in cls.types)  # type: ignore[union-attr]

    @classmethod
    def of(cls, *components) -> Self:
        """This tuple's value, built from one variable or value per component, in `types`
        order.

        Each component is accepted as `pylea.syntax.variable.as_variable` accepts a value
        for its type, so a plain value becomes a constant.  `TypeError` if a component is
        a variable of another type, `ValueError` if the count is wrong or a value is not
        one of its type's.  Must be called inside an active DPP definition.

            >>> from pylea import DPP, Enum
            >>> class Coin(Enum):
            ...     elements = ["H", "T"]
            >>> class CoinPair(Tuple):
            ...     types = (Coin, Coin)
            >>> @DPP
            ... def both_heads() -> CoinPair:
            ...     return CoinPair.of(Coin.const("H"), "H")
            >>> both_heads.distribution().tolist()
            [1.0, 0.0, 0.0, 0.0]
        """
        _require_types(cls, f"{cls.__name__}.of")
        types: tuple[DataType, ...] = cls.types  # type: ignore[assignment]
        if len(components) != len(types):
            raise ValueError(
                f"{cls.__name__} has {len(types)} components; got {len(components)}.")
        parts = [as_variable(x, t, f"{cls.__name__}.of: component {i}")
                 for i, (x, t) in enumerate(zip(components, types))]
        return cls.make(*(b for x in parts for b in x.bits()))  # type: ignore[return-value]

    def components(self) -> tuple[Variable, ...]:
        """This tuple's value, taken apart into one variable per component, in `types` order.

            >>> from pylea import DPP, Enum
            >>> class Coin(Enum):
            ...     elements = ["H", "T"]
            >>> class CoinPair(Tuple):
            ...     types = (Coin, Coin)
            >>> @DPP
            ... def first_of(pair: CoinPair) -> Coin:
            ...     head, _tail = pair.components()
            ...     return head
            >>> @DPP
            ... def first_when_heads_tails() -> Coin:
            ...     return first_of(CoinPair.of(Coin.const("H"), Coin.const("T")))
            >>> first_when_heads_tails.distribution().tolist()
            [1.0, 0.0]
        """
        types: tuple[DataType, ...] = type(self).types  # type: ignore[assignment]
        at = 0
        parts: list[Variable] = []
        for t in types:
            n = t.nbits()
            parts.append(t.make(*self.bits()[at:at + n]))
            at += n
        return tuple(parts)

    @classmethod
    def encoding(cls) -> Encoding:
        cached = vars(cls).get("_encoding")
        if cached is not None:
            return cached
        _require_types(cls, f"{cls.__name__}.encoding")
        ts: tuple[DataType, ...] = cls.types  # type: ignore[assignment]
        name = cls.__name__
        widths = [t.nbits() for t in ts]
        offsets = [sum(widths[:i]) for i in range(len(ts) + 1)]

        class EncodedTuple(BitEncoded):
            """A value of a `Tuple`, a plain tuple encoded component after component."""

            def __init__(self, value: tuple) -> None:
                if not isinstance(value, tuple):
                    raise TypeError(f"a value of {name} is a tuple of {len(ts)} values, one "
                                    f"per component; got {value!r}")
                if len(value) != len(ts):
                    raise ValueError(f"a value of {name} has {len(ts)} components; got "
                                     f"{value!r}")
                parts = []
                for i, (v, t) in enumerate(zip(value, ts)):
                    try:
                        parts.append(t.encoding()(v))
                    except (TypeError, ValueError) as exc:
                        raise _prefixed(exc, f"{name} component {i}") from None
                self._parts = tuple(parts)

            @property
            def value(self) -> tuple:
                return tuple(p.value for p in self._parts)

            @classmethod
            def nbits(cls) -> int:
                return offsets[-1]

            @classmethod
            def n_values(cls) -> int:
                return math.prod(t.encoding().n_values() for t in ts)

            @classmethod
            def base_type(cls) -> type:
                return tuple

            @classmethod
            def from_bits(cls, *bits: bool) -> EncodedTuple:
                if len(bits) != offsets[-1]:
                    raise ValueError(f"Expected {offsets[-1]} bits; got {len(bits)}.")
                return EncodedTuple(tuple(
                    t.encoding().from_bits(*bits[offsets[i]:offsets[i + 1]]).value
                    for i, t in enumerate(ts)))

            def to_bits(self: EncodedTuple) -> tuple[bool, ...]:
                return tuple(bit for p in self._parts for bit in p.to_bits())

        cls._encoding = EncodedTuple
        return EncodedTuple
