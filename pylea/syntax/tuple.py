"""Product types.

A `Tuple` subclass declares its component types in the class variable `types`.  The encoding
is the components' encodings concatenated in declaration order, so the width is their sum.
`Tuple.of` builds a value from one variable per component, and `Tuple.components` takes one
apart again; neither needs the caller to know how the components are packed into bits.

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
"""
from __future__ import annotations
from pylea.syntax.dpp import Bool
from pylea.syntax.encoding import Encoding, BitEncoded
from pylea.syntax.variable import Variable, DataType
from typing import ClassVar, Self


class Tuple(Variable):
    """A fixed-length tuple of variables, each of its own type.

    A subclass declares its component types by setting the class variable `types`.  The
    encoding is the components' encodings concatenated in order, so `nbits` is their sum.

        >>> from pylea import Enum
        >>> class Coin(Enum):
        ...     elements = ["H", "T"]
        >>> class Colour(Enum):
        ...     elements = ["R", "G", "B"]
        >>> class CoinAndColour(Tuple):
        ...     types = (Coin, Colour)
        >>> CoinAndColour.nbits()
        3

    Instantiating a subclass that has not set `types` raises `NotImplementedError`.
    """

    types: tuple[DataType, ...] | None = None
    # The built encoding, stored on the class that built it and read from its own
    # namespace only, so a subclass declaring its own parameters never inherits it.
    _encoding: ClassVar[Encoding | None] = None

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        if type(self).types is None:
            raise NotImplementedError(
                f"Cannot instantiate subclass \"{type(self).__name__}\" of Tuple; "
                "class variable `types` unspecified."
            )
        else:
            super().__init__(*xs, name=name)

    @classmethod
    def of(cls, *components: Variable) -> Self:
        """This tuple's value, built from one variable per component, in `types` order.

        `TypeError` if a component is not of the matching `types` entry, `ValueError` if
        the count is wrong.  Must be called inside an active DPP definition.

            >>> from pylea import DPP, Enum
            >>> class Coin(Enum):
            ...     elements = ["H", "T"]
            >>> class CoinPair(Tuple):
            ...     types = (Coin, Coin)
            >>> @DPP
            ... def both_heads() -> CoinPair:
            ...     return CoinPair.of(Coin.const("H"), Coin.const("H"))
            >>> both_heads.distribution().tolist()
            [1.0, 0.0, 0.0, 0.0]
        """
        assert cls.types is not None
        if len(components) != len(cls.types):
            raise ValueError(
                f"{cls.__name__} has {len(cls.types)} components; got {len(components)}.")
        for i, (x, t) in enumerate(zip(components, cls.types)):
            if type(x) is not t:
                raise TypeError(
                    f"{cls.__name__} component {i} must be of type {t.__name__}; "
                    f"got {type(x).__name__}.")
        return cls.make(*(b for x in components for b in x.bits()))  # type: ignore[return-value]

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
        types = type(self).types
        assert types is not None
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
        if cls.types is None:
            raise NotImplementedError("Class variable `types` unspecified.")
        else:
            ts = cls.types

        class EncodedTuple(BitEncoded):

            def __init__(self, value: tuple[BitEncoded, ...]) -> None:
                if len(ts) != len(value):
                    raise ValueError(f"Expected tuple of length {len(ts)}; got {value}.")
                for x, t in zip(value, ts):
                    if not isinstance(x, t.encoding()):
                        raise TypeError(
                            f"Expected value of type \"{t.encoding().__name__}\"; got {x}.")
                self._value = value

            @property
            def value(self) -> tuple:
                return self._value

            @classmethod
            def nbits(cls) -> int:
                return sum(t.nbits() for t in ts)

            @classmethod
            def base_type(cls) -> type:
                return tuple

            @classmethod
            def from_bits(cls, *bits: bool) -> EncodedTuple:
                offsets = [0]
                for t in ts:
                    offsets.append(offsets[-1] + t.nbits())
                return EncodedTuple(
                    tuple(
                        t.encoding().from_bits(*bits[offsets[i]:offsets[i + 1]])
                        for i, t in enumerate(ts)
                    )
                )

            def to_bits(self: EncodedTuple) -> tuple[bool, ...]:
                return tuple(
                    bit for i, _ in enumerate(ts)
                    for bit in self.value[i].to_bits()
                )

        cls._encoding = EncodedTuple
        return EncodedTuple
