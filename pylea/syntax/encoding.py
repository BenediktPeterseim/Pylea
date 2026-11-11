"""Bit encodings.

A `BitEncoded` subclass gives each value of its `base_type` a pattern of `nbits` bits and
reads that pattern back.  Every `pylea.syntax.variable.Variable` subclass declares one, and
that is what connects the DSL to the wires beneath it.

Not every one of the ``2**nbits()`` patterns need be valid, and neither need every value of
the base type.  `n_values` says how many patterns are, and a subclass must make `from_bits`
a bijection from the first `n_values` patterns, read as integers, onto the valid values,
with `to_bits` its inverse.  `decoded` returns that map from pattern to value.
"""
from abc import abstractmethod, ABC
from typing import Any, ClassVar, Self


class BitEncoded(ABC):
    """A fixed-width bit representation of the values of some Python type.

    A subclass gives each value of its `base_type`, such as a `bool` or a string element,
    a bit pattern of length `nbits`, and reads that pattern back.  This is the bridge
    between the DSL and the bit level: every `pylea.syntax.variable.Variable` subclass
    declares its encoding, and `nbits`, `from_bits` and `to_bits` translate across.

    Not every one of the ``2**nbits()`` patterns need be valid, and neither need every
    value of the base type.  `n_values` reports how many patterns are valid, and a
    subclass must make `from_bits` a bijection from the first `n_values` patterns, read as
    integers, onto the valid values, with `to_bits` its inverse.
    """

    _decoded: ClassVar[dict[int, object] | None] = None

    @abstractmethod
    def __init__(self, value: Any) -> None:
        """An instance holding `value`, which every subclass is built from; a subclass
        raises where it has no bit pattern for `value`."""

    @classmethod
    @abstractmethod
    def nbits(cls) -> int:
        """The width of this encoding, in bits."""

    @classmethod
    def n_values(cls) -> int:
        """How many bit patterns this encoding treats as valid.

        ``2**nbits()`` unless a subclass overrides it, which one must where not every
        pattern is valid.  See the class docstring for the bijection to maintain.

            >>> from pylea import Enum
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> Colour.encoding().nbits(), Colour.encoding().n_values()
            (2, 3)
        """
        return 2 ** cls.nbits()

    @classmethod
    def decoded(cls) -> dict[int, object]:
        """The value each bit pattern encodes, keyed by the pattern read as an unsigned
        integer, MSB first.

        A pattern that encodes no value is left out:

            >>> from pylea import Enum
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> Colour.encoding().decoded()
            {0: 'R', 1: 'G', 2: 'B'}

        The default reads every pattern through `from_bits` and holds the result; a
        subclass whose values are already in pattern order overrides it.
        """
        # Held on the exact class, so a subclass does not inherit its parent's table, and
        # copied out, so the caller cannot alter what the next call returns.
        table = cls.__dict__.get("_decoded")
        if table is None:
            n = cls.nbits()
            table = {}
            for pattern in range(1 << n):
                bits = tuple(bool((pattern >> (n - 1 - i)) & 1) for i in range(n))
                try:
                    table[pattern] = cls.from_bits(*bits).value
                except ValueError:
                    continue
            cls._decoded = table
        return dict(table)

    @classmethod
    @abstractmethod
    def base_type(cls) -> type:
        """The Python type of the decoded values, such as `bool` or `str`."""

    @classmethod
    @abstractmethod
    def from_bits(cls, *bits: bool) -> Self:
        """The instance `bits` encodes, MSB first."""

    @abstractmethod
    def to_bits(self: Self) -> tuple[bool, ...]:
        """The `nbits()` bits encoding `self`, MSB first."""

    @property
    @abstractmethod
    def value(self) -> object:
        """The base-type value this instance holds."""
        ...

type Encoding = type[BitEncoded]