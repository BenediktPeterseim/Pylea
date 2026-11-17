"""Bit encodings.

A `BitEncoded` subclass gives each value of a type a pattern of `nbits` bits and reads that
pattern back.  Every `pylea.syntax.variable.Variable` subclass declares one, and that is
what connects the variable types to the bits beneath them.

Not every one of the ``2**nbits()`` patterns need encode a value.  The contract every
subclass keeps: `from_bits` raises `ValueError` for a pattern that encodes no value and is a
bijection from the remaining patterns onto the values; `to_bits` is its inverse; `n_values`
counts those patterns, and `decoded` maps each of them, read as an integer, to its value.
Constructing an instance from something that is not a value raises `ValueError`, or
`TypeError` where it is not even of the right kind.
"""
from abc import abstractmethod, ABC
from typing import Any, ClassVar, Self


class BitEncoded(ABC):
    """A fixed-width bit representation of the values of a variable type.

    A subclass gives each value, such as a `bool`, an element of an `Enum` or a tuple, a
    bit pattern of length `nbits`, and reads that pattern back.  Every
    `pylea.syntax.variable.Variable` subclass declares its encoding, and `from_bits` and
    `to_bits` convert between its values and their bits.  The module docstring states the
    contract a subclass keeps.
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
        """How many bit patterns encode a value.

        ``2**nbits()`` unless a subclass overrides it, which one must where not every
        pattern encodes a value.

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