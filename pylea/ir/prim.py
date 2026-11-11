"""Primitives: the leaves of the IR.

A `Prim` is a `Tag` together with a non-negative matrix of shape
``(2**out_bits, 2**in_bits)``, whose entry at ``[o, i]`` is the weight of output pattern `o`
given input pattern `i`.  Build one with a classmethod rather than with the constructor:

    >>> Prim.flip(0.25).matrix.tolist()
    [[0.75], [0.25]]
    >>> Prim.and_().matrix.tolist()
    [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

Every tag but `MATRIX` derives its matrix from the tag and `params`, and shares one array
between equal primitives.  `MATRIX` carries its own, and is identified by a hash of the
bytes, so two equal arrays are interchangeable.

Two properties of the matrix decide what `pylea.simplify` may do with an op.  Both are read
off the array for `MATRIX` and taken from the tag otherwise:

  * `is_deterministic`: every column is 0/1 with at most one 1, so the output is fixed by
    the input wherever the matrix is non-zero.  A zero column excludes an input rather than
    mapping it, which leaves the property intact.
  * `is_discardable`: every column sums to 1, so summing out every output leaves 1.

Neither implies the other:

    >>> Prim.equate(1).is_deterministic, Prim.equate(1).is_discardable
    (True, False)
    >>> Prim.flip(0.25).is_deterministic, Prim.flip(0.25).is_discardable
    (False, True)

There are no `id`, `copy` or `delete` primitives.  In a `pylea.ir.body.Body`, fan-out is a
wire read by several ops, discard is a wire absent from the outputs, and identity is one
wire named twice.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from pylea.semantics import matrices


class Tag(Enum):
    """Which matrix a `Prim` stands for.  `MATRIX` is opaque; the rest are derived."""

    AND = auto()
    OR = auto()
    NOT = auto()
    CONST = auto()          # deterministic 0 -> k
    FLIP = auto()           # 0 -> 1, Bernoulli(p)
    UNIFORM = auto()        # 0 -> k, uniform
    MATRIX = auto()         # an opaque matrix: a CPT, or a table a caller supplied
    OBSERVE = auto()        # 1 -> 0, conditioning
    EQUATE = auto()         # 2k -> k, conditioning on equality
    ITE = auto()            # (1 + 2k) -> k


STOCHASTIC = frozenset({Tag.FLIP, Tag.UNIFORM})
CONDITIONING = frozenset({Tag.OBSERVE, Tag.EQUATE})
BOOLEAN = frozenset({Tag.AND, Tag.OR, Tag.NOT, Tag.CONST})


def _digest(arr: np.ndarray) -> bytes:
    """A content hash of `arr`, serving as the identity of a `MATRIX` primitive."""
    a = np.ascontiguousarray(arr, dtype=float)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.digest()


@dataclass(frozen=True, eq=False)
class Prim:
    """A primitive operation and the matrix it stands for.

    The constructors fix the arity, in bits, of what the op reads and writes:

        >>> [(p.in_bits, p.out_bits) for p in (Prim.and_(), Prim.not_(), Prim.observe(),
        ...                                    Prim.equate(2), Prim.uniform(3))]
        [(2, 1), (1, 1), (1, 0), (4, 2), (0, 3)]
    """

    tag: Tag
    in_bits: int
    out_bits: int
    params: tuple = ()
    # Set only for MATRIX; other tags derive their matrix from `tag` and `params`.
    data: np.ndarray | None = field(default=None, repr=False)

    @property
    def matrix(self) -> np.ndarray:
        """The ``(2**out_bits, 2**in_bits)`` matrix this primitive stands for."""
        if self.data is not None:
            return self.data
        return _DERIVED[self.tag](self)

    def _key(self) -> tuple:
        """The value this primitive is equal and hashed by."""
        # A derived primitive is determined by its tag and params; MATRIX needs its content.
        # The digest is computed on demand, since hashing a large CPT is not free.
        if self.tag is not Tag.MATRIX:
            return (self.tag, self.in_bits, self.out_bits, self.params)
        assert self.data is not None            # MATRIX always carries its array
        d = getattr(self, "_digest_cache", None)
        if d is None:
            d = _digest(self.data)
            object.__setattr__(self, "_digest_cache", d)
        return (self.tag, self.in_bits, self.out_bits, self.params, d)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Prim) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    @property
    def is_deterministic(self) -> bool:
        """Whether every column of the matrix is 0/1 with at most one 1."""
        # Every built-in but FLIP and UNIFORM is a function or a constraint, both of
        # which have 0/1 columns.
        if self.tag in STOCHASTIC:
            return False
        if self.tag is Tag.MATRIX:
            return self._array_property("_det_cache", is_indicator)
        return True

    @property
    def is_discardable(self) -> bool:
        """Whether every column of the matrix sums to 1."""
        # Every built-in but OBSERVE and EQUATE has columns summing to 1.
        if self.tag is Tag.MATRIX:
            return self._array_property("_disc_cache", is_stochastic)
        return self.tag not in CONDITIONING

    def _array_property(self, slot: str, test) -> bool:
        """`test` applied to this primitive's array, computed once and cached.

        A CPT can be large enough that testing it twice is worth avoiding.
        """
        value = getattr(self, slot, None)
        if value is None:
            value = test(self.matrix)
            object.__setattr__(self, slot, value)
        return value

    # -- constructors -------------------------------------------------------------

    @classmethod
    def and_(cls) -> Prim:
        return cls(Tag.AND, 2, 1)

    @classmethod
    def or_(cls) -> Prim:
        return cls(Tag.OR, 2, 1)

    @classmethod
    def not_(cls) -> Prim:
        return cls(Tag.NOT, 1, 1)

    @classmethod
    def observe(cls) -> Prim:
        return cls(Tag.OBSERVE, 1, 0)

    @classmethod
    def equate(cls, k: int) -> Prim:
        return cls(Tag.EQUATE, 2 * k, k, (k,))

    @classmethod
    def ite(cls, k: int) -> Prim:
        return cls(Tag.ITE, 1 + 2 * k, k, (k,))

    @classmethod
    def uniform(cls, k: int) -> Prim:
        return cls(Tag.UNIFORM, 0, k, (k,))

    @classmethod
    def const(cls, value: int, bits: int) -> Prim:
        if not 0 <= value < (1 << bits):
            raise ValueError(f"const {value} does not fit in {bits} bits")
        return cls(Tag.CONST, 0, bits, (value, bits))

    @classmethod
    def flip(cls, p: float) -> Prim:
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"coin flip probability must be in [0, 1]; got {p!r}")
        return cls(Tag.FLIP, 0, 1, (p,))

    @classmethod
    def from_matrix(cls, arr) -> Prim:
        """A primitive standing for `arr`.

        Raises `ValueError` unless `arr` is two-dimensional, non-negative, and has a power
        of two rows and columns, those being the sides of ``(2**out_bits, 2**in_bits)``.

            >>> Prim.from_matrix([[0.9, 0.4], [0.1, 0.6]]).in_bits
            1
        """
        a = np.ascontiguousarray(arr, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"a matrix must be 2-d; got shape {a.shape}")
        rows, cols = a.shape
        out_bits, in_bits = _exact_log2(rows, "rows"), _exact_log2(cols, "columns")
        if a.min() < 0.0:
            raise ValueError("a matrix must be non-negative")
        a.flags.writeable = False
        return cls(Tag.MATRIX, in_bits, out_bits, (), a)

    def __post_init__(self) -> None:
        if self.in_bits < 0 or self.out_bits < 0:
            raise ValueError(f"negative arity in {self!r}")


def _exact_log2(n: int, what: str) -> int:
    k = n.bit_length() - 1
    if n <= 0 or (1 << k) != n:
        raise ValueError(f"number of {what} must be a power of two; got {n}")
    return k


def is_indicator(arr: np.ndarray) -> bool:
    """Whether every column of `arr` is 0/1 with at most one 1.

    `arr` is then a function on the inputs it admits, a zero column being an input it
    excludes.  This is the condition `Prim.is_deterministic` reports.

        >>> is_indicator([[1.0, 0.0], [0.0, 1.0]])
        True
        >>> is_indicator([[0.5, 0.0], [0.5, 1.0]])
        False
    """
    a = np.asarray(arr, dtype=float)
    return bool(np.all((a == 0.0) | (a == 1.0)) and np.all(a.sum(axis=0) <= 1.0))


def constant_value(prim: Prim) -> int | None:
    """The output pattern `prim` always produces, or `None` if it is not a constant.

    Read off the matrix rather than the tag, so a constant that is not tagged as one is
    still found:

        >>> constant_value(Prim.flip(0.0)), constant_value(Prim.flip(0.5))
        (0, None)
        >>> constant_value(Prim.const(3, 2))
        3
    """
    if prim.in_bits:
        return None
    column = np.asarray(prim.matrix, dtype=float)[:, 0]
    (nonzero,) = np.nonzero(column)
    if len(nonzero) == 1 and column[nonzero[0]] == 1.0:
        return int(nonzero[0])
    return None


def is_stochastic(arr: np.ndarray) -> bool:
    """Whether every column of `arr` sums to 1, so that `arr` is a conditional distribution.

    Read up to the round-off of the floating-point arithmetic that built `arr`, and no
    further: deleting an op with this property scales the answer by its columns' departure
    from 1.  A distribution declared in a DPP is normalised where it is declared, by
    `pylea.syntax.enum.Enum.random` and `pylea.syntax.enum.Enum.cpt`, so a table that
    departs from 1 by more is a weight as written and is kept.

        >>> is_stochastic([[0.75], [0.25]])
        True
        >>> is_stochastic([[0.3], [0.700001]])
        False
        >>> is_stochastic([[0.0, 1.0]])
        False
    """
    sums = np.asarray(arr, dtype=float).sum(axis=0)
    return bool(np.all(np.abs(sums - 1.0) <= _ROUND_OFF))


# Largest departure from 1 of a column sum `is_stochastic` reads as round-off.  The largest
# measured was 1.2e-15: every table of the Bayesian-network corpus, the examples and the
# Aralia fault trees, before and after `pylea.simplify.simplify`.  A margin of about 1000
# over that; re-measure if a rule composes tables in a way the corpus does not exercise.
_ROUND_OFF = 1e-12


_DERIVED = {
    Tag.AND: lambda p: matrices.and_(),
    Tag.OR: lambda p: matrices.or_(),
    Tag.NOT: lambda p: matrices.not_(),
    Tag.OBSERVE: lambda p: matrices.observe(),
    Tag.EQUATE: lambda p: matrices.equate(p.params[0]),
    Tag.ITE: lambda p: matrices.if_then_else(p.params[0]),
    Tag.UNIFORM: lambda p: matrices.uniform(p.params[0]),
    Tag.CONST: lambda p: matrices.const(*p.params),
    Tag.FLIP: lambda p: matrices.flip(p.params[0]),
}
