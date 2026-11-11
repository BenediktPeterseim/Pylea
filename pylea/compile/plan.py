"""Plans: the steps the evaluator runs.

A `Plan` is a list of `Step`s in SSA form: step `i` produces value `i`, whose logical shape
is ``shapes[i]``, and `result` is the value that reshapes to the semantics of the body the
plan came from, a ``(2**out_bits, 2**in_bits)`` matrix.

Values are dense arrays.  `Contract` is one batched matrix multiplication: it permutes each
operand so that the axes of the first read ``batch, left, sum`` and those of the second
``batch, sum, right``, and produces ``batch, left, right``.  Every other step is a
permutation, a sum, an index selection, or a move onto or off a diagonal.

`sliced_dims` holds the sizes of the indices the plan iterates over instead of holding.  The
evaluator runs the plan once per combination of their values and adds the results, and
``varying[i]`` marks the steps whose value depends on those values; the rest are computed
once.  `contracted` is the value those results are summed into.  The steps after it reshape
that sum, and being linear they may be applied to the sum rather than to each slice.
"""
from __future__ import annotations

from math import prod

from dataclasses import dataclass

from pylea.compile.network import Leaf


@dataclass(frozen=True, slots=True)
class Load:
    """The array of leaf `leaf`, with one axis per index."""

    leaf: int


@dataclass(frozen=True, slots=True)
class Select:
    """Value `src` restricted to one position of `axis`, chosen by sliced index `which`."""

    src: int
    axis: int
    which: int


@dataclass(frozen=True, slots=True)
class Reduce:
    """Value `src` summed over `axes`."""

    src: int
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Diagonal:
    """Value `src` where the axes of each group agree, keeping one axis per group.

    ``groups[k]`` are the axes of `src` that the k-th result axis ties together, so a group
    of one is an axis carried through unchanged.  The diagonal is read rather than built, so
    the work is the size of the result, which is no larger than `src`.
    """

    src: int
    groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Spread:
    """Value `src` placed on the diagonal of an array of shape `shape`, zero elsewhere.

    ``positions[k]`` are the axes of `shape` that the k-th axis of `src` fills together.
    This is how a body returning one wire twice, or returning an argument, gets its
    semantics: the contraction runs over the distinct wires and only the result is ever this
    wide.
    """

    src: int
    positions: tuple[tuple[int, ...], ...]
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Transpose:
    """Value `src` with its axes reordered by `perm`."""

    src: int
    perm: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Contract:
    """One batched matrix multiplication of values `a` and `b`.

    `perm_a` brings `a`'s axes into ``batch, left, sum`` order and `perm_b` brings `b`'s
    into ``batch, sum, right``; the four fields are the sizes of those blocks.  The result
    has axes ``batch, left, right``.
    """

    a: int
    b: int
    perm_a: tuple[int, ...]
    perm_b: tuple[int, ...]
    batch: int
    left: int
    sum: int
    right: int


Step = Load | Select | Reduce | Diagonal | Spread | Transpose | Contract


@dataclass(frozen=True, slots=True)
class Plan:
    """Steps in SSA form, together with where the slice sum ends.

    `contracted` is the value the slices are summed into.  The steps after it reshape that
    sum into the matrix, being a transpose and, where the output names a wire twice, a
    spread.  Both are linear, so they commute with the sum and run once rather than once
    per slice.
    """

    leaves: tuple[Leaf, ...]
    steps: tuple[Step, ...]
    shapes: tuple[tuple[int, ...], ...]
    varying: tuple[bool, ...]
    sliced_dims: tuple[int, ...]
    contracted: int
    result: int
    out_bits: int
    in_bits: int

    @property
    def n_slices(self) -> int:
        n = 1
        for d in self.sliced_dims:
            n *= d
        return n

    def peak_entries(self) -> int:
        """The largest number of array entries a single value holds.

            >>> from pylea import DPP, Bool, flip
            >>> from pylea.compile.lower import compile_program
            >>> @DPP
            ... def both() -> Bool:
            ...     return flip(0.5) & flip(0.25)
            >>> plans, root = compile_program(both.program())
            >>> plans[root].n_slices, plans[root].peak_entries()
            (1, 8)
        """
        return max((prod(s) for s in self.shapes), default=1)

    def subplans(self) -> tuple[int, ...]:
        """The body indices this plan loads as matrices, each listed once."""
        from pylea.compile.network import Tensor
        seen: dict[int, None] = {}
        for leaf in self.leaves:
            if isinstance(leaf, Tensor) and isinstance(leaf.callee, int):
                seen[leaf.callee] = None
        return tuple(seen)

