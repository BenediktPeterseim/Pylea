"""The tensor network of a body.

`build_network` turns body `i` of a DPP into a `Network`: one tensor per op, whose axes
are the op's output wires followed by its argument wires, and one index per wire, of size
``2**bits[w]``.  Contracting the network down to the body's outputs followed by its inputs
gives the semantics of the body as a ``(2**out_bits, 2**in_bits)`` matrix.

A wire read by several ops is one index carried by several tensors.  A wire occurring twice
on one tensor, as in ``f(x, x)``, is one index carried twice by that tensor, which the plan
resolves by taking the diagonal of the leaf as it is built, a contraction step being able to
multiply two arrays but not also to take a diagonal.

A wire that only one tensor carries and that is not part of the interface can only be summed
away, so it is summed as the leaf is formed and never reaches the order finder, which also
keeps cotengra's slicer from being handed an index it cannot slice.

An interface wire that no op carries gets a tensor of ones, so that every index of the
result is present in the network.  A body with no ops and no interface leaves a single
scalar tensor of ones.

The output axes may name a wire twice, which happens for a body returning one wire twice or
returning an argument.  The contraction runs over the distinct wires and the plan spreads
the result onto the diagonal at the end, so only the result is ever that wide.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pylea.ir.prim import Prim
from pylea.ir.program import Program


@dataclass(frozen=True, slots=True)
class Tensor:
    """One op's matrix, reshaped to one axis per wire: its outputs, then its arguments.

    `reduce` names the axes to sum out as the leaf is formed, those being the op's wires
    that nothing else in the body reads and that are outside its interface.  They are summed
    here rather than left to the contraction, since no contraction tree can do anything but
    sum them.
    """

    callee: Prim | int
    shape: tuple[int, ...]
    reduce: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Ones:
    """The constant-one tensor of shape `dims`."""

    dims: tuple[int, ...]


Leaf = Tensor | Ones


@dataclass(frozen=True, slots=True)
class Network:
    """Tensors, the output axes, the index sizes, and how to form each tensor.

    `axes[t]` are the labels of tensor `t` in the order its array is stored, once its dead
    axes are summed.  A label may repeat there, which is a diagonal.  `output_axes` are the
    result's axes and may also repeat; `output` is the distinct ones in order of first
    occurrence.

    Every label either occurs on at least two tensors or is free, so a contraction tree can
    sum each one out where its last occurrence is consumed.
    """

    axes: tuple[tuple[int, ...], ...]
    output_axes: tuple[int, ...]
    output: tuple[int, ...]
    size: dict[int, int]
    leaves: tuple[Leaf, ...]

    @property
    def n_tensors(self) -> int:
        return len(self.leaves)


def distinct(labels) -> tuple[int, ...]:
    """`labels` with later repeats removed, order preserved.

        >>> distinct((3, 1, 3, 2, 1))
        (3, 1, 2)
    """
    return tuple(dict.fromkeys(labels))


def build_network(prog: Program, body: int) -> Network:
    """The network of body `body`, whose output axes are its outputs then its inputs.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> net = build_network(both.program(), 0)
        >>> net.n_tensors, net.output
        (3, (2,))
    """
    b = prog.bodies[body]
    size: dict[int, int] = {w: 1 << b.bits[w] for w in range(len(b.bits))}
    axes: list[tuple[int, ...]] = []
    leaves: list[Leaf] = []

    for op in b.ops:
        wires = op.outs + op.args
        axes.append(wires)
        leaves.append(Tensor(op.callee, tuple(size[w] for w in wires)))

    output_axes = b.outputs + b.inputs
    free = set(output_axes)

    # A wire read nowhere else and outside the interface can only be summed, so it is summed
    # as the leaf is formed and never reaches the order finder.  cotengra's slicer fails on
    # an index that only one tensor carries.
    occurrences: Counter = Counter(ix for t in axes for ix in set(t))
    for t, leaf in enumerate(leaves):
        dead = tuple(ax for ax, ix in enumerate(axes[t])
                     if occurrences[ix] == 1 and ix not in free)
        if dead:
            assert isinstance(leaf, Tensor)
            leaves[t] = Tensor(leaf.callee, leaf.shape, dead)
            axes[t] = tuple(ix for ax, ix in enumerate(axes[t]) if ax not in set(dead))

    incident = {ix for t in axes for ix in t}
    for ix in distinct(output_axes):
        if ix not in incident:
            axes.append((ix,))
            leaves.append(Ones((size[ix],)))
            incident.add(ix)

    if not axes:                         # a body with no ops and no interface is just 1
        axes.append(())
        leaves.append(Ones(()))

    return Network(tuple(axes), output_axes, distinct(output_axes), size, tuple(leaves))
