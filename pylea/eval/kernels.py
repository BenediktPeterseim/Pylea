"""Array kernels.

Every value is a one-dimensional array whose logical shape the caller passes in, so a
reshape moves nothing and only `permute` and `contract` touch memory.  Each kernel views the
array at the lowest rank that expresses its work, which keeps it inside the backend's
`max_rank` however many indices the logical shape has: a reduction or a selection needs three
axes whatever the shape, and a permutation needs one axis per block of axes that move
together.

`coalesce` finds those blocks.  Where a permutation still needs more blocks than the backend
allows, `_passes` splits it into several, each placing as long a prefix of the target as
fits, since a pass costs one traverse of memory however many axes it fixes.

`diagonal` and `spread` move on and off a diagonal by gathering and scattering flat indices.
Those indices are computed with numpy whatever the backend, being arithmetic over the plan's
shapes rather than over any matrix's data, and their number is the size of the diagonal
rather than of the array.
"""
from __future__ import annotations

from math import prod

import functools


def coalesce(perm, shape):
    """`perm` as a permutation of the fewest blocks, with the block sizes.

    Axes staying adjacent are merged: a run of target axes that is also a run of source axes
    in the same order moves as one block, at the same element order and a lower rank.

        >>> coalesce((0, 1, 2), (2, 2, 2))
        ([0], [8])
        >>> coalesce((2, 0, 1), (2, 4, 8))
        ([1, 0], [8, 8])
    """
    perm = tuple(perm)
    n = len(perm)
    runs, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and perm[j + 1] == perm[j] + 1:
            j += 1
        runs.append((perm[i], perm[j]))
        i = j + 1
    src = sorted(runs)
    sizes = [prod(shape[s:e + 1]) for (s, e) in src]
    position = {s: k for k, (s, _e) in enumerate(src)}
    return [position[s] for (s, _e) in runs], sizes


@functools.lru_cache(maxsize=4096)
def _passes(shape, perm, max_rank):
    """`perm` as a sequence of permutations of at most `max_rank` contiguous blocks.

    Each costs one pass over memory however many axes it fixes, so the aim is to use as few
    as possible, greedily placing the longest prefix of the target that still fits.
    """
    cur, target = list(range(len(shape))), list(perm)
    plan = []
    while cur != target:
        for k in range(len(target), 0, -1):
            prefix = target[:k]
            nxt = prefix + [a for a in cur if a not in set(prefix)]
            blocks, sizes = coalesce([cur.index(a) for a in nxt], [shape[a] for a in cur])
            if len(blocks) <= max_rank:
                plan.append((tuple(sizes), tuple(blocks)))
                cur = nxt
                break
        else:
            raise RuntimeError(f"no admissible permutation pass for {shape} {perm}")
    return tuple(plan)


def permute(B, t, shape, perm):
    """`t`, viewed as `shape`, with its axes reordered by `perm`.  Flat in, flat out.

        >>> from pylea.eval.backend import NumpyBackend
        >>> B = NumpyBackend()
        >>> permute(B, B.asarray([1, 2, 3, 4]), (2, 2), (1, 0)).tolist()
        [1.0, 3.0, 2.0, 4.0]
    """
    blocks, sizes = coalesce(perm, shape)
    if blocks == list(range(len(blocks))):
        return t
    if len(blocks) <= B.max_rank:
        return B.flatten(B.permute(B.view(t, sizes), blocks))
    for sizes, blocks in _passes(tuple(shape), tuple(perm), B.max_rank):
        t = B.flatten(B.permute(B.view(t, sizes), blocks))
    return t


def reduce_sum(B, t, shape, axes):
    """`t`, viewed as `shape`, summed over `axes`, each axis through a rank-3 view.

        >>> from pylea.eval.backend import NumpyBackend
        >>> B = NumpyBackend()
        >>> reduce_sum(B, B.asarray([1, 2, 3, 4]), (2, 2), (1,)).tolist()
        [3.0, 7.0]
    """
    shape = list(shape)
    for ax in sorted(axes, reverse=True):
        before, here, after = prod(shape[:ax]), shape[ax], prod(shape[ax + 1:])
        t = B.flatten(B.sum_axis(B.view(t, (before, here, after)), 1))
        shape.pop(ax)
    return t


def select(B, t, shape, axis, index):
    """`t` restricted to position `index` of `axis`, through a rank-3 view.

    This is what a sliced index costs per slice: one row of the array is taken and the axis
    disappears.

        >>> from pylea.eval.backend import NumpyBackend
        >>> B = NumpyBackend()
        >>> select(B, B.asarray([1, 2, 3, 4]), (2, 2), axis=1, index=0).tolist()
        [1.0, 3.0]
    """
    before, here, after = prod(shape[:axis]), shape[axis], prod(shape[axis + 1:])
    return B.flatten(B.take(B.view(t, (before, here, after)), 1, index))


def _offsets(shape, groups) -> tuple[list[int], list[int]]:
    """Per group, the flat-index step of moving one position along all its axes at once,
    and the group's extent."""
    stride, acc = [1] * len(shape), 1
    for ax in range(len(shape) - 1, -1, -1):
        stride[ax] = acc
        acc *= shape[ax]
    return ([sum(stride[ax] for ax in group) for group in groups],
            [shape[group[0]] for group in groups])


def _flat_indices(offsets, extents):
    """The flat index of every position of the diagonal, as one array.

    Built with numpy whatever the backend and then handed to `B.indices`.  This is index
    arithmetic over the plan's shapes rather than over any matrix's data, so it belongs on
    the host, and its size is the diagonal's rather than the array's.
    """
    import numpy as np

    index = np.zeros(1, dtype=np.intp)
    for k, (offset, extent) in enumerate(zip(offsets, extents)):
        step = (np.arange(extent, dtype=np.intp) * offset).reshape(
            [extent if i == k else 1 for i in range(len(extents))])
        index = index + step
    return index.reshape(-1)


def diagonal(B, t, shape, groups):
    """`t` where the axes of each group agree, one axis per group.

    This is what an op naming one wire twice needs, a contraction step being unable to take
    a diagonal itself:

        >>> from pylea.eval.backend import NumpyBackend
        >>> B = NumpyBackend()
        >>> diagonal(B, B.asarray([1, 2, 3, 4]), (2, 2), ((0, 1),)).tolist()
        [1.0, 4.0]
    """
    offsets, extents = _offsets(shape, groups)
    return B.gather(t, B.indices(_flat_indices(offsets, extents)))


def spread(B, t, shape, positions):
    """`t` on the diagonal of an array of shape `shape`, zero elsewhere.

    The inverse of `diagonal`, and what a body returning one wire twice needs:

        >>> from pylea.eval.backend import NumpyBackend
        >>> B = NumpyBackend()
        >>> spread(B, B.asarray([1, 2]), (2, 2), ((0, 1),)).tolist()
        [1.0, 0.0, 0.0, 2.0]
    """
    offsets, extents = _offsets(shape, positions)
    return B.scatter(B.indices(_flat_indices(offsets, extents)), t, prod(shape))


def contract(B, a, b, batch, left, sum_, right):
    """One batched matrix multiplication of operands already in ``batch, left, sum`` order.

    Every contraction step in a plan is one of these; the permutations that bring the
    operands into that order are separate steps.

        >>> from pylea.eval.backend import NumpyBackend
        >>> B = NumpyBackend()
        >>> identity = B.asarray([1, 0, 0, 1])
        >>> contract(B, B.asarray([1, 2, 3, 4]), identity, 1, 2, 2, 2).tolist()
        [1.0, 2.0, 3.0, 4.0]
    """
    return B.flatten(B.matmul(B.view(a, (batch, left, sum_)), B.view(b, (batch, sum_, right))))

