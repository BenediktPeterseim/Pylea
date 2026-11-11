"""Lowering: a DPP becomes one plan per body.

`compile_program` first splices in every callee too wide to hold as a single matrix, then
lowers each body that remains: it builds the body's network, asks the order finder for a
contraction tree, and turns that tree into plan steps.  A call surviving that first step
stays a leaf of its caller's network, and the evaluator contracts it once however many
callers load it.

`assemble` walks the tree and emits the steps.  It tracks how many live tensors still carry
each index, so that an index is summed away at the contraction where its last carrier is
consumed, and a free index is never summed at all.  Leaves are formed first, taking a
diagonal where the op names a wire twice and selecting a position where the index is sliced.

Lowering does no arithmetic and touches no array, so a DPP compiles the same way
whatever it is later evaluated on.
"""
from __future__ import annotations

from math import prod

from collections import Counter

from pylea.compile.network import Tensor, Network, build_network, distinct as _distinct
from pylea.compile.order import AdaptiveOrderFinder, Order, OrderFinder
from pylea.compile.plan import (
    Contract, Diagonal, Load, Plan, Reduce, Select, Spread, Transpose,
)
from pylea.ir.inline import flatten_wide_calls
from pylea.ir.program import Program, reachable

# A callee wider than this is inlined into its caller rather than held as one matrix.
# ``2**22`` float64 entries is 32 MiB, and a body at the ceiling is contracted once rather
# than once per use.
MAX_LEAF_BITS = 22


class PlanTooWide(ValueError):
    """Raised when a plan's largest intermediate exceeds the entry budget."""


def compile_program(prog: Program, order_finder: OrderFinder | None = None,
                    kappa: int | None = None, max_leaf_bits: int = MAX_LEAF_BITS,
                    max_entries: int | None = None) -> tuple[dict[int, Plan], int]:
    """Plans for every body of `prog` surviving inlining, together with the root's index.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, root = compile_program(both.program())
        >>> len(plans[root].steps), plans[root].peak_entries()
        (5, 8)
    """
    finder = AdaptiveOrderFinder() if order_finder is None else order_finder
    prog = flatten_wide_calls(prog, max_leaf_bits)
    plans = {i: lower_body(prog, i, finder, kappa, max_entries) for i in reachable(prog)}
    return plans, prog.root


def lower_body(prog: Program, body: int, order_finder: OrderFinder,
               kappa: int | None = None, max_entries: int | None = None) -> Plan:
    """The plan contracting body `body` to its semantics.

    Raises `PlanTooWide` if `max_entries` is given and the plan's peak exceeds it.
    """
    net = build_network(prog, body)
    order = order_finder.find(net.axes, net.output, net.size, kappa=kappa)
    plan = assemble(prog, body, net, order)
    if max_entries is not None and plan.peak_entries() > max_entries:
        raise PlanTooWide(
            f"body {body} contracts through {plan.peak_entries()} entries, "
            f"over the budget of {max_entries}")
    return plan


def assemble(prog: Program, body: int, net: Network, order: Order) -> Plan:
    """The `Plan` the contraction tree `order` over `net` becomes."""
    b = prog.bodies[body]
    steps: list = []
    shapes: list[tuple[int, ...]] = []
    varying: list[bool] = []

    def emit(step, shape, var: bool) -> int:
        steps.append(step)
        shapes.append(tuple(shape))
        varying.append(var)
        return len(steps) - 1

    slice_at = {ix: k for k, ix in enumerate(order.sliced)}
    free = set(net.output)
    if free & set(slice_at):
        raise ValueError("a free index was sliced; slices could not be summed")

    # How many live tensors carry each index.  An index leaves a contraction's result once
    # no other live tensor and no free position still needs it.
    live: Counter = Counter()
    for ixs in net.axes:
        for ix in set(ixs):
            if ix not in slice_at:
                live[ix] += 1

    def keep(ix: int) -> bool:
        return live[ix] > 0 or ix in free

    values: list[tuple[int, list[int]]] = []
    for t, axes in enumerate(net.axes):
        leaf = net.leaves[t]
        shape = leaf.shape if isinstance(leaf, Tensor) else leaf.dims
        vid = emit(Load(t), shape, False)
        if isinstance(leaf, Tensor) and leaf.reduce:
            vid = emit(Reduce(vid, leaf.reduce), _drop(shape, leaf.reduce), False)
        cur = list(axes)
        if len(set(cur)) != len(cur):   # `f(x, x)`, and a contraction cannot take a diagonal
            groups = tuple(tuple(ax for ax, ix in enumerate(cur) if ix == label)
                           for label in _distinct(cur))
            cur = list(_distinct(cur))
            vid = emit(Diagonal(vid, groups), [net.size[ix] for ix in cur], False)
        for ix in list(cur):
            if ix in slice_at:
                ax = cur.index(ix)
                vid = emit(Select(vid, ax, slice_at[ix]), _drop(shapes[vid], [ax]), True)
                cur.pop(ax)
        values.append((vid, cur))

    for a, c in order.path:
        va, ia = values[a]
        vb, ib = values[c]
        for ix in ia:
            live[ix] -= 1
        for ix in ib:
            live[ix] -= 1

        shared = set(ia) & set(ib)
        if any(not keep(ix) for ix in ia if ix not in shared) or \
                any(not keep(ix) for ix in ib if ix not in shared):
            raise AssertionError("an index survived to a contraction that cannot sum it")
        batch = [ix for ix in ia if ix in shared and keep(ix)]
        summed = [ix for ix in ia if ix in shared and not keep(ix)]
        left = [ix for ix in ia if ix not in shared]
        right = [ix for ix in ib if ix not in shared]

        perm_a = _perm(ia, batch + left + summed)
        perm_b = _perm(ib, batch + summed + right)
        out = batch + left + right
        vid = emit(
            Contract(va, vb, perm_a, perm_b,
                     prod(net.size[i] for i in batch), prod(net.size[i] for i in left),
                     prod(net.size[i] for i in summed), prod(net.size[i] for i in right)),
            [net.size[ix] for ix in out],
            varying[va] or varying[vb],
        )
        for ix in out:
            live[ix] += 1
        values.append((vid, out))

    vid, cur = values[-1]
    contracted = vid
    if cur != list(net.output):
        perm = _perm(cur, list(net.output))
        vid = emit(Transpose(vid, perm), [shapes[vid][p] for p in perm], varying[vid])

    if len(net.output_axes) != len(net.output):
        # The output names a wire more than once, so it is a diagonal.  The contraction ran
        # over the distinct wires, and only the result is this wide.
        shape = tuple(net.size[ix] for ix in net.output_axes)
        positions = tuple(tuple(ax for ax, ix in enumerate(net.output_axes) if ix == label)
                          for label in net.output)
        vid = emit(Spread(vid, positions, shape), shape, varying[vid])

    return Plan(
        leaves=net.leaves,
        steps=tuple(steps),
        shapes=tuple(shapes),
        varying=tuple(varying),
        sliced_dims=tuple(net.size[ix] for ix in order.sliced),
        contracted=contracted,
        result=vid,
        out_bits=b.out_bits,
        in_bits=b.in_bits,
    )


def _drop(shape, axes) -> tuple[int, ...]:
    drop = set(axes)
    return tuple(d for ax, d in enumerate(shape) if ax not in drop)


def _perm(current: list[int], target: list[int]) -> tuple[int, ...]:
    at = {ix: ax for ax, ix in enumerate(current)}
    return tuple(at[ix] for ix in target)

