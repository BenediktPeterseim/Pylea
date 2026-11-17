"""Lowering: a DPP becomes one plan per body.

`compile_program` first splices in every callee too wide to hold as a single matrix, then
lowers each body that remains: it builds the body's network, asks the order finder for a
contraction tree, and turns that tree into plan steps.  A call surviving that first step
stays a leaf of its caller's network, and the evaluator contracts it once however many
callers load it.  The order finder is asked once for all bodies, through its `find_all`
where it has one, so that a search bounded by wall clock is bounded as a whole.

`assemble` walks the tree and emits the steps.  It tracks how many live tensors still carry
each index, so that an index is summed away at the contraction where its last carrier is
consumed, and a free index is never summed at all.  Leaves are formed first, taking a
diagonal where the op names a wire twice and selecting a position where the index is sliced.

**A plan is refused when a tensor exceeds `max_entries`.**  The body's result and its
largest leaf are tensors that no contraction tree and no slicing avoid, so where either
exceeds `max_entries` the DPP is refused before any search, with
`ExactInferenceTooLarge`.  After assembly the plan's largest intermediate is checked as
well, except for an order the finder left unsliced because its cost was beyond the
finder's cap (`pylea.compile.order.Order.over_cost_cap`): such a plan serves only as an
estimate, and `pylea.compile.planner` decides what it means.

Lowering does no arithmetic and touches no array, so a DPP compiles the same way
whatever it is later evaluated on.
"""
from __future__ import annotations

import math
from collections import Counter
from math import prod

from pylea.compile.network import (
    Tensor, Network, build_network, distinct as _distinct,
)
from pylea.compile.order import (
    GreedyOrderFinder, Order, OrderFinder, check_finder_and_kappa, find_orders,
    slice_target,
)
from pylea.compile.plan import (
    Contract, Diagonal, Load, Plan, Reduce, Select, Spread, Transpose,
)
from pylea.ir.inline import flatten_wide_calls
from pylea.ir.program import Program, reachable

# A callee wider than this, in bits of input and output, is inlined into its caller rather
# than held as one matrix.  Chosen, not fitted: ``2**22`` float64 entries is 32 MiB, and a
# body at the ceiling is contracted once rather than once per use.
MAX_LEAF_BITS = 22

# The largest tensor exact inference creates whatever `kappa` is, in entries.  Chosen, not
# fitted: evaluating a DPP whose result has ``2**22`` to ``2**26`` entries holds 2.3 to 3.4
# times the result's memory at its peak (numpy, float64), so ``2**29`` entries, 4 GiB,
# need 9 to 14 GiB, and ``2**29`` is the largest power of two that fits a machine with
# 16 GiB of memory.  Re-measure after changing `pylea.eval`.
MAX_TENSOR_ENTRIES = 2**29

# Bytes per entry of a float64 tensor, the dtype exact inference evaluates in.
_ENTRY_BYTES = 8


class ExactInferenceInfeasible(ValueError):
    """Raised where exact inference on a DPP would need more memory or time than its
    limit.

    The message says what was asked, what it would need, what the limit is and what can be
    done instead.  `trace` holds the compilation steps that ran before the refusal, in
    the form of `pylea.syntax.compiled.Compiled.trace`.
    """

    def __init__(self, message: str, trace: tuple[str, ...] = ()):
        super().__init__(message)
        self.trace = tuple(trace)

    @property
    def reason(self) -> str:
        """The message's first sentence: what was asked, what it would need and the limit.

            >>> ExactInferenceTooSlow("exact inference on 'grid' is estimated to take "
            ...     "about 3 years; the limit is 11.6 days.  Sample instead.").reason
            "exact inference on 'grid' is estimated to take about 3 years; the limit is 11.6 days"
        """
        return str(self).split(".  ")[0]


class ExactInferenceTooLarge(ExactInferenceInfeasible):
    """Raised where exact inference on a DPP needs a tensor with more entries than the
    limit, ``max(2**29, 2**kappa)``, and no slicing avoids it."""


class ExactInferenceTooSlow(ExactInferenceInfeasible):
    """Raised where exact inference on a DPP is estimated to take longer than the limit."""


def entry_limit(kappa: int | None) -> int | None:
    """The most entries a tensor of exact inference may have under `kappa`:
    ``max(MAX_TENSOR_ENTRIES, 2**kappa)``, and no limit for `kappa` `None` or where
    ``2**kappa`` exceeds every float, as for `pylea.compile.order.slice_target`.

        >>> entry_limit(28) == 2**29, entry_limit(40) == 2**40, entry_limit(None)
        (True, True, None)
        >>> entry_limit(2**40) is None
        True
    """
    if kappa is None or slice_target(kappa) is None:
        return None
    return max(MAX_TENSOR_ENTRIES, 1 << kappa)


def subject(name: str | None) -> str:
    """How a refusal names the DPP: by `name` where one is given.

        >>> subject("wide"), subject(None)
        ("'wide'", 'this DPP')
    """
    return "this DPP" if name is None else repr(name)


def power_of_two(n: int) -> str:
    """`n` written as a power of two where it is one, else in digits.

        >>> power_of_two(2**40), power_of_two(12)
        ('2**40', '12')
    """
    if n > 0 and n & (n - 1) == 0:
        return f"2**{n.bit_length() - 1}"
    return str(n)


def memory(entries: int) -> str:
    """The memory `entries` float64 entries take, in binary units.

        >>> memory(2**29), memory(2**40), memory(2**100)
        ('4 GiB', '8 TiB', '2**103 bytes')
    """
    size = entries * _ENTRY_BYTES
    if size >= 1 << 90:
        return f"{power_of_two(size)} bytes"
    for power, unit in ((80, "YiB"), (70, "ZiB"), (60, "EiB"), (50, "PiB"), (40, "TiB"),
                        (30, "GiB"), (20, "MiB"), (10, "KiB")):
        if size >= 1 << power:
            return f"{size / (1 << power):.3g} {unit}"
    return f"{size} bytes"


def _too_large(name: str | None, entries: int, limit: int, cause: str = "",
               outputs: bool = False) -> ExactInferenceTooLarge:
    """The refusal of a tensor of `entries` entries, `cause` saying why no contraction
    tree avoids it, and `outputs` whether returning fewer variables would."""
    kappa = math.ceil(math.log2(entries))
    fewer = ", return fewer variables at once" if outputs else ""
    return ExactInferenceTooLarge(
        f"exact inference on {subject(name)} needs a tensor with {power_of_two(entries)} "
        f"entries ({memory(entries)} of memory); the limit is {power_of_two(limit)} "
        f"entries ({memory(limit)}).  {cause}Sample with method=\"smc\" or the default "
        f"method=\"auto\"{fewer}, or raise the limit with kappa={kappa}.")


def compile_program(prog: Program, order_finder: OrderFinder | None = None,
                    kappa: int | None = None, max_leaf_bits: int = MAX_LEAF_BITS,
                    max_entries: int | None = None,
                    name: str | None = None) -> tuple[dict[int, Plan], int]:
    """Plans for every body of `prog` surviving inlining, together with the root's index.

    `order_finder` chooses the contraction trees, `GreedyOrderFinder` by default, and
    `kappa` is the size limit slicing aims for, ``2**kappa`` entries per intermediate.
    Where `max_entries` is given, a plan holding a larger tensor raises
    `ExactInferenceTooLarge`, its message naming the DPP by `name`; the module docstring
    says which plans are checked.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, root = compile_program(both.program())
        >>> len(plans[root].steps), plans[root].peak_entries()
        (5, 8)

    Raises what `pylea.compile.order.check_finder_and_kappa` raises for an invalid
    `order_finder` or `kappa`.
    """
    kappa = check_finder_and_kappa(order_finder, kappa)
    finder = GreedyOrderFinder() if order_finder is None else order_finder
    prog = flatten_wide_calls(prog, max_leaf_bits)
    bodies = reachable(prog)
    networks = [build_network(prog, i) for i in bodies]
    for i, net in zip(bodies, networks):
        _check_unavoidable(prog, i, net, max_entries, name)
    # Slicing never aims below a body's largest table, which slicing cannot shrink, so
    # each body is searched under its own floor; bodies sharing a floor share a search.
    floors = [_slicing_floor(net, kappa) for net in networks]
    orders: list[Order | None] = [None] * len(networks)
    for floor in dict.fromkeys(floors):
        group = [k for k, f in enumerate(floors) if f == floor]
        found = find_orders(finder, [(networks[k].axes, networks[k].output,
                                      networks[k].size) for k in group], kappa=floor)
        for k, order in zip(group, found):
            orders[k] = order
    return ({i: _checked(assemble(prog, i, net, order), order, max_entries, name)
             for i, net, order in zip(bodies, networks, orders)}, prog.root)


def _largest_leaf(net: Network) -> int:
    """The entries of the largest table among `net`'s leaves."""
    return max((prod(leaf.shape if isinstance(leaf, Tensor) else leaf.dims)
                for leaf in net.leaves), default=1)


def _slicing_floor(net: Network, kappa: int | None) -> int | None:
    """`kappa`, raised to the bits of `net`'s largest table: a smaller target multiplies
    the work without lowering the largest tensor held."""
    if kappa is None:
        return None
    return max(kappa, math.ceil(math.log2(max(1, _largest_leaf(net)))))


def lower_body(prog: Program, body: int, order_finder: OrderFinder,
               kappa: int | None = None, max_entries: int | None = None,
               name: str | None = None) -> Plan:
    """The plan contracting body `body` to its semantics.

    Raises `ExactInferenceTooLarge` where `max_entries` is given and the plan holds a
    larger tensor, as `compile_program` does.
    """
    net = build_network(prog, body)
    _check_unavoidable(prog, body, net, max_entries, name)
    order = order_finder.find(net.axes, net.output, net.size,
                              kappa=_slicing_floor(net, kappa))
    return _checked(assemble(prog, body, net, order), order, max_entries, name)


def _check_unavoidable(prog: Program, body: int, net: Network, max_entries: int | None,
                       name: str | None) -> None:
    """Nothing, or `ExactInferenceTooLarge` where the body's result or one of its leaves
    exceeds `max_entries`: those tensors exist whatever the contraction tree and slicing."""
    if max_entries is None:
        return
    result = prod(net.size[ix] for ix in net.output_axes)
    leaf = _largest_leaf(net)
    if max(result, leaf) <= max_entries:
        return
    b = prog.bodies[body]
    outputs = False
    if leaf > result:
        cause = (f"One of its assignments alone has a table of {power_of_two(leaf)} "
                 "entries.  ")
    elif body != prog.root:
        cause = (f"A definition it calls alone has {power_of_two(result)} joint values of "
                 "its input and output variables.  ")
    elif b.inputs:
        cause = (f"Its input and output variables alone have {power_of_two(result)} joint "
                 "values.  ")
        outputs = True
    else:
        cause = f"Its output variables alone have {power_of_two(result)} joint values.  "
        outputs = True
    raise _too_large(name, max(result, leaf), max_entries, cause, outputs)


def _checked(plan: Plan, order: Order, max_entries: int | None, name: str | None) -> Plan:
    """`plan`, or `ExactInferenceTooLarge` where its largest tensor exceeds `max_entries`
    and its order was sliced as far as the finder could."""
    if max_entries is not None and not order.over_cost_cap \
            and plan.peak_entries() > max_entries:
        raise _too_large(name, plan.peak_entries(), max_entries)
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

