"""Plan cost estimation.

`contraction_cost` counts the scalar multiplications the plans perform and `step_count` the
steps the interpreter executes.  `tree_cost` counts the cost of the contraction trees the
plans follow as if no index were sliced, which depends on neither the slicing nor the
machine.  A step depending on no sliced index runs once outside the
slice loop rather than once per slice, and both counts follow the interpreter in that.  Both
are returned as floats, infinite where the count exceeds the float range, so that a plan
whose work no float can hold still compiles and is recognised by its cost rather than
failing in the arithmetic of the estimate.

`predicted_ms` converts the two counts to milliseconds and takes the larger.  A plan doing
little arithmetic is bound by per-step overhead, at `MS_PER_STEP` each.  One doing a great
deal is bound by the throughput of the contraction kernel, which `SLOPE` and `INTERCEPT`
describe as a power law in the operation count.  `predicted_ms_from_cost` applies that power
law alone, for a cost arrived at without building a plan, and returns the overhead floor
below it, there being no step count to fall back on.

The result is good to an order of magnitude, which is what `pylea.compile.planner` needs to
place an instance on one side or the other of a multi-second search.  It orders instances
and does not time them.  Every constant here belongs to one machine and to `pylea.eval`.
"""
from __future__ import annotations

import math
from math import prod

from pylea.compile.plan import Contract, Load, Plan, Select

# Scalar multiplications per millisecond once the contraction kernel saturates, and the wall
# clock of one plan step where it does not.  Timed over the fault-tree corpus on an
# Apple M3 Pro; re-time on another machine or after changing `pylea.eval`.
OPS_PER_MS = 1.0e8
MS_PER_STEP = 3.5e-3

# The power law ``predicted = 10 ** (SLOPE * log10(cost) + INTERCEPT)``, fitted over the
# arithmetic-bound plans of the fault-tree corpus on an Apple M3 Pro.  Re-fit on another machine
# or after changing `pylea.eval`.  A third coefficient for step or element count cannot be
# added: those two and the operation count are collinear across the corpus.
SLOPE = 0.8389
INTERCEPT = -5.6853

# The cost at which one step's worth of arithmetic equals one step's overhead.  Below it the
# step count governs rather than the power law.
OVERHEAD_FLOOR_COST = MS_PER_STEP * OPS_PER_MS


def contraction_cost(plans: dict[int, Plan]) -> float:
    """The scalar multiplications `plans` perform, slices included.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, _root = compile_program(both.program())
        >>> contraction_cost(plans)
        12.0
    """
    return _as_float(sum(_body_work(p)[1] for p in plans.values()))


def step_count(plans: dict[int, Plan]) -> float:
    """The steps the interpreter executes, a step counting once per slice it varies in.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, _root = compile_program(both.program())
        >>> step_count(plans)
        5.0
    """
    return _as_float(sum(_body_work(p)[0] for p in plans.values()))


def tree_cost(plans: dict[int, Plan]) -> float:
    """The cost of the contraction trees `plans` follow, each index held rather than
    sliced: the entries of every leaf plus the multiplications of every pairwise
    contraction, each counted once.

    A sliced index is absent from every value downstream of the `Select` that removed it,
    so a contraction whose operands lack it would, unsliced, multiply its count by the
    index's size.  For a plan that slices nothing this is `contraction_cost` plus the
    leaves' entries:

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, _root = compile_program(both.program())
        >>> tree_cost(plans)
        24.0
    """
    return _as_float(sum(_tree_work(p) for p in plans.values()))


def _tree_work(plan: Plan) -> int:
    # removed[v]: the sliced indices value v lacks that its unsliced counterpart holds.  A
    # sliced index is summed, unsliced, by the contraction after which no live value holds
    # it any more, sliced indices never being free; `holders` counts the live values.
    uses = [0] * len(plan.steps)
    for step in plan.steps:
        for src in _operands(step):
            uses[src] += 1
    removed: list[frozenset[int]] = []
    holders = [0] * len(plan.sliced_dims)
    total = 0

    def consume(v: int) -> None:
        uses[v] -= 1
        if uses[v] == 0:
            for j in removed[v]:
                holders[j] -= 1

    for i, step in enumerate(plan.steps):
        if isinstance(step, Load):
            held: frozenset[int] = frozenset()
            total += prod(plan.shapes[i])
        elif isinstance(step, Contract):
            involved = removed[step.a] | removed[step.b]
            total += (step.batch * step.left * step.sum * step.right
                      * prod(plan.sliced_dims[j] for j in involved))
            consume(step.a)
            consume(step.b)
            held = frozenset(j for j in involved if holders[j] > 0)
        else:
            src = step.src
            held = removed[src] | ({step.which} if isinstance(step, Select) else set())
            consume(src)
        removed.append(held)
        if uses[i]:
            for j in held:
                holders[j] += 1
    return total


def _operands(step) -> tuple[int, ...]:
    if isinstance(step, Load):
        return ()
    if isinstance(step, Contract):
        return (step.a, step.b)
    return (step.src,)


def _as_float(count: int) -> float:
    """`count` as a float, or infinity where it exceeds the float range.

    Converting such an integer raises `OverflowError` rather than rounding to infinity.
    """
    try:
        return float(count)
    except OverflowError:
        return math.inf


def _body_work(plan: Plan) -> tuple[int, int]:
    """``(steps run, scalar multiplications performed)`` for one plan."""
    slices = max(plan.n_slices, 1)
    steps = ops = 0
    for i, step in enumerate(plan.steps):
        reps = slices if plan.varying[i] else 1
        steps += reps
        if isinstance(step, Contract):
            ops += step.batch * step.left * step.sum * step.right * reps
    return steps, ops


def predicted_ms(plans: dict[int, Plan]) -> float:
    """The estimated wall clock of evaluating `plans`, in milliseconds.

    A DPP this small is bound by its step count rather than by its arithmetic, so the
    estimate is `MS_PER_STEP` times the steps:

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, _root = compile_program(both.program())
        >>> round(predicted_ms(plans), 6)
        0.0175
    """
    steps, ops = 0, 0
    for p in plans.values():
        s, o = _body_work(p)
        steps += s
        ops += o
    return max(_as_float(steps) * MS_PER_STEP, predicted_ms_from_cost(_as_float(ops)))


def predicted_ms_from_cost(cost: float) -> float:
    """Milliseconds from an operation count alone, for a cost obtained without a plan.

    Below `OVERHEAD_FLOOR_COST` there is no step count to fall back on, so this returns what
    the floor itself would cost rather than a number below anything observable.

        >>> predicted_ms_from_cost(1e3)
        1e-05
        >>> round(predicted_ms_from_cost(1e12))
        24071
    """
    if cost <= 0.0:
        return 0.0
    if cost < OVERHEAD_FLOOR_COST:
        return cost / OPS_PER_MS
    return 10.0 ** (SLOPE * math.log10(cost) + INTERCEPT)
