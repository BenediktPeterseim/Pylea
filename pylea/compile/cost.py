"""Plan cost estimation.

`contraction_cost` counts the scalar multiplications the plans perform and `step_count` the
steps the interpreter executes.  A step depending on no sliced index runs once outside the
slice loop rather than once per slice, and `contraction_cost` and `step_count` follow the
interpreter in that.  Both are returned as floats, infinite where the count exceeds the
float range, so that a plan whose work no float can hold still compiles and is recognised
by its cost rather than failing in the arithmetic of the estimate.

`predicted_ms` converts the two counts to milliseconds and takes the larger.  A plan doing
little arithmetic is bound by per-step overhead, at `MS_PER_STEP` each.  One doing a great
deal is bound by the throughput of the contraction kernel, which `SLOPE` and `INTERCEPT`
describe as a power law in the operation count.  `predicted_ms_from_cost` estimates from an
operation count alone, for a cost arrived at without building a plan: the power law from
`OVERHEAD_FLOOR_COST` upwards, and below it, where there is no step count to fall back on,
the operations at `OPS_PER_MS`.  `cost_for_ms` is the inverse of the power law.

`search_gain_ms` is the part of the estimate that a different contraction tree can change:
the arithmetic, and the steps that slicing repeats.  Every tree over a network has one leaf
per tensor and one contraction fewer than it has tensors, so the steps counted once do not
depend on the tree, and no search can remove them.

The result is good to an order of magnitude, which is what `pylea.compile.planner` needs to
place an instance on one side or the other of a multi-second search.  It orders instances
and does not time them.  Every constant here belongs to one machine and to `pylea.eval`.
"""
from __future__ import annotations

import math

from pylea.compile.plan import Contract, Plan

# Scalar multiplications per millisecond once the contraction kernel saturates, and the wall
# clock of one plan step, in milliseconds, where it does not.  Fitted by timing the
# evaluation of fault trees with numpy and its default BLAS on an Apple M3 Pro; not
# reproducible from this repository.  Re-measure on another machine, or after changing
# `pylea.eval`, numpy or the BLAS it links.
OPS_PER_MS = 1.0e8
MS_PER_STEP = 3.5e-3

# The power law ``predicted = 10 ** (SLOPE * log10(cost) + INTERCEPT)``, in milliseconds.
# Fitted on log10 wall clock against log10 operation count, over the plans
# of fault trees whose arithmetic exceeds their step overhead, evaluated with numpy and its
# default BLAS on an Apple M3 Pro; not reproducible from this repository.  Re-fit on
# another machine, or after changing `pylea.eval`, numpy or the BLAS it links.  A third
# coefficient for step or element count cannot be added: on those plans the two are
# collinear with the operation count.
SLOPE = 0.8389
INTERCEPT = -5.6853

# The cost at which one step's worth of arithmetic, at `OPS_PER_MS`, equals one step's
# overhead.  Derived, not fitted.  `predicted_ms_from_cost` uses the power law from here
# upwards.
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

    From `OVERHEAD_FLOOR_COST` upwards this is the power law of `SLOPE` and `INTERCEPT`.
    Below it, there being no step count to fall back on, it is the operations at
    `OPS_PER_MS`, which is less than the power law gives at `OVERHEAD_FLOOR_COST`, so the
    estimate rises abruptly there:

        >>> predicted_ms_from_cost(1e3)
        1e-05
        >>> round(predicted_ms_from_cost(3e5), 4), round(predicted_ms_from_cost(4e5), 4)
        (0.003, 0.1033)
        >>> round(predicted_ms_from_cost(1e12))
        24071
    """
    if cost <= 0.0:
        return 0.0
    if cost < OVERHEAD_FLOOR_COST:
        return cost / OPS_PER_MS
    return 10.0 ** (SLOPE * math.log10(cost) + INTERCEPT)


def cost_for_ms(ms: float) -> float:
    """The operation count `predicted_ms_from_cost` turns into `ms` milliseconds, on its
    power law.

        >>> round(predicted_ms_from_cost(cost_for_ms(1e9)))
        1000000000
    """
    return 10.0 ** ((math.log10(ms) - INTERCEPT) / SLOPE)


def search_gain_ms(plans: dict[int, Plan]) -> float:
    """The part of `predicted_ms` that a different contraction tree can change, in
    milliseconds: the arithmetic, or the steps slicing repeats beyond the first slice,
    whichever is larger.

    A plan that slices nothing and does almost no arithmetic leaves a search nothing to
    gain:

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, _root = compile_program(both.program())
        >>> search_gain_ms(plans)
        1.2e-07
    """
    steps, ops, once = 0, 0, 0
    for p in plans.values():
        s, o = _body_work(p)
        steps += s
        ops += o
        once += len(p.steps)
    return max(_as_float(steps - once) * MS_PER_STEP, predicted_ms_from_cost(_as_float(ops)))
