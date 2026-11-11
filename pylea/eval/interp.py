"""The plan interpreter.

`evaluate` returns the semantics of one body as ``(array, log_scale)``, whose value is
``array * exp(log_scale)``.  The bodies the root reaches are run once each, every body
after the bodies it calls, so a body called from many places is contracted once and a call
graph of any depth is evaluated without recursion.

`run` executes one plan.  Steps that do not depend on the values of the sliced indices are
run first and once.  The rest are run per combination of those values, up to the value the
slices are summed into, and the results are brought to a common magnitude and added.  The
steps past that value reshape the sum and are run once, being linear.

A value is dropped once the steps that consume it have run, counted separately for the
fixed and the varying steps, so a run holds the contraction's peak rather than everything it
has computed.

`underflowed` reports whether an array holds an exact zero, which is how too narrow a dtype
fails here.  Every matrix is non-negative, so a contraction sums products of non-negative
numbers and cannot cancel; what a narrow dtype cannot do is hold one entry far below the
largest in the same array, and no per-array scale helps, the scale being common to both.
Testing the result rather than predicting the range is what makes the check affordable: an
underflow gives exactly 0 and never a wrong non-zero, so an array without a zero did not
underflow, and one with a zero is either an impossibility or the failure, which only a wider
dtype separates.
"""
from __future__ import annotations

from math import prod

import numpy as np

from pylea.compile.network import Tensor, Ones
from pylea.compile.plan import (
    Contract, Diagonal, Load, Plan, Reduce, Select, Spread, Transpose,
)
from pylea.eval import kernels as K
from pylea.eval.backend import Backend, NumpyBackend


def evaluate(plans: dict[int, Plan], root: int, backend=None) -> tuple[np.ndarray, float]:
    """The semantics of body `root` as a matrix, with its log magnitude.

    `backend` defaults to numpy rather than to whatever device happens to be present, so
    that evaluation is reproducible unless a caller asks otherwise.  Pass
    `pylea.eval.backend.select` for the best available.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def both() -> Bool:
        ...     return flip(0.5) & flip(0.25)
        >>> plans, root = compile_program(both.program())
        >>> array, log_scale = evaluate(plans, root)
        >>> array.round(4).tolist(), log_scale
        ([[0.875], [0.125]], 0.0)
    """
    B: Backend = NumpyBackend() if backend is None else backend
    memo: dict[int, tuple] = {}
    # Callees before callers, so that a call graph of any depth needs no recursion.
    for i in _callees_first(plans, root):
        memo[i] = run(plans[i], memo.__getitem__, B)

    array, scale = memo[root]
    plan = plans[root]
    # The one place a scale is read.  On an accelerator it is a device scalar until here, and
    # reading it once per step is what would stall the pipeline.
    return (B.to_numpy(array).reshape(1 << plan.out_bits, 1 << plan.in_bits),
            B.scale_float(scale))


def _callees_first(plans: dict[int, Plan], root: int) -> list[int]:
    """The bodies `root` reaches, `root` included, each after every body it calls."""
    order: list[int] = []
    seen = {root}
    stack = [(root, iter(_callees(plans[root])))]
    while stack:
        body, pending = stack[-1]
        callee = next(pending, None)
        if callee is None:
            stack.pop()
            order.append(body)
        elif callee not in seen:
            seen.add(callee)
            stack.append((callee, iter(_callees(plans[callee]))))
    return order


def _callees(plan: Plan) -> list[int]:
    return [leaf.callee for leaf in plan.leaves
            if isinstance(leaf, Tensor) and isinstance(leaf.callee, int)]


def run(plan: Plan, matrix_of, B=None) -> tuple:
    """One plan's value, as ``(flat array, log_scale)``.

    `matrix_of` resolves a called body's index to its already-evaluated result.
    """
    B = NumpyBackend() if B is None else B
    fixed_uses, sliced_uses = _use_counts(plan)

    fixed: dict[int, tuple] = {}
    for i, step in enumerate(plan.steps):
        if plan.varying[i]:
            continue
        fixed[i] = _apply(step, plan, fixed, matrix_of, (), B)
        for src in _operands(step):
            fixed_uses[src] -= 1
            if fixed_uses[src] == 0 and sliced_uses[src] == 0:
                fixed.pop(src, None)

    if not plan.sliced_dims:
        return fixed[plan.result]

    total = None
    for assignment in _assignments(plan.sliced_dims):
        values = dict(fixed)
        remaining = list(sliced_uses)
        for i in range(plan.contracted + 1):
            if not plan.varying[i]:
                continue
            values[i] = _apply(plan.steps[i], plan, values, matrix_of, assignment, B)
            for src in _operands(plan.steps[i]):
                if plan.varying[src]:
                    remaining[src] -= 1
                    if remaining[src] == 0:
                        values.pop(src, None)
        array, scale = values[plan.contracted]
        if total is None:
            total = (array, scale)
        else:
            a, b, scale = B.align(total[0], total[1], array, scale)
            total = (B.add(a, b), scale)
    assert total is not None

    # The steps past the sum reshape the result.  They are linear, so applying them to the
    # sum gives the same answer as applying them per slice, at one pass instead of many.
    values = dict(fixed)
    values[plan.contracted] = total
    for i in range(plan.contracted + 1, len(plan.steps)):
        values[i] = _apply(plan.steps[i], plan, values, matrix_of, (), B)
    return values[plan.result]


def _use_counts(plan: Plan) -> tuple[list[int], list[int]]:
    """Per value, how many fixed and how many varying steps consume it."""
    fixed = [0] * len(plan.steps)
    sliced = [0] * len(plan.steps)
    for i, step in enumerate(plan.steps):
        for src in _operands(step):
            (sliced if plan.varying[i] else fixed)[src] += 1
    (sliced if plan.varying[plan.result] else fixed)[plan.result] += 1
    return fixed, sliced


def _operands(step) -> tuple[int, ...]:
    if isinstance(step, Load):
        return ()
    if isinstance(step, Contract):
        return (step.a, step.b)
    return (step.src,)


def _assignments(dims):
    """Every combination of positions of the sliced indices, the last index fastest."""
    assignment = [0] * len(dims)
    while True:
        yield tuple(assignment)
        for k in range(len(dims) - 1, -1, -1):
            assignment[k] += 1
            if assignment[k] < dims[k]:
                break
            assignment[k] = 0
        else:
            return


def _apply(step, plan: Plan, values: dict, matrix_of, assignment, B) -> tuple:
    if isinstance(step, Load):
        return _leaf(plan.leaves[step.leaf], matrix_of, B)

    if isinstance(step, Contract):
        a, scale_a = values[step.a]
        b, scale_b = values[step.b]
        left = K.permute(B, a, plan.shapes[step.a], step.perm_a)
        right = K.permute(B, b, plan.shapes[step.b], step.perm_b)
        array = K.contract(B, left, right, step.batch, step.left, step.sum, step.right)
        return B.renormalise(array, B.scale_add(scale_a, scale_b))

    array, scale = values[step.src]
    shape = plan.shapes[step.src]
    if isinstance(step, Select):
        return K.select(B, array, shape, step.axis, assignment[step.which]), scale
    if isinstance(step, Reduce):
        return K.reduce_sum(B, array, shape, step.axes), scale
    if isinstance(step, Transpose):
        return K.permute(B, array, shape, step.perm), scale
    if isinstance(step, Diagonal):
        return K.diagonal(B, array, shape, step.groups), scale
    if isinstance(step, Spread):
        return K.spread(B, array, step.shape, step.positions), scale
    raise TypeError(f"unknown plan step {step!r}")


def _leaf(leaf, matrix_of, B) -> tuple:
    if isinstance(leaf, Tensor):
        if isinstance(leaf.callee, int):
            array, scale = matrix_of(leaf.callee)
            return B.asarray(array), scale
        return B.asarray(leaf.callee.matrix), B.scale_zero()
    if isinstance(leaf, Ones):
        return B.ones(prod(leaf.dims)), B.scale_zero()
    raise TypeError(f"unknown leaf {leaf!r}")



def underflowed(array) -> bool:
    """Whether `array` holds an exact zero, which is how too narrow a dtype fails here.

        >>> import numpy as np
        >>> underflowed(np.array([0.9, 0.1])), underflowed(np.array([1.0, 0.0]))
        (False, True)

    A narrow dtype fails by range, not by precision.  Every matrix is non-negative, so a
    contraction sums products of non-negative numbers and cannot cancel.  What it cannot do
    is hold one entry far below the largest in the same array: an output of
    ``[0.9, 1e-61]`` spans sixty-one orders of magnitude where float32 covers thirty-eight,
    and the per-array scale is common to both entries, so it moves the pair together and
    cannot close the gap.

    The failure is silent but not misleading: an underflow gives exactly 0 and never a wrong
    non-zero.  An array with no zero in it therefore did not underflow, and one with a zero
    is either an impossibility or an underflow, which only a wider dtype separates.  This is
    why the result is tested and its range is not predicted beforehand.
    """
    return bool((np.asarray(array) == 0.0).any())
