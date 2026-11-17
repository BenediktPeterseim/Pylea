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

**Every value carries a certificate of the error underflow may have caused.**  One log scale
per array keeps an array's largest entry in range, but a product inside one contraction can
still fall below the smallest normal float and lose its precision or become zero.  Beside
each array three numbers are kept on the host, natural logs in the units of the stored
array: `M` bounds the largest entry from above, `L` the smallest positive entry from below,
and `E` the absolute error of any entry from underflow (minus infinity: none).  A
contraction can only lose a product where ``L_a + L_b`` falls below the dtype's smallest
normal number; only then are the operands' smallest entries measured, and the error a lost
product can carry added to `E`.  Slice sums account for factors too small to store in the
same way.  Every matrix is non-negative, so nothing cancels: a sum is never smaller than
any of its terms.  The invariant the rules keep is that every positive stored entry is
either at least ``e**L`` or at most ``e**E / 2``.

`evaluate_certified` accepts a result whose `E` is minus infinity, or whose `E` lies
`CERTIFIED_LOG_RTOL` below its smallest positive entry and, where it has a zero entry,
below what a float64 array can hold beside its largest entry.  Otherwise it evaluates
again in float64 on the CPU, and where that is not certified either, with an exponent per
entry (`pylea.eval.backend.ExtendedRangeBackend`), which keeps the relative precision of
float64 for any range at several times the cost.  A zero entry is accepted only where its
true value, at most ``e**E``, is too small to be held beside the largest entry in any
float64 array, so that zero is the value the array would hold anyway.  Inside one
evaluation every value carries its record, so a called body's tiny entry that a later step
selects is accounted for there.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from math import prod

import numpy as np

from pylea.compile.network import Tensor, Ones
from pylea.compile.plan import (
    Contract, Diagonal, Load, Plan, Reduce, Select, Spread, Transpose, operands,
)
from pylea.eval import kernels as K
from pylea.eval.backend import LO, HI, Backend, ExtendedRangeBackend, NumpyBackend

# The relative error underflow may add to a certified result: ``2**-40``, about 1e-12, far
# below what any query reports and far above float64 rounding.  One value for every dtype,
# since it bounds what underflow adds, not the dtype's own rounding.
CERTIFIED_LOG_RTOL = math.log(2.0 ** -40)

# The log of the smallest ratio between two entries of one float64 array: a subnormal
# number's least step, ``2**-1074``, against an entry of 1, halved for rounding.
_LOG_SMALLEST_RATIO = -1075 * math.log(2.0)

# A product of two lower bounds is rounded on the host; this keeps `L` a lower bound.
_ROUNDING = 2.0 ** -50

_NO_ERROR = -math.inf


@dataclass(frozen=True)
class Evaluation:
    """An evaluated matrix, its log scale, and which evaluation produced it: ``"backend"``
    for the backend asked for, ``"float64"`` for the repetition in float64 on the CPU,
    ``"extended"`` for the repetition with an exponent per entry."""

    array: np.ndarray
    log_scale: float
    path: str


def evaluate(plans: dict[int, Plan], root: int, backend=None) -> tuple[np.ndarray, float]:
    """The semantics of body `root` as a matrix, with its log magnitude, certified.

    `backend` defaults to numpy rather than to whatever device happens to be present, so
    that evaluation is reproducible unless a caller asks otherwise.  For a device chosen
    from the plans, pass ``pylea.eval.backend.select(steps=..., ops=...)``, or a backend
    built directly; `select()` without those counts returns numpy.  The result is that of
    `evaluate_certified`; the array belongs to the caller.

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
    result = evaluate_certified(plans, root, backend)
    return result.array, result.log_scale


def evaluate_certified(plans: dict[int, Plan], root: int, backend=None) -> Evaluation:
    """The semantics of body `root`, from `backend` where its certificate holds, else from
    float64 on the CPU, else with an exponent per entry; the module docstring gives the
    rule.

    Here two observed reports of rare events give products near 1e-340, below the range
    of float64, so the answer, 0.1 for `a`, comes from the evaluation with an exponent per
    entry:

        >>> from pylea import DPP, Bool, flip, observe
        >>> from pylea.compile.lower import compile_program
        >>> @DPP
        ... def two_reports() -> Bool:
        ...     a = flip(0.5)
        ...     b = (flip(1e-170) & a) | (flip(3e-170) & ~a)
        ...     c = (flip(1e-170) & a) | (flip(3e-170) & ~a)
        ...     observe(b & c)
        ...     return a
        >>> result = evaluate_certified(*compile_program(two_reports.program()))
        >>> column = result.array[:, 0]
        >>> result.path, (column / column.sum()).round(12).tolist()
        ('extended', [0.9, 0.1])
    """
    B: Backend = NumpyBackend() if backend is None else backend
    array, scale, certified = _run_all(plans, root, B)
    if certified:
        return Evaluation(array, scale, "backend")
    if type(B) is not NumpyBackend:
        array, scale, certified = _run_all(plans, root, NumpyBackend())
        if certified:
            return Evaluation(array, scale, "float64")
    array, scale, _ = _run_all(plans, root, ExtendedRangeBackend())
    return Evaluation(array, scale, "extended")


def _run_all(plans: dict[int, Plan], root: int, B) -> tuple[np.ndarray, float, bool]:
    """Every body `root` reaches run on `B`: the root's matrix, its finite log scale (0
    for an array of zeros), and whether its certificate holds."""
    memo: dict[int, tuple] = {}
    # Callees before callers, so that a call graph of any depth needs no recursion.
    for i in _callees_first(plans, root):
        memo[i] = run(plans[i], memo.__getitem__, B)

    array, scale, record = memo[root]
    plan = plans[root]
    # The one place a scale is read.  On an accelerator it is a device scalar until here,
    # and reading it once per step is what would stall the pipeline.
    array, log_scale = B.finish(array, scale)
    array = array.reshape(1 << plan.out_bits, 1 << plan.in_bits)
    certified = _certified(array, record)
    if log_scale == -math.inf:
        log_scale = 0.0
    return array, log_scale, certified


def _certified(array: np.ndarray, record) -> bool:
    _M, _L, E = record
    if not np.isfinite(array).all():
        return False                    # a narrow dtype overflowed
    if E == _NO_ERROR:
        return True
    top = float(array.max()) if array.size else 0.0
    if top <= 0.0:
        return False
    positive = array[array > 0.0]
    if E > CERTIFIED_LOG_RTOL + math.log(float(positive.min())):
        return False
    # A zero entry is right where its true value, at most ``e**E``, lies below what a
    # float64 array can hold beside its largest entry.
    return positive.size == array.size or E <= math.log(top) + _LOG_SMALLEST_RATIO


def _callees_first(plans: dict[int, Plan], root: int) -> list[int]:
    """The bodies `root` reaches, `root` included, each after every body it calls."""
    order: list[int] = []
    seen = {root}
    stack = [(root, iter(plans[root].subplans()))]
    while stack:
        body, pending = stack[-1]
        callee = next(pending, None)
        if callee is None:
            stack.pop()
            order.append(body)
        elif callee not in seen:
            seen.add(callee)
            stack.append((callee, iter(plans[callee].subplans())))
    return order


def run(plan: Plan, matrix_of, B=None) -> tuple:
    """One plan's value, as ``(flat array, log_scale, record)``, `record` being the
    certificate ``(M, L, E)`` of the module docstring.

    `matrix_of` resolves a called body's index to its already-evaluated value.
    """
    B = NumpyBackend() if B is None else B
    fixed_uses, sliced_uses = _use_counts(plan)

    fixed: dict[int, tuple] = {}
    for i, step in enumerate(plan.steps):
        if plan.varying[i]:
            continue
        fixed[i] = _apply(step, plan, fixed, matrix_of, (), B)
        for src in operands(step):
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
            for src in operands(plan.steps[i]):
                if plan.varying[src]:
                    remaining[src] -= 1
                    if remaining[src] == 0:
                        values.pop(src, None)
        value = values[plan.contracted]
        total = value if total is None else _slice_sum(total, value, B)
    assert total is not None

    # The steps past the sum reshape the result.  They are linear, so applying them to the
    # sum gives the same answer as applying them per slice, at one pass instead of many.
    values = dict(fixed)
    values[plan.contracted] = total
    for i in range(plan.contracted + 1, len(plan.steps)):
        values[i] = _apply(plan.steps[i], plan, values, matrix_of, (), B)
    return values[plan.result]


def _slice_sum(x: tuple, y: tuple, B) -> tuple:
    """The sum of two slices' values, with its certificate."""
    (ta, sa, (Ma, La, Ea)), (tb, sb, (Mb, Lb, Eb)) = x, y
    ta, tb, s, la, lb = B.align(ta, sa, tb, sb)
    lu = B.log_underflow_unit
    terms = [la + Ea, lb + Eb]
    # Entries multiplied below the smallest normal number may have been lost, and so may
    # the whole operand where its factor itself is below it.
    if la + La < lu or lb + Lb < lu:
        terms.append(math.log(2.0) + lu)
    if la < lu:
        terms.append(la + Ma)
    if lb < lu:
        terms.append(lb + Mb)
    return (B.add(ta, tb), s, (_log_sum_exp(la + Ma, lb + Mb), min(la + La, lb + Lb),
                               _log_sum_exp(*terms)))


def _use_counts(plan: Plan) -> tuple[list[int], list[int]]:
    """Per value, how many fixed and how many varying steps consume it."""
    fixed = [0] * len(plan.steps)
    sliced = [0] * len(plan.steps)
    for i, step in enumerate(plan.steps):
        for src in operands(step):
            (sliced if plan.varying[i] else fixed)[src] += 1
    (sliced if plan.varying[plan.result] else fixed)[plan.result] += 1
    return fixed, sliced


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
        return _contract(step, plan, values, B)

    array, scale, (M, L, E) = values[step.src]
    shape = plan.shapes[step.src]
    if isinstance(step, Select):
        return K.select(B, array, shape, step.axis, assignment[step.which]), scale, (M, L, E)
    if isinstance(step, Reduce):
        # A sum of `k` entries: the largest and the error grow by at most `k` times.
        k = math.log(prod(shape[axis] for axis in step.axes))
        return K.reduce_sum(B, array, shape, step.axes), scale, (M + k, L, E + k)
    if isinstance(step, Transpose):
        return K.permute(B, array, shape, step.perm), scale, (M, L, E)
    if isinstance(step, Diagonal):
        return K.diagonal(B, array, shape, step.groups), scale, (M, L, E)
    if isinstance(step, Spread):
        return K.spread(B, array, step.shape, step.positions), scale, (M, L, E)
    raise TypeError(f"unknown plan step {step!r}")


def _contract(step: Contract, plan: Plan, values: dict, B) -> tuple:
    """A contraction and its certificate, per the module docstring."""
    a, scale_a, (Ma, La, Ea) = values[step.a]
    b, scale_b, (Mb, Lb, Eb) = values[step.b]
    lu = B.log_underflow_unit
    if La + Lb < lu:
        # The bounds say a product may fall below the smallest normal number; the
        # operands' own smallest entries decide.
        La, Lb = B.log_min_positive(a), B.log_min_positive(b)
    danger = La + Lb < lu
    left = K.permute(B, a, plan.shapes[step.a], step.perm_a)
    right = K.permute(B, b, plan.shapes[step.b], step.perm_b)
    array = K.contract(B, left, right, step.batch, step.left, step.sum, step.right)
    if Ea == _NO_ERROR and Eb == _NO_ERROR and not danger:
        E = _NO_ERROR
    else:
        E = math.log(step.sum) + _log_sum_exp(
            Ea + Mb, Ma + Eb, Ea + Eb, (math.log(2.0) + lu) if danger else _NO_ERROR)
    L = La + Lb - _ROUNDING
    scale = B.scale_add(scale_a, scale_b)
    array, new_scale, log_max, removed = B.renormalise(array, scale)
    if log_max == -math.inf:
        # An array of zeros.  Where none of them carries an error its scale is minus
        # infinity; otherwise the scale moves to the error, so that a sum of slices still
        # weighs it.
        if E == _NO_ERROR:
            return array, new_scale, (-math.inf, math.inf, _NO_ERROR)
        return array, B.scale_add(scale, E), (-math.inf, math.inf, 0.0)
    if removed is None:
        # An accelerator divided by the largest entry without telling the host its size.
        # `L` then drops by at most the bound on that entry, and an error, which is rare,
        # is worth one read from the device.
        L -= Ma + Mb + math.log(step.sum)
        if E != _NO_ERROR:
            E -= B.scale_float(new_scale) - B.scale_float(scale)
        return array, new_scale, (log_max, L, E)
    return array, new_scale, (log_max, L - removed, E - removed)


def _leaf(leaf, matrix_of, B) -> tuple:
    if isinstance(leaf, Tensor):
        if isinstance(leaf.callee, int):
            array, scale, record = matrix_of(leaf.callee)
            return B.asarray(array), scale, record
        if getattr(B, "keeps_every_range", False):
            # The matrix as it is: dividing by its largest entry in float64 first could
            # lose the smallest ones.
            matrix = np.asarray(leaf.callee.matrix, dtype=np.float64)
            array, scale = B.leaf(matrix, 0.0)
            return array, scale, (0.0, 0.0, _NO_ERROR)
        host, log_scale, M, L, E = _in_band(leaf.callee, always=not B.exact)
        if L < B.log_underflow_unit:
            # A narrow dtype holds entries below its smallest normal number imprecisely.
            E = _log_sum_exp(E, B.log_underflow_unit + math.log(2.0))
        array, scale = B.leaf(host, log_scale)
        return array, scale, (M, L, E)
    if isinstance(leaf, Ones):
        return B.ones(prod(leaf.dims)), B.scale_zero(), (0.0, 0.0, _NO_ERROR)
    raise TypeError(f"unknown leaf {leaf!r}")


# Per `Prim`, its matrix brought into range, for the band of float64 and for division by
# its largest entry: the read-only array, the log of what was divided out, the logs of
# its largest and smallest positive entries, and the error the division caused.  A pure
# function of the primitive, so it is computed once while the primitive lives.
_LEAVES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

_LOG_TINY64 = math.log(float(np.finfo(np.float64).tiny))


def _in_band(prim, always: bool = False) -> tuple[np.ndarray, float, float, float, float]:
    """`prim`'s matrix divided by its largest entry where that entry lies outside the
    band, or with `always` in any case, as `_LEAVES` holds it."""
    cached = _LEAVES.setdefault(prim, {})
    hit = cached.get(always)
    if hit is None:
        host = np.array(prim.matrix, dtype=np.float64).reshape(-1)
        top = float(host.max()) if host.size else 0.0
        positive = host[host > 0.0]
        smallest = float(positive.min()) if positive.size else 0.0
        log_scale = 0.0
        if top > 0.0 and (always or not (LO <= top <= HI)):
            host = host / top
            log_scale = math.log(top)
        M = math.log(top) - log_scale if top > 0.0 else -math.inf
        L = math.log(smallest) - log_scale if smallest > 0.0 else math.inf
        # An entry the division took below the smallest normal float64 kept fewer digits,
        # or none; its stored error is at most that number.
        E = (_LOG_TINY64 + math.log(2.0)) if L < _LOG_TINY64 else _NO_ERROR
        host.flags.writeable = False
        hit = cached[always] = (host, log_scale, M, L, E)
    return hit


def _log_sum_exp(*logs: float) -> float:
    """``log(sum(exp(x) for x in logs))`` on the host, minus infinity for no term."""
    finite = [x for x in logs if x != -math.inf]
    if not finite:
        return -math.inf
    top = max(finite)
    if top == math.inf:
        return math.inf
    return top + math.log(sum(math.exp(x - top) for x in finite))
