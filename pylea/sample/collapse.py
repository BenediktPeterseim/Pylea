"""Exact sub-tables inside the sampler.

`collapse` walks a DPP's bodies callees-first and replaces every body it can afford
to evaluate exactly by the single MATRIX primitive standing for its likelihood table.
The residual DPP is what the sampler runs: a hybrid where the narrow parts are exact
and only the wide remainder is estimated.  Two effects pay for it.  A collapsed body's
internal conditioning becomes a substochastic column, which the sweep interpreter
consumes as a soft weight, so observations that would reject or starve particles inside the
callee weight them instead; and a call tree whose interfaces are narrow stops being
unfolded at all, which is what gives the sampling lane the exact engine's depth scaling
on such DPPs.

A body qualifies when every call inside it has already been collapsed, its interface
wires are single bits with at most `COLLAPSE_IN_BITS` in and `COLLAPSE_OUT_BITS` out,
the greedy plan prices its contraction under `COLLAPSE_COST_CAP` scalar multiplications,
and the resulting table fits float64.  The output cap is tighter than the input cap
because the sampler's generic column path draws from ``2**out_bits`` rows per
particle.  A body that is a pure function of its inputs is also left alone: as gates the
packed executor advances 64 particles per word through it, while as a table it would
force a per-particle gather.  The caps bound the pass's own cost: each attempted body
costs one greedy compile and at most one capped evaluation.

Only exactly evaluated tables are spliced.  An estimated sub-table would compose
soundly only through monotone two-run propagation, running the caller once with the
entrywise lower table and once with the upper, since plugging a point estimate into a
body called more than once biases the product; that machinery is not built, and a body
too wide to evaluate exactly is simply left to the residual sampler.
"""
from __future__ import annotations

from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern, reachable
from pylea.semantics.likelihood_table import scaled_matrix

# Interface caps, as bits, and the per-body greedy contraction-cost cap.  The bit caps
# bound a spliced table to 2**20 entries, the same bound `pylea.sample.cost_model`
# puts on a sampled table's width, with the output side tighter because the sampler's
# column draw costs a row gather per particle per halving of the row count.  The cost
# cap is in scalar multiplications of the greedy plan, priced by
# `pylea.compile.cost.OPS_PER_MS`: 1e6 is 0.01 ms of arithmetic on the numpy backend
# of the machine that constant was timed on, below the per-step dispatch floor, so an
# attempted collapse costs its greedy compile and dispatch rather than its evaluation.
# Re-measure with `OPS_PER_MS`, after changing the evaluator or the cost model.
COLLAPSE_IN_BITS = 12
COLLAPSE_OUT_BITS = 8
COLLAPSE_COST_CAP = 1e6


def collapse(prog: Program) -> Program:
    """`prog` with every affordable body replaced by its exact table.

    The root is never collapsed; collapsing it would be the exact engine's whole job.
    The result is interned, so calls to bodies that became equal tables merge.

        >>> from pylea import DPP, Bool, flip, observe
        >>> from pylea.ir.program import op_count
        >>> @DPP
        ... def noisy_and() -> Bool:
        ...     x, y = flip(0.5), flip(0.5)
        ...     observe(x | y)
        ...     return x & y
        >>> @DPP
        ... def twice() -> Bool:
        ...     return noisy_and() | noisy_and()
        >>> before = twice.program()
        >>> after = collapse(before)
        >>> len(after.bodies), op_count(after)
        (1, 3)

    The callee's internal observe survives as the substochastic column of its table:

        >>> import numpy as np
        >>> table = [op.callee.matrix for op in after.main.ops
        ...          if op.callee.tag.name == "MATRIX"][0]
        >>> np.round(table.sum(), 4)
        np.float64(0.75)
    """
    tables: dict[int, Prim] = {}
    bodies = list(prog.bodies)
    for i in reachable(prog):
        rewritten = _with_tables(bodies[i], tables)
        bodies[i] = rewritten
        if i == prog.root:
            continue
        if any(op.is_call for op in rewritten.ops):
            continue                    # an uncollapsed callee survives inside
        table = _exact_table(rewritten)
        if table is not None:
            tables[i] = table
    if not tables:
        return prog
    return intern(Program(tuple(bodies), prog.root))


def _with_tables(body: Body, tables: dict[int, Prim]) -> Body:
    """`body` with every call to a collapsed callee replaced by its table primitive."""
    if not any(not isinstance(op.callee, Prim) and op.callee in tables
               for op in body.ops):
        return body
    ops = tuple(op if isinstance(op.callee, Prim) or op.callee not in tables
                else Op(tables[op.callee], op.args, op.outs)
                for op in body.ops)
    return Body(body.bits, body.inputs, body.outputs, ops)


def _exact_table(body: Body) -> Prim | None:
    """The MATRIX primitive standing for `body`, or None where it is not affordable.

    Affordability is decided before evaluating: the interface must be single-bit wires
    within the caps, and the greedy plan's contraction cost must fall under the cap.  A
    table whose magnitude leaves float64 is also declined; the sampler needs a plain
    array.
    """
    if body.in_bits > COLLAPSE_IN_BITS or body.out_bits > COLLAPSE_OUT_BITS:
        return None
    if any(body.bits[w] != 1 for w in body.inputs + body.outputs):
        return None
    if body.inputs and all(op.callee.is_deterministic for op in body.ops
                           if isinstance(op.callee, Prim)):
        # A pure function of its inputs is a gate cluster: as gates it runs packed at a
        # word per 64 particles, while as a table it forces a per-particle gather, so
        # collapsing it would slow the sampler it means to help.
        return None

    from pylea.compile.cost import contraction_cost
    from pylea.compile.lower import compile_program
    from pylea.compile.order import GreedyOrderFinder
    from pylea.eval.backend import NumpyBackend
    from pylea.eval.interp import evaluate
    from pylea.syntax.compiled import DEFAULT_KAPPA

    single = Program((body,), root=0)
    try:
        plans, root = compile_program(single, GreedyOrderFinder(), kappa=DEFAULT_KAPPA)
    except Exception:                   # noqa: BLE001, an unplannable body stays sampled
        return None
    if contraction_cost(plans) > COLLAPSE_COST_CAP:
        return None
    array, scale = evaluate(plans, root, backend=NumpyBackend())
    try:
        matrix = scaled_matrix(array, scale)
    except ValueError:                  # the magnitude leaves float64
        return None
    return Prim.from_matrix(matrix)
