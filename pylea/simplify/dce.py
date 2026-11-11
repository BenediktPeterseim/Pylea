"""Dead code elimination and specialisation.

Dead code elimination deletes an op none of whose results is read.  Specialisation deletes
an output of a body that no call site reads, and the result wire from each call.  `dce`
does both in one pass, because each exposes the other: an output no caller reads makes the
ops producing it dead, and deleting an op can leave a call's result unread.

An op may be deleted only where summing out its outputs leaves 1, which is
`pylea.ir.prim.Prim.is_discardable`; for a call, every op of the callee must have that
property, recursively.  An op that conditions scales the answer whatever reads it, so it is
kept however dead its surroundings.  The property is read off the matrix, so a table whose
columns sum to 1 is deleted like a `flip`, and one whose columns do not is kept like an
`observe`.

Liveness is computed callers before callees, which takes one pass because the call graph is
acyclic: by the time a body is reached, every site that could read it has been seen.

A body is narrowed once, to the union of the output positions read over all of its call
sites, so narrowing never splits one body into two specialised copies.  The root keeps all
of its outputs, those being the answer.
"""
from __future__ import annotations

from pylea.ir.body import Body, Op, prune_wires
from pylea.ir.prim import Prim
from pylea.ir.program import Program, discardable, reachable, with_bodies


def dce(prog: Program) -> Program:
    """`prog` with the unread ops deleted and every body narrowed to its live outputs.

    A coin flip nothing reads goes; an `observe` stays, since it scales the answer:

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> from pylea.ir.program import Program, op_count
        >>> b = BodyBuilder()
        >>> (kept,) = b.emit(Prim.flip(0.3), ())
        >>> (unread,) = b.emit(Prim.flip(0.7), ())
        >>> prog = Program((b.finish(inputs=(), outputs=(kept,)),), root=0)
        >>> op_count(prog), op_count(dce(prog))
        (2, 1)
    """
    keeps_op = discardable(prog)
    live_positions = _live_positions(prog, keeps_op)
    bodies = list(prog.bodies)
    for i in reachable(prog):
        rebuilt = _rebuild(prog.bodies[i], live_positions[i], live_positions, keeps_op)
        bodies[i] = prog.bodies[i] if rebuilt == prog.bodies[i] else rebuilt
    return with_bodies(prog, bodies)


def _live_positions(prog: Program, keeps_op) -> dict[int, set[int]]:
    """The output positions of each body that some call site reads."""
    live: dict[int, set[int]] = {i: set() for i in reachable(prog)}
    live[prog.root] = set(range(len(prog.main.outputs)))

    for i in reversed(reachable(prog)):                     # callers first
        body = prog.bodies[i]
        keep, live_wires = _analyse(body, live[i], keeps_op)
        for op, kept in zip(body.ops, keep):
            if kept and not isinstance(op.callee, Prim):
                live[op.callee].update(p for p, w in enumerate(op.outs) if w in live_wires)
    return live


def _analyse(body: Body, live_out: set[int], keeps_op) -> tuple[list[bool], set[int]]:
    """Which ops of `body` to keep, and which of its wires are read, given its live outputs."""
    live = {body.outputs[p] for p in live_out}
    keep = [False] * len(body.ops)
    for j in reversed(range(len(body.ops))):
        op = body.ops[j]
        if any(w in live for w in op.outs) or not _discardable(op, keeps_op):
            keep[j] = True
            live.update(op.args)
    return keep, live


def _rebuild(body: Body, live_out: set[int], live_positions: dict[int, set[int]],
             keeps_op) -> Body:
    """`body` restricted to its kept ops and live outputs, with its calls narrowed to match."""
    keep, _reads = _analyse(body, live_out, keeps_op)
    ops = []
    for op, kept in zip(body.ops, keep):
        if not kept:
            continue
        if isinstance(op.callee, Prim):
            ops.append(op)                  # a primitive's arity is fixed
        else:
            positions = sorted(live_positions[op.callee])
            ops.append(Op(op.callee, op.args, tuple(op.outs[p] for p in positions)))
    outputs = tuple(body.outputs[p] for p in sorted(live_out))
    return prune_wires(Body(body.bits, body.inputs, outputs, tuple(ops)))


def _discardable(op: Op, keeps_op: dict[int, bool]) -> bool:
    return op.callee.is_discardable if isinstance(op.callee, Prim) else keeps_op[op.callee]
