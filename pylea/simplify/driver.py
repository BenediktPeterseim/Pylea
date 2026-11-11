"""The rewriting loop.

`simplify` applies the rules in turn, over and over, until a full round leaves the DPP
unchanged.  A rule is a `Program -> Program` function that may rewrite at any number of
sites, and reports firing nowhere by returning its argument.

Every rewrite step strictly lowers `size`, the number of symbols in the DPP, so the loop
terminates whatever order the rules run in, after at most `size` steps; the order is
chosen for speed alone.  `check=True` asserts the decrease after each rule, at the cost
of a sweep, which is meant for use while writing a rule.

Besides the rules, the loop merges equal bodies and drops bodies the root no longer calls,
through `pylea.ir.program.intern`, after every rule that changed the DPP.  Dropping a
body nothing calls is the removal rule; merging two equal definitions is not one of the
rules but lowers `size` and the contraction cost alike.

`budget_ms` bounds the phase by wall clock.  A rule has completed by the time it returns,
so stopping between rules leaves a valid DPP simplified less.  The deadline is tested
between rules, and is passed on to every rule taking a `deadline` argument.  Those are the
rules whose cost grows faster than the size of a body: `pylea.simplify.fold.block_fold`,
`pylea.simplify.fold.bottom_cse`, `pylea.simplify.mask.mask`,
`pylea.simplify.absorb.absorb` and `pylea.simplify.lift.lift`.  Once the deadline has
passed, each of these stops looking for sites and applies the rewrites it has found, which
takes one sweep of each body.  The phase therefore outlasts its budget by about one sweep
of the DPP, whichever rule is running when the budget runs out.
"""
from __future__ import annotations

from inspect import signature
from time import perf_counter

from pylea.ir.program import Program, intern, reachable

from pylea.simplify.absorb import absorb
from pylea.simplify.boolean import bool_rules
from pylea.simplify.cse import flip_cse
from pylea.simplify.dce import dce
from pylea.simplify.fold import block_fold, bottom_cse, const_fold
from pylea.simplify.inline import inline
from pylea.simplify.lift import lift
from pylea.simplify.observe import observe_rules
from pylea.simplify.mask import mask

# Cheapest first, so that a rule removing work leaves less for the later ones.  `dce` leads
# because narrowing a callee is what brings the others' preconditions about.  The three
# rules over syntactic implication, `mask`, `absorb` and `lift`, come last and cost the
# most.
RULES = (dce, inline, bool_rules, observe_rules, const_fold, block_fold, bottom_cse,
         flip_cse, mask, absorb, lift)


class SizeDidNotDecrease(AssertionError):
    """Raised under `check` where a rule changed the DPP without lowering `size`."""


def size(prog: Program) -> int:
    """The number of symbols of `prog`: one per op argument, result, body input and output,
    plus one per op and one per body, summed over the reachable bodies.

    Every rule lowers it, so it falls monotonically across the phase:

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def shared() -> Bool:
        ...     v = flip(0.3)
        ...     return (v & v) | (v & v)
        >>> size(shared.program()), size(simplify(shared.program()))
        (16, 4)
    """
    total = 0
    for i in reachable(prog):
        b = prog.bodies[i]
        total += 1 + len(b.inputs) + len(b.outputs)
        total += sum(1 + len(op.args) + len(op.outs) for op in b.ops)
    return total


def simplify(prog: Program, rules=RULES, check: bool = False,
             budget_ms: float | None = None) -> Program:
    """`prog` with `rules` applied until none of them changes it, or the budget runs out.

    `budget_ms` is a wall-clock bound on the whole phase, counted from this call.  A rule
    taking a `deadline` argument is given the end of the budget as one.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.ir.program import op_count
        >>> @DPP
        ... def redundant() -> Bool:
        ...     x = flip(0.3)
        ...     return (x & x) | (x & x)
        >>> op_count(redundant.program()), op_count(simplify(redundant.program()))
        (4, 1)
    """
    deadline = None if budget_ms is None else perf_counter() + budget_ms / 1000.0
    if deadline is not None:
        rules = tuple(_until(r, deadline) if "deadline" in signature(r).parameters else r
                      for r in rules)
    return fixpoint(intern(prog), rules, check, deadline)


def fixpoint(prog: Program, rules, check: bool = False,
             deadline: float | None = None) -> Program:
    """`prog` with `rules` applied in turn, over and over, until a round changes nothing.

    `deadline` is a `time.perf_counter` reading past which no further rule is started, so
    at most one rule runs beyond it.  A rule whose result interns back to the DPP it
    was given counts as no change.

    Unlike `simplify`, it runs the rules it is given on `prog` as it stands, without
    interning it first, which is how a caller wanting a few cheap rules uses it:

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> from pylea.ir.program import op_count
        >>> from pylea.simplify import bool_rules, dce
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (one,) = b.emit(Prim.const(1, 1), ())
        >>> (y,) = b.emit(Prim.and_(), (x, one))         # AND(x, 1) is x
        >>> (z,) = b.emit(Prim.not_(), (y,))
        >>> (w,) = b.emit(Prim.not_(), (z,))             # NOT(NOT(x)) is x
        >>> prog = Program((b.finish(inputs=(), outputs=(w,)),), root=0)
        >>> op_count(prog), op_count(fixpoint(prog, (bool_rules, dce)))
        (5, 1)
    """
    while True:
        changed = False
        for rule in rules:
            if deadline is not None and perf_counter() >= deadline:
                return prog
            result = rule(prog)
            if result is prog:                  # the rule reports firing nowhere
                continue
            result = intern(result)
            if result == prog:
                continue
            if check and size(result) >= size(prog):
                raise SizeDidNotDecrease(
                    f"{rule.__name__} changed the DPP but size went "
                    f"{size(prog)} -> {size(result)}")
            prog, changed = result, True
        if not changed:
            return prog


def _until(rule, deadline: float):
    """`rule` bound by `deadline`, in the plain `Program -> Program` form the loop wants."""
    def bounded(prog: Program) -> Program:
        return rule(prog, deadline=deadline)

    bounded.__name__ = rule.__name__
    return bounded
