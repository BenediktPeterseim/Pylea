"""Inlining.

`inline` splices a body into its caller where the body has exactly one call site, counted
over the whole DPP, and drops the body.  The call and the definition go, and the
definition's ops take the call's place, so the DPP loses two symbols per interface
position of the body.

A body with several sites is left alone whatever its size: splicing it everywhere would
copy it, and a chain of definitions each calling the one below it twice would unfold to a
DPP exponential in its depth.  This holds for a body of one op as well.  A copy of its
op can take more symbols than the call it replaces, as where the op reads a wire twice or
has results the body does not return, so splicing it at every site does not lower `size`
in general.

Splicing a callee that is too wide to hold as one matrix is a separate matter, done out of
necessity rather than profit; `pylea.ir.inline.flatten_wide_calls` is that one.
"""
from __future__ import annotations

from pylea.ir.body import prune_wires
from pylea.ir.inline import inline_all
from pylea.ir.program import Program, intern, reachable, use_counts


def inline(prog: Program) -> Program:
    """`prog` with every body called from exactly one site spliced into its caller.

        >>> from pylea import DPP, Bool, flip
        >>> from pylea.ir.program import reachable
        >>> @DPP
        ... def helper() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def uses_once() -> Bool:
        ...     return helper()
        >>> len(reachable(uses_once.program())), len(reachable(inline(uses_once.program())))
        (2, 1)
    """
    sites = use_counts(prog)
    chosen = {i for i in reachable(prog) if sites[i] == 1}
    if not chosen:
        return prog

    bodies = list(prog.bodies)
    for i in reachable(prog):           # callees first: what is spliced in is already flat
        spliced = inline_all(Program(tuple(bodies), prog.root), i, chosen.__contains__)
        bodies[i] = prune_wires(spliced)
    return intern(Program(tuple(bodies), prog.root))
