"""Masked reads.

A disjunction `y = OR(x, c)` differs from `c` only while `x` is 1.  If `x` being 1
decides, through the other gates, a wire on every circuit path from `y` to a wire that
leaves the circuit, then nothing further up can tell `y` from `c`, and `y` is replaced by
`c`.  A circuit path runs from a wire to the result of a gate reading it, and on; a wire
leaves the circuit where it is a body output or an argument of any op that is not a gate.
Dually, a conjunction `y = AND(x, c)` masked while `x` is 0 is `c`.  A gate masked at the
value of `x` that decides it is that constant: `OR(x, c)` masked while `x` is 0 is 1, and
`AND(x, c)` masked while `x` is 1 is 0.

What `x` decides is found by propagating its value forwards through the gates other than
`y`'s own, so that the gate is never the reason it is masked.  This is syntactic
implication in the sense of `pylea.simplify.implication` without that gate, restricted to
what follows from one literal.

Each rewrite deletes a gate and renames its result to its other argument, or replaces a
gate by a constant over its result alone; nothing moves, which is what lets `x` be any
wire at all.  Flip lifting, in `pylea.simplify.lift`, remains for the reads of a coin flip
that no masking removes, where forcing the dominator needs every one of them.

`mask` works through the wires read more than once, dropping as many of their reads as
the remaining ones keep masked, one pass per body until none can be dropped.  What one
value of a wire decides is propagated once for all of the wire's readers, and only as far
as their tests need; a dropped read is taken out of it rather than propagated afresh.
"""
from __future__ import annotations

from pylea.ir.body import Body
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program

from pylea.simplify.common import (
    Circuit, Propagation, Sweep, constant_bits, each_body, expired, gate_tag, leaving,
)


def mask(prog: Program, deadline: float | None = None) -> Program:
    """`prog` with every masked read dropped.

    `deadline` is a `time.perf_counter` reading past which no further masked read is
    looked for; the reads dropped before it are complete rewrites.

    Here `x` is read by two disjunctions under a third.  While `x` is 1 the first forces
    the top, so the second is masked and becomes `q`:

        >>> import numpy as np
        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.program import Program, consumer_counts
        >>> from pylea.ir.reference import semantics
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (p,) = b.emit(Prim.flip(0.4), ())
        >>> (q,) = b.emit(Prim.flip(0.5), ())
        >>> (a,) = b.emit(Prim.or_(), (x, p))
        >>> (c,) = b.emit(Prim.or_(), (x, q))
        >>> (d,) = b.emit(Prim.or_(), (a, c))
        >>> prog = Program((b.finish(inputs=(), outputs=(d,)),), root=0)
        >>> out = mask(prog)
        >>> consumer_counts(prog.main)[x], consumer_counts(out.main)[x]
        (2, 1)
        >>> bool(np.allclose(semantics(prog), semantics(out)))
        True
    """
    return each_body(prog, lambda body: _drop_all(body, deadline), deadline)


def _drop_all(body: Body, deadline: float | None) -> Body:
    current = body
    while not expired(deadline):
        rewrites = _masked(current, deadline)
        if not rewrites:
            break
        current = _apply(current, rewrites)
    return current


# ------------------------------------------------------------------------- the test

def _escapes(circuit: Circuit, out: set[int]) -> set[int]:
    """The wires from which some circuit path reaches a wire in `out`."""
    escapes = set(out)
    for j in reversed(range(len(circuit.outs))):
        y = circuit.outs[j]
        if y in escapes and circuit.tags[j] is not None:
            escapes.update(circuit.args[j])
    return escapes


def _is_masked(circuit: Circuit, out: set[int], escapes: set[int], y: int,
               propagation: Propagation, without: int) -> bool:
    """Whether every circuit path from `y` to a wire in `out` meets a wire decided without
    the gates in the bit set `without`.

    The propagation is advanced only as far as the wire being visited.  The search ends at
    a wire from which a path reaches `out` once the propagation has decided all it ever
    will, none of it above that wire: no wire on the path is then decided.
    """
    readers, outs, made_at = circuit.readers, circuit.outs, circuit.made_at
    known, needs = propagation.known, propagation.needs
    stack = [y]
    seen = {y}
    while stack:
        w = stack.pop()
        at = made_at[w]
        propagation.advance(at)
        if w in known and not needs[w] & without:
            continue
        if w in out or (propagation.done and propagation.last <= at and w in escapes):
            return False
        for k in readers.get(w, ()):
            z = outs[k]
            if z not in seen:
                seen.add(z)
                stack.append(z)
    return True


# ------------------------------------------------------------------------- the sites

def _masked(body: Body, deadline: float | None = None) -> dict[int, int | None]:
    """The gates to rewrite, as op index -> the wire that replaces the result, or `None`
    for a gate that becomes its decisive constant.

    The readers of each wire are tried in op order for each value, each against what the
    gates not already rewritten decide, never counting what its own gate decides, so that
    what one rewrite relied on stays.  The rewrites found before `deadline` are therefore
    valid together, however few they are.
    """
    circuit = Circuit(body)
    out = leaving(body)
    escapes = _escapes(circuit, out)
    constant = {w for op in body.ops if constant_bits(op.callee) is not None for w in op.outs}
    rewrites: dict[int, int | None] = {}
    # The gates rewritten: every propagation leaves them out, and `exclude` adds to them.
    excluded: set[int] = set()
    for x, js in circuit.readers.items():
        if len(js) < 2 or x in constant or body.bits[x] != 1:
            continue
        eligible = [j for j in js if circuit.tags[j] in (Tag.AND, Tag.OR)
                    and body.ops[j].args.count(x) == 1]
        if not eligible:
            continue
        bit = {j: 1 << n for n, j in enumerate(js)}
        for value in (1, 0):
            propagation = Propagation(circuit, x, value, excluded, bit)
            propagation.advance(js[-1])
            if len(propagation.known) == 1:
                continue            # only a gate that nothing reads would be masked
            for j in eligible:
                if expired(deadline):
                    return rewrites
                if j in excluded:
                    continue
                y = circuit.outs[j]
                if not _is_masked(circuit, out, escapes, y, propagation, bit[j]):
                    continue
                decides = (circuit.tags[j] is Tag.OR) == (value == 0)
                (c,) = [w for w in body.ops[j].args if w != x]
                rewrites[j] = None if decides else c
                propagation.exclude(j)
    return rewrites


def _apply(body: Body, rewrites: dict[int, int | None]) -> Body:
    s = Sweep(body)
    for j, op in enumerate(body.ops):
        args = s.begin(op)
        target = rewrites.get(j)
        if j not in rewrites:
            s.keep(op, args)
        elif target is None:
            s.replace(op, Prim.const(1 if gate_tag(op) is Tag.OR else 0, 1))
        else:
            s.alias(op.outs[0], s.resolve(target))
    return s.finish()
