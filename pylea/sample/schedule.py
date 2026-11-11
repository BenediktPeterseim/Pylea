"""Schedule rewrites: lower-variance conditioning at unchanged semantics.

`schedule` applies four `Body -> Body` passes to an unfolded body, in this order:

  1. `fold`: constant propagation through the Boolean gates, deletion of a gate reading
     one wire twice, and removal of dead deterministic ops, by the rules of
     `pylea.simplify` on the wrapped body.  A traced comparison against a constant
     carries the constant's bits through an equality circuit; folding reduces that
     circuit to a conjunction of literals, which is what the two passes below match on.
  2. `split_observes`: an OBSERVE whose argument is the single-use output of an AND
     becomes two OBSERVEs on the AND's inputs, recursively, and the consumed AND is
     dropped.  One terminal rejection test becomes one survival test per conjunct.
     Splitting never increases the stage rarity `R`: for a product ``p = p1 * p2`` the
     difference ``(1/p - 1) - (1/p1 - 1) - (1/p2 - 1)`` equals
     ``(1 - p1)(1 - p2) / (p1 p2)``, which is non-negative.
  3. `push_evidence`: a table op every one of whose output bits is observed at a constant
     value becomes the observed row of the table, a soft weight on the op's inputs, plus
     a CONST producing the outputs.  This is likelihood weighting for that node: the
     particle's weight picks up the row entry and never becomes zero unless that entry
     is, where the plain observes kill every particle that misses the observed value.
     Without it, sweeps on DPPs with many observed table nodes die outright.
  4. `asap_observes`: a legal reordering that runs every OBSERVE as soon as its ancestor
     cone allows, emitting each observe's cone and then the observe, in observe order.
     Stage probabilities depend only on the relative order of the observes, so this moves
     resampling points earlier and never changes `R`.

`fold` runs a second time between the last two passes, clearing the negations and
constants `push_evidence` strands.

Every pass preserves the unnormalised semantics exactly; the differential tests check
each against `pylea.ir.reference.semantics`.  A pass that removes ops prunes the wires it
strands, since a wire named by nothing must not survive into an enumeration oracle.

The recognition in `push_evidence` works on what the passes in front of it produce, not
on the comparator circuit itself: after folding and splitting, an observed table node
appears as one OBSERVE per output bit, reading the bit directly for an observed 1 and
through a NOT for an observed 0.  If the comparator's emitted circuit ever changes shape,
the recognition silently stops firing and sweeps with deep observations collapse again;
a regression test therefore asserts that the pass fires on a traced
``observe(node == value)``.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from pylea.ir.body import Body, Op, prune_wires
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program
from pylea.simplify.absorb import idempotence
from pylea.simplify.boolean import bool_rules
from pylea.simplify.dce import dce
from pylea.simplify.driver import fixpoint


def schedule(body: Body) -> Body:
    """`body` rewritten by all four passes, ready for `pylea.sample.sweep.SweepProgram`.

        >>> from pylea import DPP, Bool, Enum, flip, observe
        >>> from pylea.sample.sweep import unfold
        >>> class Coin(Enum):
        ...     elements = ["heads", "tails"]
        >>> @DPP
        ... def observed_toss() -> Bool:
        ...     first = Coin.random({"heads": 0.3, "tails": 0.7})
        ...     observe(first == "heads")
        ...     return flip(0.5)
        >>> scheduled = schedule(unfold(observed_toss.program()))
        >>> [op.callee.tag.name for op in scheduled.ops]
        ['MATRIX', 'FLIP']

    The hard observe is gone: the observation became the observed row of the coin flip's
    table, a soft weight that no particle can miss.  The constant standing for the observed
    value was read by nothing further and folded away with it.
    """
    body = fold(body)
    body = split_observes(body)
    body = push_evidence(body)
    body = fold(body)
    return asap_observes(body)


def fold(body: Body) -> Body:
    """`body` with constants propagated through its gates and dead deterministic ops gone.

    Runs `pylea.simplify.boolean.bool_rules`, `pylea.simplify.absorb.idempotence` and
    `pylea.simplify.dce.dce` on the wrapped body until none of them changes it; all three
    preserve the unnormalised semantics exactly.  Multi-bit constants are split into one
    CONST per bit first, since the propagation tracks single-bit wires; without the split,
    a comparison against a wide value never folds.

        >>> from pylea.ir.body import BodyBuilder
        >>> b = BodyBuilder()
        >>> (x,) = b.emit(Prim.flip(0.3), ())
        >>> (y,) = b.emit(Prim.and_(), (x, x))
        >>> _ = b.emit(Prim.observe(), (y,))
        >>> [op.callee.tag.name for op in fold(b.finish((), (x,))).ops]
        ['FLIP', 'OBSERVE']
    """
    prog = fixpoint(Program((_split_constants(body),), root=0), (bool_rules, idempotence, dce))
    return prune_wires(prog.main)


def _split_constants(body: Body) -> Body:
    """`body` with every multi-bit CONST replaced by one single-bit CONST per wire."""
    ops: list[Op] = []
    for op in body.ops:
        prim = op.callee
        if isinstance(prim, Prim) and prim.tag is Tag.CONST and prim.out_bits > 1:
            value, bits = int(prim.params[0]), int(prim.params[1])
            ops.extend(Op(Prim.const((value >> (bits - 1 - k)) & 1, 1), (), (w,))
                       for k, w in enumerate(op.outs))
        else:
            ops.append(op)
    return Body(body.bits, body.inputs, body.outputs, tuple(ops))


def split_observes(body: Body) -> Body:
    """`body` with every OBSERVE of a single-use AND output split into its conjuncts.

    Each observe's cone of single-use ANDs is walked once, depth first, so an AND tree
    becomes one observe per leaf in left-to-right order.  The walk is linear in the op
    count: replacing an observed AND by observes on its arguments preserves every use
    count, so the counts taken up front stay valid throughout.  The consumed ANDs are
    dropped; keeping one would both waste work and hold its output's use count above
    one, blocking the nested splits.
    """
    uses = Counter(w for op in body.ops for w in op.args)
    uses.update(body.outputs)
    producer = {w: i for i, op in enumerate(body.ops) for w in op.outs}
    replacement: dict[int, list[Op]] = {}
    dropped: set[int] = set()
    for i, op in enumerate(body.ops):
        if not (isinstance(op.callee, Prim) and op.callee.tag is Tag.OBSERVE):
            continue
        leaves: list[int] = []
        stack = [op.args[0]]
        while stack:
            wire = stack.pop()
            at = producer.get(wire)
            feed = body.ops[at] if at is not None else None
            if (feed is not None and at not in dropped
                    and isinstance(feed.callee, Prim) and feed.callee.tag is Tag.AND
                    and uses[wire] == 1):
                assert at is not None
                dropped.add(at)
                stack.append(feed.args[1])
                stack.append(feed.args[0])
            else:
                leaves.append(wire)
        if len(leaves) > 1:
            replacement[i] = [Op(Prim.observe(), (w,), ()) for w in leaves]
    if not replacement:
        return body
    ops = [new for i, op in enumerate(body.ops) if i not in dropped
           for new in replacement.get(i, [op])]
    return prune_wires(Body(body.bits, body.inputs, body.outputs, tuple(ops)))


def push_evidence(body: Body) -> Body:
    """`body` with fully observed table nodes replaced by their observed row and a CONST.

    A table op qualifies when every one of its output bits carries exactly one observed
    value: an OBSERVE reading the bit directly observes 1, and one reading it through a
    NOT observes 0.  A bit observed at both values is a contradiction, and the op
    is left alone rather than rewritten; the plain observes then correctly assign the
    DPP zero mass.  Other readers of the node's outputs are unaffected: they read the
    CONST wires, which under the observations is what the node is.
    """
    producer: dict[int, int] = {w: i for i, op in enumerate(body.ops) for w in op.outs}

    def prim_at(site: int | None, tag: Tag) -> Prim | None:
        """The primitive at `site` where it carries `tag`, else None."""
        if site is None:
            return None
        callee = body.ops[site].callee
        return callee if isinstance(callee, Prim) and callee.tag is tag else None

    # site of a table op -> {bit position: (value, observe site)}
    evidence: dict[int, dict[int, tuple[int, int]]] = {}
    contradicted: set[tuple[int, int]] = set()
    for i, op in enumerate(body.ops):
        if prim_at(i, Tag.OBSERVE) is None:
            continue
        wire = op.args[0]
        value = 1
        feed_at = producer.get(wire)
        if prim_at(feed_at, Tag.NOT) is not None:
            assert feed_at is not None
            value = 0
            wire = body.ops[feed_at].args[0]
            feed_at = producer.get(wire)
        if feed_at is None or prim_at(feed_at, Tag.MATRIX) is None:
            continue
        node = body.ops[feed_at]
        bit = node.outs.index(wire)
        known = evidence.setdefault(feed_at, {})
        if bit in known and known[bit][0] != value:
            contradicted.add((feed_at, bit))
        known.setdefault(bit, (value, i))

    replacement: dict[int, list[Op]] = {}
    drop: set[int] = set()
    for site, known in evidence.items():
        node = body.ops[site]
        prim = prim_at(site, Tag.MATRIX)
        assert prim is not None
        bits = len(node.outs)
        if len(known) != bits or any((site, b) in contradicted for b in known):
            continue
        pattern = 0
        for b in range(bits):
            pattern = (pattern << 1) | known[b][0]
        matrix = np.asarray(prim.matrix, dtype=float)
        replacement[site] = [
            Op(Prim.from_matrix(matrix[pattern:pattern + 1, :]), node.args, ()),
            Op(Prim.const(pattern, bits), (), node.outs),
        ]
        drop.update(observe_site for _value, observe_site in known.values())

    if not replacement:
        return body
    ops = [new for i, op in enumerate(body.ops) if i not in drop
           for new in replacement.get(i, [op])]
    return prune_wires(Body(body.bits, body.inputs, body.outputs, tuple(ops)))


def asap_observes(body: Body) -> Body:
    """`body` reordered so each OBSERVE runs as soon as its ancestor cone is complete.

    Cone by cone: for each OBSERVE in order, everything its argument depends on and then
    the observe itself; the remaining ops follow.  This pulls a comparison circuit ahead
    of unrelated sampling, where a greedy readiness order would leave a terminal observe
    terminal.
    """
    producer = {w: op for op in body.ops for w in op.outs}
    emitted: set[int] = set()
    ordered: list[Op] = []

    def emit_cone(op: Op) -> None:
        stack = [op]
        while stack:
            top = stack[-1]
            missing = [producer[a] for a in top.args
                       if a in producer and id(producer[a]) not in emitted]
            if missing:
                stack.extend(missing)
                continue
            stack.pop()
            if id(top) not in emitted:
                emitted.add(id(top))
                ordered.append(top)

    for op in body.ops:
        if isinstance(op.callee, Prim) and op.callee.tag is Tag.OBSERVE:
            emit_cone(op)
    for op in body.ops:
        if id(op) not in emitted:
            emit_cone(op)
    return Body(body.bits, body.inputs, body.outputs, tuple(ordered))
