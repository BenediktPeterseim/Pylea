"""Syntactic implication between the literals of a body.

A literal is a one-bit wire or its negation.  `Implications` computes the smallest relation
on the literals of a body that is reflexive and transitive, closed under contraposition,
and satisfies, for each gate of the body,

    y = AND(x1, x2):  y implies x1 and x2;  l implies y whenever l implies x1 and x2
    y = OR(x1, x2):   x1 and x2 imply y;    y implies l whenever x1 and x2 imply l
    y = NOT(x):       y implies NOT x, and NOT x implies y

Where a literal implies another in this sense, the first being true forces the second to
be true for every choice of values of the wires that the gates admit; the converse fails,
since the relation reads the gates one at a time and never reasons by cases over two
literals at once.  Ops other than gates contribute nothing, so their results are literals
that imply and are implied by themselves alone.

The relation is held as a bit matrix, a row per literal over the literals it implies, and
computed by rounds until a round changes nothing.  A round runs up the body once and down
it once, each gate joining the rows of its inputs and its result in the direction the rules
read, and then applies, to each row changed since it was last looked at, the rule that
adds a conjunction to a row holding both of its inputs.  Contraposition holds of the result
because the rules come in contrapositive pairs; in a body without negations only the
literals of the wires themselves are needed, their negations never entering a derivation,
and only those are held.  Only the literals of wires that some gate reads or produces are
held at all.  The matrix takes ``n * n / 8`` bytes for `n` literals held; a sparse form
would take less only where few pairs of literals are related.

A literal implying itself is left implicit, each join adding the literals it joins where
they belong to the result, so that building the matrix writes nothing up front: memory is
taken as rows are joined.  Given a
deadline, nothing is allocated where it has passed already, and the computation stops once
it passes, part way through a round if need be.  Every bit set is a valid implication at any
point, so the relation then holds a subset of the implications; a rule reading it fires at
fewer sites and nowhere it should not.  Without a deadline the whole matrix is computed,
whatever its size.
"""
from __future__ import annotations

import numpy as np

from pylea.ir.body import Body
from pylea.ir.prim import Tag

from pylea.simplify.common import expired, gate_tag, neg, pos


class Implications:
    """The syntactic implications of one body; see the module docstring.

        >>> from pylea.ir.body import BodyBuilder
        >>> from pylea.ir.prim import Prim
        >>> b = BodyBuilder()
        >>> x, y = b.fresh(), b.fresh()
        >>> (a,) = b.emit(Prim.and_(), (x, y))
        >>> (o,) = b.emit(Prim.or_(), (x, y))
        >>> rel = Implications(b.finish(inputs=(x, y), outputs=(a, o)))
        >>> rel.implies(pos(a), pos(o)), rel.implies(pos(o), pos(a))
        (True, False)
    """

    def __init__(self, body: Body, deadline: float | None = None) -> None:
        stopped = expired(deadline)
        gates = [] if stopped else [
            op for op in body.ops
            if gate_tag(op) is not None and all(body.bits[w] == 1 for w in op.args + op.outs)]
        self.monotone = monotone = all(gate_tag(op) is not Tag.NOT for op in gates)
        touched = sorted({w for op in gates for w in op.args + op.outs})
        held = [pos(w) for w in touched] if monotone else [
            lit for w in touched for lit in (pos(w), neg(w))]
        self.n = n = len(held)
        # The row of each literal held, and -1 for the others.
        self.index = np.full(2 * len(body.bits), -1, dtype=np.intp)
        self.index[held] = np.arange(n)
        self.rows = np.zeros((n, (n + 7) // 8), dtype=np.uint8)
        self.complete = not stopped
        if not gates:
            return

        # The joins of one pass up the body: each entry is (target, left, right, both),
        # the target row gaining the intersection of the other two where `both`, else
        # their union; and the pass down, where a target gains one other row.
        up: list[tuple[int, int, int, bool]] = []
        down: list[tuple[int, int]] = []
        pairs: list[tuple[int, int, int]] = []       # (target, left, right): both imply
        for op in gates:
            (y,) = op.outs
            tag = gate_tag(op)
            if tag is Tag.NOT:
                (x,) = op.args
                up += [(pos(y), neg(x), neg(x), False), (neg(y), pos(x), pos(x), False)]
                down += [(neg(x), pos(y)), (pos(x), neg(y))]
                continue
            a, b = op.args
            if tag is Tag.AND:
                up.append((pos(y), pos(a), pos(b), False))
                down += [(neg(a), neg(y)), (neg(b), neg(y))]
                pairs.append((pos(y), pos(a), pos(b)))
                if not monotone:
                    up.append((neg(y), neg(a), neg(b), True))
            else:
                up.append((pos(y), pos(a), pos(b), True))
                down += [(pos(a), pos(y)), (pos(b), pos(y))]
                if not monotone:
                    up.append((neg(y), neg(a), neg(b), False))
                    pairs.append((neg(y), neg(a), neg(b)))
        index = self.index.tolist()
        up = [(index[t], index[l], index[r], both) for t, l, r, both in up if index[t] >= 0]
        down = [(index[t], index[s]) for t, s in down if index[t] >= 0][::-1]
        targets, left, right = _indices([(index[t], index[l], index[r])
                                         for t, l, r in pairs if index[t] >= 0])

        self.complete = _close(self.rows, up, down, (targets, left, right), deadline)

    def implies(self, first: int, second: int) -> bool:
        """Whether literal `first` syntactically implies literal `second`."""
        if first == second:
            return True
        i, j = self.index[first], self.index[second]
        if self.monotone and first & 1 and second & 1:
            # Two negations in a body without negations: by contraposition.
            i, j = self.index[second ^ 1], self.index[first ^ 1]
        if i < 0 or j < 0:
            return False                        # a literal held by no row implies itself alone
        return bool((self.rows[i, j >> 3] >> (j & 7)) & 1)


def _close(rows: np.ndarray, up, down, pairs, deadline: float | None) -> bool:
    """Whether `rows` were closed under the joins `up` and `down` and the pair rule over
    `pairs` before `deadline`, rounds being run on them until a round changes nothing."""
    n = len(rows)
    derived: dict[int, np.ndarray] = {}          # row -> targets the pair rule added
    # Rows only ever gain bits, so a row has changed exactly where its count has grown.
    seen = np.zeros(n, dtype=np.int64)            # each row's count as the pair rule left it
    first = True
    while True:
        # A row stands for its literal's implications without the literal itself, so each
        # join adds the literals it joins where they belong to the result.  Those of a
        # union belong to it unconditionally and are added in the first round; those of an
        # intersection where each literal implies the other or the two are one.
        for t, l, r, both in up:
            if expired(deadline):
                return False
            row = rows[t]
            if both:
                np.bitwise_or(row, rows[l] & rows[r], out=row)
                if l == r or _holds(rows[r], l):
                    _add(row, l)
                if l != r and _holds(rows[l], r):
                    _add(row, r)
            else:
                np.bitwise_or(row, rows[l] | rows[r], out=row)
                if first:
                    _add(row, l)
                    _add(row, r)
        for t, s in down:
            if expired(deadline):
                return False
            np.bitwise_or(rows[t], rows[s], out=rows[t])
            if first:
                _add(rows[t], s)
        # A target the pair rule added to a row is implied by it, so the row holds the
        # target's row as well, as that row grows.
        for i, ys in derived.items():
            if expired(deadline):
                return False
            _join(rows, i, ys)
        sizes = _sizes(rows)
        dirty = range(n) if first else np.flatnonzero(sizes != seen).tolist()
        if not first and not dirty:
            return True                           # the round changed nothing
        first = False
        for i in dirty:
            if expired(deadline):
                return False
            added = _pair_rule(rows, i, n, *pairs)
            if len(added):
                derived[i] = np.concatenate((derived[i], added)) if i in derived else added
                sizes[i] = _sizes(rows[i:i + 1])[0]
        seen = sizes


def _holds(row: np.ndarray, j: int) -> bool:
    return bool((row[j >> 3] >> (j & 7)) & 1)


def _add(row: np.ndarray, j: int) -> None:
    row[j >> 3] |= np.uint8(1 << (j & 7))


# Rows read at a time by `_sizes` and `_join`, bounding the memory either takes.
_CHUNK = 1024


def _join(rows: np.ndarray, i: int, others) -> None:
    """Nothing, row `i` now holding each of the rows `others` as well."""
    for start in range(0, len(others), _CHUNK):
        chunk = rows[others[start:start + _CHUNK]]
        np.bitwise_or(rows[i], np.bitwise_or.reduce(chunk, axis=0), out=rows[i])


def _sizes(rows: np.ndarray) -> np.ndarray:
    """The number of bits set in each row."""
    sizes = np.empty(len(rows), dtype=np.int64)
    for start in range(0, len(rows), _CHUNK):
        sizes[start:start + _CHUNK] = np.bitwise_count(rows[start:start + _CHUNK]).sum(axis=1)
    return sizes


def _indices(triples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not triples:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, empty
    array = np.asarray(triples, dtype=np.intp)
    return array[:, 0], array[:, 1], array[:, 2]


def _pair_rule(rows: np.ndarray, i: int, n: int, targets, left, right) -> np.ndarray:
    """The targets row `i` gained, now holding every target both of whose sources it holds,
    its own literal counting as held, together with each such target's own row.

    A target gained may be a source of another, so the rule is repeated on the row until
    it adds nothing.
    """
    added: list[np.ndarray] = []
    if not len(targets):
        return targets
    row = rows[i]
    bits = np.unpackbits(row, count=n, bitorder="little").astype(bool)
    while True:
        bits[i] = True
        hit = targets[bits[left] & bits[right] & ~bits[targets]]
        if not len(hit):
            return np.concatenate(added) if added else hit
        added.append(hit)
        _join(rows, i, hit)
        bits = np.unpackbits(row, count=n, bitorder="little").astype(bool)
        bits[hit] = True
        row[:] = np.packbits(bits, bitorder="little")
