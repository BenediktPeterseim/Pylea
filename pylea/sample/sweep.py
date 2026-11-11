"""The particle interpreter.

`SweepProgram` preprocesses an unfolded body once; `SweepProgram.run` then executes
batches of particles through it.  After unfolding every op is a primitive and every wire a
primitive writes is one bit, so the particle state is one uint8 plane of `n` values per
live wire; only root input wires can be wider, and those hold one scalar shared by all
particles.

Execution has two modes.  Without resampling every particle is an independent draw, and
the result reports per-particle weight sums from which `pylea.sample.certify` builds
particle-level certificates.  With resampling the batch is one sweep, or, through
`SweepProgram.run_blocks`, many independent sweeps side by side in one array: whenever a
block's effective sample size falls below `pylea.sample.cost_model.ESS_FRACTION` times
its size, that block's mean weight is folded into its running log factor and its
particles are redrawn in proportion to their weights.  Batching blocks is what lets the
statistically correct small sweep size be run without paying Python dispatch per sweep.
Either way the estimate is unbiased for the unnormalised semantics, for every size,
schedule and trigger, a dead sweep counting as the value zero: the partial estimate times
the true remaining conditional expectation is a martingale in the op index, and
resampling at an effective-sample-size trigger, a stopping time of the particle history,
preserves its conditional mean.

Fast paths cover FLIP, UNIFORM, AND, OR, NOT, CONST and OBSERVE.  Every other primitive
goes through one generic path that needs no per-tag knowledge: gather the matrix column
each particle's input pattern selects, multiply the particle's weight by the column mass,
which is how soft observations and substochastic tables work with no extra machinery, and
sample the output pattern from the normalised column.  A deterministic matrix, one whose
columns hold at most a single 1, maps patterns through a precomputed table and draws no
randomness at all.

Randomness for a coin flip is drawn as bytes and refined only at the boundary: one byte
per particle is compared against the probability's first base-256 digit, and only the
particles that hit the digit exactly draw a further byte against the next digit.  This is
exact for every float probability at about one byte per particle, and it is a layout
contract, not an optimisation: `pylea.sample.packed` consumes the identical byte stream,
so the two executors are compared for exact equality rather than statistically.

Preprocessing keeps peak memory at the live wire frontier rather than the wire count: an
op with no inputs and positive mass is moved to just before its first reader, or dropped
when nothing reads it, only its mass factor remaining, and every plane is freed after
its last read.  A traced DPP materialises all coin flips before any gate, so without this
the state would be the number of coin flips times `n`.  Moving such an op past a
weight-changing one is sound because it is independent of everything already drawn; in
sweep mode it additionally means the op is drawn fresh for each resampled particle rather
than copied.

Weights stay in [0, 1] and away from needless zero by construction: every table op is
divided by its largest column mass, and the log of the product of these divisors is
`SweepProgram.log_scale`, to be folded back into any unconditioned mass.  This caps a
caller-supplied table's masses at one, keeps a long product of small soft weights
representable, and, where an op's column masses do not depend on the state, removes its
weighting entirely: conditioning whose strength is state-independent is a scale, not a
variance source, and such an op certifies as if it were not conditioning at all.
Certification happens on the rescaled scale, and for a conditional query the factor
cancels in the quotient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pylea.ir.body import Body, Op
from pylea.ir.prim import Prim, Tag
from pylea.ir.program import Program
from pylea.sample.cost_model import ESS_FRACTION

# Step kinds of the preprocessed op list.
_FLIP, _AND, _OR, _NOT, _OBSERVE, _CONST, _UNIFORM, _COLUMN, _FREE = range(9)

# Weights whose largest entry falls below this are renormalised into the running log
# factor, keeping long products of soft weights representable.  Sound only in sweep mode,
# where the factorisation is exact; i.i.d. samples must keep their fixed [0, 1] scale.
_RENORMALISE_BELOW = 1e-100


def unfold(prog: Program) -> Body:
    """The root body of `prog` with every call spliced in, so only primitives remain.

    One pass over the call tree, linear in the unfolded size: each call site's callee is
    emitted with its interface wires mapped onto the caller's, in the caller's op order,
    so the result runs the ops in the same order splicing in place would.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> len(unfold(both.program()).ops)
        3
    """
    import sys

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 20 * len(prog.bodies) + 1000))
    try:
        bits: list[int] = []
        ops: list[Op] = []

        def fresh(width: int) -> int:
            bits.append(width)
            return len(bits) - 1

        def emit(index: int, inputs: list[int]) -> list[int]:
            body = prog.bodies[index]
            wire: dict[int, int] = dict(zip(body.inputs, inputs))
            for op in body.ops:
                args = [wire[w] for w in op.args]
                if isinstance(op.callee, Prim):
                    outs = [fresh(body.bits[w]) for w in op.outs]
                    ops.append(Op(op.callee, tuple(args), tuple(outs)))
                else:
                    outs = emit(op.callee, args)
                wire.update(zip(op.outs, outs))
            return [wire[w] for w in body.outputs]

        root = prog.bodies[prog.root]
        root_inputs = [fresh(root.bits[w]) for w in root.inputs]
        outputs = emit(prog.root, root_inputs)
        return Body(tuple(bits), tuple(root_inputs), tuple(outputs), tuple(ops))
    finally:
        sys.setrecursionlimit(limit)


def unfolded_op_count(prog: Program) -> int:
    """The op count of `unfold(prog)`, computed from the call DAG without unfolding.

    One memoised pass: a body's count is its own primitive ops plus, per call site, the
    callee's count.  A feasibility guard can therefore refuse a DPP whose unfolding
    is exponential without paying for the unfolding.

        >>> from pylea import DPP, Bool, flip
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.5)
        >>> @DPP
        ... def both() -> Bool:
        ...     return coin() & coin()
        >>> unfolded_op_count(both.program())
        3
    """
    counts: dict[int, int] = {}
    stack = [prog.root]
    while stack:
        index = stack[-1]
        if index in counts:
            stack.pop()
            continue
        pending = {c for c in prog.bodies[index].callees() if c not in counts}
        if pending:
            stack.extend(pending)
            continue
        counts[index] = sum(1 if isinstance(op.callee, Prim) else counts[op.callee]
                            for op in prog.bodies[index].ops)
        stack.pop()
    return counts[prog.root]


def flip_digits(p: float) -> tuple[int, ...]:
    """The base-256 digits a byte stream is compared against to draw Bernoulli(`p`).

    Exact for every float: a float's fractional part terminates in base 256, so at most
    a handful of digits remain and the last comparison is decisive.

        >>> flip_digits(0.5), flip_digits(1.0), flip_digits(0.0)
        ((128,), (256,), (0,))
    """
    if p >= 1.0:
        return (256,)
    digits = []
    remainder = p
    while True:
        remainder *= 256.0
        digit = math.floor(remainder)
        remainder -= digit
        digits.append(digit)
        if remainder <= 0.0:
            return tuple(digits)


def draw_flips(rng: np.random.Generator, n: int, digits: tuple[int, ...]) -> np.ndarray:
    """`n` exact coin flips as a uint8 array, consuming bytes from `rng`.

    One byte per particle against the first digit; only the particles hitting the digit
    exactly draw further bytes.  Every executor draws coin flips through this, which is
    what makes their streams comparable bit for bit.
    """
    u = rng.integers(0, 256, n, dtype=np.uint8)
    out = u < digits[0]
    if len(digits) > 1:
        boundary = np.flatnonzero(u == digits[0])
        if boundary.size:
            out[boundary] = draw_flips(rng, int(boundary.size), digits[1:]).astype(bool)
    return out.view(np.uint8)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """What one batch of particles produced.

    `counts` is the weighted mass per output pattern and `total` the weight summed over
    all particles, both on the rescaled scale and, in sweep mode, still to be multiplied
    by ``exp(log_z)``.  `counts_sq` and `total_sq` are the same sums over squared
    weights, present only when requested; `pylea.sample.certify.WeightedParticleStream`
    needs them for its variance.  `stage_fractions` holds the surviving weight fraction
    at each weight-changing op, present only when requested.
    """

    counts: np.ndarray
    total: float
    counts_sq: np.ndarray | None
    total_sq: float | None
    log_z: float
    n: int
    stage_fractions: np.ndarray | None


@dataclass(frozen=True, slots=True)
class BlockResult:
    """What a batch of side-by-side sweeps produced, one row per sweep.

    ``counts[s]`` is sweep `s`'s weighted mass per output pattern and ``log_z[s]`` its
    running log factor, so the sweep's value vector is ``counts[s] * exp(log_z[s]) / n``
    with `n` the particles per sweep.
    """

    counts: np.ndarray
    log_z: np.ndarray
    n: int


class SweepProgram:
    """An unfolded body preprocessed for particle execution.

    `binary_weights` says whether every particle weight stays exactly 0 or 1, which is
    what admits the Clopper-Pearson certificate; `conditions` whether any op can change a
    weight at all; `log_scale` the log factor taken out of the weights, zero unless some
    table op's largest column mass differs from one; `peak_planes` the largest number
    of wire planes live at once, which bounds the particle state at ``peak_planes * n``
    bytes.  A body with inputs needs `input_pattern`, the one column of the matrix the
    particles run, since every input wire holds a single scalar; leaving it out raises
    `ValueError`.

        >>> from pylea import DPP, Bool, flip, observe
        >>> @DPP
        ... def either() -> Bool:
        ...     a, b = flip(0.5), flip(0.5)
        ...     observe(a | b)
        ...     return a
        >>> program = SweepProgram(unfold(either.program()))
        >>> program.binary_weights, program.conditions, program.out_bits
        (True, True, 1)

    A batch of an unconditioned DPP reproduces its distribution:

        >>> import numpy as np
        >>> @DPP
        ... def coin() -> Bool:
        ...     return flip(0.25)
        >>> batch = SweepProgram(unfold(coin.program())).run(4096, np.random.default_rng(0))
        >>> batch.total == 4096.0, bool(abs(batch.counts[1] / 4096 - 0.25) < 0.03)
        (True, True)
    """

    def __init__(self, body: Body, input_pattern: int | None = None):
        if body.inputs and input_pattern is None:
            raise ValueError(
                "a SweepProgram over a body with inputs needs an explicit input_pattern; "
                "sampling a query fixes every input wire to one scalar column")
        prims: list[Prim] = []
        for op in body.ops:
            if not isinstance(op.callee, Prim):
                raise ValueError(
                    "SweepProgram requires an unfolded body; it still has calls")
            prims.append(op.callee)
        self.body = body
        self.out_bits = body.out_bits
        self.log_scale = 0.0
        self._binary = True
        self._inputs = _input_values(body, input_pattern or 0)
        self._steps, self.n_stages = self._preprocess(prims)
        self.binary_weights = self._binary
        self.peak_planes = self._peak_planes()

    @property
    def conditions(self) -> bool:
        """Whether any op can change a particle's weight."""
        return self.n_stages > 0

    # -- preprocessing --------------------------------------------------------------

    def _preprocess(self, prims: list[Prim]):
        body = self.body
        first_read: dict[int, int] = {}
        for i, op in enumerate(body.ops):
            for w in op.args:
                first_read.setdefault(w, i)

        # Movable ops, keyed by the op index they now precede; `len(ops)` is the end.
        moved_to: dict[int, list] = {}
        dropped_or_moved: set[int] = set()
        for i, op in enumerate(body.ops):
            prim = prims[i]
            # A zero-input op's single column mass is a state-independent factor, so
            # after the mass moves into the static log factor the op is a pure
            # independent draw: movable to its first reader, droppable when unread.
            if prim.in_bits == 0 and _single_column_mass(prim) > 0.0:
                dropped_or_moved.add(i)
                reads = [first_read[w] for w in op.outs if w in first_read]
                if not reads and not any(w in body.outputs for w in op.outs):
                    mass = _single_column_mass(prim)
                    if mass != 1.0:
                        self.log_scale += math.log(mass)
                    continue           # read by nothing: only its mass factor remains
                moved_to.setdefault(min(reads) if reads else len(body.ops), []).append(op)

        ordered: list = []
        for i, op in enumerate(body.ops):
            ordered.extend(moved_to.get(i, ()))
            if i not in dropped_or_moved:
                ordered.append(op)
        ordered.extend(moved_to.get(len(body.ops), ()))

        last_read: dict[int, int] = {}
        for i, op in enumerate(ordered):
            for w in op.args:
                last_read[w] = i
        keep = set(body.outputs) | set(body.inputs)

        steps: list = []
        n_stages = 0
        for i, op in enumerate(ordered):
            step, is_stage = self._encode(op)
            n_stages += is_stage
            steps.append(step)
            frees = tuple(dict.fromkeys(
                w for w in op.args if last_read.get(w) == i and w not in keep))
            if frees:
                steps.append((_FREE, frees))
        return steps, n_stages

    def _encode(self, op) -> tuple[tuple, bool]:
        """One preprocessed step for `op`, and whether the op can change weights."""
        prim = op.callee
        assert isinstance(prim, Prim)           # the constructor rejected calls
        tag = prim.tag
        if tag is Tag.FLIP:
            return (_FLIP, flip_digits(float(prim.params[0])), op.outs[0]), False
        if tag is Tag.UNIFORM:
            return (_UNIFORM, op.outs), False
        if tag is Tag.AND:
            return (_AND, op.args[0], op.args[1], op.outs[0]), False
        if tag is Tag.OR:
            return (_OR, op.args[0], op.args[1], op.outs[0]), False
        if tag is Tag.NOT:
            return (_NOT, op.args[0], op.outs[0]), False
        if tag is Tag.OBSERVE:
            return (_OBSERVE, op.args[0]), True
        if tag is Tag.CONST:
            value, bits = int(prim.params[0]), int(prim.params[1])
            planes = tuple((value >> (bits - 1 - k)) & 1 for k in range(bits))
            return (_CONST, op.outs, planes), False
        matrix = np.asarray(prim.matrix, dtype=float)
        masses = matrix.sum(axis=0)
        top = float(masses.max(initial=0.0))
        # Dividing by the largest column mass moves what every particle would pay
        # anyway into the static log factor.  It keeps weights in [0, 1] where a mass
        # exceeds one, keeps long soft products representable where all masses are
        # small, and where the masses are column-independent it removes the weighting
        # altogether: conditioning whose strength does not depend on the state is a scale,
        # not a variance source.
        if top > 0.0 and top != 1.0:
            matrix = matrix / top
            masses = masses / top
            self.log_scale += float(np.log(top))
        weighting = not bool(np.all(masses == 1.0))
        self._binary &= bool(np.all((masses == 0.0) | (masses == 1.0)))
        # A deterministic matrix maps each admitted pattern to one output and draws no
        # randomness; a zero column's image is arbitrary, its weight being zero.
        table = np.argmax(matrix, axis=0) if prim.is_deterministic else None
        sampler = table if table is not None else np.cumsum(matrix, axis=0)
        return (_COLUMN, sampler, masses, op.args, op.outs, weighting,
                table is not None, tag), weighting

    def _peak_planes(self) -> int:
        live = peak = 0
        scalar: set[int] = set()        # wires holding one shared value, not a plane
        for step in self._steps:
            kind = step[0]
            if kind is _FREE:
                live -= sum(1 for w in step[1] if w not in scalar)
            elif kind in (_FLIP, _AND, _OR, _NOT):
                live += 1
            elif kind is _CONST:
                scalar.update(step[1])
            elif kind is _UNIFORM:
                live += len(step[1])
            elif kind is _COLUMN:
                live += len(step[4])
            peak = max(peak, live)
        return peak

    # -- execution ------------------------------------------------------------------

    def run(self, n: int, rng: np.random.Generator, *, resample: bool = False,
            squares: bool = False, record_stages: bool = False) -> BatchResult:
        """One batch of `n` particles, as a `BatchResult`.

        Without `resample` every particle is independent; with it the batch is one
        sweep.  `squares` additionally accumulates squared-weight sums, and
        `record_stages` the surviving weight fraction at each weight-changing op.
        """
        counts, counts_sq, log_z, fractions = self._execute(
            1, n, rng, resample=resample, squares=squares,
            record_stages=record_stages)
        row = counts[0]
        if self.binary_weights:
            row = row.round()
            if counts_sq is not None:
                counts_sq = counts_sq.round()
        row_sq = counts_sq[0] if counts_sq is not None else None
        return BatchResult(row, float(row.sum()), row_sq,
                           float(row_sq.sum()) if row_sq is not None else None,
                           float(log_z[0]), n, fractions if record_stages else None)

    def run_blocks(self, n_sweeps: int, n: int, rng: np.random.Generator) -> BlockResult:
        """`n_sweeps` independent sweeps of `n` particles each, run side by side.

        One array of ``n_sweeps * n`` particles with block-wise resampling, so the
        statistically correct small sweep size pays Python dispatch once per op rather
        than once per sweep.
        """
        counts, _sq, log_z, _fractions = self._execute(
            n_sweeps, n, rng, resample=True, squares=False, record_stages=False)
        return BlockResult(counts, log_z, n)

    def _execute(self, blocks: int, n: int, rng: np.random.Generator, *,
                 resample: bool, squares: bool, record_stages: bool):
        """The op loop over ``blocks * n`` particles; see `run` and `run_blocks`."""
        if record_stages and blocks != 1:
            raise ValueError("stage recording is defined for a single sweep")
        total_n = blocks * n
        bits = self.body.bits
        vals: dict[int, np.ndarray | int] = dict(self._inputs)
        weights = np.ones(total_n)
        log_z = np.zeros(blocks)
        fractions = np.zeros(self.n_stages)
        stage_at = 0
        previous_total = float(total_n)

        for step in self._steps:
            kind = step[0]
            weighted = False
            if kind is _FLIP:
                vals[step[2]] = draw_flips(rng, total_n, step[1])
            elif kind is _AND:
                vals[step[3]] = vals[step[1]] & vals[step[2]]
            elif kind is _OR:
                vals[step[3]] = vals[step[1]] | vals[step[2]]
            elif kind is _NOT:
                vals[step[2]] = vals[step[1]] ^ 1
            elif kind is _FREE:
                for w in step[1]:
                    del vals[w]
                continue
            elif kind is _CONST:
                # A constant is one scalar shared by all particles, exactly like a root
                # input; the gates, `_pack` and the resampling gather all broadcast it.
                for w, bit in zip(step[1], step[2]):
                    vals[w] = bit
            elif kind is _UNIFORM:
                for w in step[1]:
                    vals[w] = draw_flips(rng, total_n, (128,))
            elif kind is _OBSERVE:
                np.multiply(weights, vals[step[1]], out=weights)
                weighted = True
            else:
                sampler, masses, args, outs, weighting, deterministic = step[1:7]
                pattern = _pack(vals, args, bits, total_n)
                if weighting:
                    np.multiply(weights, masses[pattern], out=weights)
                    weighted = True
                if outs:
                    chosen = _column_choice(sampler, masses, pattern, weighting,
                                            deterministic, rng)
                    for w, plane in _bit_planes(chosen, outs):
                        vals[w] = plane

            if not weighted or not (resample or record_stages):
                # Independent particles need no per-op bookkeeping: a dead particle
                # keeps its zero weight to the final count.
                continue
            by_block = weights.reshape(blocks, n)
            totals = by_block.sum(axis=1)
            if record_stages:
                total = float(totals[0])
                fractions[stage_at] = total / previous_total if previous_total else 0.0
                stage_at += 1
                previous_total = total
            if not totals.any():
                # Every block is dead: the value is exactly zero, so the remaining ops
                # are skipped.
                size = 1 << self.out_bits
                zeros = np.zeros((blocks, size))
                return zeros, (np.zeros((blocks, size)) if squares else None), \
                    np.zeros(blocks), fractions
            if resample:
                tops = by_block.max(axis=1)
                renormalise = (tops > 0.0) & (tops < _RENORMALISE_BELOW)
                for b in np.flatnonzero(renormalise):
                    log_z[b] += math.log(tops[b])
                    by_block[b] /= tops[b]
                    totals[b] *= 1.0 / tops[b]
                if record_stages and renormalise[0]:
                    # The recorded totals must stay on one scale, or the next fraction
                    # is inflated by the factor just moved into the log.
                    previous_total /= tops[0]
                squared = (by_block * by_block).sum(axis=1)
                triggered = (totals > 0.0) & (totals * totals
                                              < ESS_FRACTION * n * squared)
                if triggered.any():
                    hit = np.flatnonzero(triggered)
                    log_z[hit] += np.log(totals[hit] / n)
                    if n <= 128:
                        # Vectorised multinomial over all triggered blocks at once: the
                        # n-squared inversion is cheap at sweep sizes, where a per-block
                        # generator call would dominate the whole step.
                        cdf = np.cumsum(by_block[hit] / totals[hit, None], axis=1)
                        u = rng.random((hit.size, n))
                        drawn = np.minimum((cdf[:, None, :] <= u[:, :, None]).sum(axis=2),
                                           n - 1)
                    else:
                        drawn = np.empty((hit.size, n), dtype=np.int64)
                        for row, b in enumerate(hit):
                            drawn[row] = rng.choice(n, n, p=by_block[b] / totals[b])
                    flat = (hit[:, None] * n + drawn).reshape(-1)
                    by_block[hit] = 1.0
                    # Only the triggered rows are gathered; untriggered blocks are left
                    # untouched.  Writing over the rows being read is safe because fancy
                    # indexing copies first, and writing in place is safe because every
                    # wire owns its plane: no step ever aliases one array to two wires.
                    for plane in vals.values():
                        if isinstance(plane, np.ndarray):
                            plane.reshape(blocks, n)[hit] = \
                                plane[flat].reshape(hit.size, n)
                    if record_stages:
                        previous_total = float(weights.reshape(blocks, n)[0].sum())

        size = 1 << self.out_bits
        out_pattern = _pack(vals, self.body.outputs, bits, total_n)
        if blocks > 1:
            out_pattern = out_pattern + (np.arange(total_n) // n) * size
        counts = np.bincount(out_pattern, weights=weights,
                             minlength=blocks * size).reshape(blocks, size)
        counts_sq = None
        if squares:
            counts_sq = np.bincount(out_pattern, weights=weights * weights,
                                    minlength=blocks * size).reshape(blocks, size)
        return counts, counts_sq, log_z, fractions


def _input_values(body: Body, input_pattern: int) -> dict[int, int]:
    """Each input wire's scalar value, unpacked MSB-first from `input_pattern`."""
    values: dict[int, int] = {}
    shift = body.in_bits
    for w in body.inputs:
        shift -= body.bits[w]
        values[w] = (input_pattern >> shift) & ((1 << body.bits[w]) - 1)
    return values


def _pack(vals, wires, bits, n: int) -> np.ndarray:
    """The values of `wires` packed MSB-first into one int64 per particle."""
    packed = np.zeros(n, dtype=np.int64)
    for w in wires:
        packed = (packed << bits[w]) | vals[w]
    return packed


def _column_choice(sampler, masses: np.ndarray, pattern: np.ndarray, weighting: bool,
                   deterministic: bool, rng: np.random.Generator) -> np.ndarray:
    """The output row each particle draws from its selected column.

    Both executors run their table ops through this, which is what keeps their random
    streams identical."""
    if deterministic:
        return sampler[pattern]
    bound = masses[pattern] if weighting else 1.0
    u = rng.random(pattern.size) * bound
    # The row is the number of cumulative masses at or below `u` in the particle's
    # column, found by a vectorised binary search over the rows: each round gathers one
    # row per particle, so the cost is logarithmic in the row count rather than linear.
    rows = sampler.shape[0]
    low = np.zeros(pattern.size, dtype=np.int64)
    high = np.full(pattern.size, rows, dtype=np.int64)
    while True:
        open_ = high > low
        if not open_.any():
            break
        mid = np.minimum((low + high) >> 1, rows - 1)   # closed particles are masked
        below = sampler[mid, pattern] <= u
        low = np.where(open_ & below, mid + 1, low)
        high = np.where(open_ & ~below, mid, high)
    return np.minimum(low, rows - 1)


def _bit_planes(chosen: np.ndarray, outs: tuple[int, ...]):
    """`chosen` split MSB-first into one fresh uint8 plane per output wire."""
    bit = len(outs)
    for w in outs:
        bit -= 1
        yield w, ((chosen >> bit) & 1).astype(np.uint8)


def _single_column_mass(prim: Prim) -> float:
    """The column mass of a zero-input primitive; zero where the op excludes everything."""
    if prim.tag is not Tag.MATRIX:
        return 1.0
    return float(np.asarray(prim.matrix, dtype=float).sum())
