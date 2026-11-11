"""The packed kernel: 64 particles per machine word, for the i.i.d. mode.

`PackedProgram` executes a `pylea.sample.sweep.SweepProgram` with every wire plane packed
into uint64 words, so one vectorised Boolean op advances 64 particles per word.  FLIP,
UNIFORM, AND, OR, NOT, CONST, OBSERVE and the deterministic EQUATE and ITE run packed;
any other table op is a boundary: its argument planes are unpacked, the byte lane's own
column path runs on them, and the outputs are packed again.  The kernel therefore serves
every DPP with 0/1 weights, at a gain set by its packable fraction; fault trees are
entirely packable, and they are where the particle counts are largest.  A DPP with
soft weights stays on the byte lane, whose float weights the word representation cannot
carry.

The kernel serves the i.i.d. mode only: resampling requires a per-particle gather at
every trigger, which at bit granularity means unpack, gather and repack per resampling
point, an overhead that defeats the purpose.  Conditioning still works, the weights being
0/1: an OBSERVE conjoins its argument into an alive mask, an EQUATE the bitwise agreement
of its arguments, and a boundary op's 0/1 column mass a gathered mask.

Coin flips are drawn through `pylea.sample.sweep.draw_flips`, byte for byte the same
stream the byte lane draws, and then packed.  The two executors therefore produce
identical counts from an identical generator state, and the equality test in
`test.sample.test_kernels` compares them exactly rather than statistically.  Coin flip
generation dominates the packed runtime, the gates having become 64 times cheaper; the
boundary-refinement draw keeps it at about one random byte per particle per coin flip.
"""
from __future__ import annotations

import numpy as np

from pylea.ir.prim import Tag
from pylea.sample.sweep import (
    _AND, _COLUMN, _CONST, _FLIP, _FREE, _NOT, _OBSERVE, _OR, _UNIFORM, BatchResult,
    SweepProgram, _bit_planes, _column_choice, draw_flips,
)

_ALL_ONES = np.uint64(0xFFFFFFFFFFFFFFFF)

# The packed op list reuses the byte lane's step kinds and tuple layouts; only the two
# deterministic table ops it runs on words have kinds of their own.
_P_EQUATE, _P_ITE = 9, 10


class NotPackable(ValueError):
    """Raised for a DPP with an op the packed kernel cannot represent."""


class PackedProgram:
    """The packed form of a `SweepProgram`, executing 64 particles per word in i.i.d. mode.

    Mirrors the `SweepProgram.run` interface for batches whose size is a multiple of 64,
    and exposes the same query-relevant attributes, so the run loops treat the two
    executors interchangeably.  Raises `NotPackable` where `program` carries soft
    weights or has input wires, whose scalars the word representation does not carry.

        >>> import numpy as np
        >>> from pylea import DPP, Bool, flip, observe
        >>> from pylea.sample.sweep import SweepProgram, unfold
        >>> @DPP
        ... def either() -> Bool:
        ...     a, b = flip(0.5), flip(0.5)
        ...     observe(a | b)
        ...     return a
        >>> byte_lane = SweepProgram(unfold(either.program()))
        >>> packed = PackedProgram(byte_lane)
        >>> a = byte_lane.run(4096, np.random.default_rng(7))
        >>> b = packed.run(4096, np.random.default_rng(7))
        >>> np.array_equal(a.counts, b.counts) and a.total == b.total
        True
    """

    def __init__(self, program: SweepProgram):
        if not program.binary_weights:
            raise NotPackable("soft weights need the byte lane's float weights")
        if program.body.inputs:
            raise NotPackable("input wires hold scalars the packed kernel does not carry")
        self._program = program
        self.body = program.body
        self.out_bits = program.out_bits
        self.log_scale = program.log_scale
        self.binary_weights = True
        self.n_stages = program.n_stages
        self.peak_planes = program.peak_planes
        self._steps = [packed for step in program._steps
                       for packed in _packed_steps(step)]

    @property
    def conditions(self) -> bool:
        return self._program.conditions

    def run(self, n: int, rng: np.random.Generator, *, resample: bool = False,
            squares: bool = False, record_stages: bool = False) -> BatchResult:
        """One i.i.d. batch of `n` particles, `n` a multiple of 64, as a `BatchResult`."""
        if resample or record_stages:
            raise ValueError("the packed kernel serves the i.i.d. mode only")
        if n % 64:
            raise ValueError(f"a packed batch must be a multiple of 64 particles; got {n}")
        vals: dict[int, np.ndarray] = {}
        alive: np.ndarray | None = None
        # Constant wires alias two shared planes; no step writes a plane in place.
        zeros = np.zeros(n // 64, dtype=np.uint64)
        ones = zeros | _ALL_ONES

        for step in self._steps:
            kind = step[0]
            if kind is _FLIP:
                vals[step[2]] = _pack_bits(draw_flips(rng, n, step[1]))
            elif kind is _AND:
                vals[step[3]] = vals[step[1]] & vals[step[2]]
            elif kind is _OR:
                vals[step[3]] = vals[step[1]] | vals[step[2]]
            elif kind is _NOT:
                vals[step[2]] = vals[step[1]] ^ _ALL_ONES
            elif kind is _CONST:
                for w, bit in zip(step[1], step[2]):
                    vals[w] = ones if bit else zeros
            elif kind is _OBSERVE:
                plane = vals[step[1]]
                alive = plane if alive is None else alive & plane
            elif kind is _P_EQUATE:
                agree = None
                for a, b, out in zip(step[1], step[2], step[3]):
                    same = (vals[a] ^ vals[b]) ^ _ALL_ONES
                    agree = same if agree is None else agree & same
                    vals[out] = vals[a]
                assert agree is not None
                alive = agree if alive is None else alive & agree
            elif kind is _P_ITE:
                condition = vals[step[1]]
                for a, b, out in zip(step[2], step[3], step[4]):
                    vals[out] = (condition & vals[a]) | (~condition & vals[b])
            elif kind is _COLUMN:
                # The boundary: unpack the arguments, run the byte lane's own column
                # path, pack the results.  The draws go through the shared
                # `_column_choice`, so the streams stay identical by construction.
                sampler, masses, args, outs, weighting, deterministic = step[1:7]
                pattern = np.zeros(n, dtype=np.int64)
                for w in args:
                    pattern = (pattern << 1) | _unpack_bits(vals[w], n)
                if weighting:
                    keep = _pack_bits((masses[pattern] != 0.0).view(np.uint8))
                    alive = keep if alive is None else alive & keep
                if outs:
                    chosen = _column_choice(sampler, masses, pattern, weighting,
                                            deterministic, rng)
                    for w, plane in _bit_planes(chosen, outs):
                        vals[w] = _pack_bits(plane)
            else:
                assert kind is _FREE
                for w in step[1]:
                    vals.pop(w, None)

        weights = (np.ones(n, dtype=np.uint8) if alive is None
                   else _unpack_bits(alive, n))
        size = 1 << self.out_bits
        pattern = np.zeros(n, dtype=np.int64)
        for w in self.body.outputs:
            pattern = (pattern << self.body.bits[w]) | _unpack_bits(vals[w], n)
        counts = np.bincount(pattern, weights=weights, minlength=size).round()
        return BatchResult(counts, float(counts.sum()),
                           counts.copy() if squares else None,
                           float(counts.sum()) if squares else None, 0.0, n, None)


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.view(bool), bitorder="little").view(np.uint64)


def _unpack_bits(words: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(words.view(np.uint8), count=n, bitorder="little")


def _packed_steps(step: tuple) -> list[tuple]:
    """The packed form of one byte-lane step.

    Most steps are the byte lane's own, executed on words; a UNIFORM becomes one fair
    coin flip per output bit in the byte lane's draw order, and a deterministic EQUATE
    or ITE a packed kind of its own.
    """
    kind = step[0]
    if kind is _UNIFORM:
        return [(_FLIP, (128,), w) for w in step[1]]
    if kind is not _COLUMN:
        return [step]
    tag = step[7]
    args, outs = step[3], step[4]
    if tag is Tag.EQUATE:
        k = len(outs)
        return [(_P_EQUATE, args[:k], args[k:], outs)]
    if tag is Tag.ITE:
        k = len(outs)
        return [(_P_ITE, args[0], args[1:1 + k], args[1 + k:], outs)]
    return [step]
