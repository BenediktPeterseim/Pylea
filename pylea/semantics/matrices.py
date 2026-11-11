"""The matrices of the built-in primitives.

Each has shape ``(2**out_bits, 2**in_bits)``, and ``M[o, i]`` is the weight of output
pattern `o` given input pattern `i`.  A pattern packs its wires MSB first, the first wire
most significant, matching `pylea.syntax.encoding`.  Every entry is non-negative.

    >>> and_().tolist()
    [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    >>> flip(0.25).tolist()
    [[0.75], [0.25]]

The arrays are read-only and shared between callers, so a caller wanting to modify one must
copy it first.  All but `flip` are cached, `flip` taking a continuous parameter.
"""
from __future__ import annotations

import functools

import numpy as np


def _deterministic(in_bits: int, out_bits: int, f) -> np.ndarray:
    """The matrix of the function `f` on bit patterns, so that ``M[f(i), i] = 1``."""
    m = np.zeros((1 << out_bits, 1 << in_bits))
    for i in range(1 << in_bits):
        m[f(i), i] = 1.0
    return _frozen(m)


def _frozen(m: np.ndarray) -> np.ndarray:
    """`m`, now marked read-only, since these matrices are shared between callers."""
    m.flags.writeable = False
    return m


@functools.lru_cache(maxsize=None)
def and_() -> np.ndarray:
    """``2 -> 1``, true where both inputs are."""
    return _deterministic(2, 1, lambda i: (i >> 1) & (i & 1))


@functools.lru_cache(maxsize=None)
def or_() -> np.ndarray:
    """``2 -> 1``, true where either input is.

        >>> or_().tolist()
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]
    """
    return _deterministic(2, 1, lambda i: (i >> 1) | (i & 1))


@functools.lru_cache(maxsize=None)
def not_() -> np.ndarray:
    """``1 -> 1``, true where the input is false.

        >>> not_().tolist()
        [[0.0, 1.0], [1.0, 0.0]]
    """
    return _deterministic(1, 1, lambda i: 1 - i)


@functools.lru_cache(maxsize=None)
def observe() -> np.ndarray:
    """``1 -> 0``, keeping only the input that is true.

    One row, since it writes nothing, and a zero wherever the input is false:

        >>> observe().tolist()
        [[0.0, 1.0]]
    """
    return _frozen(np.array([[0.0, 1.0]]))


@functools.lru_cache(maxsize=None)
def equate(k: int) -> np.ndarray:
    """``2k -> k``, keeping only inputs whose two k-bit halves agree.

    Columns for disagreeing inputs are zero, so this conditions rather than maps.

        >>> equate(1).tolist()
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    """
    m = np.zeros((1 << k, 1 << (2 * k)))
    for half in range(1 << k):
        m[half, (half << k) + half] = 1.0
    return _frozen(m)


@functools.lru_cache(maxsize=None)
def uniform(k: int) -> np.ndarray:
    """``0 -> k``, every pattern equally likely.

        >>> uniform(2).ravel().tolist()
        [0.25, 0.25, 0.25, 0.25]
    """
    return _frozen(np.full((1 << k, 1), 2.0 ** -k))


@functools.lru_cache(maxsize=None)
def const(value: int, bits: int) -> np.ndarray:
    """``0 -> bits``, concentrated on the pattern `value`.

        >>> const(2, 2).ravel().tolist()
        [0.0, 0.0, 1.0, 0.0]
    """
    m = np.zeros((1 << bits, 1))
    m[value, 0] = 1.0
    return _frozen(m)


def flip(p: float) -> np.ndarray:
    """``0 -> 1``, true with probability `p`."""
    # Not cached, `p` being continuous: a cache would grow without bound.
    return _frozen(np.array([[1.0 - p], [p]]))


@functools.lru_cache(maxsize=None)
def if_then_else(k: int) -> np.ndarray:
    """``(1 + 2k) -> k``, selecting between two k-bit inputs.

    Inputs are ``(condition, then_k, else_k)``, the condition most significant.  The width
    grows as ``2**k`` by ``2**(2k + 1)``, which is why `pylea.syntax.dpp.DPP._emit` expands a
    wide one into k of these rather than building it:

        >>> if_then_else(1).shape, if_then_else(3).shape
        ((2, 8), (8, 128))
    """
    def pick(i: int) -> int:
        cond = i >> (2 * k)
        then_ = (i >> k) & ((1 << k) - 1)
        else_ = i & ((1 << k) - 1)
        return then_ if cond else else_
    return _deterministic(2 * k + 1, k, pick)
