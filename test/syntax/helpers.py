"""Shared helpers for the front-end tests."""
import numpy as np


def matrix_of(dpp, **kwargs) -> np.ndarray:
    """The ``(2**out_bits, 2**in_bits)`` matrix of a DPP, at its true magnitude."""
    array, scale = dpp.compiled(**kwargs).matrix()
    return np.asarray(array, dtype=float) * np.exp(scale)


def _assert_proportional(a: np.ndarray, b: np.ndarray) -> None:
    """Nothing, or an assertion failure unless ``a == c * b`` for some positive `c`."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 and b_norm == 0:
        return
    assert a_norm > 0 and b_norm > 0, (
        f"One matrix is zero but the other is not:\n{a}\nvs\n{b}")
    assert np.allclose(a / a_norm, b / b_norm), (
        f"Matrices are not proportional:\n{a}\nvs\n{b}")
