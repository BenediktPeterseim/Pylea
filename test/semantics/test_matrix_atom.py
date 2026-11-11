"""A `LikelihoodTable` called as one op inside a model.

Two properties are pinned: a model calling a single table evaluates to that table's matrix
up to scale, and a model calling two in sequence evaluates to their product up to scale.
"""
import numpy as np
from pylea.semantics.likelihood_table import LikelihoodTable

from test.syntax.helpers import matrix_of
from pylea.syntax.dpp import Bool, DPP
from pylea.syntax.flip import flip  # type: ignore[attr-defined]


def _proportional(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff a == k*b for some k > 0, up to tol."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    if np.allclose(b, 0):
        return bool(np.allclose(a, 0))
    mask = np.abs(b) > tol
    if not np.any(mask):
        return bool(np.allclose(a, 0))
    ratios = a[mask] / b[mask]
    return bool(np.allclose(ratios, ratios[0], atol=tol) and ratios[0] > 0)


def _make_noisy_channel() -> LikelihoodTable:
    """Simple 2x2 noisy-channel table: outputs 1 with prob 0.8 when input is 1."""
    return LikelihoodTable(
        input_types=(Bool,),
        output_types=(Bool,),
        matrix=np.array([[0.8, 0.3],
                        [0.2, 0.7]]),
    )


def test_a_model_calling_one_table_evaluates_to_that_tables_matrix():
    f = _make_noisy_channel()

    @DPP
    def prog(x: Bool) -> Bool:
        return f(x)

    result = matrix_of(prog)
    assert _proportional(result, f.matrix), (
        f"Expected result proportional to f.matrix, got:\n{result}\nvs\n{f.matrix}"
    )


def test_a_model_calling_two_tables_evaluates_to_their_product():
    f = _make_noisy_channel()
    g = LikelihoodTable(
        input_types=(Bool,),
        output_types=(Bool,),
        matrix=np.array([[0.6, 0.4],
                        [0.4, 0.6]]),
    )
    expected = f.matrix @ g.matrix

    @DPP
    def prog(x: Bool) -> Bool:
        y = g(x)
        return f(y)

    result = matrix_of(prog)
    assert _proportional(result, expected), (
        f"Expected result proportional to f@g, got:\n{result}\nvs\n{expected}"
    )


def test_a_table_applied_to_a_flip_gives_the_marginal():
    f = _make_noisy_channel()
    p = 0.3
    prior = np.array([[1 - p], [p]])
    expected = f.matrix @ prior

    @DPP
    def prog() -> Bool:
        x = flip(p)
        return f(x)

    result = matrix_of(prog)
    assert _proportional(result, expected), (
        f"Got:\n{result}\nvs expected:\n{expected}"
    )
