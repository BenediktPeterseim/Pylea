"""`is_deterministic` decides whether CSE may merge two calls, so it must never over-claim.

Deterministic means that the output is a function of the input, which for a dense matrix is
exactly that every column is 0/1 with at most one 1.  Reporting a stochastic matrix as
deterministic would let CSE collapse two independent draws into one shared draw, which
changes the answer rather than merely the cost.
"""
import numpy as np
import pytest

from test.syntax.helpers import matrix_of

from pylea.semantics.likelihood_table import LikelihoodTable
from pylea.syntax.dpp import DPP, Bool


def _atom(lower, n_in, n_out):
    return LikelihoodTable(
        input_types=tuple(Bool for _ in range(n_in)),
        output_types=tuple(Bool for _ in range(n_out)),
        matrix=np.asarray(lower, dtype=float),
    ).compiled()


def _prob(dpp) -> float:
    v = matrix_of(dpp).ravel()
    return float(v[1] / v.sum())


def test_a_stochastic_matrix_is_not_deterministic():
    coin = _atom([[0.5], [0.5]], 0, 1)
    assert coin.is_deterministic is False


def test_a_permutation_matrix_is_deterministic():
    negate = _atom([[0.0, 1.0], [1.0, 0.0]], 1, 1)
    assert negate.is_deterministic is True


def test_entries_in_zero_and_one_are_not_enough_without_one_per_column():
    leaky = _atom([[1.0, 1.0], [0.0, 1.0]], 1, 1)      # the second column sums to 2
    assert leaky.is_deterministic is False


def test_two_draws_of_a_stochastic_table_stay_independent():
    """Merging the two calls would give 0.5 here rather than 0.25."""
    coin = _atom([[0.5], [0.5]], 0, 1)

    @DPP
    def both() -> Bool:
        return coin() & coin()

    assert _prob(both) == pytest.approx(0.25)


def test_a_repeated_deterministic_table_still_agrees_with_its_semantics():
    """Merging the two calls is sound here, ``f(x) & f(x)`` being ``f(x)``."""
    ident = _atom([[1.0, 0.0], [0.0, 1.0]], 1, 1)

    @DPP
    def twice() -> Bool:
        from pylea.syntax.flip import flip
        x = flip(0.3)
        return ident(x) & ident(x)

    assert _prob(twice) == pytest.approx(0.3)
