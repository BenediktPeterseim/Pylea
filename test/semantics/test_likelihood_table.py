"""The `LikelihoodTable` surface: what evaluation returns, and what its views mean.

`test_matrix_atom.py` covers a table used as one op inside a model.  Here the table itself
is the subject: what `evaluated` returns, what `probabilities` and `normalised` mean, and
that a round trip through another model preserves the semantics.
"""
import numpy as np
import pytest

from test.syntax.helpers import matrix_of

from pylea import DPP, Bool, Enum, LikelihoodTable, flip, observe


class Coin(Enum):
    elements = ["heads", "tails"]


@DPP
def biased() -> Coin:
    return Coin.random({"heads": 0.25, "tails": 0.75})


def test_evaluated_holds_the_true_matrix():
    table = biased.evaluated()
    assert table.input_types() == ()
    assert table.output_types() == (Coin,)
    assert np.allclose(table.matrix, matrix_of(biased))


def test_probabilities_normalise_each_column():
    @DPP
    def conditioned(x: Bool) -> Bool:
        y = flip(0.5)
        observe(x | y)
        return y

    table = conditioned.evaluated()
    # Given x = 1 any y survives; given x = 0 only y = 1 does.
    assert np.allclose(table.probabilities.sum(axis=0), 1.0)
    assert table.probabilities[1, 0] == pytest.approx(1.0)
    assert table.probabilities[1, 1] == pytest.approx(0.5)


def test_a_zero_column_stays_zero_rather_than_dividing():
    table = LikelihoodTable((Bool,), (Bool,), np.array([[0.0, 0.3], [0.0, 0.7]]))
    assert np.allclose(table.probabilities[:, 0], 0.0)


def test_normalised_turns_conditioning_into_the_conditional():
    """The german-tank idiom: uniform-then-condition becomes uniform on the admitted set."""
    class Three(Enum):
        elements = [0, 1, 2]

    @DPP
    def truncated(n: Three) -> Three:
        s = Three.uniform()
        observe((s < n) | (s == n))
        return s

    channel = truncated.evaluated().normalised()
    # Given n = 1 the draw is uniform on {0, 1}, whatever mass the evidence had.
    assert channel.probabilities[0, 1] == pytest.approx(0.5)
    assert channel.probabilities[1, 1] == pytest.approx(0.5)
    assert channel.probabilities[2, 1] == pytest.approx(0.0)


def test_an_evaluated_table_reused_in_a_model_is_the_matrix_it_came_from():
    table = biased.evaluated()

    @DPP
    def indirectly() -> Coin:
        return table()

    assert np.allclose(matrix_of(indirectly), matrix_of(biased))


def test_the_matrix_must_match_the_declared_types():
    with pytest.raises(ValueError, match="must be"):
        LikelihoodTable((Bool,), (Bool,), np.ones((2, 4)))


def test_the_matrix_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        LikelihoodTable((), (Bool,), np.array([[0.5], [-0.5]]))


def test_a_magnitude_beyond_float64_raises_rather_than_underflowing():
    """The scale is folded into the matrix only where that loses nothing."""
    @DPP
    def rare() -> None:
        for _ in range(8):
            observe(flip(1e-60))

    with pytest.raises(ValueError, match="matrix"):
        rare.evaluated()


def test_str_decodes_patterns_to_the_declared_values():
    text = str(biased.evaluated())
    assert "heads" in text and "tails" in text
    assert "0.25" in text and "0.75" in text
