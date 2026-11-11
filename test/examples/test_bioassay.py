"""The bioassay example against an independent numpy replication of the grid analysis."""
import numpy as np
import pytest

from pylea import prob, probability_mass_function
from pylea.examples.bioassay import (
    DEATHS, DOSES, GROUP_SIZE, _LD50_GRID, Alpha, Beta, harmful, lethal_dose,
)


def reference():
    """The grid posterior, its positive-slope mass, and the discretised LD50 pmf,
    computed with plain numpy arrays and nothing from the example's model."""
    A = np.array(list(Alpha))
    B = np.array(list(Beta))
    x = np.array(DOSES)
    y = np.array(DEATHS)

    AA, BB = np.meshgrid(A, B, indexing="ij")
    theta = 1 / (1 + np.exp(-(AA[:, :, None] + BB[:, :, None] * x[None, None, :])))
    likelihood = np.prod(theta ** y * (1 - theta) ** (GROUP_SIZE - y), axis=2)
    posterior = likelihood / likelihood.sum()

    positive = posterior[:, B > 0].sum()

    conditional = posterior[:, B > 0] / posterior[:, B > 0].sum()
    ld50 = -AA[:, B > 0] / BB[:, B > 0]
    grid = np.array(_LD50_GRID)
    binned = np.zeros(len(grid))
    for weight, value in zip(conditional.ravel(), ld50.ravel()):
        binned[np.abs(grid - value).argmin()] += weight
    return positive, binned


def test_the_positive_slope_probability_matches_the_replication():
    positive, _ = reference()
    assert prob(harmful, method="exact") == pytest.approx(positive, rel=1e-12)


def test_the_discretised_ld50_posterior_matches_the_replication_entrywise():
    _, binned = reference()
    pmf = probability_mass_function(lethal_dose, method="exact")
    for value, mass in zip(_LD50_GRID, binned):
        assert pmf[value] == pytest.approx(mass, rel=1e-9, abs=1e-15), value


def test_the_grid_ends_hold_negligible_mass():
    # The LD50 grid clamps what lies beyond it onto its ends; this pins that the choice
    # of range keeps that distortion small.
    pmf = probability_mass_function(lethal_dose, method="exact")
    assert pmf[_LD50_GRID[0]] + pmf[_LD50_GRID[-1]] < 1e-3
