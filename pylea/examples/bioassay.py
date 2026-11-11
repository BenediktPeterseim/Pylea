"""Dose-response by discretised Bayesian regression: the bioassay of Racine et al.

Four groups of five animals were exposed to a toxin at four doses, and 0, 1, 3 and 5 of
them died [1].  An animal at dose `x` dies with probability ``logistic(alpha + beta * x)``,
and the dose at which that probability crosses one half, called LD50, is the standard
summary of the toxin's strength.  Gelman et al. treat this experiment as their bioassay
example and compute the posterior of ``(alpha, beta)`` by discretising it on a grid [2];
the DPPs below are that same analysis, the inference on the grid being done by the exact
engine, and LD50, a nonlinear function of both parameters, being one more variable:

    >>> from pylea import expectation, prob
    >>> round(expectation(lethal_dose, method="exact"), 4)
    -0.1065

So half of the animals die near dose ``10**-0.11`` g/ml.  That the toxin kills at all,
which is the event of a positive slope, is practically certain given the data:

    >>> round(prob(harmful, method="exact"), 7)
    0.9999997

LD50 is read on the dose scale, log g/ml, and is only meaningful where the slope is
positive, so `lethal_dose` conditions on that as [2] does.  The likelihood conditions on
each animal's own outcome; which of a group's animals died is arbitrary, and choosing a
different set changes the weight of every ``(alpha, beta)`` by the same binomial factor,
which normalising cancels.

    [1] Racine A, Grieve AP, Fluehler H, Smith AFM.  Bayesian methods in practice:
        experiences in the pharmaceutical industry.  Appl Stat. 1986;35(2):93-150.
    [2] Gelman A, Carlin JB, Stern HS, Dunson DB, Vehtari A, Rubin DB.  Bayesian Data
        Analysis.  3rd ed.  Chapman and Hall/CRC; 2013.  Section 3.7.
"""
import math

from pylea import DPP, Bool, Enum, observe

# The experiment: dose in log g/ml, five animals per group, deaths observed.
DOSES = (-0.86, -0.30, -0.05, 0.73)
DEATHS = (0, 1, 3, 5)
GROUP_SIZE = 5


def _grid(low: float, high: float, points: int) -> list[float]:
    """`points` equally spaced values from `low` to `high`, both included."""
    step = (high - low) / (points - 1)
    return [low + k * step for k in range(points)]


class Alpha(Enum):
    """The intercept, on the grid Gelman et al. use; the prior is uniform on it."""
    elements = _grid(-4.0, 8.0, 50)


class Beta(Enum):
    """The slope per unit log dose, on the same reference's grid."""
    elements = _grid(-10.0, 40.0, 50)


_LD50_GRID = _grid(-0.6, 0.6, 61)


class LethalDose(Enum):
    """LD50 in log g/ml, discretised; the ends absorb the negligible mass beyond them."""
    elements = _LD50_GRID


class Died(Enum):
    elements = [False, True]


def _death_probability(alpha: float, beta: float, dose: float) -> float:
    return 1.0 / (1.0 + math.exp(-(alpha + beta * dose)))


def _dies(dose: float) -> DPP:
    """Whether an animal at `dose` dies, given the two parameters, as a DPP of its own.

    The conditional probability table wrapped as a `@DPP`, so a diagram labels each
    call's box with the dose.
    """
    table = Died.cpt(
        [Alpha, Beta],
        {(a, b): {True: _death_probability(a, b, dose),
                  False: 1.0 - _death_probability(a, b, dose)}
         for a in Alpha for b in Beta})

    def dies(alpha: Alpha, beta: Beta) -> Died:
        return table(alpha, beta)

    return DPP(dies, name=f"dies_at({dose:g})")


# One conditional probability table per dose.
_dies_at = {dose: _dies(dose) for dose in DOSES}


def _fitted_parameters() -> tuple[Alpha, Beta]:
    """The parameters, drawn from the prior and conditioned on every animal's outcome.

    Must be called inside an active DPP definition.
    """
    alpha, beta = Alpha.uniform(), Beta.uniform()
    for dose, deaths in zip(DOSES, DEATHS):
        for died in [True] * deaths + [False] * (GROUP_SIZE - deaths):
            observe(_dies_at[dose](alpha, beta) == died)
    return alpha, beta


@DPP
def harmful() -> Bool:
    """The event that the death probability rises with the dose, given the data."""
    _alpha, beta = _fitted_parameters()
    return beta.map(lambda b: b > 0.0, data_type=Bool)


def _nearest_dose(alpha: float, beta: float) -> float:
    """The `LethalDose` value nearest ``-alpha / beta``, its ends taking what lies
    beyond; the value for a non-positive slope is arbitrary, all its mass being
    conditioned away."""
    if beta <= 0.0:
        return _LD50_GRID[0]
    return min(_LD50_GRID, key=lambda d: abs(d - (-alpha / beta)))


@DPP
def lethal_dose() -> LethalDose:
    """The dose at which half the animals die, in log g/ml, given the data."""
    alpha, beta = _fitted_parameters()
    observe(beta.map(lambda b: b > 0.0, data_type=Bool))
    return alpha.map(_nearest_dose, beta, data_type=LethalDose)
