# %% [markdown]
# # Discrete Bayesian statistics
#
# [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics) treats unknown
# quantities as random variables: a prior distribution states what is believed about them
# before the data are seen, and conditioning on the data gives the posterior distribution.
#
# In Pylea, a model is a DPP that draws the unknown quantities from their prior,
# conditions on each observation with `observe`, and returns what is asked about;
# `probability_mass_function`, `prob` and `expectation` compute its posterior exactly.
# Every variable takes finitely many values, so a continuous parameter is discretised on a
# grid, as in the first and the third example.
#
# ## A biased coin
#
# A coin shows heads with an unknown probability, its bias.  The data are 3000 tosses of a
# coin with bias 0.7, generated at random.  The prior on the bias is uniform on a grid of
# 50 values.  Each update takes the posterior so far as its prior and conditions on one
# more toss.

# %%
import matplotlib.pyplot as plt
import numpy as np

from pylea import DPP, Enum, observe, probability_mass_function

TRUE_BIAS = 0.7
rng = np.random.default_rng(seed=0)
tosses = ["heads" if u < TRUE_BIAS else "tails" for u in rng.random(3000)]


class Bias(Enum):
    elements = [k / 50 for k in range(1, 51)]  # 0.02, 0.04, ..., 1


class Side(Enum):
    elements = ["tails", "heads"]


@DPP
def side_given_bias(bias: Bias) -> Side:
    return Side.cpt([Bias], {(p,): {"heads": p, "tails": 1 - p} for p in Bias})(bias)


def update(prior: dict, side: str) -> dict:
    """The posterior of the bias, given `prior` and one toss that showed `side`."""

    @DPP
    def bias() -> Bias:
        b = Bias.random(prior)
        observe(side_given_bias(b) == side)
        return b

    return probability_mass_function(bias)


posteriors = [{p: 1 / len(Bias) for p in Bias}]  # the uniform prior
for side in tosses:
    posteriors.append(update(posteriors[-1], side))

# %% [markdown]
# The posterior after `n` tosses, for `n` from 0 to 3000, concentrates on the true bias:

# %%
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()


def draw(n: int):
    ax.clear()
    ax.axvline(TRUE_BIAS, color="grey", linestyle=":")
    ax.set(xlim=(0, 1.02), ylim=(0, 1), xlabel="bias", ylabel="posterior probability",
           title=f"after {n} tosses")
    return ax.bar(list(Bias), list(posteriors[n].values()), width=0.015)


shown = [0, *np.unique(np.geomspace(1, len(tosses), 60).astype(int))]
animation = FuncAnimation(fig, draw, frames=shown)
plt.close(fig)
HTML(animation.to_jshtml())

# %% [markdown]
# A beta prior on the bias gives the posterior in closed form, which is the more efficient
# way to solve this problem; the example only illustrates the method.  Pylea computes
# posteriors that have no closed form, such as that of the bioassay below.
#
# ## The German tank problem
#
# During the Second World War, Allied statisticians estimated the monthly production of
# German tanks from the serial numbers on parts of captured tanks, such as gearboxes and
# wheels [1].  For June 1940 they estimated 169 tanks, where conventional intelligence
# estimated 1,000; German records later showed 122.
#
# In the [German tank problem](https://en.wikipedia.org/wiki/German_tank_problem), the
# size `N` of a fleet is estimated from the serial numbers `1, ..., N` of captured tanks.
# The prior on `N` is uniform on 1 to 40, and each captured tank's serial number is
# uniform on 1 to `N`.  Unlike the coin's bias, `N` is discrete to begin with: the
# parameter space is already finite, and no discretisation is needed.

# %%
class N(Enum):
    elements = range(1, 41)  # fleet sizes, and serial numbers


@DPP
def serial(n: N) -> N:
    """The serial number of one captured tank, uniform on 1 to `n`."""
    s = N.uniform()
    observe(s <= n)
    return s


# On its own, `serial` gives every serial number up to `n` weight 1/40.  Normalising its
# table divides by the probability n/40 of the condition `s <= n`, which gives 1/n, the
# probability of a serial number given `n`.  A table is called inside a DPP like a DPP.
serial_given_fleet = serial.evaluated().normalised()


def posterior(serials: list[int]) -> DPP:
    """The fleet size, given the observed serial numbers."""

    @DPP
    def fleet() -> N:
        n = N.uniform()
        for s in serials:
            observe(serial_given_fleet(n) == s)
        return n

    return fleet


pmf = probability_mass_function(posterior([15, 14, 3]))
print("most probable N:", max(pmf, key=lambda n: pmf[n]))
print("P(N > 30):", round(sum(p for n, p in pmf.items() if n > 30), 4))

# %%
for serials in ([15, 14, 3], [15, 14, 3, 6, 2, 10, 5, 16, 8, 1, 4, 19]):
    pmf = probability_mass_function(posterior(serials))
    plt.plot(list(pmf), list(pmf.values()), marker=".", label=f"{len(serials)} serials")
plt.xlabel("fleet size N")
plt.ylabel("posterior probability")
plt.legend()
plt.show()

# %% [markdown]
# ## A bioassay
#
# Four groups of five animals were given a toxin at four doses, and 0, 1, 3 and 5 of them
# died [2].  An animal at log dose `x` dies with probability `logistic(α + β x)`.  As in
# [3], the prior on the parameters `α` and `β` is uniform on a grid of 50 × 50 values.
# The outcome for one animal is a conditional probability table over that grid, one for
# each dose, and `parameters` returns the parameters given every animal's outcome.

# %%
import math

from pylea import Bool, expectation, prob

DOSES = [-0.86, -0.30, -0.05, 0.73]  # log g/ml
DEATHS = [0, 1, 3, 5]  # out of five animals at each dose


def grid(low: float, high: float, points: int) -> list[float]:
    """`points` equally spaced values from `low` to `high`, both included."""
    step = (high - low) / (points - 1)
    return [low + k * step for k in range(points)]


class Alpha(Enum):
    elements = grid(-4, 8, 50)


class Beta(Enum):
    elements = grid(-10, 40, 50)


class Outcome(Enum):
    elements = ["survived", "died"]


def outcome_given_parameters(dose: float) -> DPP:
    """The outcome for one animal at `dose`, given the parameters."""
    rows = {}
    for a in Alpha:
        for b in Beta:
            p = 1 / (1 + math.exp(-(a + b * dose)))  # logistic(a + b * dose)
            rows[a, b] = {"survived": 1 - p, "died": p}

    @DPP
    def outcome(alpha: Alpha, beta: Beta) -> Outcome:
        return Outcome.cpt([Alpha, Beta], rows)(alpha, beta)

    return outcome


outcome_at = {dose: outcome_given_parameters(dose) for dose in DOSES}


@DPP
def parameters() -> tuple[Alpha, Beta]:
    alpha, beta = Alpha.uniform(), Beta.uniform()
    for dose, deaths in zip(DOSES, DEATHS):
        for outcome in ["died"] * deaths + ["survived"] * (5 - deaths):
            observe(outcome_at[dose](alpha, beta) == outcome)
    return alpha, beta


# %% [markdown]
# A query calls `parameters` and returns what is asked about.  `map` applies an ordinary
# Python function to the values of variables.  The probability that the toxin is harmful,
# that is, that the probability of death rises with the dose, `β > 0`:


# %%
@DPP
def harmful() -> Bool:
    alpha, beta = parameters()
    return beta.map(lambda b: b > 0, data_type=Bool)


prob(harmful)

# %% [markdown]
# The dose at which half of the animals die, LD50, is `-α / β` where `β > 0`.  It is one
# more variable, on a grid whose ends take the values beyond them:


# %%
class LethalDose(Enum):
    elements = grid(-0.6, 0.6, 25)  # log g/ml


def nearest_dose(a: float, b: float) -> float:
    """The grid value nearest to `-a / b`."""
    return min(LethalDose, key=lambda d: abs(d + a / b))


@DPP
def lethal_dose() -> LethalDose:
    alpha, beta = parameters()
    observe(beta.map(lambda b: b > 0, data_type=Bool))
    return alpha.map(nearest_dose, beta, data_type=LethalDose)


print("E[LD50]:", round(expectation(lethal_dose), 4))
pmf = probability_mass_function(lethal_dose)
plt.bar(list(pmf), list(pmf.values()), width=0.04)
plt.xlabel("LD50 (log g/ml)")
plt.ylabel("posterior probability")
plt.show()

# %% [markdown]
# ## References
#
# [1] R. Ruggles, H. Brodie.  An empirical approach to economic intelligence in World War
# II.  *J. Am. Stat. Assoc.* 42(237), 1947.
#
# [2] A. Racine, A. P. Grieve, H. Flühler, A. F. M. Smith.  Bayesian methods in practice:
# experiences in the pharmaceutical industry.  *Appl. Stat.* 35(2), 1986.
#
# [3] A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, D. B. Rubin.
# *Bayesian Data Analysis*, 3rd ed., Section 3.7.  Chapman and Hall/CRC, 2013.
