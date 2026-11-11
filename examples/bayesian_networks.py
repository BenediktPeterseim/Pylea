# %% [markdown]
# # Bayesian networks
#
# A Bayesian network is written as a DPP: each variable is an `Enum`, each conditional
# probability table a DPP of the variable's parents built with `Enum.cpt`, and the network
# a DPP that calls the tables in order and returns every variable.
#
# ## The sprinkler network
#
# Whether it is cloudy decides how likely the sprinkler is on and how much it rains, and
# both decide how wet the grass is.

# %%
from pylea import DPP, Enum, observe, probability_mass_function


class Cloudiness(Enum):
    elements = ["cloudy", "sunny"]


class SprinklerState(Enum):
    elements = ["on", "off"]


class Rain(Enum):
    elements = ["none", "light", "heavy"]


class GrassWetness(Enum):
    elements = ["dry", "damp", "wet"]


@DPP
def sprinkler_given_cloudy(c: Cloudiness) -> SprinklerState:
    return SprinklerState.cpt(
        [Cloudiness],
        {
            ("cloudy",): [0.1, 0.9],
            ("sunny",):  [0.5, 0.5],
        },
    )(c)


@DPP
def rain_given_cloudy(c: Cloudiness) -> Rain:
    return Rain.cpt(
        [Cloudiness],
        {
            ("cloudy",): [0.2, 0.4, 0.4],
            ("sunny",):  [0.7, 0.2, 0.1],
        },
    )(c)


@DPP
def wetgrass_given_sprinkler_rain(s: SprinklerState, r: Rain) -> GrassWetness:
    return GrassWetness.cpt(
        [SprinklerState, Rain],
        {
            ("on",  "none"):  [0.2, 0.5, 0.3],
            ("on",  "light"): [0.0, 0.2, 0.8],
            ("on",  "heavy"): [0.0, 0.1, 0.9],
            ("off", "none"):  [0.9, 0.1, 0.0],
            ("off", "light"): [0.1, 0.6, 0.3],
            ("off", "heavy"): [0.0, 0.3, 0.7],
        },
    )(s, r)


@DPP
def sprinkler_network() -> tuple[Cloudiness, SprinklerState, Rain, GrassWetness]:
    cloudiness = Cloudiness.random({"cloudy": 0.4, "sunny": 0.6})
    sprinkler_state = sprinkler_given_cloudy(cloudiness)
    rain = rain_given_cloudy(cloudiness)
    grass_wetness = wetgrass_given_sprinkler_rain(sprinkler_state, rain)
    return cloudiness, sprinkler_state, rain, grass_wetness


sprinkler_network.dependency_graph()

# %% [markdown]
# `cpt` maps each tuple of parent states to the probabilities of the child's states, in
# the order `elements` lists them.  Printed, a table shows one row per tuple of parent
# states:

# %%
print(wetgrass_given_sprinkler_rain.evaluated())

# %% [markdown]
# A query is a DPP that calls the network, observes the evidence and returns the variable
# asked about; `probability_mass_function` gives its posterior.  Given wet grass, learning
# that it rained heavily makes it less likely that the sprinkler was on:


# %%
@DPP
def sprinkler_given_wet_grass() -> SprinklerState:
    cloudiness, sprinkler_state, rain, grass_wetness = sprinkler_network()
    observe(grass_wetness == "wet")
    return sprinkler_state


@DPP
def sprinkler_given_wet_grass_and_heavy_rain() -> SprinklerState:
    cloudiness, sprinkler_state, rain, grass_wetness = sprinkler_network()
    observe(grass_wetness == "wet")
    observe(rain == "heavy")
    return sprinkler_state


print(probability_mass_function(sprinkler_given_wet_grass))
print(probability_mass_function(sprinkler_given_wet_grass_and_heavy_rain))

# %% [markdown]
# ## The Asia network
#
# A network from medical diagnosis [1], with the tables of [2]: a visit to Asia and
# smoking, three lung diseases, an x-ray and shortness of breath (dyspnoea).

# %%
class YesNo(Enum):
    elements = ["yes", "no"]


@DPP
def tub_given_asia(asia: YesNo) -> YesNo:
    return YesNo.cpt([YesNo], {("yes",): [0.05, 0.95], ("no",): [0.01, 0.99]})(asia)


@DPP
def lung_given_smoke(smoke: YesNo) -> YesNo:
    return YesNo.cpt([YesNo], {("yes",): [0.1, 0.9], ("no",): [0.01, 0.99]})(smoke)


@DPP
def bronc_given_smoke(smoke: YesNo) -> YesNo:
    return YesNo.cpt([YesNo], {("yes",): [0.6, 0.4], ("no",): [0.3, 0.7]})(smoke)


@DPP
def either_given_lung_tub(lung: YesNo, tub: YesNo) -> YesNo:
    return YesNo.cpt([YesNo, YesNo], {
        ("yes", "yes"): [1.0, 0.0], ("no", "yes"): [1.0, 0.0],
        ("yes", "no"): [1.0, 0.0], ("no", "no"): [0.0, 1.0],
    })(lung, tub)


@DPP
def xray_given_either(either: YesNo) -> YesNo:
    return YesNo.cpt([YesNo], {("yes",): [0.98, 0.02], ("no",): [0.05, 0.95]})(either)


@DPP
def dysp_given_bronc_either(bronc: YesNo, either: YesNo) -> YesNo:
    return YesNo.cpt([YesNo, YesNo], {
        ("yes", "yes"): [0.9, 0.1], ("no", "yes"): [0.7, 0.3],
        ("yes", "no"): [0.8, 0.2], ("no", "no"): [0.1, 0.9],
    })(bronc, either)


@DPP
def asia_network() -> tuple[YesNo, YesNo, YesNo, YesNo, YesNo, YesNo, YesNo, YesNo]:
    asia = YesNo.random({"yes": 0.01, "no": 0.99})
    tub = tub_given_asia(asia)
    smoke = YesNo.random({"yes": 0.5, "no": 0.5})
    lung = lung_given_smoke(smoke)
    bronc = bronc_given_smoke(smoke)
    either = either_given_lung_tub(lung, tub)
    xray = xray_given_either(either)
    dysp = dysp_given_bronc_either(bronc, either)
    return asia, tub, smoke, lung, bronc, either, xray, dysp


asia_network.dependency_graph()

# %% [markdown]
# Lung cancer, for a smoker with a positive x-ray and dyspnoea:


# %%
@DPP
def lung_given_findings() -> YesNo:
    asia, tub, smoke, lung, bronc, either, xray, dysp = asia_network()
    observe(smoke == "yes")
    observe(xray == "yes")
    observe(dysp == "yes")
    return lung


probability_mass_function(lung_given_findings)

# %% [markdown]
# ## Networks in the `.bif` format
#
# `bif_to_python` turns a network in the `.bif` format, here the Asia network as the
# file [asia.bif](asia.bif) of [2], into the source of a module of the form above.

# %%
from pathlib import Path

from pylea.utility.bif_to_dpp import bif_to_python

print(bif_to_python(Path("asia.bif").read_text(), network_name="asia_network"))

# %% [markdown]
# ## References
#
# [1] S. L. Lauritzen, D. J. Spiegelhalter.  Local computations with probabilities on
# graphical structures and their application to expert systems.  *J. R. Stat. Soc. B*
# 50(2), 1988.
#
# [2] M. Scutari.  bnlearn Bayesian Network Repository.
# <https://www.bnlearn.com/bnrepository/>
