"""The sprinkler network, a Bayesian network built from conditional probability tables.

    Cloudiness  ->  SprinklerState  ->  GrassWetness
        |                                    ^
        +--------->  Rain  ------------------+

Each variable's states are an `pylea.syntax.enum.Enum`.  Each conditional probability table
is a DPP of the variable's parents applying an `pylea.syntax.enum.Enum.cpt` call, whose rows
map a tuple of parent states to probabilities over the child's states, in the order
`elements` declares them.  `sprinkler_network` returns all four variables, and a query is a
DPP that calls it, observes what was seen and returns what is asked about:

    >>> from pylea import prob
    >>> round(prob(rain_given_wet_or_damp_grass), 6)
    0.70302
"""
from pylea import DPP, Bool, Enum, observe


class Cloudiness(Enum):
    elements = ["cloudy", "sunny"]


class SprinklerState(Enum):
    elements = ["on", "off"]


class Rain(Enum):
    elements = ["none", "light", "heavy"]


class GrassWetness(Enum):
    elements = ["dry", "damp", "wet"]


# --- the conditional probability tables ---

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


# --- the joint distribution ---

@DPP
def sprinkler_network() -> tuple[Cloudiness, SprinklerState, Rain, GrassWetness]:
    cloudiness = Cloudiness.random({"cloudy": 0.4, "sunny": 0.6})
    sprinkler_state = sprinkler_given_cloudy(cloudiness)
    rain = rain_given_cloudy(cloudiness)
    grass_wetness = wetgrass_given_sprinkler_rain(sprinkler_state, rain)
    return cloudiness, sprinkler_state, rain, grass_wetness

# --- a query over it ---

@DPP
def rain_given_wet_or_damp_grass() -> Bool:
    """The event that it rained, given that the grass is wet or damp."""
    _, _, rain, grass_wetness = sprinkler_network()
    observe((grass_wetness == "wet") | (grass_wetness == "damp"))
    return (rain == "light") | (rain == "heavy")
