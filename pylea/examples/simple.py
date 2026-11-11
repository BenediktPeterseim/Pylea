"""One small model per construction of the language.

Between them these cover Boolean gates, sharing and redundancy, unused inputs and outputs,
conditioning, several inputs and outputs, and every `pylea.syntax.enum.Enum` construction
over types whose element counts both are and are not powers of two.
"""

from pylea.syntax.dpp import DPP, Bool
from pylea.syntax.enum import Enum
from pylea.syntax.flip import flip
from pylea.syntax.observe import observe


class Coin(Enum):
    elements = ["H", "T"]                       # 1 bit, exactly two patterns


class Color(Enum):
    elements = ["R", "G", "B"]                  # 2 bits, one pattern encoding no element


class Direction(Enum):
    elements = ["N", "E", "S", "W"]             # 2 bits, exactly four patterns


class Season(Enum):
    # 3 bits, three patterns encoding no element
    elements = ["Spring", "Summer", "Autumn", "Winter", "Monsoon"]


@DPP
def empty_dpp() -> None:
    pass


@DPP
def identity_dpp(x: Bool) -> Bool:
    return x


@DPP
def redundant_ors_dpp() -> Bool:
    x = flip(0.1)
    return x | x | x


@DPP
def redundant_ands_dpp() -> Bool:
    x = flip(0.4)
    return x & x


@DPP
def mixed_redundant_dpp() -> Bool:
    x = flip(0.2)
    y = flip(0.5)
    u = x & x
    return u | y | y


@DPP
def fault_tree_and_only() -> Bool:
    x = flip(0.1)
    y = flip(0.2)
    return x & y


@DPP
def tree_structured_fault_tree_small() -> Bool:
    x = flip(0.1)
    y = flip(0.2)
    z = flip(0.3)
    w = flip(0.4)
    u = x & y
    v = z & w
    return u | v


@DPP
def fault_tree_small_with_sharing() -> Bool:
    x = flip(0.1)
    y = flip(0.2)
    z = flip(0.3)
    u = x & y
    v = y & z
    return u | v


@DPP
def tree_structured_fault_tree_larger():
    x1 = flip(0.1)
    x2 = flip(0.2)
    x3 = flip(0.3)
    x4 = flip(0.3)
    x5 = flip(0.3)
    x6 = flip(0.3)
    x7 = flip(0.3)
    x8 = flip(0.3)
    y1 = x1 | x2
    y2 = x3 | x4
    y3 = x5 | x6
    y4 = x7 | x8
    z1 = y1 & y2
    z2 = y3 & y4
    return z1 | z2


@DPP
def unused_output_dpp() -> Bool:
    x = flip(0.1)
    y = flip(0.2)                               # read by nothing
    return x


@DPP
def unused_input_dpp(unused: Bool) -> Bool:
    x = flip(0.1)
    return x


@DPP
def dpp_with_observe(x: Bool) -> Bool:
    c = flip(0.5)
    y = c | x
    observe(y)
    return x

@DPP
def two_inputs_dpp(x: Bool, y: Bool) -> Bool:
    return x & y


@DPP
def two_outputs_dpp() -> tuple[Bool, Bool]:
    x = flip(0.3)
    y = flip(0.7)
    return x, y


@DPP
def obs_dpp() -> Bool:
    x = flip(0.3)
    observe(x)
    return x


@DPP
def two_input_output_dpp(x: Bool, y: Bool) -> tuple[Bool, Bool]:
    a = flip(0.5)
    b = flip(0.5)
    u = a | x
    v = b | y
    return u | v, u & v


# The same model, with its input types given by the `types` keyword argument.
@DPP
def two_input_output_dpp_type_tuple(x, types=(Bool, Bool)):
    a = flip(0.5)
    b = flip(0.5)
    u = a | x[0]
    v = b | x[1]
    return u | v, u & v


@DPP
def not_after_flip() -> Bool:
    x = flip(0.3)
    y = ~x
    return y


@DPP
def double_not_after_flip() -> Bool:
    x = flip(0.3)
    y = ~x
    z = ~y
    return z


# --- drawing from an Enum ---

@DPP
def coin_random() -> Coin:
    # P(H) is 0.3 and P(T) is 0.7.
    return Coin.random({"H": 0.3, "T": 0.7})


@DPP
def color_random() -> Color:
    # The fourth bit pattern encodes no element, so it gets probability 0.
    return Color.random({"R": 0.2, "G": 0.5, "B": 0.3})


@DPP
def color_random_missing_element() -> Color:
    # "G" is absent, so it has probability 0.
    return Color.random({"R": 0.6, "B": 0.4})


@DPP
def color_random_unnormalised() -> Color:
    # The weights do not sum to 1, so they are normalised to 0.5, 0.25 and 0.25.
    return Color.random({"R": 2.0, "G": 1.0, "B": 1.0})


# --- branching on an Enum ---

@DPP
def coin_match_swap(x: Coin) -> Coin:
    # Swaps the two elements, which is a negation of the single bit.
    return x.match({"H": "T", "T": "H"}, data_type=Coin)


@DPP
def color_match_to_bool(x: Color) -> Bool:
    # Selects between constants rather than between variables.
    return x.match({"R": True, "G": False, "B": True}, data_type=Bool)


@DPP
def coin_match_with_variable(x: Coin) -> Bool:
    # Selects between variables rather than between constants.
    b = flip(0.4)
    return x.match({"H": b, "T": ~b})


# --- drawing and branching together ---

@DPP
def color_random_then_match() -> Bool:
    c = Color.random({"R": 0.5, "G": 0.25, "B": 0.25})
    return c.match({"R": True, "G": False, "B": False}, data_type=Bool)


# --- types wider than one bit ---

@DPP
def color_identity(x: Color) -> Color:
    # A two-bit input returned unchanged.
    return x


@DPP
def direction_random() -> Direction:
    # Two bits, every pattern encoding an element.
    return Direction.random({"N": 0.1, "E": 0.4, "S": 0.4, "W": 0.1})


@DPP
def season_random() -> Season:
    # Three bits, and "Monsoon" is absent, so it has probability 0.
    return Season.random({"Spring": 0.25, "Summer": 0.35, "Autumn": 0.25, "Winter": 0.15})


@DPP
def color_unused_input(unused: Color) -> Bool:
    # A two-bit input read by nothing, so its wires are summed out.
    return flip(0.3)


@DPP
def color_unused_output() -> Bool:
    # Drawn and then discarded, so dead code elimination removes it.
    c = Color.random({"R": 0.5, "G": 0.3, "B": 0.2})  # noqa: F841, deliberately unread
    return flip(0.4)


@DPP
def coin_and_color() -> tuple[Coin, Color]:
    # Two independent outputs, each wider than one bit.
    c = Coin.random({"H": 0.6, "T": 0.4})
    k = Color.random({"R": 0.5, "G": 0.3, "B": 0.2})
    return c, k


@DPP
def direction_to_coin(x: Direction) -> Coin:
    # Branching from a two-bit type down to a one-bit one.
    return x.match({"N": "H", "E": "T", "S": "T", "W": "H"}, data_type=Coin)


@DPP
def color_two_matches(x: Color) -> tuple[Bool, Coin]:
    # One two-bit input branched on twice, so its wires have several readers.
    b = x.match({"R": True, "G": False, "B": False}, data_type=Bool)
    c = x.match({"R": "H", "G": "T", "B": "H"}, data_type=Coin)
    return b, c
