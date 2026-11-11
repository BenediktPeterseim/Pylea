r"""Seepage through a bed of gravel: how much water gets through.

Water is poured onto the middle stone of the top layer of a bed `width` stones across and
`layers` deep.  A wet stone wets each of the three stones beneath it, left, centre and
right, along a channel that is open with probability `p`, the channels being independent.
The stones at the two edges have fewer channels.

        o             the top layer, where the water goes in
       /|\
      o o o           every channel is open with probability `p`
     /|\|/|\
    o o o o o         the bottom layer, whose wet stones are counted

`wet_stones` counts the stones of the bottom layer that end up wet, and its expected value
is how much of the bed's floor the water reaches.  The digits below are statements about
the model rather than estimates, so every query pins the exact engine:

    >>> round(expectation(wet_stones(width=10, layers=40, p=0.5), method="exact"), 4)
    4.3058

That bed has 1120 channels, so its ``2**1120`` configurations are far past enumerating.
Each layer depends only on the layer above it, so the cost of the answer grows in
proportion to the depth of the bed, and exponentially in its width.

How much gets through turns sharply on `p`:

    >>> [round(expectation(wet_stones(10, 40, p), method="exact"), 3)
    ...  for p in (0.3, 0.45, 0.6)]
    [0.0, 1.752, 7.691]

Below the turn the water stops inside the bed, so a deeper bed lets through less and less:

    >>> [round(expectation(wet_stones(10, layers, 0.35), method="exact"), 4)
    ...  for layers in (10, 40, 80)]
    [0.5087, 0.0144, 0.0001]

Above it the flow reaches any depth, thinning only very slowly, since a bed of finite width
does dry up in the end:

    >>> [round(expectation(wet_stones(10, layers, 0.6), method="exact"), 3)
    ...  for layers in (40, 80, 160)]
    [7.691, 7.689, 7.687]

Percolation processes were introduced by Broadbent and Hammersley for fluid in a porous
medium [1].  This bed is directed bond percolation on the triangular lattice, and reading it
layer by layer makes it a stochastic cellular automaton, the correspondence Domany and
Kinzel set out [2].  Neither the numbers above nor the value of `p` where the turn happens
has a closed form: the best estimate of that threshold, 0.52198(1), comes from a series
expansion [3], and Hinrichsen reviews what is known about the class [4].

    [1] Broadbent SR, Hammersley JM.  Percolation processes: I. Crystals and mazes.
        Math Proc Camb Philos Soc. 1957;53(3):629-41.  doi:10.1017/S0305004100032680
    [2] Domany E, Kinzel W.  Equivalence of cellular automata to Ising models and
        directed percolation.  Phys Rev Lett. 1984;53(4):311-14.
        doi:10.1103/PhysRevLett.53.311
    [3] Jensen I, Guttmann AJ.  Series expansions of the percolation probability on the
        directed triangular lattice.  J Phys A Math Gen. 1996;29(3):497.
        doi:10.1088/0305-4470/29/3/006
    [4] Hinrichsen H.  Non-equilibrium critical phenomena and phase transitions into
        absorbing states.  Adv Phys. 2000;49(7):815-958.  doi:10.1080/00018730050198152
"""
from pylea import DPP, Bool, Enum, at_least, expectation, flip


def wet_stones(width: int, layers: int, p: float) -> DPP:
    """The number of stones in the bottom layer of the bed that end up wet.

    A stone is wet where a channel to it from a wet stone above is open, so each layer
    depends only on the one above.  With every channel open the whole bottom layer is wet,
    at any depth:

        >>> expectation(wet_stones(6, 20, 1.0), method="exact")
        6.0
    """
    class Wet(Enum):
        elements = list(range(width + 1))

    # One more stone counted.  The row for `width` is unreachable, no stone being left to
    # count, but a table needs a row for every state of its parent.
    one_more = Wet.cpt([Wet], {(k,): {min(k + 1, width): 1.0} for k in range(width + 1)})

    @DPP
    def bottom_layer() -> Wet:
        # The top layer: the stone the water is poured on is wet, the rest are dry.
        wet = [Bool.const(j == width // 2) for j in range(width)]

        for _ in range(layers):
            # One layer down.  Stone `j` is wet where some stone above it is wet and the
            # channel from that stone to `j` is open.  Every `flip(p)` is one channel of
            # the bed, drawn where it is used, so no two stones share a channel.
            wet = [at_least(1, *[wet[k] & flip(p)
                                 for k in (j - 1, j, j + 1) if 0 <= k < width])
                   for j in range(width)]

        # The stones of the bottom layer, counted one at a time.
        total = Wet.const(0)
        for stone in wet:
            total = stone.match({True: one_more(total), False: total}, data_type=Wet)
        return total

    return bottom_layer
