"""Voting power in the US Electoral College.

Each state casts a fixed number of electoral votes as one bloc, and the presidency needs
270 of the 538.  A state's voting power is not proportional to its votes: it is the
probability that the state decides the outcome, which Penrose [1] and Banzhaf [2] measure
with every other state voting either way independently with probability one half, so that
every split of the others is equally likely.  `pivotal` is the event that a state's votes
are decisive under that measure.  The digits below are statements about the model rather
than estimates, so the queries pin the exact engine:

    >>> from pylea import prob
    >>> wyoming = prob(pivotal("Wyoming"), method="exact")
    >>> california = prob(pivotal("California"), method="exact")
    >>> round(wyoming, 6), round(california, 6)
    (0.022692, 0.460711)

Banzhaf's point was that weighted voting distorts power beyond the weights: an elector of
the largest state carries more power than one of the smallest, not less:

    >>> round((california / 54) / (wyoming / 3), 2)
    1.13

The probability sums over the ``2**50`` splits of the other states, far past enumerating.
Written as a running total, the DPP is a chain, so the exact engine answers it as a
dynamic programme over totals without that being stated anywhere.

`decisive` poses the same question for any weighted voting game, such as a council in
which shareholders or member states carry different vote counts.

    [1] Penrose LS.  The elementary statistics of majority voting.  J R Stat Soc.
        1946;109(1):53-57.
    [2] Banzhaf JF.  Weighted voting doesn't work: a mathematical analysis.
        Rutgers Law Rev. 1965;19:317-343.
"""
from pylea import DPP, Bool, UInt, flip

# Electoral votes per state for the 2024 election, apportioned from the 2020 census;
# they sum to 538.  Maine and Nebraska in fact split their votes by district; treating
# every state as one bloc is the standard simplification in this literature.
ELECTORAL_VOTES = {
    "Alabama": 9, "Alaska": 3, "Arizona": 11, "Arkansas": 6, "California": 54,
    "Colorado": 10, "Connecticut": 7, "Delaware": 3, "District of Columbia": 3,
    "Florida": 30, "Georgia": 16, "Hawaii": 4, "Idaho": 4, "Illinois": 19,
    "Indiana": 11, "Iowa": 6, "Kansas": 6, "Kentucky": 8, "Louisiana": 8, "Maine": 4,
    "Maryland": 10, "Massachusetts": 11, "Michigan": 15, "Minnesota": 10,
    "Mississippi": 6, "Missouri": 10, "Montana": 4, "Nebraska": 5, "Nevada": 6,
    "New Hampshire": 4, "New Jersey": 14, "New Mexico": 5, "New York": 28,
    "North Carolina": 16, "North Dakota": 3, "Ohio": 17, "Oklahoma": 7, "Oregon": 8,
    "Pennsylvania": 19, "Rhode Island": 4, "South Carolina": 9, "South Dakota": 3,
    "Tennessee": 11, "Texas": 40, "Utah": 6, "Vermont": 3, "Virginia": 13,
    "Washington": 12, "West Virginia": 4, "Wisconsin": 10, "Wyoming": 3,
}


def decisive(member: str, weights: dict[str, int], quota: int) -> DPP:
    """The event that `member`'s votes decide a weighted voting game.

    `weights` gives each voter's whole vote count and `quota` the total a side needs to
    win.  Every other voter joins the side independently with probability one half, the
    convention behind the Penrose and Banzhaf measures of a priori power.  `member` is
    decisive where the others' total falls short of the quota by at most `member`'s own
    votes: with those votes the side wins, without them it does not.

    Raises `KeyError` for an unknown member and `ValueError` for a quota outside 1 to
    the total vote count.
    """
    votes = weights[member]
    total_votes = sum(weights.values())
    if not 1 <= quota <= total_votes:
        raise ValueError(f"the quota must be between 1 and {total_votes}; got {quota}.")

    class Total(UInt):
        width = max(1, total_votes.bit_length())     # wide enough for every vote at once

    @DPP
    def swings() -> Bool:
        # The other voters' total for one side: each joins it by a fair coin flip and then
        # contributes either all its votes or none of them.
        others = Total.const(0)
        for name, count in weights.items():
            if name != member:
                others = others + flip(0.5).match({True: count, False: 0},
                                                  data_type=Total)
        # Decisive: with `member` the side reaches the quota, without it it falls short.
        return (others + votes >= quota) & (others <= quota - 1)

    return swings


def pivotal(state: str) -> DPP:
    """The event that `state` decides the presidency, its Banzhaf voting power."""
    return decisive(state, ELECTORAL_VOTES, quota=270)
