"""Secret Santa: how often a random gift-exchange draw needs no do-over.

Drawing names from a hat is the simplest way to run a Secret Santa gift exchange, but two
outcomes usually count as failures: someone drawing their own name, and, in a group that
includes couples, someone drawing their own partner's.  `valid_draw` is the probability that
one draw, made uniformly at random and without replacement, needs no do-over.

    >>> from pylea import prob
    >>> round(prob(valid_draw(n_couples=2, n_singles=1)), 6)
    0.133333

A family of ten, four couples and two singles, is past what the exact engine reaches within
minutes.  Each person's own forbidden names do not depend on how many other people are
drawing, so the chance of a valid draw stays about the same size as the group grows, and the
sampling engine converges quickly regardless of the exact engine's reach:

    >>> chance = prob(valid_draw(n_couples=4, n_singles=2), rel_err=0.03, seed=0)
    >>> 0.14 < chance.low < chance.high < 0.16
    True
"""
from pylea import DPP, Bool, Enum, pick


def valid_draw(n_couples: int, n_singles: int) -> DPP:
    """The event that a random draw, without replacement, assigns nobody their own name or
    their partner's.

    The group has ``2 * n_couples + n_singles`` people, positions ``0`` to ``n - 1``.
    Couples occupy consecutive pairs of positions, ``2k`` and ``2k + 1``; the remaining
    positions, from ``2 * n_couples`` up, are singles, who only need to avoid themselves.
    """
    n = 2 * n_couples + n_singles
    partner = {}
    for k in range(n_couples):
        a, b = 2 * k, 2 * k + 1
        partner[a], partner[b] = b, a

    class Person(Enum):
        elements = list(range(n))

    @DPP
    def draw() -> Bool:
        # `pick` draws each recipient from whoever is left, so nobody's name is given
        # out twice without anything needing to be conditioned on; whether a drawn name
        # is the giver's own or their partner's is checked below instead.
        recipients, _ = pick([Person.const(i) for i in range(n)], n)
        no_self = Bool.all(recipients[i] != i for i in range(n))
        no_partner = Bool.all(recipients[i] != partner[i] for i in partner)
        return no_self & no_partner

    return draw
