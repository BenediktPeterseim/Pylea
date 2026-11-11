"""The birthday paradox: how likely a small group shares a birthday.

Among 20 people, each born on a uniformly random day of a 365-day year, what is the
probability that at least two share a birthday?

For `n` people the event is a clique of ``n * (n - 1) / 2`` pairwise equality constraints
over `n` independent 9-bit variables, so its treewidth is far past the reach of standard
exact inference methods.  Still, Pylea can solve this example using a sampling-based
algorithm under the hood:

    >>> from pylea import prob
    >>> chance = prob(two_with_same_birthday(n_people=20), rel_err=0.02, seed=0)
    >>> 0.40 < chance.low < chance.high < 0.43
    True

The exact value, from the closed form ``1 - prod((365 - i) / 365 for i in range(20))``, is
0.4114.
"""
from pylea import DPP, Bool, Enum


def two_with_same_birthday(n_people: int, n_days: int = 365) -> DPP:
    """The event that at least two of `n_people`, each born on a uniformly random one of
    `n_days` days, share a birthday."""

    class Day(Enum):
        elements = range(1, n_days + 1)

    @DPP
    def shared_birthday() -> Bool:
        birthdays: list[Day] = [Day.uniform() for _ in range(n_people)]
        return Bool.any(a == b for i, a in enumerate(birthdays)
                               for j, b in enumerate(birthdays) if i < j)

    return shared_birthday
