"""Drawing without replacement."""
from __future__ import annotations

from typing import Sequence, TypeVar, overload

from pylea.syntax.enum import Enum
from pylea.syntax.variable import Variable

_V = TypeVar('_V', bound=Variable)


@overload
def pick(choices: Sequence[_V]) -> tuple[_V, list[_V]]: ...
@overload
def pick(choices: Sequence[_V], n: int) -> tuple[list[_V], list[_V]]: ...
def pick(choices: Sequence[_V], n: int | None = None):
    """`n` of `choices`, drawn uniformly at random without replacement, with the rest.

    With `n` left out, one draw: the result is the drawn value and the list of those not
    drawn, in their original order, so calling `pick` again on the rest deals a second
    value without replacement from the first.  With `n` given, the first value is instead
    the list of the `n` draws, in the order they were drawn; drawing every element gives
    a uniformly random ordering of `choices`.

    `choices` must be non-empty, every element of the same
    `pylea.syntax.variable.Variable` type, and `n`, where given, between 0 and
    ``len(choices)``; `TypeError` or `ValueError` says otherwise.

        >>> from pylea import DPP, Bool, Enum, prob, probability_mass_function
        >>> class Card(Enum):
        ...     elements = ["A", "K", "Q", "J"]
        >>> @DPP
        ... def distinct() -> Bool:
        ...     deck = [Card.const(e) for e in Card.elements]
        ...     (first, second), _ = pick(deck, 2)
        ...     return first != second
        >>> prob(distinct)
        1.0

    Every card is equally likely to be the one drawn first:

        >>> @DPP
        ... def first_card() -> Card:
        ...     deck = [Card.const(e) for e in Card.elements]
        ...     first, _ = pick(deck)
        ...     return first
        >>> {c: round(p, 4) for c, p in probability_mass_function(first_card).items()}
        {'A': 0.25, 'K': 0.25, 'Q': 0.25, 'J': 0.25}

    Which position was drawn is a fresh uniform `pylea.syntax.enum.Enum` over
    ``range(len(choices))``, read off by `pylea.syntax.enum.Enum.match`: the drawn value is
    the match at that position, and slot `m` of the remainder is `choices[m]` where the
    draw landed above `m` and `choices[m + 1]` otherwise, the standard reindexing after one
    element is deleted from an array.  No event is conditioned on, so every particle a
    sampling query draws is accepted; this is the reason to prefer `pick` over drawing a
    candidate and rejecting it with `pylea.syntax.observe.observe` where it collides with
    what was already dealt, whose acceptance rate falls sharply as more values are dealt.

    Must be called inside an active DPP definition.
    """
    remaining = list(choices)
    if not remaining:
        raise ValueError("pick: choices must be non-empty.")
    output_type = type(remaining[0])
    for i, x in enumerate(remaining):
        if type(x) is not output_type:
            raise TypeError(
                f"pick: every choice must be of the same type; choices[0] is "
                f"{output_type.__name__}, choices[{i}] is {type(x).__name__}.")

    if n is None:
        return _pick_one(remaining, output_type)
    if not 0 <= n <= len(remaining):
        raise ValueError(
            f"pick: cannot draw {n} of {len(remaining)} choices without replacement.")
    drawn: list[_V] = []
    for _ in range(n):
        one, remaining = _pick_one(remaining, output_type)
        drawn.append(one)
    return drawn, remaining


def _pick_one(choices: list[_V], output_type: type) -> tuple[_V, list[_V]]:
    """One uniform draw from the already-validated `choices`, with the rest."""
    n = len(choices)
    if n == 1:
        return choices[0], []

    class _Position(Enum):
        elements = list(range(n))

    position = _Position.uniform()
    chosen = position.match({i: choices[i] for i in range(n)}, data_type=output_type)
    remaining = [
        (position > m).match({True: choices[m], False: choices[m + 1]},
                             data_type=output_type)
        for m in range(n - 1)
    ]
    return chosen, remaining
