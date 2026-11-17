"""Drawing without replacement.

`pick` draws from a list of variables of one type, each draw uniform over the choices not
drawn before, and returns the drawn values in the order they were drawn together with the
rest, a `Remainder`.  No event is conditioned on, so every sample a sampling query draws
is accepted; this is the reason to prefer `pick` over drawing a value and rejecting it
with `pylea.syntax.observe.observe` where it collides with one already drawn, whose
acceptance probability falls sharply as more values are drawn.

A draw is recorded as arithmetic on positions in the original list: a position uniform
among the choices left, moved past each position drawn before, and the drawn value
selected by one `pylea.syntax.enum.Enum.match` of that position over the choices.  No
assignment grows with the number of draws.  An exact query carries the drawn positions it
depends on jointly, so the exact engine reaches a few draws from many choices; a query
about a whole ordering of more than a few choices is one for the sampling engine.

A `Remainder` holds the choices and the positions drawn from them.  Slot `m` is built
when it is first read, as the position `m` moved past every drawn position, so slots that
are never read add nothing to the DPP.  `pick` applied to a remainder continues drawing
from the same choices.  A remainder belongs to the DPP definition in which it was made,
and reading it anywhere else raises `RuntimeError`.

    >>> from pylea import DPP, Bool, Enum, prob
    >>> class Card(Enum):
    ...     elements = ["A", "K", "Q", "J"]
    >>> @DPP
    ... def ace_after_king() -> Bool:
    ...     deck = [Card.const(e) for e in Card.elements]
    ...     first, rest = pick(deck)
    ...     second, rest = pick(rest)
    ...     return (first == "K") & (second == "A") & (rest[0] == "Q")
    >>> round(prob(ace_after_king), 6)
    0.083333
"""
from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from typing import Iterator, TypeVar, overload

from pylea.syntax.dpp import DPP, Bool, active_definition
from pylea.syntax.enum import Enum
from pylea.syntax.variable import Variable, require_definition

_V = TypeVar('_V', bound=Variable)

# A position is a constant `int` where no draw has made it random, or an `Enum` variable
# over the positions of the choices.
type _Position = int | Enum
# A condition is a Python `bool` where both positions compared are constants.
type _Condition = bool | Bool


class Remainder(Sequence[_V]):
    """The choices a `pick` has not drawn, in their original order.

    A remainder supports `len`, indexing, slicing and iteration; an index or a slice gives
    variables of the type of the choices, and a slice gives them as a list.  Each slot is
    built when it is first read, inside the DPP definition in which `pick` was called,
    and is the same variable on every later read.  Reading a slot outside that definition
    raises `RuntimeError`.  `pick` accepts a remainder in place of a list and continues
    drawing from it.

        >>> from pylea import DPP, Enum, probability_mass_function
        >>> class Card(Enum):
        ...     elements = ["A", "K", "Q"]
        >>> @DPP
        ... def lowest_left() -> Card:
        ...     _, rest = pick([Card.const(e) for e in Card.elements])
        ...     return rest[0]
        >>> {c: round(p, 4) for c, p in probability_mass_function(lowest_left).items()}
        {'A': 0.6667, 'K': 0.3333, 'Q': 0.0}
    """

    def __init__(self, choices: tuple[_V, ...], index_type: type[Enum],
                 ascending: list[_Position],
                 last: tuple[_Position, list[_Condition]] | None,
                 definition: DPP) -> None:
        self._choices = choices
        self._index_type = index_type
        # The drawn positions in ascending order, apart from `last`.
        self._ascending = ascending
        # The position drawn last, with the conditions recorded when it was moved past
        # each entry of `_ascending`; merged into `_ascending` when it is first needed.
        self._last = last
        self._definition = definition
        self._slots: dict[int, _V] = {}

    def __len__(self) -> int:
        return len(self._choices) - len(self._ascending) - (self._last is not None)

    @overload
    def __getitem__(self, key: int) -> _V: ...
    @overload
    def __getitem__(self, key: slice) -> list[_V]: ...
    def __getitem__(self, key: int | slice) -> _V | list[_V]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        i = operator.index(key)
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"pick: the rest has {n} choices; index {key} is out of range")
        i %= n
        self._require_definition()
        if i not in self._slots:
            position, _ = _moved_past(i, self._sorted(), self._index_type)
            self._slots[i] = _value_at(position, self._choices)
        return self._slots[i]

    def __iter__(self) -> Iterator[_V]:
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        return f"<rest of a pick: {len(self)} of {len(self._choices)} choices>"

    def _require_definition(self) -> None:
        """Nothing inside the definition this remainder belongs to; `RuntimeError`
        otherwise."""
        if active_definition() is not self._definition:
            raise RuntimeError(
                f"pick: the rest of a pick was used outside the definition of "
                f"{self._definition.name()}, in which pick was called; use it only there")

    def _sorted(self) -> list[_Position]:
        """Every drawn position, in ascending order."""
        if self._last is not None:
            position, conditions = self._last
            self._ascending = _inserted(position, conditions, self._ascending,
                                        self._index_type)
            self._last = None
        return self._ascending

    def _drawn(self, k: int) -> tuple[list[_V], Remainder[_V]]:
        """`k` draws from this remainder, in drawing order, with what is left after them."""
        if k == 0:
            return [], self
        index_type = self._index_type
        n = len(self._choices)
        ascending = self._sorted()
        last: tuple[_Position, list[_Condition]] | None = None
        drawn: list[_V] = []
        for j in range(k):
            if last is not None:
                ascending = _inserted(*last, ascending, index_type)
            left = len(self) - j
            # A position uniform over the first `left` positions, which counts among the
            # choices not drawn yet.
            start: _Position = (
                0 if left == 1 else index_type.random([1.0] * left + [0.0] * (n - left)))
            position, conditions = _moved_past(start, ascending, index_type)
            drawn.append(_value_at(position, self._choices))
            last = (position, conditions)
        return drawn, Remainder(self._choices, index_type, ascending, last,
                                self._definition)


def _moved_past(position: _Position, ascending: list[_Position],
                index_type: type[Enum]) -> tuple[_Position, list[_Condition]]:
    """`position`, counted among the positions not in `ascending`, as a position among all
    of them, with the condition recorded at each entry of `ascending`.

    Moving past the entries in ascending order, each adds one where the position is at or
    above it.  The condition at an entry is therefore true exactly where the result lies
    above that entry, and the conditions are true for a first part of `ascending` and
    false for the rest.
    """
    conditions: list[_Condition] = []
    for s in ascending:
        condition: _Condition
        if isinstance(position, int) and isinstance(s, int):
            condition = position >= s
            position += condition
        elif isinstance(position, int):
            assert isinstance(s, Enum)
            above = s <= position
            position = above.match({True: position + 1, False: position},
                                   data_type=index_type)
            condition = above
        else:
            condition = position >= s
            position = _plus(position, condition, index_type)
        conditions.append(condition)
    return position, conditions


def _plus(position: Enum, bit: Bool, index_type: type[Enum]) -> Enum:
    """`position` plus `bit`, as a ripple-carry circuit from the lowest bit up.

    The carry out of the top bit is dropped; no position that a draw produces reaches it.
    """
    bits = list(position.bits())
    carry = bit
    for i in reversed(range(1, len(bits))):
        bits[i], carry = bits[i] ^ carry, bits[i] & carry
    bits[0] = bits[0] ^ carry
    return index_type(*bits)


def _inserted(position: _Position, conditions: list[_Condition],
              ascending: list[_Position], index_type: type[Enum]) -> list[_Position]:
    """`ascending` with `position` inserted in order, read off `conditions`, which
    `_moved_past` recorded for `position` against each entry of `ascending`.

    Entry `i` of the result is `ascending[i]` where the position lies above it, the
    position itself where it lies above the entry before only, and ``ascending[i - 1]``
    otherwise.
    """
    def select(condition: _Condition, then: _Position, otherwise: _Position) -> _Position:
        if isinstance(condition, bool):
            return then if condition else otherwise
        return condition.match({True: then, False: otherwise}, data_type=index_type)

    r = len(ascending)
    if r == 0:
        return [position]
    result = [select(conditions[0], ascending[0], position)]
    for i in range(1, r):
        result.append(select(conditions[i], ascending[i],
                             select(conditions[i - 1], position, ascending[i - 1])))
    result.append(select(conditions[r - 1], position, ascending[r - 1]))
    return result


def _value_at(position: _Position, choices: tuple[_V, ...]) -> _V:
    """The choice at `position`."""
    if isinstance(position, int):
        return choices[position]
    return position.match(dict(enumerate(choices)),  # type: ignore[return-value]
                          data_type=type(choices[0]))


@overload
def pick(choices: Sequence[_V]) -> tuple[_V, Remainder[_V]]: ...
@overload
def pick(choices: Sequence[_V], n: int) -> tuple[list[_V], Remainder[_V]]: ...
def pick(choices: Sequence[_V], n: int | None = None):
    """`n` of `choices`, drawn uniformly at random without replacement, with the rest.

    With `n` left out, one draw: the result is the drawn value and the `Remainder` of
    those not drawn, in their original order, so calling `pick` again on the rest deals a
    second value without replacement from the first.  With `n` given, the first value is
    instead the list of the `n` draws, in the order they were drawn; drawing every element
    gives a uniformly random ordering of `choices`.

    `choices` is a non-empty list of variables of one `pylea.syntax.variable.Variable`
    type, or a `Remainder`; a fixed value is given as a constant through
    `pylea.syntax.variable.Variable.const`.  `n`, where given, is an integer between 0 and
    ``len(choices)``.  `TypeError` or `ValueError` says otherwise.

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

    Must be called inside an active DPP definition.
    """
    require_definition("pick")
    if isinstance(choices, Remainder):
        choices._require_definition()
        start: Remainder = choices
    else:
        start = _fresh(choices)
    if n is None:
        k = 1
    else:
        try:
            k = operator.index(n)
        except TypeError:
            raise TypeError(f"pick: n must be an integer; got {n!r}") from None
    if not 0 <= k <= len(start):
        raise ValueError(
            f"pick: cannot draw {k} of {len(start)} choices without replacement.")
    drawn, rest = start._drawn(k)
    if n is None:
        return drawn[0], rest
    return drawn, rest


def _fresh(choices: Iterable[_V]) -> Remainder[_V]:
    """`choices`, checked, as a remainder from which nothing is drawn yet."""
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Iterable):
        raise TypeError(f"pick: choices must be a list of variables; got a "
                        f"{type(choices).__name__}")
    items = tuple(choices)
    if not items:
        raise ValueError("pick: choices must be non-empty.")
    for i, x in enumerate(items):
        if not isinstance(x, Variable):
            raise TypeError(
                f"pick: choices must be variables; got {type(x).__name__} at position "
                f"{i}; write a fixed value v of a type T as T.const(v)")
    output_type = type(items[0])
    for i, x in enumerate(items):
        if type(x) is not output_type:
            raise TypeError(
                f"pick: every choice must be of the same type; choices[0] is "
                f"{output_type.__name__}, choices[{i}] is {type(x).__name__}.")

    class _Index(Enum):
        elements = tuple(range(len(items)))

    definition = active_definition()
    assert definition is not None
    return Remainder(items, _Index, [], None, definition)
