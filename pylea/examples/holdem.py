"""Texas hold'em: the probability that a fixed hand wins at showdown.

Each player owns two cards and shares five more with the opponent; whoever makes the better
five-card hand out of their seven wins.  The rules of the game are stated as comments, next
to the constructions that model them.  `showdown` decides a fixed, fully known deal, and the
tests check it card for card against an independent evaluator.  `hero_wins` takes the cards
already visible and deals every other one at random, without replacement.  Here the hero
holds the king and queen of hearts, and the first three shared cards are the jack of
hearts, the ten of hearts and the two of clubs:

    >>> from pylea import prob
    >>> equity = prob(hero_wins(("Kh", "Qh"), board=("Jh", "Th", "2c")),
    ...               rel_err=0.05, seed=7)
    >>> 0.7 < equity.low < equity.high < 0.82
    True
"""
from __future__ import annotations

from typing import Sequence

from pylea import DPP, Bool, Enum, Tuple, at_least, pick

# A card has a rank and a suit.  Ranks are listed lowest first, 2 up to 10, then jack,
# queen, king, ace, so that `<` on two `Rank` variables orders cards as the game does.
# The suits, clubs, hearts, diamonds, spades, are not ordered.  A card is written as its
# rank then its suit, "Kh" for the king of hearts.
RANKS = "23456789TJQKA"
SUITS = "chds"


class Rank(Enum):
    elements = list(RANKS)


class Suit(Enum):
    elements = list(SUITS)


class Card(Tuple):
    types = (Rank, Suit)


class Category(Enum):
    """The nine kinds of hand, weakest first, so that `>` says which category wins."""
    elements = [
        "high card",        # none of the below; only the individual cards count
        "one pair",         # two cards of the same rank
        "two pair",         # two cards of one rank and two of another
        "three of a kind",  # three cards of the same rank
        "straight",         # five ranks in a row, such as 6, 7, 8, 9, 10
        "flush",            # five cards of the same suit
        "full house",       # three cards of one rank and two of another
        "four of a kind",   # four cards of the same rank
        "straight flush",   # five ranks in a row, all of the same suit
    ]


def _card(name: str) -> Card:
    """The card written like ``"Kh"`` as a constant `Card`."""
    return Card.of(Rank.const(name[0]), Suit.const(name[1]))


def _highest(is_there: list[Bool]) -> Rank:
    """The highest rank present, given one flag per rank, lowest rank first.

    Where no flag is true the result is the lowest rank; callers compare it only under a
    category that guarantees at least one true flag.
    """
    result = Rank.const(RANKS[0])
    for rank, there in zip(RANKS, is_there):
        result = there.match({True: rank, False: result}, data_type=Rank)
    return result


def _best(is_there: list[Bool], count: int) -> list[Rank]:
    """The `count` highest ranks present, best first: take the highest, strike out its
    rank, repeat."""
    best: list[Rank] = []
    for _ in range(count):
        top = _highest(is_there)
        best.append(top)
        is_there = [there & (top != rank) for rank, there in zip(RANKS, is_there)]
    return best


def _higher(a: list[Rank], b: list[Rank]) -> Bool:
    """Whether the ranks `a` beat the ranks `b`, compared pairwise from the front.

    The first pair that differs decides; equal throughout is not a win.
    """
    beats = Bool.const(False)
    tied_so_far = Bool.const(True)
    for x, y in zip(a, b):
        beats = beats | (tied_so_far & (x > y))
        tied_so_far = tied_so_far & (x == y)
    return beats


# The ten runs of five consecutive ranks a straight can be, weakest first.  The ace may
# also play below the 2; that straight is topped by the 5.
_RUNS = [["A", "2", "3", "4", "5"]] + [[RANKS[i + k] for k in range(5)] for i in range(9)]


def _best_run(is_there: list[Bool]) -> tuple[Bool, Rank]:
    """Whether five ranks in a row are all present, and the top rank of the best such run."""
    found = Bool.const(False)
    top = Rank.const(RANKS[0])
    for run in _RUNS:
        complete = Bool.all(is_there[RANKS.index(r)] for r in run)
        found = found | complete
        top = complete.match({True: run[-1], False: top}, data_type=Rank)
    return found, top


class _Hand:
    """One player's seven cards, as what showdown needs to know about them.

    `category` is the best `Category` the cards contain, and `deciding[c]` the ranks that
    settle a tie against another hand of category `c`, most important first.
    """

    def __init__(self, cards: list) -> None:
        # How often each rank occurs among the seven cards, as one flag per rank per
        # threshold: `paired[i]` says at least two of the cards have rank `i`.
        def occurs(times: int) -> list[Bool]:
            return [at_least(times, *(r == rank for r, _s in cards)) for rank in RANKS]

        present, paired, tripled, quadrupled = (occurs(k) for k in (1, 2, 3, 4))

        # A flush needs five cards of one suit.  Seven cards can manage that in at most
        # one suit, so the per-suit answers can simply be merged.
        def of_suit(s: str) -> list[Bool]:
            return [Bool.any((r == rank) & (suit == s) for r, suit in cards)
                    for rank in RANKS]

        suit_cards = {s: of_suit(s) for s in SUITS}
        has_five = {s: at_least(5, *(suit == s for _r, suit in cards)) for s in SUITS}
        flush = Bool.any(has_five.values())
        flush_cards = [Bool.any(has_five[s] & suit_cards[s][i] for s in SUITS)
                       for i in range(len(RANKS))]

        straight, straight_high = _best_run(present)
        straight_flush = Bool.const(False)
        straight_flush_high = Rank.const(RANKS[0])
        for s in SUITS:
            run_here, run_top = _best_run(suit_cards[s])
            straight_flush = straight_flush | run_here
            straight_flush_high = run_here.match({True: run_top,
                                                  False: straight_flush_high})

        # The ranks the categories are made of.  Four of a kind also counts as a pair and
        # as three of a kind, so `pair` is simply the highest rank occurring at least
        # twice, whatever else that rank occurs as.
        pair = _highest(paired)
        second_pair = _highest([p & (pair != rank) for rank, p in zip(RANKS, paired)])
        three = _highest(tripled)
        full_pair = _highest([p & (three != rank) for rank, p in zip(RANKS, paired)])
        four = _highest(quadrupled)
        two_pair = at_least(2, *paired)

        # The hand's category: start from "high card" and overwrite with each stronger
        # description that applies, so the strongest one that applies wins.
        applies = [
            ("one pair", Bool.any(paired)),
            ("two pair", two_pair),
            ("three of a kind", Bool.any(tripled)),
            ("straight", straight),
            ("flush", flush),
            ("full house", Bool.any(tripled) & two_pair),
            ("four of a kind", Bool.any(quadrupled)),
            ("straight flush", straight_flush),
        ]
        category = Category.const("high card")
        for name, holds in applies:
            category = holds.match({True: name, False: category}, data_type=Category)
        self.category = category

        # The highest cards whose ranks were not already counted for the category; they
        # settle what the category's own ranks leave tied.
        def rest(taken: list[Rank], count: int) -> list[Rank]:
            left = [there & Bool.all(t != rank for t in taken)
                    for rank, there in zip(RANKS, present)]
            return _best(left, count)

        # Per category, the ranks that settle a tie, most important first: first the
        # ranks the category is made of, then the highest remaining cards, five cards
        # counting in total.  This table is the tie-breaking rule of the game.
        self.deciding: dict[str, list[Rank]] = {
            "high card":       _best(present, 5),
            "one pair":        [pair] + rest([pair], 3),
            "two pair":        [pair, second_pair] + rest([pair, second_pair], 1),
            "three of a kind": [three] + rest([three], 2),
            "straight":        [straight_high],
            "flush":           _best(flush_cards, 5),
            "full house":      [three, full_pair],
            "four of a kind":  [four] + rest([four], 1),
            "straight flush":  [straight_flush_high],
        }


def _beats(a: _Hand, b: _Hand) -> Bool:
    """Whether hand `a` strictly beats hand `b`; a tie is not a win.

    The better category wins outright; equal categories are settled by that category's
    deciding ranks, compared one by one.
    """
    tie_broken = Bool.any(
        (a.category == name) & (b.category == name)
        & _higher(a.deciding[name], b.deciding[name])
        for name in Category)
    return (a.category > b.category) | tie_broken


def showdown(hero_cards, villain_cards, board) -> Bool:
    """The event that `hero_cards` strictly beats `villain_cards` on `board`.

    Cards are written names like ``"Kh"``.  Must be called inside a ``@DPP`` definition;
    the deal is fixed, so the traced circuit is constant and the tests enumerate it.
    """
    hero = _Hand([_card(c).components() for c in hero_cards + board])
    villain = _Hand([_card(c).components() for c in villain_cards + board])
    return _beats(hero, villain)


DECK = [rank + suit for rank in RANKS for suit in SUITS]


def hero_wins(hero: Sequence[str], board: Sequence[str]) -> DPP:
    """The event that `hero` beats one opponent at showdown, everything unknown dealt at
    random.

    `hero` is the hero's two cards and `board` the shared cards revealed so far, up to
    five, all written as names like ``"Kh"``.  The opponent's two cards and the shared
    cards still to come are dealt uniformly at random, without replacement, from the rest
    of the deck.  Raises `ValueError` for a misspelt card, a repeated one, or the wrong
    count.
    """
    known = tuple(hero) + tuple(board)
    if len(hero) != 2 or len(board) > 5:
        raise ValueError(f"hero holds two cards and the board shows at most five; "
                         f"got {len(hero)} and {len(board)}.")
    for name in known:
        if name not in DECK:
            raise ValueError(f"{name!r} is not a card; write rank then suit, as in 'Kh'.")
    if len(set(known)) != len(known):
        raise ValueError(f"a card cannot appear twice; got {known}.")

    @DPP
    def wins() -> Bool:
        # The hero's two cards and the board are visible; each other card of the deck is
        # equally likely to be anywhere.
        unseen = [_card(name) for name in DECK if name not in known]

        # Deal the missing cards, each drawn from the cards not yet dealt, so no card
        # appears twice: two for the opponent, then whatever the board still lacks.
        cards, _ = pick(unseen, 2 + 5 - len(board))
        dealt = [card.components() for card in cards]
        opponent, still_to_come = dealt[:2], dealt[2:]

        # Each player's hand is their own two cards plus the five shared ones.
        shared = [_card(c).components() for c in board] + still_to_come
        return _beats(_Hand([_card(c).components() for c in hero] + shared),
                      _Hand(opponent + shared))

    return wins
