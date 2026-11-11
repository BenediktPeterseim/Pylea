"""An independent seven-card poker evaluator, for checking the hold'em example.

`scores` maps arrays of seven-card hands to comparable integers: the category in the top
digit and up to five base-13 tie-break digits below it, so `scores(a) > scores(b)` says
hand `a` strictly beats hand `b`.  Everything is vectorised over deals, which is what
lets the equity of a scenario be enumerated exactly over a million completions in
seconds.  The implementation shares nothing with `pylea.examples.holdem` beyond the
rules of poker; agreement between the two is the test.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "chds"


def card(name: str) -> tuple[int, int]:
    """The written card as ``(rank index, suit index)``."""
    return RANKS.index(name[0]), SUITS.index(name[1])


def _top_digits(mask: np.ndarray, count: int) -> np.ndarray:
    """The `count` highest set ranks per row, packed as base-13 digits, high first."""
    n = mask.shape[0]
    digits = np.zeros(n, dtype=np.int64)
    taken = np.zeros(n, dtype=np.int64)
    for r in range(12, -1, -1):
        use = mask[:, r] & (taken < count)
        digits = np.where(use, digits * 13 + r, digits)
        taken += use
    # Pad short rows so every row carries exactly `count` digits.
    for _ in range(count):
        pad = taken < count
        digits = np.where(pad, digits * 13, digits)
        taken += pad
    return digits


def _straight_top(present: np.ndarray) -> np.ndarray:
    """The best straight's window index per row, 0 the wheel and 9 the broadway; -1 none."""
    n = present.shape[0]
    best = np.full(n, -1, dtype=np.int64)
    windows = [[12, 0, 1, 2, 3]] + [[low + k for k in range(5)] for low in range(9)]
    for w, window in enumerate(windows):
        best = np.where(present[:, window].all(axis=1), w, best)
    return best


def scores(ranks: np.ndarray, suits: np.ndarray) -> np.ndarray:
    """The comparable score of each seven-card hand; rows are deals, columns cards."""
    n = ranks.shape[0]
    counts = np.zeros((n, 13), dtype=np.int64)
    suit_counts = np.zeros((n, 4), dtype=np.int64)
    for c in range(ranks.shape[1]):
        counts[np.arange(n), ranks[:, c]] += 1
        suit_counts[np.arange(n), suits[:, c]] += 1
    present = counts >= 1
    pairs = counts >= 2
    trips = counts >= 3
    quads = counts >= 4

    flush_suit = np.argmax(suit_counts >= 5, axis=1)
    has_flush = np.asarray((suit_counts >= 5).any(axis=1))
    flush_ranks = np.zeros((n, 13), dtype=bool)
    for c in range(ranks.shape[1]):
        hit = np.asarray(suits[:, c] == flush_suit)
        np.logical_or.at(flush_ranks, (np.arange(n), np.asarray(ranks[:, c])), hit)
    flush_ranks &= has_flush[:, None]

    straight = _straight_top(present)
    straight_flush = _straight_top(flush_ranks)

    top_trips = _top_digits(trips, 1)
    trips_mask = np.zeros((n, 13), dtype=bool)
    trips_mask[np.arange(n), top_trips % 13] = trips.any(axis=1)
    pair_below_trips = pairs & ~trips_mask
    full = trips.any(axis=1) & pair_below_trips.any(axis=1)
    two_pair = pairs.sum(axis=1) >= 2

    top_pairs2 = _top_digits(pairs, 2)
    pair_mask2 = np.zeros((n, 13), dtype=bool)
    pair_mask2[np.arange(n), top_pairs2 // 13] = True
    pair_mask2[np.arange(n), top_pairs2 % 13] |= two_pair

    quad_rank = _top_digits(quads, 1)
    quad_mask = np.zeros((n, 13), dtype=bool)
    quad_mask[np.arange(n), quad_rank % 13] = quads.any(axis=1)

    pair_top = _top_digits(pairs, 1)
    pair_mask1 = np.zeros((n, 13), dtype=bool)
    pair_mask1[np.arange(n), pair_top % 13] = pairs.any(axis=1)

    base = 13**5
    tie = np.select(
        [straight_flush >= 0,
         quads.any(axis=1),
         full,
         has_flush,
         straight >= 0,
         trips.any(axis=1),
         two_pair,
         pairs.any(axis=1)],
        [straight_flush * base // 13,
         quad_rank * 13**4 + _top_digits(present & ~quad_mask, 1) * 13**3,
         top_trips * 13**4 + _top_digits(pair_below_trips, 1) * 13**3,
         _top_digits(flush_ranks, 5),
         straight * base // 13,
         top_trips * 13**4 + _top_digits(present & ~trips_mask, 2) * 13**2,
         top_pairs2 * 13**3 + _top_digits(present & ~pair_mask2, 1) * 13**2,
         pair_top * 13**4 + _top_digits(present & ~pair_mask1, 3) * 13],
        default=_top_digits(present, 5))
    category = np.select(
        [straight_flush >= 0, quads.any(axis=1), full, has_flush, straight >= 0,
         trips.any(axis=1), two_pair, pairs.any(axis=1)],
        [8, 7, 6, 5, 4, 3, 2, 1], default=0)
    return category * 13**6 + tie


def equity(hero: tuple[str, str], board: tuple[str, ...]) -> tuple[float, int]:
    """Hero's exact strict-win probability given `board`, and the completion count.

    Enumerates the opponent's two cards and the remaining board cards from the unseen
    deck, uniformly without replacement.
    """
    seen = [card(c) for c in hero + board]
    deck = [(r, s) for r in range(13) for s in range(4) if (r, s) not in seen]
    missing = 5 - len(board)
    board_cards = [card(c) for c in board]

    wins = 0
    total = 0
    rows_r: list[list[int]] = []
    rows_s: list[list[int]] = []

    def drain() -> tuple[int, int]:
        ranks = np.array(rows_r)
        suits_arr = np.array(rows_s)
        rows_r.clear()
        rows_s.clear()
        hero_scores = scores(ranks[:, :7], suits_arr[:, :7])
        villain_scores = scores(ranks[:, 7:], suits_arr[:, 7:])
        return int((hero_scores > villain_scores).sum()), len(ranks)

    for villain in combinations(range(len(deck)), 2):
        rest = [i for i in range(len(deck)) if i not in villain]
        for extra in combinations(rest, missing):
            common = [deck[i] for i in extra]
            hero_hand = seen + common
            villain_hand = [deck[i] for i in villain] + board_cards + common
            rows_r.append([r for r, _ in hero_hand] + [r for r, _ in villain_hand])
            rows_s.append([s for _, s in hero_hand] + [s for _, s in villain_hand])
        if len(rows_r) >= 100_000:
            won, seen_rows = drain()
            wins += won
            total += seen_rows
    if rows_r:
        won, seen_rows = drain()
        wins += won
        total += seen_rows
    return wins / total, total
