"""The hold'em example's comparator against an independent evaluator, and its equity.

The showdown circuit for a fixed complete deal is constant, so one particle batch
through the sampler evaluates it deterministically; the reference is
`test.sample.poker_reference`, which shares nothing with the example beyond the rules.
The random-deal sweep covers the categories broadly; the named deals pin the classic
traps: the wheel, the board playing for both, kickers, a six-card flush, and the full
house made of two triples.
"""
from __future__ import annotations

import numpy as np
import pytest

from pylea import DPP, Bool
from pylea.examples.holdem import RANKS, SUITS, showdown
from pylea.sample.sweep import SweepProgram, unfold
from test.sample.poker_reference import card, scores


def circuit_says(hero, villain, board) -> bool:
    @DPP
    def deal() -> Bool:
        return showdown(hero, villain, board)

    result = SweepProgram(unfold(deal.program())).run(64, np.random.default_rng(0))
    return bool(result.counts[1] > 0)


def reference_says(hero, villain, board) -> bool:
    hero_cards = [card(c) for c in hero + board]
    villain_cards = [card(c) for c in villain + board]
    ranks = np.array([[r for r, _ in hero_cards], [r for r, _ in villain_cards]])
    suits = np.array([[s for _, s in hero_cards], [s for _, s in villain_cards]])
    both = scores(ranks, suits)
    return bool(both[0] > both[1])


NAMED_DEALS = (
    (("Kh", "Qh"), ("2s", "2d"), ("Jh", "Th", "Ah", "2h", "2c")),   # royal over quads
    (("Ah", "Kd"), ("As", "Qd"), ("Ac", "7h", "5s", "3d", "2c")),   # the king kicker
    (("2h", "3d"), ("2s", "4d"), ("9c", "Tc", "Jd", "Qh", "Ks")),   # the board plays
    (("Ah", "2d"), ("6s", "6d"), ("3c", "4c", "5d", "Kh", "Ks")),   # the wheel
    (("2h", "9h"), ("Ah", "As"), ("3h", "4h", "5h", "6s", "Kh")),   # six hearts, top five
    (("Ac", "Ad"), ("Ks", "Kd"), ("Ah", "Kc", "Kh", "2c", "3d")),   # trips against trips
    (("7c", "8c"), ("7d", "8d"), ("9c", "Tc", "Jc", "2d", "2h")),   # straight flush edge
    (("As", "Kd"), ("Ad", "Ks"), ("Ac", "Kc", "2h", "3h", "7d")),   # two pair tie
)


def test_the_circuit_matches_the_reference_on_the_named_deals():
    for hero, villain, board in NAMED_DEALS:
        assert circuit_says(hero, villain, board) == \
            reference_says(hero, villain, board), (hero, villain, board)


def random_deal(rng):
    deck = [r + s for r in RANKS for s in SUITS]
    picked = rng.choice(len(deck), size=9, replace=False)
    names = [deck[i] for i in picked]
    return tuple(names[:2]), tuple(names[2:4]), tuple(names[4:])


def test_the_circuit_matches_the_reference_on_random_deals():
    rng = np.random.default_rng(0)
    for _ in range(12):
        hero, villain, board = random_deal(rng)
        assert circuit_says(hero, villain, board) == \
            reference_says(hero, villain, board), (hero, villain, board)


@pytest.mark.slow
def test_the_circuit_matches_the_reference_on_many_random_deals():
    rng = np.random.default_rng(1)
    for _ in range(200):
        hero, villain, board = random_deal(rng)
        assert circuit_says(hero, villain, board) == \
            reference_says(hero, villain, board), (hero, villain, board)
