"""`BitEncoded.decoded`, against the `from_bits` it has to agree with."""
import pytest

from pylea import Bool, Tuple
from pylea.syntax.encoding import BitEncoded
from pylea.examples.simple import Coin, Color


class CoinAndColor(Tuple):
    types = (Coin, Color)


def _plain(value: object) -> object:
    """`value` with any encoded component replaced by the value it holds.

    A `pylea.syntax.tuple.Tuple` decodes to a tuple of encoded components, and those
    compare by identity, so two decodings of one pattern are equal only once unwrapped.
    """
    if isinstance(value, BitEncoded):
        return _plain(value.value)
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    return value


def _from_bits(enc, pattern: int) -> object:
    """The value `enc` gives `pattern`, or `None` where it gives none."""
    n = enc.nbits()
    bits = tuple(bool((pattern >> (n - 1 - i)) & 1) for i in range(n))
    try:
        return _plain(enc.from_bits(*bits).value)
    except ValueError:
        return None


@pytest.mark.parametrize("cls", [Bool, Coin, Color, CoinAndColor])
def test_decoded_agrees_with_from_bits_on_every_pattern(cls):
    enc = cls.encoding()
    decoded = enc.decoded()
    for pattern in range(1 << enc.nbits()):
        expected = _from_bits(enc, pattern)
        if expected is None:
            assert pattern not in decoded
        else:
            assert _plain(decoded[pattern]) == expected


def test_a_pattern_encoding_no_element_is_left_out():
    assert Color.encoding().decoded() == {0: "R", 1: "G", 2: "B"}


def test_a_tuple_leaves_out_the_patterns_its_components_leave_out():
    # The colour is the low two bits, so the patterns whose colour part is 3 encode nothing.
    assert set(CoinAndColor.encoding().decoded()) == {0, 1, 2, 4, 5, 6}


def test_repeated_calls_give_the_same_map():
    first, second = CoinAndColor.encoding().decoded(), CoinAndColor.encoding().decoded()
    assert {k: _plain(v) for k, v in first.items()} == {k: _plain(v) for k, v in second.items()}
