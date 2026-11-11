"""Sets of non-negative integers held as the bits of a Python integer.

The set ``{0, 1, 3}`` is the integer ``0b1011``, so union, intersection and difference are
``|``, ``&`` and ``& ~``, each taking time in proportion to the largest member.  The passes
over a body use such sets for wires and op positions.
"""
from __future__ import annotations

import numpy as np


def set_bits(mask: int) -> list[int]:
    """The members of the set `mask`, lowest first.

        >>> set_bits(0b1011)
        [0, 1, 3]
        >>> set_bits(0)
        []
    """
    if not mask:
        return []
    raw = np.frombuffer(mask.to_bytes((mask.bit_length() + 7) // 8, "little"), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(raw, bitorder="little")).tolist()
