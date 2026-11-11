"""Subprocess worker running one bounded `pylea.compile.order.AdaptiveOrderFinder` search.

An `AdaptiveOrderFinder` with a `max_time` runs its search here, so that the search can be
stopped at that bound: cotengra tests its own budget only between trials, and on a wide
enough network a single trial runs far beyond it.  The protocol is one pickled request on
stdin, ``((optimize, escalate_cost, max_time), (indices, output, size, kappa))``, and one
pickled response on stdout, ``{"order": order}`` or ``{"error": message}``.  The finder's
settings travel rather than the finder itself, so a subclass defined anywhere still runs.
"""
from __future__ import annotations

import pickle
import sys


def main() -> None:
    from pylea.compile.order import AdaptiveOrderFinder

    (optimize, escalate_cost, max_time), (indices, output, size, kappa) = pickle.load(
        sys.stdin.buffer)
    finder = AdaptiveOrderFinder(optimize, escalate_cost, max_time)
    try:
        response = {"order": finder.find_here(indices, output, size, kappa=kappa)}
    except Exception as exc:                    # noqa: BLE001, reported to the parent
        response = {"error": f"{type(exc).__name__}: {exc}"}
    pickle.dump(response, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
