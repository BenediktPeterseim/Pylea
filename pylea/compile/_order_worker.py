"""Subprocess worker running one bounded contraction tree search, for all the networks of
a DPP together.

A `pylea.compile.order.BoundedOrderFinder`, and an `AdaptiveOrderFinder` with a
`max_time`, run their search here, so that the search can be stopped at its bound:
cotengra tests its own budget only between trials, and on a wide enough network a single
trial runs far beyond it.  The protocol is one pickled request on stdin,
``(kind, settings, networks, kappa)`` with `networks` a list of
``(indices, output, size)``, and one pickled response on stdout,
``{"orders": orders, "starved": starved}`` or ``{"error": message}``; the work is
`pylea.compile.order.search_all`.  The finder's settings travel, and the worker builds
the finder of the kind named in the request from them, so a subclass's own `find` does
not run in the worker.  The warnings of cotengra that `pylea.compile.order` hides are
hidden here and in the processes its search starts.

The worker is served by `pylea.compile._process_tree.serve` and run by
`pylea.compile._process_tree.ProcessTree`.  It starts no process before it has read the
request, which is what lets the tree stop every process the search starts, and once it has
read it, the end of its standard input stops it, since the parent keeps that open while it
waits.
"""
from __future__ import annotations


def _handle(request) -> dict:
    from pylea.compile.order import hide_cotengra_warnings, search_all

    kind, settings, networks, kappa = request
    hide_cotengra_warnings()
    orders, starved = search_all(kind, settings, networks, kappa)
    return {"orders": orders, "starved": starved}


def main() -> None:
    from pylea.compile._process_tree import serve

    serve(_handle)


if __name__ == "__main__":
    main()
