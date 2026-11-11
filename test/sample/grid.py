"""The grid family of models, on which the sampling tests pin the portfolio's decisions."""
from __future__ import annotations

from pylea.syntax.dpp import DPP, Bool
from pylea.syntax.flip import flip


def grid_model(w: int, p_cell: float = 0.98, theta: float | None = None) -> DPP:
    """The grid family: cells, adjacent-pair disjunctions, optionally observed per row.

    The event is the conjunction over all horizontally and vertically adjacent pairs
    that at least one cell of the pair is up.  With `theta`, each row's conjunct is
    observed in disjunction with an independent coin flip of bias `theta`, the
    staged-evidence variant the scheduler is measured on.
    """
    from pylea.syntax.observe import observe

    @DPP
    def grid() -> Bool:
        cells = [[flip(p_cell) for _ in range(w)] for _ in range(w)]
        ok = Bool.const(True)
        for i in range(w):
            rowok = Bool.const(True)
            for j in range(w):
                if j + 1 < w:
                    rowok = rowok & (cells[i][j] | cells[i][j + 1])
                if i + 1 < w:
                    rowok = rowok & (cells[i][j] | cells[i + 1][j])
            if theta is not None:
                observe(rowok | flip(theta))
            ok = ok & rowok
        return ok

    return grid
