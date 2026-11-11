"""Subprocess worker evaluating a DPP on the exact engine.

The portfolio algorithm's hedge runs the exact lane in a fresh process so that it can be
killed the moment the sampling lane certifies, or the moment a deadline passes; a worker
thread would instead hold the interpreter lock through the Python-heavy order search and
starve the sampler.  The protocol is one pickled request on stdin, the triple ``(mode,
program, compile_kwargs)`` with `program` a `pylea.ir.program.Program`, and one pickled
response on stdout.  Mode ``"evaluate"`` compiles with the full ladder and evaluates,
responding with `array` and `scale` as `pylea.syntax.compiled.Compiled.matrix` returns
them, plus `trace`; mode ``"predict"`` compiles a greedy unsimplified plan and responds with
`predicted_ms`, so the portfolio algorithm can budget the predictor by killing this
process.  Any failure becomes an `error` string.  Pickling the traced `Program` rather than
re-importing the DPP keeps the worker independent of where the DPP was defined.
"""
from __future__ import annotations

import pickle
import sys


def evaluate_program(program, compile_kwargs: dict):
    """`program`'s root matrix and log scale, through the same `planned`,
    `chosen_backend` and `evaluated_matrix` pipeline `Compiled` evaluates with."""
    from pylea.syntax.compiled import (
        DEFAULT_KAPPA, chosen_backend, evaluated_matrix, planned,
    )

    _program, plans, root, trace = planned(
        program, compile_kwargs.get("order_finder"),
        compile_kwargs.get("kappa", DEFAULT_KAPPA),
        compile_kwargs.get("simplify", True))
    backend = chosen_backend(plans, compile_kwargs.get("device"))
    array, scale, _underflowed = evaluated_matrix(plans, root, backend)
    return array, scale, trace


def probe_compile_kwargs(compile_kwargs: dict) -> dict:
    """The compile options of the greedy exact-cost probe: cotengra's greedy tree, no
    simplification, and the query's `kappa`.

    `pylea.sample.portfolio` compiles through `pylea.syntax.dpp.DPP.compiled` with these
    in this process and this worker through `pylea.syntax.compiled.planned` with the
    same, so the two probes price the exact lane identically; `GreedyOrderFinder`'s
    value equality is what lets the in-process route hit the per-DPP compiled cache.
    """
    from pylea.compile.order import GreedyOrderFinder
    from pylea.syntax.compiled import DEFAULT_KAPPA

    return {"order_finder": GreedyOrderFinder(), "simplify": False,
            "kappa": compile_kwargs.get("kappa", DEFAULT_KAPPA)}


def predict_program(program, compile_kwargs: dict) -> float:
    """The greedy unsimplified plan's predicted evaluation cost, in milliseconds."""
    from pylea.compile.cost import predicted_ms
    from pylea.syntax.compiled import planned

    options = probe_compile_kwargs(compile_kwargs)
    _program, plans, _root, _trace = planned(
        program, options["order_finder"], options["kappa"], options["simplify"])
    return predicted_ms(plans)


def main() -> None:
    mode, program, compile_kwargs = pickle.load(sys.stdin.buffer)
    try:
        if mode == "predict":
            response = {"predicted_ms": predict_program(program, compile_kwargs)}
        else:
            array, scale, trace = evaluate_program(program, compile_kwargs)
            response = {"array": array, "scale": scale, "trace": trace}
    except Exception as exc:                    # noqa: BLE001, reported to the parent
        response = {"error": f"{type(exc).__name__}: {exc}"}
    pickle.dump(response, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
