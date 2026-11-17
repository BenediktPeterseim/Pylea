"""Subprocess worker evaluating a DPP on the exact engine.

The portfolio algorithm's hedge runs the exact lane in a fresh process so that it can be
killed the moment the sampling lane certifies, or the moment a deadline passes; a worker
thread would instead hold the interpreter lock through the Python-heavy order search and
starve the sampler.  The protocol is one pickled request on stdin, ``(mode, program,
compile_kwargs, name)`` with `program` a `pylea.ir.program.Program` and `name` the DPP's
name, for messages, and one pickled response on stdout.  Mode ``"evaluate"`` compiles with
the full ladder and evaluates, responding with `array` and `scale` as
`pylea.syntax.compiled.Compiled.matrix` returns them, plus `trace`; mode ``"predict"``
compiles a greedy unsimplified plan and responds with `predicted_ms`, so the portfolio
algorithm can budget the predictor by killing this process.  Any failure becomes an `error`
string; a refusal, `pylea.compile.lower.ExactInferenceInfeasible`, also carries the
exception itself under `infeasible`, so that the caller raises it as it is and the
portfolio algorithm can tell a refusal from a failure.  Pickling the traced `Program`
rather than re-importing the DPP keeps the worker independent of where the DPP was
defined.

The worker is served by `pylea.compile._process_tree.serve`, which reads the request
inside its error handling, so a request that cannot be read, such as one naming a class
the worker cannot import, is answered with an `error` string too.  It is run by
`pylea.compile._process_tree.ProcessTree`, so that killing it also kills the processes it
started: the order search of the planner's last stage, and the processes that search
starts.  It starts no process before it has read the request, and once it has read it, the
end of its standard input stops it, since the parent keeps that open while it waits.
"""
from __future__ import annotations


def evaluate_program(program, compile_kwargs: dict, name: str | None = None):
    """`program`'s root matrix and log scale, through the same `planned`,
    `chosen_backend` and `evaluated_matrix` pipeline `Compiled` evaluates with."""
    from pylea.syntax.compiled import (
        DEFAULT_KAPPA, chosen_backend, evaluated_matrix, planned,
    )

    _program, plans, root, trace = planned(
        program, compile_kwargs.get("order_finder"),
        compile_kwargs.get("kappa", DEFAULT_KAPPA),
        compile_kwargs.get("simplify", True), name)
    backend = chosen_backend(plans, compile_kwargs.get("device"))
    array, scale, _path = evaluated_matrix(plans, root, backend)
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


def predict_program(program, compile_kwargs: dict, name: str | None = None) -> float:
    """The greedy unsimplified plan's predicted evaluation cost, in milliseconds.

    Raises `pylea.compile.lower.ExactInferenceTooLarge` where the DPP exceeds the size
    limit of the query's `kappa`.
    """
    from pylea.compile.cost import predicted_ms
    from pylea.syntax.compiled import planned

    options = probe_compile_kwargs(compile_kwargs)
    _program, plans, _root, _trace = planned(
        program, options["order_finder"], options["kappa"], options["simplify"], name)
    return predicted_ms(plans)


def _handle(request) -> dict:
    from pylea.compile._process_tree import error_response
    from pylea.compile.lower import ExactInferenceInfeasible

    mode, program, compile_kwargs, name = request
    try:
        if mode == "predict":
            return {"predicted_ms": predict_program(program, compile_kwargs, name)}
        array, scale, trace = evaluate_program(program, compile_kwargs, name)
    except ExactInferenceInfeasible as exc:
        return {**error_response(exc), "infeasible": exc}
    return {"array": array, "scale": scale, "trace": trace}


def main() -> None:
    from pylea.compile._process_tree import serve

    serve(_handle)


if __name__ == "__main__":
    main()
