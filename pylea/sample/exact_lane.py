"""The exact engine in a subprocess that can be stopped at any time.

`ExactLane` evaluates a DPP, or predicts the cost of evaluating it, in a fresh
`pylea.sample._exact_worker` process.  The portfolio algorithm races it against the
sampling engine and stops it when the sampling engine answers first; an exact query with a
`max_time` runs in it so that the deadline holds however long compilation takes.  This
module does not import the sampling engine, so an exact query with a deadline does not
load it.
"""
from __future__ import annotations

import time

from pylea.compile._process_tree import (
    DeadlineExceeded, start_worker,
)


class ExactLane:
    """The exact engine evaluating a DPP in a subprocess that can be stopped.

    The subprocess is a `pylea.compile._process_tree.ProcessTree`, so `kill` also stops
    the processes the worker started, and every caller calls `kill` once it no longer
    needs the lane.  A lane that nothing holds any more, as when an interrupt arrives
    before the caller has it, is collected, and its worker then ends by itself.  A request
    that cannot be pickled becomes an error response rather than an exception, so a racing
    caller handles it like any other worker failure.  `name`, the DPP's name, is what a
    refusal of exact inference calls the DPP; a refusal's response carries the exception
    under `infeasible` (`pylea.sample._exact_worker`).
    """

    def __init__(self, program, compile_kwargs: dict, mode: str = "evaluate",
                 name: str | None = None):
        self.started = time.monotonic()
        self._unsent: dict | None = None
        if mode == "predict":
            # The prediction compiles with its own finder and no simplification; of the
            # caller's options it reads `kappa` alone.
            compile_kwargs = {k: v for k, v in compile_kwargs.items() if k == "kappa"}
        finder = compile_kwargs.get("order_finder")
        if finder is not None and type(finder).__module__ in ("__main__", "__mp_main__"):
            # The subprocess imports the request's classes by module; `__main__` there is
            # the worker itself, so a class defined in the user's script cannot be found.
            self._unsent = {"error": (
                f"order_finder is an instance of {type(finder).__name__}, a class defined "
                "in the script being run, which the exact engine cannot load in the "
                "subprocess it runs in under max_time or in parallel with the sampling "
                "engine; define the class in a module of its own")}
            self._tree = None
            return
        self._tree = start_worker("pylea.sample._exact_worker")
        try:
            self._tree.send((mode, program, compile_kwargs, name))
        except Exception as exc:                # noqa: BLE001, reported as a response
            self._unsent = {
                "error": f"the request could not be pickled for the subprocess: {exc}"}
            self.kill()
        except BaseException:
            self.kill()
            raise

    def done(self) -> bool:
        """Whether the worker has answered or ended; `response` then returns at once."""
        return self._unsent is not None or self._tree is None or self._tree.answered()

    def finished_ok(self) -> bool:
        """Whether the worker answered with a response that is not an error."""
        if not self.done():
            return False
        try:
            return "error" not in self.response()
        except RuntimeError:
            return False

    def response(self, timeout: float | None = None) -> dict:
        """The worker's response, waiting at most `timeout` seconds; `None`, infinity and
        any value beyond what `threading` accepts wait without a limit.

        Raises `pylea.compile._process_tree.DeadlineExceeded`, a `TimeoutError`, when the
        worker has not answered in time, having stopped it, and `RuntimeError` where it
        ended without an answer.
        """
        if self._unsent is not None or self._tree is None:
            return self._unsent or {"error": "the request was not sent"}
        try:
            return self._tree.receive(timeout)
        except DeadlineExceeded:
            raise DeadlineExceeded(
                "the exact engine did not finish within its deadline") from None
        except RuntimeError as exc:
            raise RuntimeError(f"the exact engine's subprocess failed: {exc}") from None

    def kill(self) -> None:
        """Stops the worker, if it is still running, with every process it started, and
        closes the pipe to it."""
        if self._tree is not None:
            try:
                self._tree.kill()
            finally:
                self._tree.close()

    @property
    def elapsed(self) -> float:
        """Seconds since the lane started."""
        return time.monotonic() - self.started


def exact_matrix_within(dpp, max_time: float | None, compile_kwargs: dict):
    """`dpp`'s root matrix and log scale from the exact engine, stopped at the deadline.

    Raises `pylea.compile._process_tree.DeadlineExceeded`, a `TimeoutError`, at the
    deadline, the `pylea.compile.lower.ExactInferenceInfeasible` the exact engine raised
    where it refused the DPP, and `RuntimeError` on any other worker failure.
    """
    started = time.monotonic()
    lane = None
    try:
        lane = ExactLane(dpp.program(), compile_kwargs, name=dpp.name())
        # The deadline counts from the call, so preparing the request is inside it.
        remaining = None if max_time is None else max_time - (time.monotonic() - started)
        response = lane.response(timeout=remaining)
    except DeadlineExceeded:
        raise DeadlineExceeded(
            "the exact engine did not finish within max_time; an exact query never "
            "falls back to sampling, so pass a larger deadline") from None
    finally:
        if lane is not None:
            lane.kill()
    if "infeasible" in response:
        raise response["infeasible"]
    if "error" in response:
        raise RuntimeError(f"the exact engine failed: {response['error']}")
    return response["array"], response["scale"]


def exact_table_within(dpp, max_time: float | None, compile_kwargs: dict):
    """The exact `LikelihoodTable` under a `max_time` deadline."""
    from pylea.semantics.likelihood_table import scaled_table

    array, scale = exact_matrix_within(dpp, max_time, compile_kwargs)
    return scaled_table(dpp.input_types(), dpp.output_types(), array, scale)
