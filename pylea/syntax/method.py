"""Method resolution and dispatch shared by the query entry points.

Every query accepts `method` (``None``, ``"exact"``, ``"smc"``, ``"auto"``), the
sampling contract's options (`rel_err`, `floor`, `confidence`, `seed`, `strict`),
`max_time`, and the compile keyword arguments of `pylea.syntax.dpp.DPP.compiled`.
`resolve` checks every one of them and turns them into a `Contract`, an effective method
with its accuracy options:

  * Any other keyword argument raises `TypeError`, naming the closest valid one, under
    every method.  A value of the wrong type raises `TypeError` and one out of range
    `ValueError`: `rel_err` must lie in ``[0, 1)``, `confidence` in ``(0, 1)``, `floor`
    on the query's scale, `max_time` at least 0, `seed` a non-negative integer.  A
    `max_time` of infinity means no deadline.
  * ``rel_err = 0.0`` is a request for exactness and forces ``"exact"``; with
    ``"smc"`` or with a sampling option it raises.
  * With no method named, an exact-engine option (`order_finder`, `kappa`, `simplify`,
    `recompile`, or a `device` naming an accelerator) pins ``"exact"``: the option
    states how the answer must be computed, and the sampling engine cannot honour it.
  * Otherwise ``None`` resolves to `DEFAULT_METHOD`, and the sampling defaults are
    `DEFAULT_REL_ERR` and `DEFAULT_CONFIDENCE`.
  * Under ``"exact"``, whether named or pinned, a sampling option (a positive
    `rel_err`, `floor`, `confidence`, `seed`, or `strict`) raises rather than being
    silently ignored, the two asking for different engines.  `max_time` applies to
    both engines.
  * ``"smc"`` with an exact-engine option raises, since honouring it is impossible;
    under an explicit ``"auto"`` the options are allowed and reach the exact engine
    only, which the portfolio algorithm may or may not run.

`query_target` checks that what a query is given is a DPP it can answer, and
`refuse_wide_output` that its output has at most `MAX_OUTPUT_BITS` bits where the query
builds one value per bit pattern of the output.

`scalar_query` is the one dispatch every scalar entry point runs: the exact value in
this process or, under `max_time`, in the killable subprocess; the portfolio algorithm
under ``"auto"``; the certified sampling run under ``"smc"``.

This module imports nothing from `pylea.sample` at module level: resolving to
``"exact"`` must leave the sampling machinery unimported, so that Python programs never
asking for it never load it.
"""
from __future__ import annotations

import difflib
import math
import numbers
import threading
import time
from dataclasses import dataclass

import numpy as np

from pylea.eval.backend import is_cpu_device
from pylea.syntax.compiled import COMPILE_OPTIONS, check_compile_options
from pylea.syntax.estimate import Estimate

# The method a query uses when none is named and no option pins one.  "auto" runs the
# portfolio algorithm, whose fast path answers every DPP the exact engine finds trivial
# with the exact value; exactness remains available by method="exact" or rel_err=0.0.
DEFAULT_METHOD = "auto"

# The portfolio algorithm's default accuracy target: relative error at the default
# confidence.
DEFAULT_REL_ERR = 0.01
DEFAULT_CONFIDENCE = 0.95

# Keyword arguments of `DPP.compiled` that only the exact engine understands.
_EXACT_ONLY = ("order_finder", "kappa", "simplify", "recompile")

# The widest output, in bits, of which `expectation` and `probability_mass_function`
# give the distribution: one value per bit pattern, ``2**20`` of them at most.  The
# sampling engine's limit on the output bits of a table is the same number.
MAX_OUTPUT_BITS = 20


def query_target(f, what: str, *, inputs: bool = False, outputs: tuple | None = None,
                 output_kind: str = ""):
    """`f`, checked to be a DPP that query `what` can answer.

    Raises `TypeError` for anything that is not a DPP, naming what to do instead, and
    `ValueError` for a DPP with inputs, unless `inputs`, or whose output types are not
    `outputs`, where that is given; `output_kind` describes them in the message.

        >>> from pylea import DPP, Bool, flip
        >>> def coin() -> Bool:
        ...     return flip(0.5)
        >>> query_target(coin, "prob")
        Traceback (most recent call last):
          ...
        TypeError: prob needs a DPP; coin is a function not decorated with @DPP.  Decorate it with @DPP.
    """
    from pylea.syntax.dpp import DPP
    from pylea.semantics.likelihood_table import LikelihoodTable

    if not isinstance(f, DPP):
        if isinstance(f, LikelihoodTable):
            raise TypeError(f"{what} needs a DPP; got a LikelihoodTable.  Call the table "
                            "inside a DPP and query that DPP.")
        if callable(f) and hasattr(f, "__name__"):
            raise TypeError(f"{what} needs a DPP; {f.__name__} is a function not "
                            "decorated with @DPP.  Decorate it with @DPP.")
        raise TypeError(f"{what} needs a DPP; got {f!r} ({type(f).__name__})")
    takes, gives = f.input_types(), f.output_types()
    if (takes and not inputs) or (outputs is not None and gives != outputs):
        raise ValueError(
            f"{what} needs a DPP with no inputs{output_kind}; {f.name()!r} takes "
            f"{_types_in_words(takes)} and returns {_types_in_words(gives)}.")
    return f


def _types_in_words(types: tuple) -> str:
    """`types` by name, for a message."""
    return ", ".join(t.__name__ for t in types) if types else "nothing"


def refuse_wide_output(f, what: str) -> None:
    """Nothing, where the single output of `f` has at most `MAX_OUTPUT_BITS` bits;
    `ValueError` otherwise, before anything of the output's size is built."""
    output = f.output_types()[0]
    bits = output.nbits()
    if bits > MAX_OUTPUT_BITS:
        raise ValueError(
            f"{what} of {f.name()!r} would need a table of 2**{bits} entries, one per bit "
            f"pattern of its {bits}-bit output {output.__name__}; the limit is "
            f"2**{MAX_OUTPUT_BITS}.  Query an event about the output instead, with prob "
            "of a DPP returning a comparison such as 'x > 100'.")


@dataclass(frozen=True)
class Contract:
    """A query's effective method and accuracy options, every value checked.

    Under ``"exact"`` the sampling options hold their defaults: `rel_err` 0.0,
    `confidence` 1.0, no `floor`, no `seed`, `strict` False.  `max_time` is `None` for
    no deadline, otherwise a finite number of seconds at least 0.
    """

    method: str
    rel_err: float
    floor: float | None
    confidence: float
    max_time: float | None
    seed: int | None
    strict: bool


# The keyword arguments every query takes besides the compile options.
QUERY_OPTIONS = ("method", "rel_err", "floor", "confidence", "max_time", "seed", "strict")


def resolve(method: str | None, rel_err: float | None, compile_kwargs: dict, *,
            floor: float | None = None, confidence: float | None = None,
            max_time: float | None = None, seed: int | None = None,
            strict: bool = False, what: str = "query",
            floor_range: tuple[float, float] = (0.0, 1.0)) -> Contract:
    """The `Contract` of a query.

        >>> resolve(None, None, {})
        Contract(method='auto', rel_err=0.01, floor=None, confidence=0.95, max_time=None, seed=None, strict=False)
        >>> resolve(None, None, {"kappa": 20}).method
        'exact'
        >>> resolve("auto", None, {}).rel_err, resolve(None, 0.05, {}).method
        (0.01, 'auto')
        >>> resolve(None, 0.0, {}).method, resolve("exact", None, {}).rel_err
        ('exact', 0.0)
        >>> resolve("exact", None, {}, seed=3)
        Traceback (most recent call last):
          ...
        ValueError: seed configure(s) the sampling engine and cannot be honoured under method='exact'; remove them or use method='smc' or 'auto'

    `what` names the query in messages.  A keyword in `compile_kwargs` that is neither a
    compile option nor a query option raises `TypeError`, with the closest valid name:

        >>> resolve(None, None, {"max_tme": 5}, what="prob")
        Traceback (most recent call last):
          ...
        TypeError: prob() got an unexpected keyword argument 'max_tme'; did you mean 'max_time'?

    A value of the wrong type raises `TypeError`, and one out of range `ValueError`.
    `floor` must lie in `floor_range`.  A `max_time` of infinity, or too large for
    `threading` to wait for, means no deadline:

        >>> resolve(None, None, {}, max_time=float("inf")).max_time is None
        True
    """
    for name in compile_kwargs:
        if name not in COMPILE_OPTIONS:
            close = difflib.get_close_matches(name, COMPILE_OPTIONS + QUERY_OPTIONS, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise TypeError(f"{what}() got an unexpected keyword argument {name!r}{hint}")
    check_compile_options(**compile_kwargs)
    if method not in (None, "exact", "smc", "auto"):
        raise ValueError(f"method must be 'exact', 'smc' or 'auto'; got {method!r}")
    if rel_err is not None:
        rel_err = _real("rel_err", rel_err)
        if not 0.0 <= rel_err < 1.0:
            raise ValueError(f"rel_err must lie in [0, 1); got {rel_err!r}")
    if confidence is not None:
        confidence = _real("confidence", confidence)
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must lie strictly in (0, 1); got {confidence!r}")
    if floor is not None:
        floor = _real("floor", floor)
        low, high = float(floor_range[0]), float(floor_range[1])
        if not low <= floor <= high:
            raise ValueError(f"floor must lie in [{low!r}, {high!r}]; got {floor!r}")
    if max_time is not None:
        max_time = _real("max_time", max_time, finite=False)
        if math.isnan(max_time) or max_time < 0.0:
            raise ValueError(f"max_time must be None or a number of seconds at least 0; "
                             f"got {max_time!r}")
        if max_time >= threading.TIMEOUT_MAX:
            max_time = None
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"seed must be None or a non-negative integer; got {seed!r}")
        if seed < 0:
            raise ValueError(f"seed must be None or a non-negative integer; got {seed!r}")
        seed = int(seed)
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError(f"strict must be True or False; got {strict!r}")
    strict = bool(strict)

    pinning = [name for name in _EXACT_ONLY if name in compile_kwargs]
    if not is_cpu_device(compile_kwargs.get("device")):
        pinning.append(f"device={compile_kwargs.get('device')!r}")
    sampling = [name for name, value in (("floor", floor), ("confidence", confidence),
                                         ("seed", seed))
                if value is not None]
    if strict:
        sampling.append("strict")
    if rel_err == 0.0:
        if method == "smc":
            raise ValueError("rel_err=0 asks for the exact value, which method='smc' "
                             "cannot give; give a positive rel_err or use method='exact'")
        if sampling:
            raise ValueError(
                f"rel_err=0 asks for the exact value, which {', '.join(sampling)} "
                f"cannot configure; remove {', '.join(sampling)} or give a positive "
                "rel_err")
        return Contract("exact", 0.0, None, 1.0, max_time, None, False)
    if rel_err is not None:
        sampling.insert(0, "rel_err")
    if method is None:
        if pinning and sampling:
            raise ValueError(
                f"{', '.join(pinning)} configure(s) the exact engine, but "
                f"{', '.join(sampling)} need(s) the sampling engine; name a method to say "
                "which is meant")
        method = "exact" if pinning else DEFAULT_METHOD
    if method == "exact":
        if sampling:
            raise ValueError(
                f"{', '.join(sampling)} configure(s) the sampling engine and cannot be "
                "honoured under method='exact'; remove them or use method='smc' or "
                "'auto'")
        return Contract("exact", 0.0, None, 1.0, max_time, None, False)
    if method == "smc" and pinning:
        raise ValueError(
            f"{', '.join(pinning)} configure(s) the exact engine and cannot be "
            "honoured under method='smc'; remove them or use method='exact' or 'auto'")
    return Contract(method, DEFAULT_REL_ERR if rel_err is None else rel_err,
                    floor, DEFAULT_CONFIDENCE if confidence is None else confidence,
                    max_time, seed, strict)


def _real(name: str, value, finite: bool = True) -> float:
    """`value` as a float; `TypeError` unless it is a real number that is not a `bool`,
    `ValueError` where `finite` and it is not finite."""
    if type(value) is not float and (isinstance(value, (bool, np.bool_))
                                     or not isinstance(value, numbers.Real)):
        raise TypeError(f"{name} must be a number; got {value!r}")
    try:
        value = float(value)
    except OverflowError:                       # an integer beyond the float range
        value = math.inf if value > 0 else -math.inf
    if finite and not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return value


@dataclass(frozen=True)
class ExactTable:
    """The exact engine's answer to a table query: the evaluator's `array` and the
    `log_scale` it took out, kept apart so that a caller can normalise before folding the
    scale in, the query's `elapsed` seconds, and its `trace`."""

    array: np.ndarray
    log_scale: float
    elapsed: float
    trace: tuple[str, ...] = ()

    @property
    def target_met(self) -> bool:
        """True: an exact answer meets every accuracy target."""
        return True


def table_query(dpp, contract: Contract, compile_kwargs: dict, *,
                normalised_contract: bool, what: str):
    """The answer to one table query over a DPP without inputs under `contract`: an
    `ExactTable`, or the `pylea.sample.run.TableOutcome` of a certified sampling run.

    Under ``"exact"`` the exact engine answers, in this process or, under `max_time`, in
    its subprocess; under ``"smc"`` the sampling engine does; under ``"auto"``
    `pylea.sample.portfolio.decide_table` chooses between them.  `normalised_contract`
    says whether the table's entries are certified relative to its total, as a
    distribution wants.
    """
    if contract.method == "smc":
        from pylea.sample.query import certified_table

        return certified_table(
            dpp, rel_err=contract.rel_err, confidence=contract.confidence,
            floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
            normalised_contract=normalised_contract, what=what)
    if contract.method == "auto":
        from pylea.sample.portfolio import decide_table

        return decide_table(
            dpp, rel_err=contract.rel_err, confidence=contract.confidence,
            floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
            normalised_contract=normalised_contract, what=what,
            compile_kwargs=compile_kwargs)
    started = time.monotonic()
    if contract.max_time is None:
        array, log_scale = dpp.compiled(**compile_kwargs).matrix()
    else:
        from pylea.sample.exact_lane import exact_matrix_within

        array, log_scale = exact_matrix_within(dpp, contract.max_time, compile_kwargs)
    return ExactTable(array, log_scale, time.monotonic() - started)


def refuse_stray_mass(distribution: np.ndarray, patterns, type_name: str,
                      what: str) -> None:
    """Nothing, where no probability in `distribution` lies outside `patterns`, the bit
    patterns that encode a value of the type named `type_name`; `ValueError` otherwise.

    `pylea.syntax.variable.Variable.uniform_bits` puts mass on such patterns for a type
    whose number of values is not a power of two, and a query over the type's values is
    then undefined.

        >>> refuse_stray_mass(np.array([0.5, 0.5, 0.0, 0.0]), [0, 1, 2], "Three", "query")
        >>> refuse_stray_mass(np.array([0.25] * 4), [0, 1, 2], "Three", "query")
        Traceback (most recent call last):
          ...
        ValueError: query is undefined: probability 0.25 lies on bit patterns that encode no value of Three.
    """
    has_value = np.zeros(distribution.shape, dtype=bool)
    has_value[list(patterns)] = True
    # Summed over the patterns that encode nothing rather than taken as one minus the
    # rest, so that the exact zeros the DPP puts there stay exactly zero.
    stray = float(distribution[~has_value].sum())
    if stray > 0.0:
        raise ValueError(
            f"{what} is undefined: probability {stray:.4g} lies on bit patterns that "
            f"encode no value of {type_name}.")


def scalar_query(dpp, contract: Contract, compile_kwargs: dict, *, what: str, values,
                 exact_value, exact_from_matrix, forbidden=None,
                 affine: tuple[float, float] = (1.0, 0.0)) -> Estimate:
    """The answer to one scalar query under `contract`.

    `exact_value` computes the exact answer in this process and `exact_from_matrix`
    maps a subprocess result's ``(array, scale)`` to it; `values`, `forbidden` and
    `affine` describe the sampled quantity as `pylea.sample.query.certified_scalar`
    takes them.  The result is an `pylea.syntax.estimate.Estimate` under every method.
    """
    if contract.method == "exact":
        started = time.monotonic()
        if contract.max_time is None:
            value = exact_value()
        else:
            from pylea.sample.exact_lane import exact_matrix_within

            array, scale = exact_matrix_within(dpp, contract.max_time, compile_kwargs)
            value = exact_from_matrix(array, scale)
        return Estimate.from_exact(value, elapsed=time.monotonic() - started)
    if contract.method == "auto":
        from pylea.sample.portfolio import decide_scalar

        result = decide_scalar(
            dpp, values, rel_err=contract.rel_err, confidence=contract.confidence,
            floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
            what=what, exact_value=exact_value,
            exact_from_matrix=exact_from_matrix, compile_kwargs=compile_kwargs,
            forbidden=forbidden, affine=affine)
    else:
        from pylea.sample.query import certified_scalar

        result = certified_scalar(
            dpp, values, rel_err=contract.rel_err, confidence=contract.confidence,
            floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
            what=what, forbidden=forbidden, affine=affine)
    report_missed_target(result, contract, what, result.target_met,
                         f"{float(result):.4g} in [{result.low:.4g}, {result.high:.4g}]")
    return result


def report_missed_target(result, contract: Contract, what: str, target_met: bool,
                         returned: str) -> None:
    """Nothing where `target_met`; otherwise the
    `pylea.sample.estimate.TargetNotMet` warning at the caller's line, or
    `pylea.sample.estimate.TargetNotMetError` carrying `result` under `contract.strict`.

    Every public query function calls this once, on the object it returns; `returned`
    describes that object in a few words.
    """
    if target_met:
        return
    from pylea.sample.estimate import notify_missed_target

    notify_missed_target(result, target_met=target_met, strict=contract.strict,
                         what=what, contract=contract, returned=returned)
