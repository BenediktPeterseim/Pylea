"""Method resolution and dispatch shared by the query entry points.

Every query accepts `method` (``None``, ``"exact"``, ``"smc"``, ``"auto"``), the
sampling contract's options (`rel_err`, `floor`, `confidence`, `seed`, `strict`),
`max_time`, and the compile keyword arguments of `pylea.syntax.dpp.DPP.compiled`.
`resolve` turns them into a `Contract`, an effective method with its accuracy options:

  * ``rel_err = 0.0`` is a request for exactness and forces ``"exact"``.
  * With no method named, an exact-engine option (`order_finder`, `kappa`, `simplify`,
    `recompile`, or a `device` naming an accelerator) pins ``"exact"``: the option
    states how the answer must be computed, and the sampling lane cannot honour it.
  * Otherwise ``None`` resolves to `DEFAULT_METHOD`, and the sampling defaults are
    `DEFAULT_REL_ERR` and `DEFAULT_CONFIDENCE`.
  * Under ``"exact"``, whether named or pinned, a sampling option (a positive
    `rel_err`, `floor`, `confidence`, `seed`, or `strict`) raises rather than being
    silently ignored, the two asking for different engines.  `max_time` applies to
    both engines.
  * ``"smc"`` with an exact-engine option raises, since honouring it is impossible;
    under an explicit ``"auto"`` the options are allowed and reach the exact lane only,
    which the portfolio algorithm may or may not run.

`scalar_query` is the one dispatch every scalar entry point runs: the exact value in
this process or, under `max_time`, in the killable subprocess; the portfolio algorithm
under ``"auto"``; the certified sampling run under ``"smc"``.

This module imports nothing from `pylea.sample` at module level: resolving to
``"exact"`` must leave the sampling machinery unimported, so that Python programs never
asking for it never load it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from pylea.eval.backend import is_cpu_device
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


@dataclass(frozen=True)
class Contract:
    """A query's effective method and accuracy options.

    Under ``"exact"`` the sampling options hold their defaults: `rel_err` 0.0,
    `confidence` 1.0, no `floor`, no `seed`, `strict` False.
    """

    method: str
    rel_err: float
    floor: float | None
    confidence: float
    max_time: float | None
    seed: int | None
    strict: bool


def resolve(method: str | None, rel_err: float | None, compile_kwargs: dict, *,
            floor: float | None = None, confidence: float | None = None,
            max_time: float | None = None, seed: int | None = None,
            strict: bool = False) -> Contract:
    """The `Contract` of a query.

        >>> resolve(None, None, {})
        Contract(method='auto', rel_err=0.01, floor=None, confidence=0.95, max_time=None, seed=None, strict=False)
        >>> resolve(None, None, {"kappa": 20}).method
        'exact'
        >>> resolve("smc", None, {}).rel_err, resolve(None, 0.05, {}).method
        (0.01, 'auto')
        >>> resolve("smc", 0.0, {}).method, resolve("exact", None, {}).rel_err
        ('exact', 0.0)
        >>> resolve("exact", None, {}, seed=3)
        Traceback (most recent call last):
          ...
        ValueError: seed configure(s) the sampling lane and cannot be honoured under method='exact'; remove them or use method='smc' or 'auto'
    """
    if method not in (None, "exact", "smc", "auto"):
        raise ValueError(f"method must be 'exact', 'smc' or 'auto'; got {method!r}")
    if rel_err is not None and not rel_err >= 0.0:
        raise ValueError(f"rel_err must be non-negative; got {rel_err!r}")
    if confidence is not None and not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly in (0, 1); got {confidence!r}")
    pinning = [name for name in _EXACT_ONLY if name in compile_kwargs]
    if not is_cpu_device(compile_kwargs.get("device")):
        pinning.append(f"device={compile_kwargs.get('device')!r}")
    sampling = [name for name, value in (("rel_err", rel_err), ("floor", floor),
                                         ("confidence", confidence), ("seed", seed))
                if value is not None and (name != "rel_err" or value > 0.0)]
    if strict:
        sampling.append("strict")
    if rel_err == 0.0:
        method = "exact"
    elif method is None:
        if pinning and sampling:
            raise ValueError(
                f"{', '.join(pinning)} configure(s) the exact engine, but "
                f"{', '.join(sampling)} need(s) a sampling lane; name a method to say "
                "which is meant")
        method = "exact" if pinning else DEFAULT_METHOD
    if method == "exact":
        if sampling:
            raise ValueError(
                f"{', '.join(sampling)} configure(s) the sampling lane and cannot be "
                "honoured under method='exact'; remove them or use method='smc' or "
                "'auto'")
        return Contract("exact", 0.0, None, 1.0, max_time, None, False)
    if method == "smc" and pinning:
        raise ValueError(
            f"{', '.join(pinning)} configure(s) the exact engine and cannot be "
            "honoured under method='smc'; remove them or use method='exact' or 'auto'")
    return Contract(method, DEFAULT_REL_ERR if rel_err is None else float(rel_err),
                    floor, DEFAULT_CONFIDENCE if confidence is None else confidence,
                    max_time, seed, strict)


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
            from pylea.sample.portfolio import exact_matrix_within

            array, scale = exact_matrix_within(dpp, contract.max_time, compile_kwargs)
            value = exact_from_matrix(array, scale)
        return Estimate.exact(value, elapsed=time.monotonic() - started)
    if contract.method == "auto":
        from pylea.sample.portfolio import decide_scalar

        return decide_scalar(
            dpp, values, rel_err=contract.rel_err, confidence=contract.confidence,
            floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
            strict=contract.strict, what=what, exact_value=exact_value,
            exact_from_matrix=exact_from_matrix, compile_kwargs=compile_kwargs,
            forbidden=forbidden, affine=affine)
    from pylea.sample.query import certified_scalar

    return certified_scalar(
        dpp, values, rel_err=contract.rel_err, confidence=contract.confidence,
        floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
        strict=contract.strict, what=what, forbidden=forbidden, affine=affine)
