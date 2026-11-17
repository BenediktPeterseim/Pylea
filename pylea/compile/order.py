"""Contraction-tree search.

An `OrderFinder` maps a network, given as one tuple of index labels per tensor with the free
indices and a table of index sizes, to an `Order`: a contraction tree as a path in SSA form,
where ids ``0..n-1`` are the tensors and ``n + i`` is what step `i` produces, plus the
indices to slice.  Only summed indices are sliced, so the evaluator adds the slices.

`GreedyOrderFinder` builds cotengra's greedy tree and stops.  `AdaptiveOrderFinder` does the
same and escalates to a longer search above `escalate_cost`.  `SlicedBestOfOrderFinder`
builds many greedy trees, scores each after slicing it to the width bound, and improves the
winner.  `find_orders` asks a finder for the trees of several networks at once, through
its `find_all` where it has one, which is how a search bounded by wall clock is bounded
for all the definitions of a DPP together.

**Slicing is skipped where it cannot help.**  Slicing never lowers a tree's cost, and on a
large network it costs more time than building the tree.  `GreedyOrderFinder` given a
`slice_cost_cap` returns a tree whose cost, unsliced, already exceeds the cap without
slicing it, and marks it with `Order.over_cost_cap`; such an order serves as an estimate
of cost, not as a plan to run.

Scoring after slicing is what a plan with a width bound pays, a tree cheaper unsliced being
able to slice far worse.  Candidates come from permuting the tensor list, which changes the
greedy preset's tie-breaking, and the unpermuted tree is scored first, so the result is
never worse than `GreedyOrderFinder`'s.  Improvement starts from the winner as it stood
before slicing, so that slicing and structure are searched together, and replaces it only
where it scores better.

**Every tree has the indices, size and cost of each of its nodes computed as soon as it is
built, from the leaves upwards.**  cotengra otherwise computes a node's indices on first use
from those of its children, recursing once per level of the tree, and its slicer starts at
the root; on a deep tree, such as a fan-out of one variable to many assignments produces,
that recursion exceeds Python's recursion limit.

An `AdaptiveOrderFinder` given a `max_time`, and a `BoundedOrderFinder`, run their search
in a subprocess, which is stopped, together with the processes cotengra starts in it, once
the bound has passed: cotengra tests its own budget only between trials, which does not
bound a single trial, and offers no time limit at all for the improvement step of
`SlicedBestOfOrderFinder`.  cotengra's search runs without a pool of processes unless
`AdaptiveOrderFinder`'s `parallel` asks for one: a pool started in the calling process
makes a script that does not guard its main code with ``if __name__ == "__main__"`` fail,
and some systems refuse the semaphores a pool needs.  cotengra's warnings about optional
libraries it cannot import, and about trials of its search that failed, are not shown:
neither says anything about the result.

Where even the best tree has an intermediate too large for a float, cotengra's scoring
overflows; the finder then warns and returns a left-to-right path.
`pylea.compile.lower.compile_program` refuses that path where a size limit applies, unless
the finder has a `slice_cost_cap`, which marks it with `Order.over_cost_cap`, its cost
exceeding every cap; `pylea.compile.planner` then refuses it for its cost.
"""
from __future__ import annotations

import inspect
import math
import numbers
import os
import random
import threading
import time
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Protocol

from pylea.compile._process_tree import (
    DeadlineExceeded, start_worker,
)

# Greedy contraction cost, in scalar multiplications, above which `AdaptiveOrderFinder`
# escalates.  Placed in the gap between the greedy costs of Bayesian networks that are
# easy and hard to compile; not reproducible from this repository.  It governs only a caller asking for this finder directly,
# `pylea.compile.planner` taking the decision itself and passing `escalate_cost=0.0`.
# Re-measure after changing the contraction tree finders or `pylea.eval`.
ESCALATE_COST = 1e8

# Seconds a bounded search is given beyond its bound before its subprocess is stopped.
# Chosen, not fitted, to cover the start of a fresh interpreter importing cotengra, and the
# trial cotengra has in progress when its own budget runs out.
BOUND_GRACE_S = 10.0


# The filters hiding cotengra's warnings about optional libraries it cannot import and
# about failed trials, in the form of `PYTHONWARNINGS`, which the processes a subprocess
# search starts inherit.  A failed trial is scored as infinitely costly and so never
# chosen; the nested searches of cotengra's partitioning trials warn about their own
# failed trials whatever `on_trial_error` the outer search was given.
_COTENGRA_WARNINGS = ("ignore:Couldn't import:UserWarning",
                      "ignore:Couldn't find:UserWarning",
                      "ignore:Trial error:UserWarning")


@dataclass(frozen=True, slots=True)
class Order:
    """A contraction tree as a binary SSA path, and the indices to slice.

    `over_cost_cap` marks a tree left unsliced because its cost exceeded the finder's
    `slice_cost_cap`; its largest intermediate may exceed ``2**kappa``.
    """

    path: tuple[tuple[int, int], ...]
    sliced: tuple[int, ...]
    over_cost_cap: bool = False


class OrderFinder(Protocol):
    """Something that chooses a contraction tree for a network."""

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order: ...


def find_orders(finder, networks, *, kappa: int | None = None) -> list[Order]:
    """`finder`'s orders for `networks`, each a triple ``(indices, output, size)`` as
    `find` takes them, through the finder's `find_all` where it has one.

        >>> find_orders(GreedyOrderFinder(), [([(0, 1), (1, 2)], (0, 2), {0: 2, 1: 2, 2: 2})])
        [Order(path=((0, 1),), sliced=(), over_cost_cap=False)]
    """
    find_all = getattr(finder, "find_all", None)
    if find_all is not None:
        return list(find_all(networks, kappa=kappa))
    return [finder.find(indices, output, size, kappa=kappa)
            for indices, output, size in networks]


def check_finder_and_kappa(order_finder, kappa) -> int | None:
    """`kappa` as an `int`, or `None`, after checking it and `order_finder`.

    `kappa` must be `None` or a non-negative integer; anything else raises `ValueError`,
    or `TypeError` for a value that is not a number.  `order_finder` must be `None` or an
    object with a method `find` taking a `kappa` keyword, as the finders of this module
    have; anything else, such as the name of a finder given as a string, raises
    `TypeError`:

        >>> check_finder_and_kappa(GreedyOrderFinder(), 20)
        20
        >>> check_finder_and_kappa("greedy", 20)
        Traceback (most recent call last):
          ...
        TypeError: order_finder must be a contraction tree finder such as GreedyOrderFinder(), AdaptiveOrderFinder() or SlicedBestOfOrderFinder() from pylea.compile.order; got 'greedy'
    """
    if kappa is not None and not (type(kappa) is int and kappa >= 0):
        if isinstance(kappa, bool) or not isinstance(kappa, numbers.Real):
            raise TypeError(f"kappa must be None or a non-negative integer; got {kappa!r}")
        if not isinstance(kappa, numbers.Integral) or kappa < 0:
            raise ValueError(f"kappa must be None or a non-negative integer; got {kappa!r}")
        kappa = int(kappa)
    if order_finder is not None and not _is_finder(order_finder):
        raise TypeError(
            "order_finder must be a contraction tree finder such as GreedyOrderFinder(), "
            "AdaptiveOrderFinder() or SlicedBestOfOrderFinder() from pylea.compile.order; "
            f"got {order_finder!r}")
    return kappa


def _is_finder(candidate) -> bool:
    """Whether `candidate` has a method `find` that takes a `kappa` keyword."""
    if isinstance(candidate, (str, bytes, type)):
        return False
    cls = type(candidate)
    if "find" not in getattr(candidate, "__dict__", ()) and callable(getattr(cls, "find",
                                                                             None)):
        # The usual case, a method of the class, is answered once per class and method:
        # reading a signature costs more than a small query.
        return _class_has_finder(cls)
    return _takes_kappa(getattr(candidate, "find", None))


# Per class, a weak reference to the `find` that was checked and the answer, so that a
# class whose `find` changes is checked again and no class is kept alive.
_CHECKED_FINDERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _class_has_finder(cls: type) -> bool:
    find = getattr(cls, "find")
    hit = _CHECKED_FINDERS.get(cls)
    if hit is not None and hit[0]() is find:
        return hit[1]
    answer = _takes_kappa(find)
    try:
        _CHECKED_FINDERS[cls] = (weakref.ref(find), answer)
    except TypeError:                           # a class or `find` without weak references
        pass
    return answer


def _takes_kappa(find) -> bool:
    if not callable(find):
        return False
    try:
        parameters = inspect.signature(find).parameters
    except (TypeError, ValueError):
        return False
    return "kappa" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _contraction_tree(ins, out, sizes, optimize) -> Any:
    """cotengra's contraction tree for the network, every node's indices, size and cost
    computed.

    `contract_stats` computes them in cotengra's bottom-up traversal, which does not recurse,
    so that no later step computes them from the root down; the module docstring says why.
    """
    import cotengra as ctg

    tree: Any = ctg.array_contract_tree(ins, out, sizes, optimize=optimize,
                                        canonicalize=False)
    tree.contract_stats()
    return tree


def _symbols(indices, output, size):
    """The network with its integer indices relabelled as the symbols cotengra requires."""
    import cotengra as ctg

    symbol: dict[int, str] = {}
    for ix in [i for t in indices for i in t] + list(output):
        if ix not in symbol:
            # opt_einsum's `get_symbol` (3.4.0) returns surrogates for 55156 to 55295,
            # which cotengra cannot encode; cotengra's own skips them.
            symbol[ix] = ctg.get_symbol(len(symbol))
    label = {s: ix for ix, s in symbol.items()}
    return ([tuple(symbol[i] for i in t) for t in indices],
            tuple(symbol[i] for i in output),
            {symbol[i]: d for i, d in size.items() if i in symbol},
            label)


def _searchable(network) -> bool:
    """Whether a network ``(indices, output, size)`` has a contraction tree to search."""
    indices, _output, size = network
    return len(indices) > 1 and bool(size)


def _tree_cost(tree) -> float:
    """`tree`'s contraction cost, unsliced, as a float: infinite beyond the float range."""
    try:
        return float(tree.contraction_cost())
    except OverflowError:
        return math.inf


def _linear(n: int) -> tuple[tuple[int, int], ...]:
    """The left-to-right path over `n` tensors, for when there is nothing to search."""
    return tuple((n + i - 1 if i else 0, i + 1) for i in range(n - 1))


class AdaptiveOrderFinder:
    """cotengra's greedy tree, escalated to a longer search where that tree is costly.

    `escalate_cost` is the greedy contraction cost above which `optimize`, a cotengra
    preset, is run instead, and `max_time` bounds that run by wall clock, cotengra's own
    budget scaling with the contraction cost rather than with the clock.  `parallel` is
    passed to cotengra's search; the module docstring says why it is off by default.  A
    search that fails keeps the greedy tree.

    With `max_time` set, the search runs in a subprocess, which is stopped `BOUND_GRACE_S`
    seconds after `max_time` if it has not finished; `find` then raises `TimeoutError`.
    `find_all` searches several networks in one such subprocess, under one bound, each
    network given an equal share of the time that remains.  `max_time` is also passed to
    cotengra, which therefore stops by itself wherever its trials are short.  A
    `max_time` that is not a number raises `TypeError`, a negative or NaN one
    `ValueError`, and one too large for any clock to reach means no bound.
    """

    def __init__(self, optimize: str = "auto-hq", escalate_cost: float = ESCALATE_COST,
                 max_time: float | None = None, parallel: bool | int | str = False):
        if max_time is not None and (isinstance(max_time, bool)
                                     or not isinstance(max_time, numbers.Real)):
            raise TypeError(f"max_time must be None or a number of seconds; "
                            f"got {max_time!r}")
        if max_time is not None and not max_time >= 0:
            raise ValueError(f"max_time must be None or a number of seconds at least 0; "
                             f"got {max_time!r}")
        if max_time is not None and max_time >= threading.TIMEOUT_MAX:
            max_time = None                     # no bound a clock could reach
        self.optimize = optimize
        self.escalate_cost = escalate_cost
        self.max_time = None if max_time is None else float(max_time)
        self.parallel = parallel

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and _settings(other) == _settings(self)

    def __hash__(self) -> int:
        return hash((type(self), _settings(self)))

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        return self.find_all([(indices, output, size)], kappa=kappa)[0]

    def find_all(self, networks, *, kappa: int | None = None) -> list[Order]:
        """The orders of `networks`, each ``(indices, output, size)``, searched under one
        bound."""
        networks = list(networks)
        if self.max_time is None or not any(_searchable(n) for n in networks):
            return [self.find_here(*n, kappa=kappa) for n in networks]
        kind, settings = _settings(self)
        orders, _starved = _bounded_search(kind, settings, networks, kappa,
                                           self.max_time + BOUND_GRACE_S)
        return orders

    def find_here(self, indices, output, size, *, kappa: int | None = None) -> Order:
        """The search `find` makes, run in the calling process whatever `max_time` is.

        This is what the subprocess of a bounded search runs.
        """
        n = len(indices)
        if not _searchable((indices, output, size)):
            return Order(_linear(n), ())
        ins, out, sizes, label = _symbols(indices, output, size)

        def work() -> Order:
            tree: Any = _contraction_tree(ins, out, sizes, "greedy")
            if _tree_cost(tree) > self.escalate_cost:
                try:
                    with warnings.catch_warnings():
                        _filter_cotengra_warnings()
                        tree = _contraction_tree(ins, out, sizes, self._optimizer())
                except (KeyError, RuntimeError, OverflowError):
                    # cotengra raises `KeyError('tree')` where no trial succeeded; the
                    # greedy tree is a valid tree, and escalation only an improvement.
                    pass
            return _finish(tree, label, kappa)

        return _guarded(work, n)

    def _optimizer(self) -> Any:
        """cotengra's optimiser for `optimize`, with this finder's bound and `parallel`,
        and failed trials skipped silently."""
        import cotengra as ctg

        kinds = {"auto-hq": ctg.AutoHQOptimizer, "auto": ctg.AutoOptimizer}
        if self.optimize not in kinds:
            return self.optimize
        options: dict[str, Any] = {"parallel": self.parallel, "on_trial_error": "ignore"}
        if self.max_time is not None:
            options["max_time"] = self.max_time
        return kinds[self.optimize](**options)


class GreedyOrderFinder:
    """cotengra's greedy tree, with no escalation.

    Two tensors sharing one index contract in a single step, and nothing is sliced:

        >>> GreedyOrderFinder().find([(0, 1), (1, 2)], output=(0, 2), size={0: 2, 1: 2, 2: 2})
        Order(path=((0, 1),), sliced=(), over_cost_cap=False)

    `slice_cost_cap`, a cost in scalar multiplications, leaves a tree whose cost already
    exceeds it unsliced, marked with `Order.over_cost_cap`; the module docstring says why.

    The finder has no state beyond `slice_cost_cap`, so two instances with the same cap
    are interchangeable and compare equal; caches keyed on the finder, such as
    `pylea.syntax.dpp.DPP.compiled`'s, therefore hit across instances:

        >>> GreedyOrderFinder() == GreedyOrderFinder()
        True
    """

    def __init__(self, slice_cost_cap: float | None = None):
        self.slice_cost_cap = slice_cost_cap

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and other.slice_cost_cap == self.slice_cost_cap

    def __hash__(self) -> int:
        return hash((type(self), self.slice_cost_cap))

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        n = len(indices)
        if n <= 1 or not size:
            return Order(_linear(n), ())
        ins, out, sizes, label = _symbols(indices, output, size)
        target = slice_target(kappa)
        cap = self.slice_cost_cap if target is not None else None

        def work() -> Order:
            tree = _contraction_tree(ins, out, sizes, "greedy")
            if cap is not None and float(tree.max_size()) > target \
                    and _tree_cost(tree) > cap:
                order = _finish(tree, label, None)
                return Order(order.path, (), over_cost_cap=True)
            return _finish(tree, label, kappa)

        return _guarded(work, n, over_cost_cap=cap is not None)


class SlicedBestOfOrderFinder:
    """The best of several greedy trees, scored after slicing, and then locally improved.

    The search stops after `patience` consecutive candidates that did not improve, after
    `max_trials` candidates, or at `budget` seconds, and `stop` reports which of the three
    ended it.

    Stopping on `patience` makes the tree a function of the network, `seed` and `kappa`
    alone.  Stopping on `budget` makes it depend on the machine and on what else is running,
    and returns a worse tree, so `starved` records that this has happened at least once and
    lets a caller say so rather than accept a load-dependent plan.

    `reconfigure` controls the improvement step, which starts from the winner as it stood
    before slicing and replaces it only where it scores better at the same width.  `best_cost` and `trials` report what the search settled on and how many
    candidates it drew, the unpermuted tree not counted among them.

    Two finders compare equal where their type and settings are equal, whatever their
    searches reported, so caches keyed on the finder, such as
    `pylea.syntax.dpp.DPP.compiled`'s, hit across instances; `AdaptiveOrderFinder`
    compares in the same way.

        >>> SlicedBestOfOrderFinder(seed=1) == SlicedBestOfOrderFinder(seed=1)
        True
    """

    def __init__(self, budget: float = 20.0, max_trials: int = 200, patience: int = 40,
                 seed: int = 0, reconfigure: bool = True):
        self.budget = budget
        self.max_trials = max_trials
        self.patience = patience
        self.seed = seed
        self.reconfigure = reconfigure
        self.stop = ""                  # why the last search stopped
        self.starved = False            # whether any search here stopped on the wall clock
        self.best_cost = 0.0            # the winning tree's score, in cotengra's cost
        self.trials = 0                 # candidates tried, the unpermuted tree not counted

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and _settings(other) == _settings(self)

    def __hash__(self) -> int:
        return hash((type(self), _settings(self)))

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        return self._find(indices, output, size, kappa, self.budget)

    def find_all(self, networks, *, kappa: int | None = None) -> list[Order]:
        """The orders of `networks`, each ``(indices, output, size)``, within `budget`
        seconds for all of them: each network is given an equal share of the time that
        remains."""
        networks = list(networks)
        started = time.perf_counter()
        orders = []
        for k, (indices, output, size) in enumerate(networks):
            remaining = max(0.0, self.budget - (time.perf_counter() - started))
            orders.append(self._find(indices, output, size, kappa,
                                     remaining / (len(networks) - k)))
        return orders

    def _find(self, indices, output, size, kappa: int | None, budget: float) -> Order:
        n = len(indices)
        if n <= 1 or not size:
            return Order(_linear(n), ())
        ins, out, sizes, label = _symbols(indices, output, size)
        target = slice_target(kappa)

        def score(tree):
            t = slice_to(tree, float(target)) if target is not None else tree
            try:
                return float(t.contraction_cost()), t
            except OverflowError:
                return float("inf"), t

        def work() -> Order:
            t0 = time.perf_counter()
            raw = _contraction_tree(ins, out, sizes, "greedy")
            best_cost, best = score(raw)
            best_raw = raw
            best_perm: list[int] | None = None

            rng = random.Random(self.seed)
            stale = 0
            tried, stop = 0, "max_trials"
            for _ in range(self.max_trials):
                if stale >= self.patience:
                    stop = "patience"
                    break
                if time.perf_counter() - t0 > budget:
                    stop = "budget"
                    break
                tried += 1
                perm = list(range(n))
                rng.shuffle(perm)
                try:
                    raw = _contraction_tree([ins[j] for j in perm], out, sizes, "greedy")
                    cost, sliced = score(raw)
                except Exception:
                    stale += 1
                    continue
                if cost < best_cost:
                    best_cost, best, best_raw, best_perm, stale = cost, sliced, raw, perm, 0
                else:
                    stale += 1

            if self.reconfigure:
                best, best_cost = _reconfigured(best, best_raw, best_cost, target)
            self.stop, self.trials, self.best_cost = stop, tried, best_cost
            self.starved = self.starved or stop == "budget"
            order = _finish(best, label, kappa, already_sliced=True)
            if best_perm is None:
                return order
            # SSA ids below `n` are positions in the permuted tensor list, so map them back.
            return Order(tuple((best_perm[a] if a < n else a, best_perm[b] if b < n else b)
                               for a, b in order.path), order.sliced)

        return _guarded(work, n)


class BoundedOrderFinder:
    """`finder`'s search, for all networks of a `find_all` call together, run in a
    subprocess that is stopped after `limit` seconds.

    `finder` is an `AdaptiveOrderFinder` or a `SlicedBestOfOrderFinder`; its settings
    travel to the subprocess, so a subclass's own `find` does not run there.  A search
    stopped at `limit` raises `TimeoutError`.  A `SlicedBestOfOrderFinder`'s `starved`
    records a search that stopped on its budget in the subprocess.
    """

    def __init__(self, finder, limit: float):
        _settings(finder)                       # refuses a finder that cannot travel
        self.finder = finder
        self.limit = limit

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        return self.find_all([(indices, output, size)], kappa=kappa)[0]

    def find_all(self, networks, *, kappa: int | None = None) -> list[Order]:
        networks = list(networks)
        if not any(_searchable(n) for n in networks):
            return find_orders(self.finder, networks, kappa=kappa)
        kind, settings = _settings(self.finder)
        orders, starved = _bounded_search(kind, settings, networks, kappa, self.limit)
        if starved:
            self.finder.starved = True
        return orders


def _settings(finder) -> tuple[str, tuple]:
    """The kind and settings by which `search_all` rebuilds `finder` in a subprocess."""
    if isinstance(finder, AdaptiveOrderFinder):
        return "adaptive", (finder.optimize, finder.escalate_cost, finder.max_time,
                            finder.parallel)
    if isinstance(finder, SlicedBestOfOrderFinder):
        return "best_of", (finder.budget, finder.max_trials, finder.patience, finder.seed,
                           finder.reconfigure)
    raise TypeError(f"a search in a subprocess takes an AdaptiveOrderFinder or a "
                    f"SlicedBestOfOrderFinder; got {finder!r}")


def search_all(kind: str, settings: tuple, networks, kappa: int | None
               ) -> tuple[list[Order], bool]:
    """The orders the finder of `kind` with `settings` finds for `networks`, and whether
    its search stopped on its budget: what the subprocess of `BoundedOrderFinder` and of a
    bounded `AdaptiveOrderFinder` runs.

    An `AdaptiveOrderFinder`'s `max_time` is shared among the networks: each is given an
    equal share of the time that remains.
    """
    if kind == "best_of":
        finder = SlicedBestOfOrderFinder(*settings)
        return finder.find_all(networks, kappa=kappa), finder.starved
    optimize, escalate_cost, max_time, parallel = settings
    networks = list(networks)
    deadline = None if max_time is None else time.monotonic() + max_time
    orders = []
    for k, (indices, output, size) in enumerate(networks):
        share = None if deadline is None else \
            max(0.0, deadline - time.monotonic()) / (len(networks) - k)
        finder = AdaptiveOrderFinder(optimize, escalate_cost, share, parallel)
        orders.append(finder.find_here(indices, output, size, kappa=kappa))
    return orders, False


def hide_cotengra_warnings() -> None:
    """Nothing; the warnings of cotengra that the module docstring names are hidden in
    this process and in every process it starts from now on."""
    _filter_cotengra_warnings()
    current = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = ",".join(
        ([current] if current else []) + list(_COTENGRA_WARNINGS))


def _filter_cotengra_warnings() -> None:
    for spec in _COTENGRA_WARNINGS:
        _action, message, _category = spec.split(":")
        warnings.filterwarnings("ignore", message=message, category=UserWarning)


def _reconfigured(sliced, raw, cost: float, target):
    """``(tree, cost)`` after improving `raw`, or `sliced` and `cost` where that did not pay.

    `raw` is the winning tree before slicing and `sliced` is the same tree after, so passing
    both is what lets slicing and structure be searched together.  Without a `target` there
    is nothing to slice and only the structure is reconfigured.

    The candidate is sliced to the width `sliced` meets: `target`, or, where the free
    indices alone exceed `target`, the wider bound `slice_to` settled on.  The two trees are
    then compared at the same bound, and the candidate's bound is one that slicing summed
    indices alone can meet, cotengra failing on any other.

    cotengra failing to slice raises `RuntimeError`, and a width beyond what a float holds
    raises `OverflowError`.  Both leave the already-scored tree in place, this step being an
    optimisation that must not fail a compilation the search survived.
    """
    try:
        candidate = raw.copy()
        if target is not None:
            width = max(int(target), int(sliced.max_size()))
            candidate.slice_and_reconfigure_(target_size=width, allow_outer=False)
        else:
            candidate.subtree_reconfigure_()
        improved = float(candidate.contraction_cost())
    except (RuntimeError, OverflowError):
        return sliced, cost
    if improved < cost:
        return candidate, improved
    return sliced, cost


def slice_to(tree, target: float):
    """`tree` sliced to `target` entries per intermediate, or to the nearest target it meets.

    Only summed indices are sliced, which leaves a floor: a target below what the free
    indices alone occupy cannot be met, and cotengra reports that by running out of indices
    to slice.  The target is then doubled and tried again, so how wide the result is remains
    for the caller to judge.
    """
    limit = float(tree.max_size())
    while target < limit:
        candidate = tree.copy()
        try:
            candidate.slice_(target_size=target, allow_outer=False)
            return candidate
        except RuntimeError:
            target *= 2
    return tree


def slice_target(kappa: int | None) -> float | None:
    """``2**kappa`` as a float, the size limit slicing aims for, or `None` where there is
    no limit: `kappa` is `None`, or ``2**kappa`` exceeds every float, which no tensor can
    reach.

        >>> slice_target(3), slice_target(None), slice_target(2000)
        (8.0, None, None)
    """
    if kappa is None or kappa > _MAX_EXPONENT:
        return None
    return math.ldexp(1.0, kappa)


# The largest `k` for which ``2.0**k`` is a finite float64.
_MAX_EXPONENT = 1023


def _finish(tree, label, kappa: int | None, already_sliced: bool = False) -> Order:
    target = slice_target(kappa)
    if target is not None and not already_sliced:
        tree = slice_to(tree, target)
    sliced = tuple(label[s] for s in getattr(tree, "sliced_inds", ()) if s in label)
    return Order(tuple(tuple(step) for step in tree.get_ssa_path()), sliced)


def _guarded(work, n: int, over_cost_cap: bool = False) -> Order:
    """``work()``, or a left-to-right path where cotengra's scoring overflowed.

    `over_cost_cap` marks that path as left unsliced over a cost cap, its cost exceeding
    every float and therefore every cap.
    """
    try:
        return work()
    except OverflowError:
        # cotengra scores trees in floating point, which overflows once even the best tree has
        # an intermediate above ``2**1023``, meaning the network's treewidth is beyond what a
        # contraction can represent.  A left-to-right path still lets compilation finish;
        # `pylea.compile.lower.compile_program` refuses it where a size limit applies, and
        # `pylea.compile.planner`, where it was left unsliced over a cap, for its cost.
        warnings.warn(
            f"contraction-tree search overflowed on a network of {n} tensors; "
            f"falling back to a left-to-right contraction tree.", stacklevel=2)
        return Order(_linear(n), (), over_cost_cap=over_cost_cap)


def _bounded_search(kind: str, settings: tuple, networks: list, kappa: int | None,
                    limit: float) -> tuple[list[Order], bool]:
    """``search_all(kind, settings, networks, kappa)``, run in a subprocess that is
    stopped after `limit` seconds.

    Raises `TimeoutError` where the search is stopped and `RuntimeError` where the worker
    fails.  Stopping the worker also stops the processes cotengra started in it.
    """
    tree = None
    try:
        tree = start_worker("pylea.compile._order_worker")
        tree.send((kind, settings, networks, kappa))
        response = tree.receive(limit)
    except DeadlineExceeded:
        raise DeadlineExceeded(
            f"the contraction-tree search did not finish within {limit:.0f} s") from None
    except RuntimeError as exc:
        raise RuntimeError(f"the contraction-tree search failed: {exc}") from None
    finally:
        if tree is not None:
            try:
                tree.kill()
            finally:
                tree.close()
    if "error" in response:
        raise RuntimeError(f"the contraction-tree search failed: {response['error']}")
    return response["orders"], response["starved"]
