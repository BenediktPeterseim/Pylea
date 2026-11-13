"""Contraction-tree search.

An `OrderFinder` maps a network, given as one tuple of index labels per tensor with the free
indices and a table of index sizes, to an `Order`: a contraction tree as a path in SSA form,
where ids ``0..n-1`` are the tensors and ``n + i`` is what step `i` produces, plus the
indices to slice.  Only summed indices are sliced, so the evaluator adds the slices.

`GreedyOrderFinder` builds cotengra's greedy tree and stops.  `AdaptiveOrderFinder` does the
same and escalates to a longer search above `escalate_cost`.  `SlicedBestOfOrderFinder`
builds many greedy trees, scores each after slicing it to the width bound, and improves the
winner.

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

An `AdaptiveOrderFinder` given a `max_time` runs its search in a subprocess, which is
stopped, together with the processes cotengra starts in it, once the bound has passed:
cotengra tests its own budget only between trials, which does not bound a single trial.

Where even the best tree has an intermediate too large for a float, cotengra's scoring
overflows; the finder then warns and returns a left-to-right path, leaving the plan for the
evaluator's width bound to reject.
"""
from __future__ import annotations

import pickle
import random
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

from pylea.compile._process_tree import ProcessTree

# Greedy contraction cost, in scalar multiplications, above which `AdaptiveOrderFinder`
# escalates.  Placed in the gap between the easy and the hard Bayesian networks of the
# corpus.  It governs only a caller asking for this finder directly, `pylea.compile.planner`
# taking the decision itself and passing `escalate_cost=0.0`.
ESCALATE_COST = 1e8

# Seconds a bounded `AdaptiveOrderFinder` search is given beyond its `max_time` before its
# subprocess is stopped: the start of a fresh interpreter importing cotengra, and the trial
# cotengra has in progress when its own budget runs out.
BOUND_GRACE_S = 10.0


@dataclass(frozen=True, slots=True)
class Order:
    """A contraction tree as a binary SSA path, and the indices to slice."""

    path: tuple[tuple[int, int], ...]
    sliced: tuple[int, ...]


class OrderFinder(Protocol):
    """Something that chooses a contraction tree for a network."""

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order: ...


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


def _linear(n: int) -> tuple[tuple[int, int], ...]:
    """The left-to-right path over `n` tensors, for when there is nothing to search."""
    return tuple((n + i - 1 if i else 0, i + 1) for i in range(n - 1))


class AdaptiveOrderFinder:
    """cotengra's greedy tree, escalated to a longer search where that tree is costly.

    `escalate_cost` is the greedy contraction cost above which `optimize` is run instead,
    and `max_time` bounds that run by wall clock, cotengra's own budget scaling with the
    contraction cost rather than with the clock.

    With `max_time` set, the search runs in a subprocess, which is stopped `BOUND_GRACE_S`
    seconds after `max_time` if it has not finished; `find` then raises `TimeoutError`.
    `max_time` is also passed to cotengra, which therefore stops by itself wherever its
    trials are short.
    """

    def __init__(self, optimize: str = "auto-hq", escalate_cost: float = ESCALATE_COST,
                 max_time: float | None = None):
        self.optimize = optimize
        self.escalate_cost = escalate_cost
        self.max_time = max_time

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        if self.max_time is None or len(indices) <= 1 or not size:
            return self.find_here(indices, output, size, kappa=kappa)
        return _bounded_search((self.optimize, self.escalate_cost, self.max_time),
                               (indices, output, size, kappa),
                               self.max_time + BOUND_GRACE_S)

    def find_here(self, indices, output, size, *, kappa: int | None = None) -> Order:
        """The search `find` makes, run in the calling process whatever `max_time` is.

        This is what the subprocess of a bounded search runs.
        """
        import cotengra as ctg

        n = len(indices)
        if n <= 1 or not size:
            return Order(_linear(n), ())
        ins, out, sizes, label = _symbols(indices, output, size)

        def work() -> Order:
            tree: Any = _contraction_tree(ins, out, sizes, "greedy")
            if tree.contraction_cost() > self.escalate_cost:
                optimize: Any = (ctg.AutoHQOptimizer(max_time=self.max_time)
                                 if self.max_time is not None else self.optimize)
                tree = _contraction_tree(ins, out, sizes, optimize)
            return _finish(tree, label, kappa)

        return _guarded(work, n)


class GreedyOrderFinder:
    """cotengra's greedy tree, with no escalation.

    Two tensors sharing one index contract in a single step, and nothing is sliced:

        >>> GreedyOrderFinder().find([(0, 1), (1, 2)], output=(0, 2), size={0: 2, 1: 2, 2: 2})
        Order(path=((0, 1),), sliced=())

    The finder is stateless, so any two instances are interchangeable and compare equal;
    caches keyed on the finder, such as `pylea.syntax.dpp.DPP.compiled`'s, therefore hit
    across instances:

        >>> GreedyOrderFinder() == GreedyOrderFinder()
        True
    """

    def __eq__(self, other) -> bool:
        return type(other) is type(self)

    def __hash__(self) -> int:
        return hash(type(self))

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        n = len(indices)
        if n <= 1 or not size:
            return Order(_linear(n), ())
        ins, out, sizes, label = _symbols(indices, output, size)
        return _guarded(lambda: _finish(
            _contraction_tree(ins, out, sizes, "greedy"), label, kappa), n)


class SlicedBestOfOrderFinder:
    """The best of several greedy trees, scored after slicing, and then locally improved.

    The search stops after `patience` consecutive candidates that did not improve, after
    `max_trials` candidates, or at `budget` seconds, and `stop` reports which of the three
    ended it.

    Stopping on `patience` makes the tree a function of the network, `seed` and `kappa`
    alone.  Stopping on `budget` makes it depend on the machine and on what else is running,
    and returns a worse tree, so `starved` records that this has happened at least once and
    lets a caller say so rather than accept a load-dependent plan.

    `reconfigure` controls the improvement step; the module docstring gives its ordering
    requirement.  `best_cost` and `trials` report what the search settled on and how many
    candidates it drew, the unpermuted tree not counted among them.
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

    def find(self, indices, output, size, *, kappa: int | None = None) -> Order:
        n = len(indices)
        if n <= 1 or not size:
            return Order(_linear(n), ())
        ins, out, sizes, label = _symbols(indices, output, size)
        target = 2 ** kappa if kappa is not None else None

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
                if time.perf_counter() - t0 > self.budget:
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


def _finish(tree, label, kappa: int | None, already_sliced: bool = False) -> Order:
    if kappa is not None and not already_sliced:
        tree = slice_to(tree, float(2 ** kappa))
    sliced = tuple(label[s] for s in getattr(tree, "sliced_inds", ()) if s in label)
    return Order(tuple(tuple(step) for step in tree.get_ssa_path()), sliced)


def _guarded(work, n: int) -> Order:
    try:
        return work()
    except OverflowError:
        # cotengra scores trees in floating point, which overflows once even the best tree has
        # an intermediate above ``2**1023``, meaning the network's treewidth is beyond what a
        # contraction can represent.  A left-to-right path still lets compilation finish, and
        # the evaluator's width guard rejects the plan if it cannot be run.
        warnings.warn(
            f"contraction-tree search overflowed on a network of {n} tensors; "
            f"falling back to a left-to-right contraction tree.", stacklevel=2)
        return Order(_linear(n), ())


def _bounded_search(settings: tuple, network: tuple, limit: float) -> Order:
    """The `Order` a bounded `AdaptiveOrderFinder` with `settings` finds for `network`,
    searched in a subprocess that is stopped after `limit` seconds.

    Raises `TimeoutError` where the search is stopped and `RuntimeError` where the worker
    fails.  Stopping the worker also stops the processes cotengra started in it.
    """
    with ProcessTree([sys.executable, "-m", "pylea.compile._order_worker"]) as tree:
        process = tree.process
        assert process.stdout is not None
        stdout = process.stdout
        received: list[bytes] = []
        # The response is read on a thread so that a large one cannot fill the pipe while
        # this process waits; the request is safe to write, the worker reading it before it
        # writes anything.
        reader = threading.Thread(target=lambda: received.append(stdout.read()), daemon=True)
        reader.start()
        tree.send((settings, network))
        reader.join(limit)
        if reader.is_alive():
            tree.kill()
            raise TimeoutError(
                f"the contraction-tree search did not finish within {limit:.0f} s")
        process.wait()
    if not received or not received[0]:
        raise RuntimeError("the contraction-tree search subprocess produced no response")
    response = pickle.loads(received[0])
    if "error" in response:
        raise RuntimeError(f"the contraction-tree search failed: {response['error']}")
    return response["order"]
