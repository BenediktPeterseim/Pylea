"""The sliced-best-of finder's local improvement step.

Reconfiguration is an optimisation over slicing and tree structure together, so what must
hold is what the ladder relies on everywhere: it never makes the winner worse, it never
changes the answer, and declining it is always possible.  How much it improves any given
network is a property of the instance and is not asserted here.  Like slicing alone, it
must slice no free index, since the evaluator sums the slices.

The bounded search of `AdaptiveOrderFinder` runs in a subprocess; what must hold is that it
finds the same kind of order as the unbounded search, and that it stops at its bound.

Every finder must find and slice a network whose greedy tree is deeper than Python's
recursion limit.
"""
import sys

import numpy as np
import pytest

from pylea import DPP, flip, observe
from pylea.compile.cost import contraction_cost
from pylea.compile.lower import compile_program
from pylea.compile.network import build_network
import pylea.compile.order as order
from pylea.compile.order import AdaptiveOrderFinder, SlicedBestOfOrderFinder
from pylea.eval.interp import evaluate
from pylea.ir.program import intern
from pylea.ir.reference import TooLarge, semantics
from test.compile.random_programs import random_program

KAPPA = 8       # small enough that these networks actually slice


def _network(seed: int):
    prog = intern(random_program(np.random.default_rng(seed)))
    return prog, build_network(prog, prog.root)


@pytest.mark.parametrize("seed", range(8))
def test_reconfiguration_never_makes_the_winner_worse(seed):
    prog, net = _network(seed)
    plain = SlicedBestOfOrderFinder(reconfigure=False)
    plain.find(net.axes, net.output, net.size, kappa=KAPPA)
    improved = SlicedBestOfOrderFinder()
    improved.find(net.axes, net.output, net.size, kappa=KAPPA)
    assert improved.best_cost <= plain.best_cost


@pytest.mark.parametrize("seed", range(8))
def test_the_reconfigured_plan_computes_the_right_factor(seed):
    prog = intern(random_program(np.random.default_rng(seed)))
    plans, root = compile_program(prog, SlicedBestOfOrderFinder(), kappa=KAPPA)
    array, scale = evaluate(plans, root)
    got = np.asarray(array, dtype=float) * float(np.exp(scale))
    try:
        want = semantics(prog)
    except TooLarge:
        pytest.skip("reference enumeration too large for this draw")
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-300)


def test_the_reconfigured_plan_respects_the_slicing_bound():
    prog, _net = _network(3)
    plans, _ = compile_program(prog, SlicedBestOfOrderFinder(), kappa=KAPPA)
    assert all(p.peak_entries() <= 2 ** KAPPA or not p.sliced_dims
               for p in plans.values())


def test_costs_agree_between_finder_and_plan():
    """The finder's score is the quantity the plan pays, so the two must not drift."""
    prog, _net = _network(5)
    finder = SlicedBestOfOrderFinder()
    plans, _ = compile_program(prog, finder, kappa=KAPPA)
    assert contraction_cost(plans) == pytest.approx(finder.best_cost, rel=0.05)


def _grid_network(side: int, outputs: int, seed: int):
    """A `side` by `side` grid of tensors, neighbours sharing an index, with `outputs` free
    indices of one tensor each, the tensors drawn with `seed`; every index has size 2."""
    index: dict[tuple, int] = {}

    def shared(a, b) -> int:
        return index.setdefault((min(a, b), max(a, b)), len(index))

    cells = [(r, c) for r in range(side) for c in range(side)]
    axes = [[shared(cell, nb) for nb in ((cell[0] + dr, cell[1] + dc)
                                         for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
             if nb in cells] for cell in cells]
    rng = np.random.default_rng(seed)
    output = tuple(len(index) + k for k in range(outputs))
    for ix in output:
        axes[int(rng.integers(len(axes)))].append(ix)
    return [tuple(t) for t in axes], output, {i: 2 for i in range(len(index) + outputs)}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("outputs", [2, 4])
@pytest.mark.parametrize("kappa", [3, 4])
def test_reconfiguration_slices_no_free_index(kappa, outputs, seed):
    axes, output, size = _grid_network(4, outputs, seed)
    found = SlicedBestOfOrderFinder(max_trials=20, patience=10).find(
        axes, output, size, kappa=kappa)
    assert not set(found.sliced) & set(output)


def test_the_reconfigured_plan_of_a_dpp_with_two_outputs_computes_the_right_factor():
    @DPP
    def lattice():
        x = [[flip(0.5) for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for a, b in ((i + 1, j), (i, j + 1)):
                    if a < 3 and b < 3:
                        same = (x[i][j] & x[a][b]) | (~x[i][j] & ~x[a][b])
                        observe(same | flip(0.3))
        return x[0][0], x[1][0]

    prog = lattice.program()
    got, got_scale = evaluate(*compile_program(prog, SlicedBestOfOrderFinder(), kappa=3))
    want, want_scale = evaluate(*compile_program(prog, order.GreedyOrderFinder(), kappa=None))
    np.testing.assert_allclose(np.asarray(got, dtype=float) * float(np.exp(got_scale)),
                               np.asarray(want, dtype=float) * float(np.exp(want_scale)),
                               rtol=1e-9)


def test_a_bounded_search_finds_an_order_computing_the_right_factor():
    for seed in range(20):
        prog = intern(random_program(np.random.default_rng(seed)))
        try:
            want = semantics(prog)
        except TooLarge:
            continue
        finder = AdaptiveOrderFinder(escalate_cost=0.0, max_time=5.0)
        plans, root = compile_program(prog, finder, kappa=KAPPA)
        array, scale = evaluate(plans, root)
        got = np.asarray(array, dtype=float) * float(np.exp(scale))
        assert np.allclose(got, want, atol=1e-12)
        return
    pytest.fail("no random program small enough to check")


def test_a_bounded_search_is_stopped_at_its_bound(monkeypatch):
    """cotengra tests its budget only between trials, so the bound must be enforced from
    outside; a bound of zero stops any search before it answers."""
    monkeypatch.setattr(order, "BOUND_GRACE_S", 0.0)
    _prog, net = _network(0)
    with pytest.raises(TimeoutError):
        AdaptiveOrderFinder(escalate_cost=0.0, max_time=0.0).find(
            net.axes, net.output, net.size, kappa=KAPPA)


def _fan_out(n: int):
    """`n` tensors sharing index 0, of size 8, each with an index of its own, of size 2.

    The greedy tree of this network is a chain ``n - 1`` levels deep, and slicing it to
    width 1 slices index 0.
    """
    return [(0, k + 1) for k in range(n)], (), {i: 8 if i == 0 else 2 for i in range(n + 1)}


def _depth(path) -> int:
    n = len(path) + 1
    depth: dict[int, int] = {}
    for k, (a, b) in enumerate(path):
        depth[n + k] = 1 + max(depth.get(a, 0), depth.get(b, 0))
    return max(depth.values())


def test_a_tree_deeper_than_the_recursion_limit_can_be_sliced_before_anything_else():
    """cotengra's slicer starts from the root, so this fails if any node's indices are left
    to be computed on first use."""
    n = 2 * sys.getrecursionlimit()
    ins, out, sizes, _label = order._symbols(*_fan_out(n))
    tree = order._contraction_tree(ins, out, sizes, "greedy")
    assert _depth(tree.get_ssa_path()) == n - 1
    tree.slice_(target_size=1)
    assert len(tree.sliced_inds) == 1


@pytest.mark.parametrize("finder", [
    order.GreedyOrderFinder(),
    SlicedBestOfOrderFinder(max_trials=2),
    AdaptiveOrderFinder(escalate_cost=0.0),
], ids=lambda f: type(f).__name__)
def test_every_finder_slices_a_network_whose_greedy_tree_exceeds_the_recursion_limit(finder):
    """Every finder builds the greedy tree first.  The escalated search's own tree is as deep
    as the trials its time budget allows make it, so its depth is not asserted."""
    found = finder.find(*_fan_out(2 * sys.getrecursionlimit()), kappa=0)
    assert found.sliced == (0,)


def test_a_network_past_55156_indices_gets_distinct_symbols_cotengra_can_encode():
    n = 56_000
    ins, out, sizes, label = order._symbols([(i, i + 1) for i in range(n)], (0, n),
                                            {i: 2 for i in range(n + 1)})
    symbols = {s for t in ins for s in t}
    assert len(symbols) == n + 1 == len(label)
    for s in symbols:
        s.encode("utf-8")


def test_an_invalid_kappa_or_finder_is_refused_with_a_message():
    from pylea.compile.lower import compile_program
    from pylea.compile.planner import plan_compilation

    prog, _net = _network(0)
    for kappa in (-1, 2.5):
        with pytest.raises(ValueError, match="non-negative integer"):
            compile_program(prog, kappa=kappa)
        with pytest.raises(ValueError, match="non-negative integer"):
            plan_compilation(prog, kappa=kappa)
    for finder in ("greedy", object(), AdaptiveOrderFinder):
        with pytest.raises(TypeError, match="GreedyOrderFinder"):
            compile_program(prog, finder, kappa=KAPPA)
    plans, _root = compile_program(prog, kappa=0)
    assert plans


def test_a_finder_reached_through_attribute_forwarding_or_added_later_is_accepted():
    from pylea.compile.order import GreedyOrderFinder, check_finder_and_kappa

    class Forwarding:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    assert check_finder_and_kappa(Forwarding(GreedyOrderFinder()), 3) == 3

    class Late:
        pass

    with pytest.raises(TypeError):
        check_finder_and_kappa(Late(), 3)
    Late.find = GreedyOrderFinder.find                       # type: ignore[attr-defined]
    assert check_finder_and_kappa(Late(), 3) == 3


def test_a_finder_whose_find_is_a_property_is_accepted():
    from pylea.compile.order import GreedyOrderFinder, check_finder_and_kappa

    class Wrapped:
        @property
        def find(self):
            return GreedyOrderFinder().find

    assert check_finder_and_kappa(Wrapped(), 3) == 3


# A script with no ``if __name__ == "__main__"`` guard that escalates in its own process and
# in a bounded subprocess.  A pool of processes started by the search in the script's own
# process would run the script again in each of its processes.
_ESCALATING_SCRIPT = """
import pylea.compile.order as order
from pylea.compile.order import AdaptiveOrderFinder

order.BOUND_GRACE_S = 3.0

def grid(n):
    index = {}
    def edge(a, b):
        return index.setdefault(tuple(sorted((a, b))), len(index))
    tensors = [tuple(edge((i, j), (i + di, j + dj))
                     for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0))
                     if 0 <= i + di < n and 0 <= j + dj < n)
               for i in range(n) for j in range(n)]
    return tensors, (), {k: 2 for k in range(len(index))}

indices, output, size = grid(6)
AdaptiveOrderFinder(escalate_cost=0.0).find(indices, output, size, kappa=8)
AdaptiveOrderFinder(escalate_cost=0.0, max_time=1.0).find(indices, output, size, kappa=8)
# A network whose cost exceeds the float range, on which cotengra's trials fail.  A search
# stopped at its bound still shows what the worker wrote until then.
n = 1100
try:
    AdaptiveOrderFinder(escalate_cost=0.0, max_time=1.0).find(
        [(k,) for k in range(n)], tuple(range(n)), {k: 2 for k in range(n)})
except TimeoutError:
    pass
print("searched")
"""


def _run_escalating_script(tmp_path, hidden=()):
    """The escalating script's output, run with the modules `hidden` made unimportable."""
    import os
    import subprocess
    from pathlib import Path

    import pylea

    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in hidden))
    root = str(Path(pylea.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), root])}
    env.pop("PYTHONWARNINGS", None)
    return subprocess.run([sys.executable, "-c", _ESCALATING_SCRIPT], env=env,
                          capture_output=True, text=True, timeout=120)


def test_escalation_in_the_callers_process_starts_no_process(tmp_path):
    result = _run_escalating_script(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["searched"]


def test_cotengra_warnings_do_not_reach_the_caller(tmp_path):
    """Without kahypar and the optional optimisation libraries cotengra warns once per
    process that searches, and its failed trials warn on a network whose cost exceeds
    the float range."""
    result = _run_escalating_script(tmp_path, ("kahypar", "optuna", "cmaes", "nevergrad"))
    assert result.returncode == 0, result.stderr
    for text in ("kahypar", "Couldn't", "Trial error", "Warning:"):
        assert text not in result.stderr.replace("OMP: Warning #179", ""), result.stderr


def test_finders_with_equal_settings_compare_equal_and_share_a_compiled_plan():
    from pylea import DPP, Bool, flip
    from pylea.compile.order import (
        AdaptiveOrderFinder, GreedyOrderFinder, SlicedBestOfOrderFinder,
    )

    @DPP
    def both() -> Bool:
        return flip(0.3) & flip(0.5)

    for make in (GreedyOrderFinder, AdaptiveOrderFinder, SlicedBestOfOrderFinder):
        assert make() == make() and hash(make()) == hash(make())
        assert both.compiled(order_finder=make()) is both.compiled(order_finder=make())
    assert SlicedBestOfOrderFinder(seed=1) != SlicedBestOfOrderFinder(seed=2)
    assert AdaptiveOrderFinder() != SlicedBestOfOrderFinder()


def test_slicing_never_aims_below_the_largest_leaf():
    import numpy as np

    from pylea import DPP, Enum, flip, observe
    from pylea.compile.lower import compile_program
    from pylea.compile.order import GreedyOrderFinder
    from pylea.simplify import simplify

    class W(Enum):
        elements = ["sun", "cloud", "rain", "snow"]

    rows = np.random.default_rng(1).dirichlet(np.ones(4), size=4)
    emits = [0.1, 0.4, 0.8, 0.6]
    step = W.cpt([W], {(w,): list(rows[i]) for i, w in enumerate(W.elements)})
    seen = list(np.random.default_rng(0).integers(0, 2, 30).astype(bool))

    @DPP
    def last_state() -> W:
        s = W.uniform()
        for k, o in enumerate(seen):
            if k > 0:
                s = step(s)
            u = s.match({w: flip(e) for w, e in zip(W.elements, emits)})
            observe(u if o else ~u)
        return s

    # The largest tensor is a table of 16 entries, which already exceeds 2**3, so slicing
    # to 2**3 could only multiply the work: 4,194,304 slices before the floor.
    plans, root = compile_program(simplify(last_state.program()), GreedyOrderFinder(),
                                  kappa=3)
    assert plans[root].peak_entries() == 16
    assert all(plan.n_slices == 1 for plan in plans.values())
