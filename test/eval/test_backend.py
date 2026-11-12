"""The backends, against each other.

A backend is tested by requiring that a plan evaluated on it agrees with the same plan
evaluated on numpy, which the rest of the suite already checks against exact enumeration.
Asserting what each operation returns would restate the implementation instead, and anything
a backend gets wrong shows up as a disagreement on some program.

torch is optional, so every test needing it skips rather than fails where it is absent.
"""
import numpy as np
import pytest

from pylea.compile.lower import compile_program
from pylea.compile.order import GreedyOrderFinder
from pylea.eval.backend import NumpyBackend, TorchBackend, select, worth_a_device
from pylea.eval.interp import evaluate, underflowed
from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern
from pylea.ir.reference import TooLarge, semantics
from test.compile.random_programs import random_program


def _corpus(seed: int = 0, n: int = 30):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        prog = random_program(rng, n_bodies=int(rng.integers(1, 3)),
                              n_ops=int(rng.integers(1, 6)))
        try:
            semantics(prog)
        except TooLarge:
            continue
        out.append(intern(prog))
    return out


CORPUS = _corpus()


def _value(prog, backend):
    plans, root = compile_program(prog, GreedyOrderFinder(), kappa=28)
    array, scale = evaluate(plans, root, backend=backend)
    return np.asarray(array, dtype=float) * float(np.exp(scale))


def _torch_backend(device: str):
    pytest.importorskip("torch")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    return TorchBackend(device)


def test_the_default_backend_is_numpy_and_not_whatever_device_is_present():
    """Evaluation is reproducible unless a caller asks for a device.

    Worth pinning: wherever an MPS device is present, a default of `select()` that used it
    would silently make every evaluation float32.
    """
    plans, root = compile_program(CORPUS[0], GreedyOrderFinder(), kappa=28)
    default, scale = evaluate(plans, root)
    explicit, explicit_scale = evaluate(plans, root, backend=NumpyBackend())
    assert isinstance(default, np.ndarray)
    assert default.dtype == np.float64
    assert np.array_equal(default, explicit) and scale == explicit_scale


def test_select_reports_a_device_and_a_dtype():
    backend = select()
    assert backend.device is not None
    assert backend.dtype is not None
    assert isinstance(backend.exact, bool)


def test_numpy_is_the_backend_asked_for_by_name():
    assert isinstance(select("numpy"), NumpyBackend)
    assert select("numpy").exact


@pytest.mark.parametrize("prog", CORPUS, ids=range(len(CORPUS)))
def test_torch_on_the_cpu_agrees_with_numpy_exactly(prog):
    """Both are float64 over the same plan, so anything but equality is a real difference."""
    backend = _torch_backend("cpu")
    assert np.array_equal(_value(prog, backend), _value(prog, None))


@pytest.mark.parametrize("prog", CORPUS, ids=range(len(CORPUS)))
def test_mps_agrees_with_numpy_to_float32(prog):
    backend = _torch_backend("mps")
    assert not backend.exact                    # MPS refuses float64 outright
    assert np.allclose(_value(prog, backend), _value(prog, None), rtol=1e-5, atol=1e-12)


def test_a_read_only_factor_is_copied_rather_than_wrapped():
    """torch calls writing through a wrapped read-only array undefined; leaves are shared."""
    backend = _torch_backend("cpu")
    a = np.ones(4, dtype=np.float64)
    a.flags.writeable = False
    t = backend.asarray(a)
    t += 1.0                                    # must not touch `a`
    assert np.array_equal(a, np.ones(4))


# ------------------------------------------------------- range, not precision

def _and_chain(n: int) -> Program:
    """`n` coins ANDed together, so the answer is exactly 0.25**n and can be dialled."""
    b = BodyBuilder()
    (w,) = b.emit(Prim.flip(0.25), ())
    for _ in range(n - 1):
        (p,) = b.emit(Prim.flip(0.25), ())
        (w,) = b.emit(Prim.and_(), (w, p))
    return intern(Program((b.finish((), (w,)),), 0))


def _p_true(prog, backend):
    plans, root = compile_program(prog, GreedyOrderFinder(), kappa=28)
    array, scale = evaluate(plans, root, backend=backend)
    return float(np.asarray(array, dtype=float).reshape(-1)[-1] * np.exp(scale))


@pytest.mark.parametrize("n", [5, 20, 40])
def test_float32_reaches_far_below_its_own_smallest_normal(n):
    """The per-array scale is what buys this, ``0.25**40`` being well under float32's floor."""
    backend = _torch_backend("mps")
    assert _p_true(_and_chain(n), backend) == pytest.approx(0.25 ** n, rel=1e-4)


def test_float32_underflows_where_no_scale_can_help_and_reports_it():
    """``0.25**80`` sits beside an entry of about 1, and no shared scale holds both.

    A silent zero on a positive probability is the worst failure this evaluator has, so both
    halves are pinned: that float32 gives zero here, and that `underflowed` sees it.
    """
    backend = _torch_backend("mps")
    plans, root = compile_program(_and_chain(80), GreedyOrderFinder(), kappa=28)
    array, _scale = evaluate(plans, root, backend=backend)
    assert underflowed(array)
    assert _p_true(_and_chain(80), NumpyBackend()) == pytest.approx(0.25 ** 80, rel=1e-6)


def test_a_healthy_result_is_not_flagged():
    plans, root = compile_program(_and_chain(20), GreedyOrderFinder(), kappa=28)
    assert not underflowed(evaluate(plans, root, backend=NumpyBackend())[0])


# ------------------------------------------------------------ device selection

def test_a_device_is_chosen_on_work_per_step_not_on_existing():
    """An accelerator pays its launch floor per step, so a plan of small steps must stay put."""
    assert not worth_a_device(steps=1000, ops=1e6, device="mps")      # 1e3 ops/step
    assert worth_a_device(steps=1000, ops=1e12, device="mps")         # 1e9 ops/step


def test_a_step_count_past_the_float_range_stays_on_the_cpu():
    """Such a plan cannot finish anywhere, so choosing it a device must neither raise nor
    load an accelerator."""
    inf = float("inf")
    assert not worth_a_device(steps=inf, ops=inf, device="mps")
    assert isinstance(select(steps=inf, ops=inf), NumpyBackend)


def test_no_device_is_chosen_without_plan_statistics():
    """`select()` knowing nothing must not silently narrow every evaluation to float32."""
    assert isinstance(select(), NumpyBackend)
