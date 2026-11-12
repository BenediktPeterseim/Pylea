"""Array backends, and the devices they run on.

A `Backend` supplies construction, reshaping, permutation, batched matrix multiplication,
reduction, gather and scatter over flat arrays, plus the arithmetic of the magnitude carried
beside each one.  `NumpyBackend` is float64 on the CPU; `TorchBackend` is the same
operations on any torch device.  Neither inherits from the other, and a differential test is
what keeps them in step.

Arrays are one-dimensional, their logical shape living in the plan, so a reshape moves
nothing and only a permutation touches memory.  `max_rank` is the highest number of axes a
view may have, which `pylea.eval.kernels` reads when splitting a permutation into passes.

Every array carries its magnitude as a natural logarithm, so that a probability far below
the smallest normal float survives.  Whether that is a Python float or a scalar on the
device is the backend's business: the evaluator asks only for `scale_zero`, `scale_add`,
`renormalise` and `align`, and converts to a float once, at the end.  Reading a scalar back
from an accelerator forces a synchronisation, so doing it per step would cost a large part
of the evaluation.  `renormalise` differs between the two for that reason: where reading the
magnitude is free the array is rescaled only on leaving the band `LO` to `HI`, and where it
is not the array is rescaled at every step, which also keeps float32 usable.

A backend is passed to whatever uses one rather than set at module level, two evaluations in
one process being free to target different devices.

Import numpy before torch.  Both link an OpenMP runtime, and on a mixed conda and pip
installation loading torch first aborts the process.  This module imports numpy at the top
and torch only inside `TorchBackend`.

`select` chooses one.  Given a device name it returns that; given a plan's step and
operation counts it weighs the arithmetic an accelerator saves per step against the launch
overhead it adds, and tests the most forgiving launch floor of any device before importing
torch, so a plan that could not pay never loads it.
"""
from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

# The band a magnitude may stay inside without the array being rescaled.  Wide enough that
# no single contraction crosses it undetected, and wide enough that most arrays never leave
# it, which saves a pass over each.
LO = 1e-100
HI = 1e100


@runtime_checkable
class Backend(Protocol):
    """What the evaluator needs of an array library."""

    name: str
    device: Any                     # where arrays live: a string or a torch.device
    dtype: Any                      # the array dtype evaluation runs in
    max_rank: int
    exact: bool                     # False where the dtype is narrower than float64
    tiny: float                     # the smallest normal magnitude the dtype represents

    def asarray(self, a) -> Any: ...
    def zeros(self, n: int) -> Any: ...
    def ones(self, n: int) -> Any: ...
    def indices(self, a) -> Any: ...
    def view(self, t, shape) -> Any: ...
    def flatten(self, t) -> Any: ...
    def permute(self, t, perm) -> Any: ...
    def matmul(self, a, b) -> Any: ...
    def sum_axis(self, t, axis: int) -> Any: ...
    def take(self, t, axis: int, index: int) -> Any: ...
    def gather(self, t, index) -> Any: ...
    def scatter(self, index, values, size: int) -> Any: ...
    def add(self, a, b) -> Any: ...
    def to_numpy(self, t) -> np.ndarray: ...

    # -- the magnitude, in whatever form this backend prefers -------------------------
    def scale_zero(self) -> Any: ...
    def scale_add(self, a, b) -> Any: ...
    def scale_float(self, s) -> float: ...
    def renormalise(self, t, scale) -> tuple[Any, Any]: ...
    def align(self, ta, sa, tb, sb) -> tuple[Any, Any, Any]: ...


class NumpyBackend:
    """Flat float64 arrays on the CPU, against which every other backend is checked."""

    name = "numpy"
    device = "cpu"
    max_rank = 32
    exact = True                    # float64, so a result needs no cross-check

    def __init__(self) -> None:
        self.dtype = np.float64
        self.tiny = float(np.finfo(self.dtype).tiny)

    # -- construction ---------------------------------------------------------------

    def asarray(self, a) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=self.dtype).reshape(-1)

    def zeros(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=self.dtype)

    def ones(self, n: int) -> np.ndarray:
        return np.ones(n, dtype=self.dtype)

    def indices(self, a) -> np.ndarray:
        """A flat integer index array, in whatever form `gather` and `scatter` want."""
        return np.asarray(a, dtype=np.intp).reshape(-1)

    # -- shape ----------------------------------------------------------------------

    def view(self, t, shape) -> np.ndarray:
        return t.reshape(tuple(shape))

    def flatten(self, t) -> np.ndarray:
        return t.reshape(-1)

    def permute(self, t, perm) -> np.ndarray:
        return np.transpose(t, tuple(perm))

    # -- arithmetic -----------------------------------------------------------------

    def matmul(self, a, b) -> np.ndarray:
        return np.matmul(a, b)

    def sum_axis(self, t, axis: int) -> np.ndarray:
        return t.sum(axis=axis)

    def take(self, t, axis: int, index: int) -> np.ndarray:
        return np.take(t, index, axis=axis)

    def gather(self, t, index) -> np.ndarray:
        """The entries of flat `t` at the flat positions `index`."""
        return t[index]

    def scatter(self, index, values, size: int) -> np.ndarray:
        """A flat array of `size` holding `values` at the flat positions `index`, else zero."""
        out = self.zeros(size)
        out[index] = values
        return out

    def add(self, a, b):
        return a + b

    def to_numpy(self, t) -> np.ndarray:
        return np.asarray(t)

    # -- magnitude ------------------------------------------------------------------

    def scale_zero(self) -> float:
        return 0.0

    def scale_add(self, a, b):
        return a + b

    def scale_float(self, s) -> float:
        return float(s)

    def renormalise(self, t, scale):
        """``(t, scale)`` with `t` brought inside the band and what was removed added to
        `scale`.

        The largest entry is the magnitude, every matrix a plan holds being non-negative and
        contraction only adding and multiplying such entries.
        """
        m = float(t.max()) if t.size else 0.0
        if m > 0.0 and not (LO <= m <= HI):
            return t / m, scale + math.log(m)
        return t, scale

    def align(self, ta, sa, tb, sb):
        """``(ta, tb, scale)``, the two arrays rescaled to a common magnitude for adding."""
        s = max(sa, sb)
        if s == -math.inf:
            return ta, tb, sa
        fa = math.exp(sa - s) if sa != -math.inf else 0.0
        fb = math.exp(sb - s) if sb != -math.inf else 0.0
        return ta * fa, tb * fb, s


class TorchBackend:
    """The same operations on a torch device, so that a plan runs unchanged on any of them.

    `device` defaults to the fastest available and `dtype` to float64, or to float32 on MPS,
    which has no float64 at all.  `exact` reports which of the two was chosen.
    """

    name = "torch"

    def __init__(self, device: str | None = None, dtype=None) -> None:
        import torch                            # imported here; see the module docstring

        self.torch = torch
        self.device = torch.device(device if device is not None else _pick_device(torch))
        # Reading a scalar back from an accelerator forces a synchronisation, where on the
        # CPU there is nothing to synchronise.  This decides how `renormalise` may work.
        self.host_reads_are_free = self.device.type == "cpu"
        # MPS refuses a tensor of more than 16 dimensions, where numpy allows 32.  This is
        # what `kernels.permute` reads to split a permutation into several passes, so a wrong
        # value here is a hard failure on wide bodies rather than a slow path.
        self.max_rank = 16 if self.device.type == "mps" else 32
        # MPS has no float64 at all, refusing the conversion rather than narrowing it, so a
        # plan running there is float32 and `exact` reports that its result is unchecked.
        if dtype is None:
            dtype = torch.float32 if self.device.type == "mps" else torch.float64
        self.dtype = dtype
        self.exact = dtype == torch.float64
        self.tiny = float(torch.finfo(dtype).tiny)

    # -- construction ---------------------------------------------------------------

    def asarray(self, a):
        if isinstance(a, self.torch.Tensor):
            return a.to(device=self.device, dtype=self.dtype).reshape(-1)
        a = np.ascontiguousarray(a)
        if not a.flags.writeable:
            # A `Prim`'s matrix is shared and read-only.  torch would wrap it without copying
            # and calls writing through such a tensor undefined.  Leaves are small.
            a = a.copy()
        return self.torch.as_tensor(a, dtype=self.dtype, device=self.device).reshape(-1)

    def zeros(self, n: int):
        return self.torch.zeros(n, dtype=self.dtype, device=self.device)

    def ones(self, n: int):
        return self.torch.ones(n, dtype=self.dtype, device=self.device)

    def indices(self, a):
        return self.torch.as_tensor(np.asarray(a).reshape(-1), dtype=self.torch.long,
                                    device=self.device)

    # -- shape ----------------------------------------------------------------------

    def view(self, t, shape):
        return t.reshape(tuple(shape))

    def flatten(self, t):
        return t.reshape(-1)

    def permute(self, t, perm):
        return t.permute(tuple(perm))

    # -- arithmetic -----------------------------------------------------------------

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def sum_axis(self, t, axis: int):
        return t.sum(dim=axis)

    def take(self, t, axis: int, index: int):
        return t.select(axis, index)

    def gather(self, t, index):
        return t[index]

    def scatter(self, index, values, size: int):
        out = self.zeros(size)
        out[index] = values
        return out

    def add(self, a, b):
        return a + b

    def to_numpy(self, t) -> np.ndarray:
        return t.detach().to("cpu").numpy()

    # -- magnitude ------------------------------------------------------------------

    def scale_zero(self):
        # The scale shares the array's dtype, MPS refusing float64 for anything at all.  A
        # float32 log scale resolves a magnitude more coarsely than float32 itself resolves a
        # value, which is harmless: the scale is common to every entry of the matrix and so
        # cancels exactly in any ratio taken from it, which is what a probability is.  `exact`
        # is already False on such a device.
        return self.torch.zeros((), dtype=self.dtype, device=self.device)

    def scale_add(self, a, b):
        return a + b

    def scale_float(self, s) -> float:
        return float(s)

    def renormalise(self, t, scale):
        """``(t, scale)`` with `t` rescaled and what was removed added to `scale`.

        On the CPU this tests the band as `NumpyBackend.renormalise` does.  On an
        accelerator it rescales at every step instead, since testing the band means reading
        the magnitude back to the host; the module docstring gives the trade.
        """
        if not t.numel():
            return t, scale
        if self.host_reads_are_free:
            m = float(t.max())
            if m > 0.0 and not (LO <= m <= HI):
                return t / m, scale + math.log(m)
            return t, scale
        m = t.max()
        # An all-zero intermediate has magnitude 0 and must not divide, so it and its scale
        # are left alone.  `where` keeps the branch on the device.
        safe = self.torch.where(m > 0, m, self.torch.ones_like(m))
        return t / safe, scale + self.torch.log(safe.to(scale.dtype))

    def align(self, ta, sa, tb, sb):
        """``(ta, tb, scale)``, the two tensors rescaled to a common magnitude for adding.

        Without the negative-infinity case `NumpyBackend.align` carries: a scale here only
        grows by the log of a magnitude `renormalise` has already found positive.
        """
        s = self.torch.maximum(sa, sb)
        return (ta * self.torch.exp(sa - s).to(ta.dtype),
                tb * self.torch.exp(sb - s).to(tb.dtype), s)


def _pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# The fixed overhead of one plan step, per device, in microseconds.  These are launch floors
# rather than throughputs: a device pays them whatever the step computes.  The CPU figure is
# `pylea.compile.cost.MS_PER_STEP` and the MPS figure was timed on an Apple M3 Pro; the
# CUDA figure is not measured.
LAUNCH_US = {"cpu": 3.5, "mps": 40.0, "cuda": 10.0}

# The lowest launch floor of any accelerator, used to ask whether a device could pay at all
# before spending seconds importing torch to find out which one is present.
_BEST_FLOOR = min((d for d in LAUNCH_US if d != "cpu"), key=lambda d: LAUNCH_US[d])


def worth_a_device(steps: float, ops: float, device: str = "mps") -> bool:
    """Whether a plan of `steps` steps doing `ops` multiplications should leave the CPU.

    An accelerator is faster per unit of arithmetic and slower per step, so it pays where
    the average step carries enough arithmetic to outweigh the extra launch cost.  A
    threshold on total predicted evaluation would not answer this: a long evaluation made of
    many small steps still belongs on the CPU.

        >>> worth_a_device(steps=1000, ops=1e4)     # many tiny steps
        False
        >>> worth_a_device(steps=10, ops=1e11)      # a few enormous ones
        True

    A step count past the float range, which `pylea.compile.cost.step_count` reports as
    infinite, belongs to a plan no device finishes, so it stays on the CPU rather than
    loading an accelerator for nothing.
    """
    if steps <= 0 or math.isinf(steps):
        return False
    floor = LAUNCH_US.get(device, 40.0) - LAUNCH_US["cpu"]
    if floor <= 0.0:
        return True
    # Arithmetic the accelerator saves per step, in microseconds, against what it adds.
    from pylea.compile.cost import OPS_PER_MS

    per_step_us = (ops / steps) / OPS_PER_MS * 1e3
    return per_step_us * (1.0 - 1.0 / _SPEEDUP) > floor


# How much faster an accelerator's arithmetic is once it is not launch-bound.  Timed on the
# MPS device of an Apple M3 Pro over batched matrix multiplication and transposes, taking the
# low end of the range so that `worth_a_device` errs towards the CPU.
_SPEEDUP = 8.0


def is_cpu_device(device: str | None) -> bool:
    """Whether `device` runs on the CPU or leaves the choice to `select`.

    ``None`` and ``"auto"`` leave it open, ``"numpy"`` and ``"cpu"`` name the CPU; any
    other name is an accelerator pin, which `select` hands to torch.

        >>> is_cpu_device("auto"), is_cpu_device("cpu"), is_cpu_device("mps")
        (True, True, False)
    """
    return device in (None, "auto", "numpy", "cpu")


def select(device: str | None = None, *, steps: float | None = None,
           ops: float | None = None) -> Backend:
    """The backend to evaluate on.

    `device` names one outright, either ``"numpy"`` or anything torch accepts.  Otherwise
    the choice follows `worth_a_device` from `steps` and `ops`, and is numpy where those are
    not given, an accelerator never being chosen merely for being present.
    """
    if device is not None and device != "auto":
        return NumpyBackend() if device == "numpy" else TorchBackend(device)
    if steps is None or ops is None:
        return NumpyBackend()
    # Importing torch costs seconds, so ask the cheapest question first: could any accelerator
    # pay here?  A plan failing this against the lowest launch floor of any device could not
    # pass for a real one either, so most instances never load torch at all.
    if not worth_a_device(steps, ops, _BEST_FLOOR):
        return NumpyBackend()
    try:
        import torch
    except ImportError:
        return NumpyBackend()
    picked = _pick_device(torch)
    if picked != "cpu" and worth_a_device(steps, ops, picked):
        return TorchBackend(picked)
    return NumpyBackend()
