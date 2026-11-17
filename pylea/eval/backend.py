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

# The band a magnitude may stay inside without the array being rescaled.  Chosen, not
# fitted: the product of two magnitudes inside the band, summed over up to ``2**100`` terms,
# stays inside the range of float64, and most arrays never leave the band, which saves a
# pass over each.
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
    # The log of the smallest normal magnitude of the dtype: a product below it may have
    # lost precision or become zero, also on a kernel that flushes subnormal numbers.
    log_underflow_unit: float

    def leaf(self, host: np.ndarray, log_scale: float = 0.0) -> tuple[Any, Any]: ...
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
    def log_min_positive(self, t) -> float: ...
    def finish(self, t, scale) -> tuple[np.ndarray, float]: ...

    # The magnitude, in whatever form the backend prefers.
    def scale_zero(self) -> Any: ...
    def scale_add(self, a, b) -> Any: ...
    def scale_float(self, s) -> float: ...
    def renormalise(self, t, scale) -> tuple[Any, Any, float, float | None]: ...
    def align(self, ta, sa, tb, sb) -> tuple[Any, Any, Any, float, float]: ...


class NumpyBackend:
    """Flat float64 arrays on the CPU, against which every other backend is checked."""

    name = "numpy"
    device = "cpu"
    max_rank = 32
    exact = True                    # float64, so a result needs no cross-check

    def __init__(self) -> None:
        self.dtype = np.float64
        self.log_underflow_unit = math.log(float(np.finfo(self.dtype).tiny))

    def leaf(self, host: np.ndarray, log_scale: float = 0.0):
        """A primitive's matrix, already brought into the band, as a flat array, and its
        log scale."""
        return np.ascontiguousarray(host, dtype=self.dtype).reshape(-1), log_scale

    def asarray(self, a) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=self.dtype).reshape(-1)

    def zeros(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=self.dtype)

    def ones(self, n: int) -> np.ndarray:
        return np.ones(n, dtype=self.dtype)

    def indices(self, a) -> np.ndarray:
        """A flat integer index array, in whatever form `gather` and `scatter` want."""
        return np.asarray(a, dtype=np.intp).reshape(-1)

    def view(self, t, shape) -> np.ndarray:
        return t.reshape(tuple(shape))

    def flatten(self, t) -> np.ndarray:
        return t.reshape(-1)

    def permute(self, t, perm) -> np.ndarray:
        return np.transpose(t, tuple(perm))

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

    def log_min_positive(self, t) -> float:
        """The log of the smallest positive entry of `t`, infinity where it has none."""
        positive = t[t > 0.0]
        return math.log(float(positive.min())) if positive.size else math.inf

    def finish(self, t, scale) -> tuple[np.ndarray, float]:
        """The result as a numpy array, which the caller owns, and its log scale as a
        float."""
        array = np.asarray(t)
        if not array.flags.writeable or not array.flags.owndata:
            array = array.copy()
        return array, float(scale)

    def scale_zero(self) -> float:
        return 0.0

    def scale_add(self, a, b):
        return a + b

    def scale_float(self, s) -> float:
        return float(s)

    def renormalise(self, t, scale):
        """``(t, scale, log_max, removed)``: `t` brought inside the band, the log of what
        was divided out, `removed`, added to `scale`, and the log of the largest entry of
        the `t` returned.

        The largest entry is the magnitude, every matrix a plan holds being non-negative and
        contraction only adding and multiplying such entries.  An array of zeros has no
        magnitude, so its scale is minus infinity, which no other array's scale loses to
        when the two are added.
        """
        m = float(t.max()) if t.size else 0.0
        if m == 0.0:
            return t, -math.inf, -math.inf, 0.0
        if not (LO <= m <= HI):
            return t / m, scale + math.log(m), 0.0, math.log(m)
        return t, scale, math.log(m), 0.0

    def align(self, ta, sa, tb, sb):
        """``(ta, tb, scale, log_fa, log_fb)``: the two arrays multiplied by the factors
        whose logs are `log_fa` and `log_fb`, to the common `scale`, for adding."""
        s = max(sa, sb)
        if s == -math.inf:
            return ta, tb, s, -math.inf, -math.inf
        la, lb = sa - s, sb - s
        fa = math.exp(la) if la != -math.inf else 0.0
        fb = math.exp(lb) if lb != -math.inf else 0.0
        return ta * fa, tb * fb, s, la, lb


class ExtendedRangeBackend(NumpyBackend):
    """Flat arrays of float64 numbers that each carry an exponent of their own.

    Entry ``x`` is held as ``m + e*1j``, a complex number whose real part is the mantissa
    `m`, in ``[0.5, 1)`` or 0, and whose imaginary part is the integer `e` with
    ``x = m * 2**e``.  Products multiply mantissas and add exponents, and sums bring
    their terms to the largest exponent first, so every operation keeps the relative
    precision of float64 while no entry can leave the range of a float, however large or
    small it is.  Reshaping, permuting, gathering and scattering act on the complex
    arrays unchanged.  The price is several times the arithmetic of `NumpyBackend`;
    `pylea.eval.interp.evaluate_certified` uses it where float64 cannot vouch for a
    result.

    The product of two entries of 1e-300, 1e-600, keeps its value, and `finish` returns
    it as an array and a log scale:

        >>> B = ExtendedRangeBackend()
        >>> t, scale = B.leaf(np.array([1e-300, 2e-300]), 0.0)
        >>> one = B.view(B.ones(1), (1, 1, 1))
        >>> product = B.matmul(B.view(t, (2, 1, 1)), B.view(t, (2, 1, 1)))
        >>> array, scale = B.finish(B.flatten(product), scale)
        >>> array.round(12).tolist(), round(scale, 3)
        ([0.25, 1.0], -1380.165)
    """

    name = "numpy-extended"
    exact = True
    # A product never underflows here, so no contraction is ever in danger, and a
    # primitive's matrix is taken as it is.
    log_underflow_unit = -math.inf
    keeps_every_range = True
    def __init__(self) -> None:
        self.dtype = np.complex128

    def leaf(self, host: np.ndarray, log_scale: float = 0.0):
        """A primitive's matrix with the factor ``exp(log_scale)`` folded into its
        exponents, and the scale 0 that remains."""
        exponent, fraction = divmod(log_scale / math.log(2.0), 1.0)
        mantissa = np.ascontiguousarray(host, dtype=np.float64).reshape(-1) * 2.0 ** fraction
        return _joined(mantissa, np.full(mantissa.shape, exponent)), 0.0

    def asarray(self, a) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.complex128).reshape(-1)

    def zeros(self, n: int) -> np.ndarray:
        return np.zeros(n, dtype=np.complex128)

    def ones(self, n: int) -> np.ndarray:
        return np.full(n, 0.5 + 1.0j)

    def matmul(self, a, b) -> np.ndarray:
        """The batched product, exact to float64 rounding in every entry.

        Each row of `a` and each column of `b` is scaled by its own largest power of two,
        and the scaled operands are multiplied in float64.  A product of two scaled
        entries is lost only below the smallest normal number, so an output entry far
        above ``inner`` times that number is accurate to rounding; only the other
        entries, which are rare, are summed term by term with their exponents.
        """
        ma, ea = _parts(a)
        mb, eb = _parts(b)
        row = ea.max(axis=2, keepdims=True)
        column = eb.max(axis=1, keepdims=True)
        row = np.where(np.isfinite(row), row, 0.0)
        column = np.where(np.isfinite(column), column, 0.0)
        product = np.matmul(_shifted(ma, ea - row), _shifted(mb, eb - column))
        exponent = row + column                         # (batch, left, right)
        inner = a.shape[2]
        trusted = product >= inner * _TRUSTED_ABOVE
        out = _joined(product, exponent)
        k, i, j = np.nonzero(~trusted)
        rows = max(1, _CHUNK // max(1, inner))
        for start in range(0, k.size, rows):
            kk, ii, jj = k[start:start + rows], i[start:start + rows], j[start:start + rows]
            out[kk, ii, jj] = _summed(ma[kk, ii, :] * mb[kk, :, jj],
                                      ea[kk, ii, :] + eb[kk, :, jj], axis=1)
        return out

    def sum_axis(self, t, axis: int) -> np.ndarray:
        return _summed(*_parts(t), axis=axis)

    def add(self, a, b):
        ma, ea = _parts(a)
        mb, eb = _parts(b)
        return _summed(np.stack([ma, mb]), np.stack([ea, eb]), axis=0)

    def log_min_positive(self, t) -> float:
        return 0.0

    def finish(self, t, scale) -> tuple[np.ndarray, float]:
        mantissa, exponent = _parts(t)
        top = float(exponent.max()) if exponent.size else -math.inf
        if top == -math.inf:
            return np.zeros(t.shape), -math.inf
        array = _shifted(mantissa, exponent - top)
        peak = float(array.max())
        return array / peak, float(scale) + top * math.log(2.0) + math.log(peak)

    def renormalise(self, t, scale):
        return t, scale, 0.0, 0.0

    def align(self, ta, sa, tb, sb):
        return ta, tb, sa, 0.0, 0.0


# Terms summed one by one per chunk in `ExtendedRangeBackend.matmul`: ``2**22`` float64.
_CHUNK = 1 << 22

# A float64 sum at least this far above the smallest normal number, per term, loses less
# than its own rounding to products that fell below that number: ``2**-1022 / 2**-53``.
_TRUSTED_ABOVE = 2.0 ** -969


def _parts(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mantissas of `t` and their exponents, minus infinity for zero entries."""
    mantissa = t.real
    return mantissa, np.where(mantissa == 0.0, -math.inf, t.imag)


def _joined(mantissa: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    """``mantissa * 2**exponent`` in the form of `ExtendedRangeBackend`."""
    m, shift = np.frexp(mantissa)
    exponent = np.where(m == 0.0, 0.0, exponent + shift)
    return m + 1j * exponent


def _shifted(mantissa: np.ndarray, by: np.ndarray) -> np.ndarray:
    """``mantissa * 2**by`` for ``by <= 0``, zero where `by` is minus infinity."""
    # 2**-1100 is below every float64, so a lower power changes nothing.
    by = np.maximum(np.where(np.isfinite(by), by, -1100.0), -1100.0)
    return np.ldexp(mantissa, by.astype(np.int64))


def _summed(mantissa: np.ndarray, exponent: np.ndarray, axis: int) -> np.ndarray:
    """The sum along `axis` of ``mantissa * 2**exponent``, in extended form."""
    top = exponent.max(axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    total = _shifted(mantissa, exponent - top).sum(axis=axis)
    return _joined(total, np.squeeze(top, axis=axis))


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
        self.log_underflow_unit = math.log(float(torch.finfo(dtype).tiny))

    def leaf(self, host: np.ndarray, log_scale: float = 0.0):
        """A primitive's matrix, already brought into the band, as a flat tensor, and its
        log scale as this backend holds scales."""
        return self.asarray(host), self.scale_add(self.scale_zero(), log_scale)

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

    def view(self, t, shape):
        return t.reshape(tuple(shape))

    def flatten(self, t):
        return t.reshape(-1)

    def permute(self, t, perm):
        return t.permute(tuple(perm))

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

    def log_min_positive(self, t) -> float:
        """The log of the smallest positive entry of `t`, infinity where it has none; one
        read from the device."""
        positive = t[t > 0]
        return math.log(float(positive.min())) if positive.numel() else math.inf

    def finish(self, t, scale) -> tuple[np.ndarray, float]:
        """The result as a numpy array, which the caller owns, and its log scale as a
        float."""
        return np.array(self.to_numpy(t), dtype=np.float64), float(scale)

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
        """``(t, scale, log_max, removed)`` as `NumpyBackend.renormalise` gives them.

        On the CPU this tests the band as `NumpyBackend.renormalise` does.  On an
        accelerator it rescales at every step instead, since testing the band means reading
        the magnitude back to the host; the module docstring gives the trade.  There
        `log_max` is 0, an upper bound once the largest entry has been divided out,
        `removed` is `None`, unknown to the host, and an all-zero tensor keeps its scale:
        telling it apart would need a read from the device, and its scale is accounted
        for where tensors are added, see `align`.
        """
        if not t.numel():
            return t, scale, -math.inf, 0.0
        if self.host_reads_are_free:
            m = float(t.max())
            if m == 0.0:
                return t, scale - math.inf, -math.inf, 0.0
            # The band suits float64; a narrower dtype is rescaled at every step, since
            # the band reaches beyond its range.
            if not (self.exact and LO <= m <= HI):
                return t / m, scale + math.log(m), 0.0, math.log(m)
            return t, scale, math.log(m), 0.0
        m = t.max()
        # `where` keeps the branch on the device.
        safe = self.torch.where(m > 0, m, self.torch.ones_like(m))
        return t / safe, scale + self.torch.log(safe.to(scale.dtype)), 0.0, None

    def align(self, ta, sa, tb, sb):
        """``(ta, tb, scale, log_fa, log_fb)`` as `NumpyBackend.align` gives them; the two
        log factors are read from the device, once per sum of slices."""
        s = self.torch.maximum(sa, sb)
        neg = self.torch.tensor(-math.inf, dtype=s.dtype, device=s.device)
        if float(s) == -math.inf:
            return ta, tb, s, -math.inf, -math.inf
        la = self.torch.where(sa == neg, neg, sa - s)
        lb = self.torch.where(sb == neg, neg, sb - s)
        return (ta * self.torch.exp(la).to(ta.dtype), tb * self.torch.exp(lb).to(tb.dtype),
                s, float(la), float(lb))


def _pick_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# The fixed overhead of one plan step, per device, in microseconds.  These are launch floors
# rather than throughputs: a device pays them whatever the step computes.  The CPU figure is
# `pylea.compile.cost.MS_PER_STEP`.  The MPS figure was timed per step on the MPS device of
# an Apple M3 Pro; not reproducible from this repository.  The CUDA figure is chosen, not
# measured.  Re-measure after changing `pylea.eval` or torch.
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
# MPS device of an Apple M3 Pro over batched matrix multiplication and transposes, against
# numpy on its CPU, taking the low end of the range so that `worth_a_device` errs towards
# the CPU; not reproducible from this repository.  Re-measure after changing numpy or
# torch.
_SPEEDUP = 8.0


def is_cpu_device(device: str | None) -> bool:
    """Whether `device` runs on the CPU or leaves the choice to `select`.

    ``None`` and ``"auto"`` leave it open, ``"numpy"`` and ``"cpu"`` name the CPU; any
    other name is an accelerator pin, which `select` hands to torch.

        >>> is_cpu_device("auto"), is_cpu_device("cpu"), is_cpu_device("mps")
        (True, True, False)
    """
    return device in (None, "auto", "numpy", "cpu")


_DEVICE_NAMES = ("'auto', 'numpy', or a torch device such as 'cpu', 'cuda', 'cuda:1' "
                 "or 'mps'")


def _torch_device(device):
    """`device` as a `torch.device`, or the errors `select` documents."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"device={device!r} evaluates with torch, which is not installed; install it "
            "with pip install 'pylea[gpu]', or pass device='numpy'") from None
    try:
        return torch.device(device)
    except (RuntimeError, TypeError):
        raise ValueError(f"device must be {_DEVICE_NAMES}; got {device!r}") from None


def select(device: str | None = None, *, steps: float | None = None,
           ops: float | None = None) -> Backend:
    """The backend to evaluate on.

    `device` names one outright, either ``"numpy"`` or a torch device such as ``"cpu"``,
    ``"cuda"``, ``"cuda:1"`` or ``"mps"``.  Otherwise the choice follows `worth_a_device`
    from `steps` and `ops`, and is numpy where those are not given, an accelerator never
    being chosen merely for being present.

    A torch device without torch installed raises `ImportError` naming the extra that
    installs it, and a name torch does not accept, such as ``"gpu"``, `ValueError`
    listing the accepted ones.
    """
    if device is not None and device != "auto":
        if device == "numpy":
            return NumpyBackend()
        return TorchBackend(_torch_device(device))
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
