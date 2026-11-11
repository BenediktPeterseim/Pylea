"""Bodies: straight-line ops over integer wires.

A `Body` is a sequence of `Op`s in topological order over wires ``0..len(bits)-1``, where
``bits[w]`` is the width of wire `w`.  At most one op produces each wire.

`args` and `outputs` may repeat a wire and may name the same one.  A wire read by several
ops is fan-out, a wire missing from `outputs` is discarded, an output naming an input
returns it unchanged, and an output naming a wire twice returns it twice.  `inputs` may not
repeat: two input positions holding one wire would define the body only where its two
arguments agree, which is a constraint and belongs in an `EQUATE` op instead.

A `Body` is immutable and hashable, so equal bodies can be shared; `pylea.ir.program.intern`
is what shares them.

Build one with a `BodyBuilder`, which checks the arity and wire widths of every call:

    >>> from pylea.ir.prim import Prim
    >>> b = BodyBuilder()
    >>> x, y = b.fresh(), b.fresh()
    >>> (z,) = b.emit(Prim.and_(), (x, y))
    >>> body = b.finish(inputs=(x, y), outputs=(z,))
    >>> body.in_bits, body.out_bits, body.n_wires
    (2, 1, 3)

A call that does not match its callee raises `ArityError`:

    >>> b.emit(Prim.not_(), (x, y))
    Traceback (most recent call last):
      ...
    pylea.ir.body.ArityError: NOT takes 1 argument(s); got 2
"""
from __future__ import annotations

from dataclasses import dataclass

from pylea.ir.prim import Prim


@dataclass(frozen=True, slots=True)
class Op:
    """``outs = callee(args)``, where `callee` is a `Prim` or an index into `Program.bodies`."""

    callee: Prim | int
    args: tuple[int, ...]
    outs: tuple[int, ...]

    @property
    def is_call(self) -> bool:
        return not isinstance(self.callee, Prim)


@dataclass(frozen=True, slots=True)
class Body:
    """Straight-line ops over wires, with the wires forming its input and output interface."""

    bits: tuple[int, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    ops: tuple[Op, ...]

    @property
    def n_wires(self) -> int:
        return len(self.bits)

    @property
    def in_bits(self) -> int:
        return sum(self.bits[w] for w in self.inputs)

    @property
    def out_bits(self) -> int:
        return sum(self.bits[w] for w in self.outputs)

    @property
    def interface_bits(self) -> int:
        """``in_bits + out_bits``, the exponent of this body's size as a single matrix.

        `pylea.ir.inline.flatten_wide_calls` splices in any callee whose value here is too
        large to hold.  An output naming a wire twice counts twice:

            >>> from pylea.ir.prim import Prim
            >>> b = BodyBuilder()
            >>> x, y = b.fresh(), b.fresh()
            >>> (z,) = b.emit(Prim.and_(), (x, y))
            >>> b.finish(inputs=(x, y), outputs=(z, z)).interface_bits
            4
        """
        return self.in_bits + self.out_bits

    def callees(self) -> tuple[int, ...]:
        """The body indices this body calls, in order, one entry per call site."""
        return tuple(op.callee for op in self.ops if not isinstance(op.callee, Prim))


def prune_wires(body: Body) -> Body:
    """`body` with the wires nothing names dropped and the rest renumbered consecutively.

    Removing an op leaves its wires behind.  Two bodies differing only in such leftovers
    are the same body, and compare equal only once the leftovers are gone.
    """
    used = set(body.inputs) | set(body.outputs)
    for op in body.ops:
        used.update(op.args)
        used.update(op.outs)
    if len(used) == len(body.bits):
        return body

    renumber = {w: i for i, w in enumerate(w for w in range(len(body.bits)) if w in used)}
    r = renumber.__getitem__
    return Body(
        tuple(body.bits[w] for w in sorted(used)),
        tuple(r(w) for w in body.inputs),
        tuple(r(w) for w in body.outputs),
        tuple(Op(op.callee, tuple(r(w) for w in op.args), tuple(r(w) for w in op.outs))
              for op in body.ops),
    )


class ArityError(ValueError):
    """Raised for a call whose arity or wire widths do not match its callee."""


def signature(callee: Prim | int, bodies) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The input and output wire widths of `callee`.

    A primitive takes and returns one wire per bit; a body uses the widths in its own `bits`.

        >>> from pylea.ir.prim import Prim
        >>> signature(Prim.and_(), bodies=())
        ((1, 1), (1,))
    """
    if isinstance(callee, Prim):
        return (1,) * callee.in_bits, (1,) * callee.out_bits
    b = bodies[callee]
    return tuple(b.bits[w] for w in b.inputs), tuple(b.bits[w] for w in b.outputs)


class BodyBuilder:
    """Incremental construction of a `Body`.

    `fresh` allocates a wire, `emit` records a call and returns its output wires, and
    `finish` produces the `Body`.  Arity and wire widths are checked as ops are emitted, so
    a `Body` obtained from a builder is well-formed.  See the module docstring for a
    worked example.

    `bodies` is the table calls are resolved against, needed to read a called body's
    signature.  Primitives need no table.
    """

    __slots__ = ("bits", "ops", "_bodies")

    def __init__(self, bodies=()) -> None:
        self.bits: list[int] = []
        self.ops: list[Op] = []
        self._bodies = bodies          # needed to resolve the signature of a called body

    def fresh(self, bits: int = 1) -> int:
        """An unused wire of width `bits`, now allocated."""
        if bits < 0:
            raise ValueError(f"wire width must be non-negative; got {bits}")
        self.bits.append(bits)
        return len(self.bits) - 1

    def emit(self, callee: Prim | int, args) -> tuple[int, ...]:
        """Fresh wires for the outputs of ``callee(args)``, which is now recorded.

        Raises `ArityError` if `args` does not match `callee`'s arity or wire widths.
        """
        args = tuple(args)
        in_w, out_w = signature(callee, self._bodies)
        if len(args) != len(in_w):
            raise ArityError(f"{_name(callee)} takes {len(in_w)} argument(s); got {len(args)}")
        for i, (w, want) in enumerate(zip(args, in_w)):
            if not 0 <= w < len(self.bits):
                raise ArityError(f"{_name(callee)}: argument {i} is not a wire of this body")
            if self.bits[w] != want:
                raise ArityError(
                    f"{_name(callee)}: argument {i} is {self.bits[w]} bits, expected {want}"
                )
        outs = tuple(self.fresh(w) for w in out_w)
        self.ops.append(Op(callee, args, outs))
        return outs

    def finish(self, inputs, outputs) -> Body:
        """The `Body` of the ops emitted so far, with the given interface wires.

        Raises `ArityError` if a wire is not of this body, if an input is also produced by
        an op, if a wire is read without being produced, or if `inputs` repeats a wire.
        """
        inputs, outputs = tuple(inputs), tuple(outputs)
        for label, ws in (("input", inputs), ("output", outputs)):
            for w in ws:
                if not 0 <= w < len(self.bits):
                    raise ArityError(f"{label} {w} is not a wire of this body")
        produced = {w for op in self.ops for w in op.outs}
        for w in inputs:
            if w in produced:
                raise ArityError(f"input wire {w} is also produced by an op")
        for w in {w for op in self.ops for w in op.args} | set(outputs):
            if w not in produced and w not in inputs:
                raise ArityError(f"wire {w} is read but neither an input nor produced")
        if len(set(inputs)) != len(inputs):
            raise ArityError(f"input wire repeated in {inputs}; use an EQUATE op instead")
        return Body(tuple(self.bits), inputs, outputs, tuple(self.ops))


def _name(callee: Prim | int) -> str:
    return callee.tag.name if isinstance(callee, Prim) else f"body#{callee}"
