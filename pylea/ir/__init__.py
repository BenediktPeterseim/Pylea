"""The intermediate representation of DPPs: programs, bodies, ops and primitives.

A `Program` is a DPP, held as a table of `Body`s with a root.  A `Body` is one definition
of the DPP, and the root body is its main function.  A `Body` is straight-line `Op`s over
integer wires.  An `Op` is an assignment: it calls either a primitive, a `Prim`, which
carries a dense matrix, or another body by index.  A wire is a variable, of the width
``bits[w]``.  A primitive reads and writes one wire per bit, and `pylea.syntax` allocates
one-bit wires, so a variable of several bits is there several wires.

`pylea.ir.program` measures and interns DPPs, `pylea.ir.inline` splices a call site, and
`pylea.ir.dependency` says which interface wires of a body are computed from which.
`pylea.ir.bits` reads the members of a set of integers held as the bits of an integer.
"""
from pylea.ir.body import ArityError, Body, BodyBuilder, Op, prune_wires, signature
from pylea.ir.dependency import direct_dependencies
from pylea.ir.prim import (
    BOOLEAN, CONDITIONING, STOCHASTIC, Prim, Tag, is_indicator, is_stochastic,
)
from pylea.ir.program import (
    Program, consumer_counts, deterministic, discardable, intern, intern_with_remap,
    multi_parent_wires, reachable, shared_reads, op_count, use_counts, validate,
    wire_slots,
)

__all__ = [
    "ArityError", "Body", "BodyBuilder", "Op", "prune_wires", "signature",
    "direct_dependencies",
    "Prim", "Tag", "BOOLEAN", "CONDITIONING", "STOCHASTIC", "is_indicator", "is_stochastic",
    "Program", "reachable", "use_counts", "op_count", "intern",
    "intern_with_remap", "validate",
    "consumer_counts", "multi_parent_wires", "shared_reads", "wire_slots",
    "discardable", "deterministic",
]
