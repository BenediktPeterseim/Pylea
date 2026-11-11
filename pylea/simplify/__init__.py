"""Simplification: rewriting a DPP before it is lowered.

Every rewrite rule is a `Program -> Program` function that preserves the semantics of the
DPP and strictly lowers `pylea.simplify.driver.size`, the number of symbols in it.
`simplify` runs the rules to a fixpoint.

The rewrite rules, each in its own module:

  * `pylea.simplify.dce`: dead code elimination, and specialisation of a body to the
    outputs its callers read.
  * `pylea.simplify.inline`: inlining of a body called once.
  * `pylea.simplify.boolean`: equational simplifications by the Boolean identities.
  * `pylea.simplify.observe`: observation splitting and propagation, and the deletion of
    repeated observations.
  * `pylea.simplify.fold`: constant folding, and common subexpression elimination at
    the bottom of a body.
  * `pylea.simplify.cse`: common subexpression elimination over a private coin flip.
  * `pylea.simplify.absorb`: absorption by implication.
  * `pylea.simplify.lift`: flip lifting, of a coin flip read by several disjunctions.
  * `pylea.simplify.mask`: masked reads.

The removal of a body nothing calls, and the merging of equal bodies, happen in the loop
itself through `pylea.ir.program.intern`.

Wherever a rule is stated for a coin flip, it takes any primitive that reads nothing and
produces one bit, whatever its weights, as `pylea.simplify.common.is_flip` says, and
constant folding takes the determinism of a table off its matrix; this is what the opaque
tables of Bayesian networks need.
"""

from pylea.simplify.absorb import absorb
from pylea.simplify.boolean import bool_rules
from pylea.simplify.cse import flip_cse
from pylea.simplify.dce import dce
from pylea.simplify.driver import RULES, SizeDidNotDecrease, fixpoint, simplify, size
from pylea.simplify.fold import block_fold, bottom_cse, const_fold
from pylea.simplify.inline import inline
from pylea.simplify.lift import lift
from pylea.simplify.observe import observe_rules
from pylea.simplify.mask import mask

__all__ = ["simplify", "fixpoint", "size", "RULES", "SizeDidNotDecrease",
           "dce", "inline", "bool_rules", "observe_rules", "const_fold", "block_fold",
           "bottom_cse", "flip_cse", "mask", "absorb", "lift"]
