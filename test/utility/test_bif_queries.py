"""The queries generated from a `.bif` network ask what their header says they ask."""
import sys
import types

import pytest

from pylea import DPP, Bool
from pylea.utility.bif_to_dpp import bif_to_python, make_query_module, query_specification

# A collider and two leaves, so the query variable has ancestors and the evidence can fall
# either side of it.
BIF = """
network net {
}
variable a { type discrete [ 2 ] { yes, no }; }
variable b { type discrete [ 2 ] { yes, no }; }
variable c { type discrete [ 2 ] { yes, no }; }
variable d { type discrete [ 2 ] { yes, no }; }
variable e { type discrete [ 2 ] { yes, no }; }
probability ( a ) { table 0.3, 0.7; }
probability ( b ) { table 0.6, 0.4; }
probability ( c | a, b ) {
  (yes, yes) 0.9, 0.1;
  (yes, no) 0.5, 0.5;
  (no, yes) 0.2, 0.8;
  (no, no) 0.05, 0.95;
}
probability ( d | c ) { (yes) 0.8, 0.2; (no) 0.1, 0.9; }
probability ( e | c ) { (yes) 0.7, 0.3; (no) 0.25, 0.75; }
"""


def _module(source: str, name: str) -> types.ModuleType:
    """`source` executed as a module registered under `name`, so others may import it."""
    mod = types.ModuleType(name)
    exec(compile(source, name, "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod


def _p_true(dpp) -> float:
    return float(dpp.evaluated().normalised().probabilities.reshape(-1)[1])


def test_a_conditioned_query_equals_the_joint_over_the_evidence():
    """`observe` must condition, not merely filter: the two ways of writing P(q | e) agree.

    The reference side never observes, so it exercises none of the machinery under test.
    """
    net = _module(bif_to_python(BIF, network_name="net"), "generated_net").net
    query, evidence = query_specification(BIF, n_evidence=2, seed=0)
    conditioned = _module(
        make_query_module(BIF, mod_name="net", pylea_import_path="generated_net",
                          n_evidence=2, seed=0),
        "generated_query",
    ).net_e2

    @DPP
    def joint() -> Bool:
        xs = net()
        hit = xs[query[0]] == query[2]
        for i, _, state in evidence:
            hit = hit & (xs[i] == state)
        return hit

    @DPP
    def evidence_only() -> Bool:
        xs = net()
        hit = Bool.const(True)
        for i, _, state in evidence:
            hit = hit & (xs[i] == state)
        return hit

    ratio = _p_true(joint) / _p_true(evidence_only)
    assert _p_true(conditioned) == pytest.approx(ratio, rel=1e-12)


def test_evidence_states_are_drawn_from_the_network_so_they_are_possible():
    """Evidence of probability zero would make the conditional undefined, silently."""
    _, evidence = query_specification(BIF, n_evidence=2, seed=0)
    net = _module(bif_to_python(BIF, network_name="net"), "generated_net_2").net

    @DPP
    def evidence_only() -> Bool:
        xs = net()
        hit = Bool.const(True)
        for i, _, state in evidence:
            hit = hit & (xs[i] == state)
        return hit

    assert _p_true(evidence_only) > 0.0


def test_a_marginal_query_observes_nothing():
    source = make_query_module(BIF, mod_name="net", pylea_import_path="generated_net")
    assert "observe" not in source
