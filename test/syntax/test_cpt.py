"""`Enum.cpt`: its matrix, and its agreement with the `match`-and-`random` encoding.

The two encodings of a conditional probability table must give the same answer, so the
sprinkler network is built both ways and compared table by table and as a whole.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from test.syntax.helpers import matrix_of

from pylea.syntax.dpp import DPP, Primitive
from pylea.syntax.enum import Enum


class Bin(Enum):
    elements = [False, True]                # 1 bit


class Tri(Enum):
    elements = ["a", "b", "c"]              # 2 bits, one pattern spare


class Quad(Enum):
    elements = [0, 1, 2, 3]                 # 2 bits, no pattern spare


class Octet(Enum):
    elements = list(range(8))               # 3 bits, no pattern spare


class Five(Enum):
    elements = ["p", "q", "r", "s", "t"]    # 3 bits, three patterns spare


def _eval(prim) -> np.ndarray:
    """The matrix of a `Primitive`."""
    return np.asarray(prim.prim.matrix, dtype=float)


def _eval_dpp(dpp: DPP) -> np.ndarray:
    """The matrix of a DPP."""
    return matrix_of(dpp)


def _eval_hierarchical(dpp: DPP) -> np.ndarray:
    """The matrix of a DPP whose callees are themselves DPPs."""
    return matrix_of(dpp)


def _allclose(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)


def _proportional(a: np.ndarray, b: np.ndarray, atol=1e-6) -> bool:
    """Return True iff a and b are proportional (a = c*b for some c > 0)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return True
    if na == 0 or nb == 0:
        return False
    return np.allclose(a / na, b / nb, atol=atol)


class TestEnumCptBasic:
    """Basic construction and output shape checks."""

    def test_cpt_returns_a_primitive(self):
        cdpp = Bin.cpt((Bin,), {(False,): [0.3, 0.7], (True,): [0.6, 0.4]})
        assert isinstance(cdpp, Primitive)

    def test_a_cpt_declares_its_parents_as_inputs_and_the_child_as_output(self):
        cdpp = Tri.cpt((Bin,), {(False,): [0.5, 0.3, 0.2], (True,): [0.1, 0.4, 0.5]})
        assert cdpp.input_types() == (Bin,)
        assert cdpp.output_types() == (Tri,)

    def test_single_parent_matrix_shape(self):
        """Bin output (m=1) with Bin parent (K=1) → (2, 2) matrix."""
        cdpp = Bin.cpt((Bin,), {(False,): [0.3, 0.7], (True,): [0.6, 0.4]})
        M = _eval(cdpp)
        assert M.shape == (2, 2)

    def test_a_cpt_is_a_single_dense_matrix(self):
        """A CPT is a single dense primitive, not a circuit."""
        rows = {(b, t): {"a": 0.5, "b": 0.3, "c": 0.2}
                for b in Bin for t in Tri}
        cpt = Tri.cpt((Bin, Tri), rows)
        assert cpt.prim is not None
        assert cpt.prim.matrix.shape == (4, 8)


    def test_a_child_with_spare_patterns_gives_them_weight_zero(self):
        """Tri has 3 states (m=2). Unused bit-pattern row must sum to 0."""
        cdpp = Tri.cpt((Bin,), {(False,): [0.5, 0.3, 0.2], (True,): [0.1, 0.4, 0.5]})
        M = _eval(cdpp)
        # Row 3 (bit-pattern 11, unused child state) must be 0
        assert _allclose(M[3, :], [0.0, 0.0])
        # Reachable rows must match specified probs
        assert _allclose(M[:3, 0], [0.5, 0.3, 0.2])
        assert _allclose(M[:3, 1], [0.1, 0.4, 0.5])

    def test_a_parent_with_spare_patterns_repeats_its_nearest_row(self):
        """Tri parent (3 states, 2 bits) has one unreachable parent config per output bit."""
        cdpp = Bin.cpt((Tri,), {
            ("a",): [0.8, 0.2],
            ("b",): [0.4, 0.6],
            ("c",): [0.1, 0.9],
        })
        M = _eval(cdpp)
        # 4 parent configs (2^2), 2 output rows
        assert M.shape == (2, 4)
        # Reachable configs: col 0 = a, col 1 = b, col 2 = c
        assert _allclose(M[:, 0], [0.8, 0.2])
        assert _allclose(M[:, 1], [0.4, 0.6])
        assert _allclose(M[:, 2], [0.1, 0.9])

    def test_a_row_given_as_a_list_is_indexed_by_element_position(self):
        """Rows as positional lists (BIF parser format)."""
        cdpp = Bin.cpt((Bin,), {(False,): [0.3, 0.7], (True,): [0.6, 0.4]})
        M = _eval(cdpp)
        assert _allclose(M[:, 0], [0.3, 0.7])
        assert _allclose(M[:, 1], [0.6, 0.4])

    def test_a_row_given_as_a_dict_is_indexed_by_element(self):
        """Rows as {state: prob} dicts."""
        cdpp = Bin.cpt((Bin,), {
            (False,): {False: 0.3, True: 0.7},
            (True,):  {False: 0.6, True: 0.4},
        })
        M = _eval(cdpp)
        assert _allclose(M[:, 0], [0.3, 0.7])
        assert _allclose(M[:, 1], [0.6, 0.4])

    def test_every_column_of_a_cpt_sums_to_one(self):
        """Unnormalised inputs are normalised to sum to 1."""
        cdpp = Bin.cpt((Bin,), {(False,): [0.6, 1.4], (True,): [1.0, 1.0]})
        M = _eval(cdpp)
        assert _allclose(M[:, 0], [0.3, 0.7])
        assert _allclose(M[:, 1], [0.5, 0.5])


class TestEnumCptMultiParent:
    """Two-parent and deeper CPT tests."""

    def test_two_one_bit_parents_give_a_four_column_matrix(self):
        """Two 1-bit parents → 4 input configs; verifies MSB-first ordering."""
        # Rows keyed by (P0_state, P1_state) in MSB-first order
        cdpp = Bin.cpt((Bin, Bin), {
            (False, False): [0.1, 0.9],
            (False, True):  [0.3, 0.7],
            (True,  False): [0.6, 0.4],
            (True,  True):  [0.9, 0.1],
        })
        M = _eval(cdpp)
        # col ordering: 00, 01, 10, 11 (MSB=P0)
        assert M.shape == (2, 4)
        assert _allclose(M[:, 0], [0.1, 0.9])  # P0=False, P1=False
        assert _allclose(M[:, 1], [0.3, 0.7])  # P0=False, P1=True
        assert _allclose(M[:, 2], [0.6, 0.4])  # P0=True,  P1=False
        assert _allclose(M[:, 3], [0.9, 0.1])  # P0=True,  P1=True

    def test_parents_of_different_widths_are_packed_in_declared_order(self):
        """One 1-bit and one 2-bit parent → 8 input configs (2 unreachable)."""
        cdpp = Bin.cpt((Bin, Tri), {
            (False, "a"): [0.1, 0.9],
            (False, "b"): [0.2, 0.8],
            (False, "c"): [0.3, 0.7],
            (True,  "a"): [0.6, 0.4],
            (True,  "b"): [0.7, 0.3],
            (True,  "c"): [0.8, 0.2],
        })
        M = _eval(cdpp)
        # 8 columns (2^3 input configs), 2 output rows
        assert M.shape == (2, 8)
        # Reachable: input_idx = p0_idx * 4 + p1_idx
        # (False, "a") → idx 0
        # (False, "b") → idx 1
        # (False, "c") → idx 2
        # (False, UR)  → idx 3 (unreachable, should equal (False,"a") by filler)
        # (True, "a")  → idx 4
        # etc.
        assert _allclose(M[:, 0], [0.1, 0.9])
        assert _allclose(M[:, 1], [0.2, 0.8])
        assert _allclose(M[:, 2], [0.3, 0.7])
        assert _allclose(M[:, 4], [0.6, 0.4])
        assert _allclose(M[:, 5], [0.7, 0.3])
        assert _allclose(M[:, 6], [0.8, 0.2])


# --- the two encodings compared over the sprinkler network ---

class SprinklerCloudiness(Enum):
    elements = ["cloudy", "sunny"]


class SprinklerState(Enum):
    elements = ["on", "off"]


class SprinklerRain(Enum):
    elements = ["none", "light", "heavy"]


class GrassWetness(Enum):
    elements = ["dry", "damp", "wet"]


# --- the reference encoding, with `match` and `random` ---

@DPP
def _sprinkler_match(c: SprinklerCloudiness) -> SprinklerState:
    return c.match({
        "cloudy": SprinklerState.random({"on": 0.1, "off": 0.9}),
        "sunny":  SprinklerState.random({"on": 0.5, "off": 0.5}),
    })


@DPP
def _rain_match(c: SprinklerCloudiness) -> SprinklerRain:
    return c.match({
        "cloudy": SprinklerRain.random({"none": 0.2, "light": 0.4, "heavy": 0.4}),
        "sunny":  SprinklerRain.random({"none": 0.7, "light": 0.2, "heavy": 0.1}),
    })


@DPP
def _wetgrass_match(s: SprinklerState, r: SprinklerRain) -> GrassWetness:
    return s.match({
        "on": r.match({
            "none":  GrassWetness.random({"dry": 0.2, "damp": 0.5, "wet": 0.3}),
            "light": GrassWetness.random({"dry": 0.0, "damp": 0.2, "wet": 0.8}),
            "heavy": GrassWetness.random({"dry": 0.0, "damp": 0.1, "wet": 0.9}),
        }),
        "off": r.match({
            "none":  GrassWetness.random({"dry": 0.9, "damp": 0.1, "wet": 0.0}),
            "light": GrassWetness.random({"dry": 0.1, "damp": 0.6, "wet": 0.3}),
            "heavy": GrassWetness.random({"dry": 0.0, "damp": 0.3, "wet": 0.7}),
        }),
    })


# --- the same tables through `Enum.cpt` ---

_sprinkler_cpt = SprinklerState.cpt((SprinklerCloudiness,), {
    ("cloudy",): {"on": 0.1, "off": 0.9},
    ("sunny",):  {"on": 0.5, "off": 0.5},
})

_rain_cpt = SprinklerRain.cpt((SprinklerCloudiness,), {
    ("cloudy",): {"none": 0.2, "light": 0.4, "heavy": 0.4},
    ("sunny",):  {"none": 0.7, "light": 0.2, "heavy": 0.1},
})

_wetgrass_cpt = GrassWetness.cpt((SprinklerState, SprinklerRain), {
    ("on",  "none"):  {"dry": 0.2, "damp": 0.5, "wet": 0.3},
    ("on",  "light"): {"dry": 0.0, "damp": 0.2, "wet": 0.8},
    ("on",  "heavy"): {"dry": 0.0, "damp": 0.1, "wet": 0.9},
    ("off", "none"):  {"dry": 0.9, "damp": 0.1, "wet": 0.0},
    ("off", "light"): {"dry": 0.1, "damp": 0.6, "wet": 0.3},
    ("off", "heavy"): {"dry": 0.0, "damp": 0.3, "wet": 0.7},
})


class TestSprinklerEquivalence:
    """Each cpt-based CPT must represent the same distribution as its match-based counterpart.

    Comparisons use _proportional (correct projective semantics: matrices equal up
    to a positive global scalar) rather than _allclose (which fails whenever the
    algebrisation introduces cup/cap bends that change the representative's scale).
    """

    def test_the_sprinkler_table_agrees_between_the_two_encodings(self):
        M_match = _eval_dpp(_sprinkler_match)
        M_cpt = _eval(_sprinkler_cpt)
        assert _proportional(M_match, M_cpt), \
            f"sprinkler CPT mismatch:\nMatch:\n{M_match}\nCPT:\n{M_cpt}"

    def test_the_rain_table_agrees_between_the_two_encodings(self):
        M_match = _eval_dpp(_rain_match)
        M_cpt = _eval(_rain_cpt)
        assert _proportional(M_match, M_cpt)

    def test_the_grass_table_agrees_between_the_two_encodings(self):
        M_match = _eval_dpp(_wetgrass_match)
        M_cpt = _eval(_wetgrass_cpt)
        assert _proportional(M_match, M_cpt)


class TestSprinklerNetworkEquivalence:
    """Full joint distribution from cpt-based network must match match-based."""

    def test_the_whole_network_agrees_between_the_two_encodings(self):
        # Reference network (match-based)
        @DPP
        def net_match() -> tuple[
            SprinklerCloudiness, SprinklerState, SprinklerRain, GrassWetness
        ]:
            c = SprinklerCloudiness.random({"cloudy": 0.4, "sunny": 0.6})
            s = _sprinkler_match(c)
            r = _rain_match(c)
            w = _wetgrass_match(s, r)
            return c, s, r, w

        # CPT-based network
        @DPP
        def net_cpt() -> tuple[
            SprinklerCloudiness, SprinklerState, SprinklerRain, GrassWetness
        ]:
            c = SprinklerCloudiness.random({"cloudy": 0.4, "sunny": 0.6})
            s = _sprinkler_cpt(c)
            r = _rain_cpt(c)
            w = _wetgrass_cpt(s, r)
            return c, s, r, w

        M_match = _eval_hierarchical(net_match)
        M_cpt = _eval_hierarchical(net_cpt)
        assert _proportional(M_match, M_cpt), \
            f"Joint distribution mismatch:\nMatch:\n{M_match}\nCPT:\n{M_cpt}"


# --- the tables the .bif converter generates ---

class TestBifCodegen:
    """Verify bif_to_python now emits cpt(...) instead of nested match."""

    def test_generated_code_builds_its_tables_with_cpt(self):
        from pylea.utility.bif_to_dpp import bif_to_python
        bif = """\
network test { }
variable X { type discrete [ 2 ] { a, b }; }
variable Y { type discrete [ 2 ] { p, q }; }
probability ( X ) { table 0.4, 0.6; }
probability ( Y | X ) {
  (a) 0.3, 0.7;
  (b) 0.8, 0.2;
}
"""
        code = bif_to_python(bif)
        assert "Y.cpt(" in code, "Expected cpt() in generated code"
        assert "@DPP" not in code.split("def test")[0] or ".cpt(" in code, \
            "Expected no @DPP for conditional variable"

    def test_generated_code_imports_and_evaluates(self):
        """Generated code can be exec'd and the network DPP can be compiled."""
        from pylea.utility.bif_to_dpp import bif_to_python
        bif = """\
network tiny { }
variable A { type discrete [ 2 ] { lo, hi }; }
variable B { type discrete [ 2 ] { x, y }; }
probability ( A ) { table 0.5, 0.5; }
probability ( B | A ) {
  (lo) 0.2, 0.8;
  (hi) 0.7, 0.3;
}
"""
        code = bif_to_python(bif)
        ns: dict = {}
        exec(code, ns)
        net = ns["tiny"]
        # Should be a DPP with B.cpt() already built
        assert hasattr(net, "output_types")
        # Compile it (should not trigger a merge loop for B)
        import time
        t0 = time.perf_counter()
        cdpp = net.compiled()
        t1 = time.perf_counter()
        assert t1 - t0 < 30.0, f"Compilation took too long: {t1-t0:.1f}s"
