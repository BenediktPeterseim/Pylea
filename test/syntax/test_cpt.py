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

    def test_a_parent_pattern_encoding_no_value_takes_the_row_of_the_first_value(self):
        """Tri parent (3 states, 2 bits) has one pattern, 3, that encodes no value."""
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
        assert _allclose(M[:, 3], M[:, 0])

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
        # (False, UR)  → idx 3 (encodes no value, takes the row of (False, "a"))
        # (True, "a")  → idx 4
        # etc.
        assert _allclose(M[:, 0], [0.1, 0.9])
        assert _allclose(M[:, 1], [0.2, 0.8])
        assert _allclose(M[:, 2], [0.3, 0.7])
        assert _allclose(M[:, 4], [0.6, 0.4])
        assert _allclose(M[:, 5], [0.7, 0.3])
        assert _allclose(M[:, 6], [0.8, 0.2])
        assert _allclose(M[:, 3], M[:, 0]) and _allclose(M[:, 7], M[:, 4])


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


# --- arguments and rows that are refused ---

class TestEnumCptChecks:
    """Mistakes in the arguments raise rather than change the table."""

    def test_a_cpt_row_naming_a_value_that_is_not_an_element_is_refused(self):
        with pytest.raises(ValueError, match="'d' is not an element of Tri"):
            Tri.cpt([Bin], {(False,): {"a": 0.5, "d": 0.5}, (True,): [1, 0, 0]})

    @pytest.mark.parametrize("row", [[0.5, 0.2, 0.2, 0.1], [1.0]])
    def test_a_list_row_of_the_wrong_length_is_refused(self, row):
        with pytest.raises(ValueError, match="needs 3 entries"):
            Tri.cpt([Bin], {(False,): row, (True,): [1, 0, 0]})

    def test_a_row_given_as_a_tuple_or_an_array_is_read_in_element_order(self):
        cdpp = Tri.cpt([Bin], {(False,): (0.5, 0.5, 0.0), (True,): np.array([0.0, 0.2, 0.8])})
        M = _eval(cdpp)
        assert _allclose(M[:3, 0], [0.5, 0.5, 0.0])
        assert _allclose(M[:3, 1], [0.0, 0.2, 0.8])

    def test_a_lone_parent_type_is_refused_with_a_list_suggested(self):
        with pytest.raises(TypeError, match=r"such as \[Tri\]"):
            Bin.cpt(Tri, {("a",): [1, 0], ("b",): [1, 0], ("c",): [1, 0]})

    def test_a_row_for_a_parent_value_that_does_not_exist_is_refused(self):
        rows = {("a",): [1, 0], ("b",): [1, 0], ("c",): [1, 0], ("hail",): [0, 1]}
        with pytest.raises(ValueError, match=r"\('hail',\)"):
            Bin.cpt([Tri], rows)

    def test_a_missing_row_is_refused(self):
        with pytest.raises(ValueError, match=r"no row for \('c',\)"):
            Bin.cpt([Tri], {("a",): [1, 0], ("b",): [1, 0]})

    def test_rows_keyed_by_bare_values_are_refused_with_one_element_tuples_suggested(self):
        with pytest.raises(ValueError, match=r"such as \('a',\)"):
            Bin.cpt([Tri], {"a": [1, 0], "b": [1, 0], "c": [1, 0]})

    def test_a_table_wider_than_the_dense_bound_is_refused_before_it_is_built(self):
        import itertools
        import time
        rows = {key: [0.5, 0.5] for key in itertools.product(Tri.elements, repeat=10)}
        start = time.perf_counter()
        with pytest.raises(ValueError, match=r"2\*\*21 entries"):
            Bin.cpt([Tri] * 10, rows)
        assert time.perf_counter() - start < 0.5

    def test_an_abstract_child_raises_not_implemented_error(self):
        class Abstract(Enum):
            pass

        with pytest.raises(NotImplementedError, match="declares no elements"):
            Abstract.cpt([Bin], {(False,): [1.0], (True,): [1.0]})


class TestEnumCptParentTypes:
    """A parent of any variable type, keyed by its values, agrees with a `match`."""

    def _agree(self, cpt, reference: DPP) -> None:
        @DPP
        def through_cpt(x: cpt.input_types()[0]) -> Tri:   # type: ignore[valid-type]
            return cpt(x)

        # Compared on the parent patterns that encode a value; on the others, the table
        # takes the row of the parent's first value, which a `match` need not.
        valid = sorted(cpt.input_types()[0].encoding().decoded())
        assert _allclose(_eval_dpp(through_cpt)[:, valid], _eval_dpp(reference)[:, valid])

    def test_a_bool_parent_is_keyed_by_true_and_false(self):
        from pylea import Bool

        cpt = Tri.cpt([Bool], {(False,): [0.2, 0.3, 0.5], (True,): [0.6, 0.4, 0.0]})

        @DPP
        def by_match(x: Bool) -> Tri:
            return x.match({False: Tri.random([0.2, 0.3, 0.5]),
                            True: Tri.random([0.6, 0.4, 0.0])})

        self._agree(cpt, by_match)

    def test_a_uint_parent_is_keyed_by_its_integers(self):
        from pylea import UInt

        class Two(UInt):
            width = 2

        rows = {(v,): [v + 1, 1, 3 - v] for v in range(4)}
        cpt = Tri.cpt([Two], rows)

        @DPP
        def by_match(x: Two) -> Tri:
            return (x < 2).match({
                True: (x < 1).match({True: Tri.random(rows[(0,)]),
                                     False: Tri.random(rows[(1,)])}),
                False: (x < 3).match({True: Tri.random(rows[(2,)]),
                                      False: Tri.random(rows[(3,)])})})

        self._agree(cpt, by_match)

    def test_a_tuple_parent_is_keyed_by_plain_tuples(self):
        from pylea import Tuple

        class Pair(Tuple):
            types = (Bin, Tri)

        # One parent, whose values are tuples: each key is a tuple of one tuple.
        rows = {((b, t),): [1 + i, 2, 3] for i, (b, t) in
                enumerate((b, t) for b in Bin.elements for t in Tri.elements)}
        cpt = Tri.cpt([Pair], rows)

        @DPP
        def by_match(p: Pair) -> Tri:
            b, t = p.components()
            return b.match({b_value: t.match({t_value: Tri.random(rows[((b_value, t_value),)])
                                              for t_value in Tri.elements})
                            for b_value in Bin.elements})

        self._agree(cpt, by_match)
