"""What a call of a DPP, a primitive or a table accepts as its arguments, and what
`Bool.match` accepts as its values.

A variable is accepted where its type is the declared one, or a subclass reading the same
bits as the same values; a value of the declared type becomes a constant; anything else is
refused with the callee and the position named.  Keyword arguments bind to parameter
names as in Python.
"""
import numpy as np
import pytest

from pylea import DPP, Bool, Enum, UInt, LikelihoodTable, at_least, flip, observe, prob
from pylea.syntax.dpp import EncodedBool


class Weather(Enum):
    elements = ["dry", "rain", "snow"]


class Grass(Enum):
    elements = ["dry", "damp", "wet"]


@DPP
def is_wet(g: Grass) -> Bool:
    return g == "wet"


@DPP
def neg(x: Bool) -> Bool:
    return ~x


@DPP
def first_not_second(a: Bool, b: Bool) -> Bool:
    return a & ~b


def _weather() -> Weather:
    return Weather.random({"dry": 0.1, "rain": 0.2, "snow": 0.7})


class TestArguments:

    def test_a_variable_of_another_type_of_the_same_width_is_refused(self):
        with pytest.raises(TypeError,
                           match=r"is_wet: argument 1 \(g\) must be a Grass; got a Weather"):
            @DPP
            def mismatch() -> Bool:
                return is_wet(_weather())

    def test_a_variable_of_another_width_is_refused_by_name_not_by_arity(self):
        with pytest.raises(TypeError,
                           match=r"neg: argument 1 \(x\) must be a Bool; got a Weather"):
            @DPP
            def mismatch() -> Bool:
                return neg(_weather())

    def test_a_python_bool_argument_becomes_a_constant(self):
        @DPP
        def negated_truth() -> Bool:
            return neg(True)

        assert prob(negated_truth, method="exact") == 0.0

    def test_a_value_outside_the_base_type_is_refused(self):
        with pytest.raises(ValueError, match=r"neg: argument 1 \(x\)"):
            @DPP
            def bad() -> Bool:
                return neg("yes")

    def test_observing_a_python_constant_conditions_on_that_constant(self):
        @DPP
        def kept() -> Bool:
            x = flip(0.3)
            observe(True)
            return x

        @DPP
        def impossible() -> Bool:
            x = flip(0.3)
            observe(False)
            return x

        assert prob(kept, method="exact") == 0.3
        with pytest.raises(ZeroDivisionError):
            prob(impossible, method="exact")

    def test_observe_refuses_a_variable_that_is_not_a_bool(self):
        class Coin(Enum):
            elements = ["heads", "tails"]

        with pytest.raises(TypeError, match="observe: argument 1 must be a Bool; got a Coin"):
            @DPP
            def bad() -> Bool:
                observe(Coin.uniform())
                return flip(0.5)

    def test_a_table_refuses_an_argument_of_a_type_with_the_same_width(self):
        table = LikelihoodTable((Grass,), (Bool,), np.full((2, 4), 0.5))
        with pytest.raises(TypeError, match="argument 1 must be a Grass; got a Weather"):
            @DPP
            def bad() -> Bool:
                return table(_weather())

    def test_a_wrong_number_of_arguments_raises_type_error_naming_the_callee(self):
        with pytest.raises(TypeError, match="first_not_second takes 2 arguments; got 1"):
            @DPP
            def bad() -> Bool:
                return first_not_second(flip(0.5))  # type: ignore[call-arg]


class TestSubclasses:
    """A subclass of the declared type is accepted only where it reads the same."""

    def test_a_subclass_with_another_width_is_refused_where_its_base_is_declared(self):
        class Narrow(UInt):
            width = 4

        class Wide(Narrow):
            width = 8

        @DPP
        def is_three(n: Narrow) -> Bool:
            return n == 3

        with pytest.raises(TypeError, match="must be a Narrow; got a Wide"):
            @DPP
            def bad() -> Bool:
                return is_three(Wide.const(3))

    def test_a_subclass_with_reordered_elements_is_refused_where_its_base_is_declared(
            self):
        class Base(Enum):
            elements = ["a", "b", "c"]

        class Reordered(Base):
            elements = ["c", "b", "a"]

        @DPP
        def is_a(x: Base) -> Bool:
            return x == "a"

        with pytest.raises(TypeError, match="must be a Base; got a Reordered"):
            @DPP
            def bad() -> Bool:
                return is_a(Reordered.const("c"))

    def test_a_subclass_with_the_same_encoding_is_accepted(self):
        class Base(Enum):
            elements = ["a", "b", "c"]

        class Same(Base):
            pass

        @DPP
        def is_a(x: Base) -> Bool:
            return x == "a"

        @DPP
        def good() -> Bool:
            return is_a(Same.const("a"))

        assert prob(good, method="exact") == 1.0


class TestKeywordArguments:

    def test_keyword_arguments_bind_to_parameter_names(self):
        @DPP
        def by_keyword() -> Bool:
            return first_not_second(b=flip(0.25), a=flip(0.5))

        @DPP
        def by_position() -> Bool:
            return first_not_second(flip(0.5), flip(0.25))

        assert prob(by_keyword, method="exact") == prob(by_position, method="exact") == 0.375

    def test_an_unknown_keyword_argument_is_refused_by_name(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'c'"):
            @DPP
            def bad() -> Bool:
                return first_not_second(flip(0.5), c=flip(0.5))  # type: ignore[call-arg]

    def test_an_argument_given_by_position_and_by_keyword_is_refused(self):
        with pytest.raises(TypeError, match="multiple values for argument 'a'"):
            @DPP
            def bad() -> Bool:
                return first_not_second(flip(0.5), a=flip(0.5))  # type: ignore[misc]

    def test_a_keyword_argument_to_a_zero_input_dpp_is_refused(self):
        @DPP
        def coin() -> Bool:
            return flip(0.5)

        with pytest.raises(TypeError, match="coin takes no arguments; got keyword"):
            @DPP
            def bad() -> Bool:
                return coin(x=flip(0.1))  # type: ignore[call-arg]

    def test_a_keyword_argument_to_a_dpp_without_parameter_names_is_refused(self):
        @DPP
        def either(xs, types=(Bool, Bool)) -> Bool:
            return xs[0] | xs[1]

        with pytest.raises(TypeError, match="either takes positional arguments only"):
            @DPP
            def bad() -> Bool:
                return either(xs=flip(0.5))  # type: ignore[call-arg]

    def test_a_keyword_argument_to_a_table_is_refused(self):
        table = LikelihoodTable((Bool,), (Bool,), np.array([[0.9, 0.2], [0.1, 0.8]]))
        with pytest.raises(TypeError):
            @DPP
            def bad() -> Bool:
                return table(x=flip(0.5))


class TestMatch:

    def test_match_refuses_a_value_of_another_type(self):
        with pytest.raises(TypeError, match="match value for False must be a Weather; "
                                            "got a Grass"):
            @DPP
            def bad() -> Weather:
                return flip(0.5).match({True: _weather(), False: Grass.const("wet")})

    def test_match_refuses_a_constant_its_output_type_cannot_hold(self):
        with pytest.raises(ValueError, match="match value for True"):
            @DPP
            def bad() -> Bool:
                return flip(0.5).match({True: "wet", False: "dry"}, data_type=Bool)

    def test_match_refuses_keys_other_than_true_and_false(self):
        with pytest.raises(ValueError, match="keys True and False only; got None"):
            @DPP
            def bad() -> Bool:
                return flip(0.5).match({True: flip(0.9), False: flip(0.1), None: flip(0.5)})

    def test_match_refuses_the_integers_zero_and_one_as_keys(self):
        with pytest.raises(ValueError, match="keys True and False only; got 1, 0"):
            @DPP
            def bad() -> Bool:
                return flip(0.5).match({1: flip(0.9), 0: flip(0.1)})

    def test_match_refuses_a_data_type_that_is_not_a_variable_type(self):
        with pytest.raises(TypeError, match="data_type must be a subclass of Variable"):
            @DPP
            def bad() -> Bool:
                return flip(0.5).match({True: True, False: False}, data_type=bool)

    def test_the_bool_encoding_refuses_a_value_that_is_not_a_bool(self):
        assert EncodedBool(np.False_).to_bits() == (False,)
        for value in (1, 0, "yes", None):
            with pytest.raises(ValueError, match="a Bool holds True or False"):
                EncodedBool(value)


class TestAtLeast:

    def test_at_least_takes_an_iterable_as_bool_any_does(self):
        @DPP
        def from_list() -> Bool:
            return at_least(2, [flip(0.5), flip(0.5), flip(0.5)])

        @DPP
        def from_generator() -> Bool:
            return at_least(2, (flip(0.5) for _ in range(3)))

        assert prob(from_list, method="exact") == prob(from_generator, method="exact") == 0.5

    def test_at_least_accepts_python_bools_among_its_events(self):
        @DPP
        def one_of() -> Bool:
            return at_least(1, flip(0.5), True)

        assert prob(one_of, method="exact") == 1.0

    def test_a_gate_built_from_at_least_without_coin_flips_is_deterministic(self):
        @DPP
        def gate(a: Bool, b: Bool, c: Bool) -> Bool:
            return at_least(2, a, b, c)

        assert gate.is_deterministic

    def test_at_least_refuses_a_threshold_that_is_not_an_integer(self):
        with pytest.raises(TypeError, match="at_least: k must be an integer; got 1.5"):
            @DPP
            def bad() -> Bool:
                return at_least(1.5, flip(0.5))  # type: ignore[arg-type]


class TestMembership:

    def test_a_membership_test_on_a_variable_names_membership_in_its_error(self):
        with pytest.raises(TypeError, match="membership tests such as 'x in"):
            @DPP
            def bad() -> Bool:
                c = Grass.uniform()
                return c in ["dry", "wet"]  # type: ignore[return-value]

    def test_a_chained_comparison_on_a_variable_names_chained_comparisons_in_its_error(
            self):
        with pytest.raises(TypeError, match="chained comparisons"):
            @DPP
            def bad() -> Bool:
                c = Grass.uniform()
                return "dry" <= c <= "wet"  # type: ignore[return-value]
