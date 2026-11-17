"""What `DPP`, `Primitive` and `Bool` refuse, and the bookkeeping around tracing.

The semantics of a model that traces successfully are covered elsewhere.  What is pinned
here is the validation: the two ways of declaring input types, the enforcement that a
variable belongs to the definition using it, the restoration of the active definition after
a failed one, the caching of a compilation, and the arity checks on a `Primitive`.
"""
import pytest
from pylea.syntax.compiled import Compiled
from pylea.syntax.dpp import DPP, Bool, Primitive, _NOT_, _AND_
from pylea.syntax.flip import flip
from pylea.ir.prim import Prim


# A primitive with no inputs, for calling outside any definition.
_ZERO_INPUT_FLIP = Primitive(
    input_types=(),
    output_types=(Bool,),
    prim=Prim.flip(0.5),
    name="_test_flip_",
)

# A primitive with neither inputs nor outputs.
_NO_IO = Primitive(
    input_types=(),
    output_types=(),
    prim=Prim.from_matrix([[1.0]]),
    name="_test_no_io_",
)


class TestDPPConstruction:

    def test_a_function_with_neither_annotations_nor_types_is_refused(self):
        """Unannotated parameter without a 'types' fallback raises ValueError."""
        with pytest.raises(ValueError, match="has no type annotation"):
            @DPP
            def bad(x):
                return x

    def test_a_types_argument_that_is_not_a_tuple_is_refused(self):
        """Providing a non-tuple as the 'types' default raises ValueError."""
        with pytest.raises(ValueError, match="must be a tuple"):
            @DPP
            def bad(*args, types=Bool):  # Bool itself, not (Bool,)
                return args[0]

    def test_a_types_argument_without_a_default_is_refused(self):
        """A 'types' parameter with no default value raises ValueError."""
        with pytest.raises(ValueError, match="no default value provided"):
            @DPP
            def bad(*args, types):
                return args[0]

    def test_an_input_annotation_that_is_not_a_variable_is_refused(self):
        """A parameter annotated with a non-Variable type raises TypeError."""
        with pytest.raises(TypeError, match="not a subclass of Variable"):
            @DPP
            def bad(x: int) -> None:
                pass

    def test_a_non_variable_inside_a_types_tuple_is_refused(self):
        """A non-Variable class in the 'types' tuple raises TypeError."""
        with pytest.raises(TypeError, match="not a subclass of Variable"):
            @DPP
            def bad(*args, types=(str,)):
                pass

    def test_returning_a_raw_value_rather_than_a_variable_is_refused(self):
        """Returning a plain Python value (not Variable/tuple/None) raises TypeError."""
        with pytest.raises(TypeError):
            @DPP
            def bad() -> None:
                return 42  # type: ignore[return-value]

    def test_returning_a_tuple_holding_a_non_variable_is_refused(self):
        """A tuple containing a non-Variable raises TypeError."""
        with pytest.raises(TypeError):
            @DPP
            def bad() -> None:
                return (flip(0.5), "oops")  # type: ignore[return-value]

    def test_the_types_argument_gives_the_same_interface_as_annotations(self):
        """DPP constructed via the types-kwarg path reports the correct input/output types."""
        @DPP
        def kwarg_identity(*args, types=(Bool,)):
            return args[0]

        assert kwarg_identity.input_types() == (Bool,)
        assert kwarg_identity.output_types() == (Bool,)

    def test_an_empty_types_argument_declares_no_inputs(self):
        """The types-kwarg path supports zero-input DPPs."""
        @DPP
        def kwarg_zero(*args, types=()):
            return flip(0.5)

        assert kwarg_zero.input_types() == ()
        assert kwarg_zero.n_inputs() == 0


class TestDPPCall:

    def test_calling_a_primitive_outside_any_definition_is_refused(self):
        """Calling any ProbabilisticCallable outside a DPP definition raises RuntimeError."""
        with pytest.raises(RuntimeError, match="outside a DPP definition"):
            _ZERO_INPUT_FLIP()

    def test_flip_called_outside_a_definition_is_named_as_written(self):
        with pytest.raises(RuntimeError, match=r"^flip\(0\.5\) was called outside a DPP"):
            flip(0.5)

    def test_calling_with_the_wrong_number_of_arguments_is_refused(self):
        """Passing the wrong number of arguments to a callable inside a DPP raises TypeError."""
        with pytest.raises(TypeError, match="takes 2 arguments; got 0"):
            @DPP
            def bad() -> Bool:
                return _AND_()  # _AND_ requires exactly 2 Bool arguments

    def test_passing_a_variable_belonging_to_another_definition_is_refused(self):
        """Using a variable that belongs to a different DPP raises ValueError."""
        @DPP
        def donor() -> Bool:
            return flip(0.5)

        foreign_var = donor.output_variables()[0]

        with pytest.raises(ValueError, match="only in the definition that made it"):
            @DPP
            def receiver() -> Bool:
                return _NOT_(foreign_var)  # foreign_var was born inside 'donor'

    def test_a_call_with_no_outputs_returns_none(self):
        """_call returns None when the called primitive produces no outputs."""
        results = []

        @DPP
        def test_no_out() -> None:
            r = _NO_IO()
            results.append(r)

        assert results[0] is None

    def test_a_call_with_several_outputs_returns_a_tuple(self):
        """_call returns a plain tuple when the callable produces more than one output."""
        # Build a 0→2 Bool primitive.
        two_flips = Primitive(
            input_types=(),
            output_types=(Bool, Bool),
            prim=Prim.from_matrix([[0.25], [0.25], [0.25], [0.25]]),
            name="_two_flips_",
        )
        captured = []

        @DPP
        def test_two_out() -> None:
            r = two_flips()
            captured.append(r)

        assert isinstance(captured[0], tuple)
        assert len(captured[0]) == 2
        assert all(isinstance(v, Bool) for v in captured[0])


# Nested models, for the structural tests below.

@DPP
def _sub_inner() -> Bool:
    return flip(0.3)


@DPP
def _sub_outer() -> Bool:
    x = _sub_inner()
    return ~x


@DPP
def _sub_diamond() -> Bool:
    """Calls _sub_inner twice to create a diamond dependency."""
    x = _sub_inner()
    y = _sub_inner()
    return x & y


class TestCompiled:

    def test_compiling_twice_returns_the_same_object(self):
        """compiled() must return the identical Primitive on every call."""
        @DPP
        def to_cache() -> Bool:
            return flip(0.4)

        c1 = to_cache.compiled()
        c2 = to_cache.compiled()
        assert c1 is c2

    def test_compiled_returns_a_compiled_object(self):
        """compiled() returns a Primitive, not a DPP."""
        @DPP
        def simple() -> Bool:
            return flip(0.6)

        assert isinstance(simple.compiled(), Compiled)


class TestContextRestoration:

    def test_a_failed_definition_leaves_no_definition_active(self):
        """
        If a DPP body raises, the active definition must be restored so that
        a subsequent DPP definition succeeds normally.
        """
        with pytest.raises(TypeError):
            @DPP
            def broken() -> None:
                return "not a variable"  # type: ignore[return-value]

        # This DPP should be constructed without error.
        @DPP
        def healthy() -> Bool:
            return flip(0.5)

        assert healthy.n_outputs() == 1

    def test_a_failed_inner_definition_restores_the_outer_one(self):
        """
        A failure inside an inner @DPP constructed within an outer @DPP body
        propagates out but leaves the outer DPP context consistent enough that
        a fresh definition succeeds afterwards.
        """
        with pytest.raises(TypeError):
            @DPP
            def outer_broken() -> Bool:
                @DPP
                def inner_broken() -> Bool:
                    return "bad"  # type: ignore[return-value]

                return flip(0.5)

        @DPP
        def after() -> Bool:
            return flip(0.7)

        assert after.n_outputs() == 1


class TestOutputVariableOwner:

    def test_returning_an_outer_definitions_variable_is_refused(self):
        """
        Returning a variable that belongs to an enclosing DPP from a nested
        @DPP raises ValueError with a message identifying the mismatch.
        """
        with pytest.raises(ValueError, match="belongs to DPP"):
            @DPP
            def outer() -> Bool:
                x = flip(0.5)

                @DPP
                def inner() -> Bool:
                    return x  # x was created inside 'outer', not 'inner'

                return inner()


class TestPrimitiveValidation:

    def test_a_primitive_whose_declared_input_bits_disagree_is_refused(self):
        """Fewer declared input bits than the primitive takes is rejected."""
        with pytest.raises(ValueError, match="but the primitive has"):
            Primitive(
                input_types=(Bool,),          # 1 bit
                output_types=(Bool,),
                prim=Prim.and_(),             # takes 2
                name="bad_input",
            )

    def test_a_primitive_whose_declared_output_bits_disagree_is_refused(self):
        """More declared output bits than the primitive produces is rejected."""
        with pytest.raises(ValueError, match="but the primitive has"):
            Primitive(
                input_types=(Bool, Bool),
                output_types=(Bool, Bool),    # 2 bits expected
                prim=Prim.and_(),             # produces 1
                name="bad_output",
            )

    def test_exactly_one_of_prim_and_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            Primitive(input_types=(), output_types=(Bool,), name="neither")


class TestBoolGuard:

    def test_bool_cast_raises_type_error(self):
        """
        Casting a Bool random variable to a Python bool must raise TypeError,
        not silently return True or False.
        """
        @DPP
        def _holder() -> Bool:
            return flip(0.5)

        x = _holder.output_variables()[0]
        with pytest.raises(TypeError):
            bool(x)

    def test_bool_in_if_raises_type_error(self):
        """Using a Bool variable as an 'if' condition raises TypeError."""
        @DPP
        def _holder2() -> Bool:
            return flip(0.5)

        x = _holder2.output_variables()[0]
        with pytest.raises(TypeError):
            if x:  # noqa: intentionally testing the TypeError
                pass


class TestNames:
    """The names a diagram labels a box with, and how they survive interning."""

    def test_a_model_is_named_after_its_function_unless_told_otherwise(self):
        def gate(a: Bool) -> Bool:
            return ~a

        assert DPP(gate).name() == "gate"
        assert DPP(gate, name="negate").name() == "negate"

    def test_the_name_recorded_for_a_callee_is_the_one_at_the_call_site(self):
        @DPP
        def helper(a: Bool) -> Bool:
            return ~a

        @DPP
        def caller() -> Bool:
            return helper(flip(0.5))

        # A call box takes the name recorded for its body where no name is given per call.
        svg = caller.diagram(op_names={}).svg()
        assert ">helper<" in svg

    def test_two_callees_that_merge_keep_the_name_of_the_first(self):
        """Interning merges bodies that are structurally equal whatever they were called."""
        @DPP
        def first() -> Bool:
            return flip(0.5)

        @DPP
        def second() -> Bool:
            return flip(0.5)

        @DPP
        def both() -> Bool:
            return first() & second()

        assert len(both.program().bodies) == 2          # the two callees merged
        svg = both.diagram(op_names={}).svg()
        assert ">first<" in svg and ">second<" not in svg

    def test_a_gate_and_a_coin_flip_have_no_name_recorded(self):
        """A gate and a coin flip are drawn and printed as themselves, so a name for them
        would be unused."""
        @DPP
        def plain() -> Bool:
            return flip(0.5) & flip(0.25)

        assert getattr(plain, "_DPP__own_op_names") == {}


class TestSimplified:
    """`DPP.simplified`: the same model, with the passes applied, still a model."""

    def test_it_means_the_same_as_the_model_it_came_from(self):
        import numpy as np
        from test.syntax.helpers import matrix_of

        @DPP
        def noisy(x: Bool) -> Bool:
            y = x & flip(0.9)
            return y | (~x & flip(0.1)) | (y & y)

        assert np.allclose(matrix_of(noisy), matrix_of(noisy.simplified()))

    def test_it_keeps_the_interface_including_an_input_nothing_reads(self):
        """Dead code elimination narrows a callee, never the root, and never an input."""
        @DPP
        def ignores(a: Bool, b: Bool) -> Bool:
            return b

        simplified = ignores.simplified()
        assert simplified.input_types() == ignores.input_types()
        assert simplified.output_types() == ignores.output_types()

    def test_it_is_built_once_and_reused(self):
        @DPP
        def coin() -> Bool:
            return flip(0.25)

        assert coin.simplified() is coin.simplified()

    def test_it_can_be_called_from_another_definition(self):
        @DPP
        def channel(x: Bool) -> Bool:
            return x & flip(0.9) | ~x & flip(0.1)

        relayed = channel.simplified()

        @DPP
        def through() -> Bool:
            return relayed(flip(0.5))

        from pylea import prob
        assert round(prob(through), 6) == 0.5

    def test_it_keeps_the_name_and_the_parameter_names(self):
        @DPP
        def screen(carrier: Bool) -> Bool:
            return carrier & flip(0.99)

        simplified = screen.simplified()
        assert simplified.name() == "screen"
        assert simplified._input_labels() == {0: "carrier"}

    def test_a_program_that_does_not_fit_the_declared_types_is_refused(self):
        @DPP
        def one_output() -> Bool:
            return flip(0.5)

        with pytest.raises(ValueError, match="output bits"):
            DPP._from_program(one_output.program(), (), (Bool, Bool), "mismatched")


_PICK = True


def _two_returns(a: Bool) -> Bool:
    if _PICK:
        y = ~a
        return y
    z = a & a
    return z


class TestReturnedNames:
    """What a model returned, read from its source so that an output wire can be named."""

    def test_a_returned_variable_gives_its_name(self):
        @DPP
        def screened() -> Bool:
            carrier = flip(0.01)
            return carrier

        assert screened._returned() == ("carrier",)
        assert screened._output_labels() == {0: "carrier"}

    def test_each_element_of_a_returned_tuple_is_named(self):
        @DPP
        def pair() -> tuple[Bool, Bool]:
            head = flip(0.5)
            tail = ~head
            return head, tail

        assert pair._returned() == ("head", "tail")

    def test_a_returned_expression_has_no_name_to_show(self):
        @DPP
        def combined(a: Bool, b: Bool) -> Bool:
            return a & b

        assert combined._returned() == (None,)
        assert combined._output_labels() == {}

    def test_a_model_with_two_return_statements_names_neither(self):
        """Which of them ran is not a property of the source, so reading one would guess."""
        assert DPP(_two_returns)._returned() == (None,)

    def test_a_return_inside_a_nested_function_is_not_the_model_s(self):
        @DPP
        def outer(a: Bool) -> Bool:
            def double(x):
                return x

            result = double(a) & a
            return result

        assert outer._returned() == ("result",)

    def test_a_model_whose_source_cannot_be_read_names_nothing(self):
        namespace: dict = {"Bool": Bool, "DPP": DPP}
        exec("def hidden(a: Bool) -> Bool:\n    b = ~a\n    return b\n", namespace)
        assert DPP(namespace["hidden"])._returned() == (None,)

    def test_a_returned_value_of_several_bits_is_one_named_position(self):
        from pylea import Enum

        class Colour(Enum):
            elements = ["R", "G", "B"]

        @DPP
        def drawn() -> Colour:
            shade = Colour.uniform()
            return shade

        assert drawn._output_labels() == {0: "shade"}

    def test_a_simplified_model_keeps_the_names_of_what_it_returns(self):
        @DPP
        def wasteful() -> Bool:
            answer = flip(0.3) & flip(0.3)
            return answer

        assert wasteful.simplified()._output_labels() == {0: "answer"}

    def test_simplifying_does_not_read_the_source(self, monkeypatch):
        @DPP
        def wasteful() -> Bool:
            answer = flip(0.3) & flip(0.3)
            return answer

        import pylea.syntax.dpp as module
        read, original = [], module.getsource
        monkeypatch.setattr(module, "getsource", lambda f: read.append(f) or original(f))
        simplified = wasteful.simplified()
        assert read == []
        assert simplified._output_labels() == {0: "answer"}
        assert len(read) == 1


def test_a_dpp_calling_itself_is_refused_with_its_name():
    from pylea.syntax.dpp import active_definition

    def itself() -> Bool:
        definition = active_definition()
        assert definition is not None
        return definition()

    with pytest.raises(ValueError, match="'itself' is called inside its own definition"):
        DPP(itself)


def test_definitions_in_several_threads_do_not_interfere():
    import sys
    import threading

    from pylea import prob

    errors: list = []
    results: dict = {}

    def define(k: int) -> None:
        p = 0.01 * (k + 1)
        try:
            for _ in range(30):
                @DPP
                def any_of() -> Bool:
                    return Bool.any(flip(p) for _ in range(20))
            results[k] = float(prob(any_of, method="exact"))
        except Exception as exc:                # noqa: BLE001, collected for the assert
            errors.append(exc)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=define, args=(k,)) for k in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
    assert errors == []
    for k, value in results.items():
        assert value == pytest.approx(1 - (1 - 0.01 * (k + 1)) ** 20, rel=1e-12)


_FIXTURES = ["postponed_annotations", "eager_annotations"]


def _fixture(name: str):
    import importlib
    return importlib.import_module(f"test.syntax.{name}")


class TestAnnotations:
    """How parameter and return annotations are looked up and checked."""

    @pytest.mark.parametrize("module", _FIXTURES)
    def test_a_type_local_to_the_defining_function_resolves_under_either_annotation_form(
            self, module):
        from pylea import prob
        assert prob(_fixture(module).local_parameter_type(), method="exact") == 0.25

    @pytest.mark.parametrize("module", _FIXTURES)
    def test_a_return_annotation_naming_a_local_type_is_checked_and_kept(self, module):
        from pylea import probability_mass_function
        payout = _fixture(module).local_return_type()
        assert payout.output_types()[0].__name__ == "Payout"
        assert {k: float(v) for k, v in probability_mass_function(
            payout, method="exact").items()} == {0: 0.5, 10: 0.5}

    @pytest.mark.parametrize("module", _FIXTURES)
    def test_an_unresolvable_return_annotation_does_not_prevent_the_definition(
            self, module):
        import sys
        if module == "eager_annotations" and sys.version_info < (3, 14):
            pytest.skip("before 3.14 the annotation fails where the function is defined")
        assert _fixture(module).unresolvable_return().output_types() == (Bool,)

    @pytest.mark.parametrize("module", _FIXTURES)
    def test_a_parameter_annotation_that_cannot_be_resolved_is_named_in_the_error(
            self, module):
        import sys
        if module == "eager_annotations" and sys.version_info < (3, 14):
            pytest.skip("before 3.14 the annotation fails where the function is defined")
        with pytest.raises(TypeError, match="annotation 'Missing' of parameter 'x'"):
            _fixture(module).unresolvable_parameter()

    def test_a_type_defined_two_functions_up_is_named_in_the_error(self):
        with pytest.raises(TypeError, match="annotation 'Weather' of parameter 'w'"):
            _fixture("postponed_annotations").type_two_functions_up()

    def test_a_generic_annotation_is_refused_with_the_parameter_named(self):
        with pytest.raises(TypeError, match=r"parameter 'xs' is annotated list\[Bool\].*"
                                            r"'types' keyword argument"):
            @DPP
            def bad(xs: list[Bool]) -> Bool:
                return xs[0]

    def test_a_keyword_only_parameter_is_refused(self):
        with pytest.raises(TypeError, match="'verbose' is keyword-only"):
            @DPP
            def bad(x: Bool, *, verbose: Bool) -> Bool:
                return x

    def test_an_annotated_variable_length_parameter_is_refused(self):
        with pytest.raises(TypeError, match=r"'\*xs' is variable-length"):
            @DPP
            def bad(*xs: Bool) -> Bool:
                return xs[0]


class TestReturnAnnotation:

    def test_a_return_annotation_that_disagrees_with_the_returned_type_is_refused(self):
        from pylea import Enum

        class Colour(Enum):
            elements = ["R", "G"]

        with pytest.raises(TypeError, match="returns Bool, but its return annotation "
                                            "says Colour"):
            @DPP
            def bad() -> Colour:
                return flip(0.5)  # type: ignore[return-value]

    def test_a_tuple_return_annotation_is_checked_position_by_position(self):
        from pylea import Enum

        class Colour(Enum):
            elements = ["R", "G"]

        @DPP
        def good() -> tuple[Bool, Colour]:
            return flip(0.5), Colour.uniform()

        assert good.output_types() == (Bool, Colour)
        with pytest.raises(TypeError, match=r"returns \(Colour, Bool\), but its return "
                                            r"annotation says \(Bool, Colour\)"):
            @DPP
            def swapped() -> tuple[Bool, Colour]:
                return Colour.uniform(), flip(0.5)  # type: ignore[return-value]
        with pytest.raises(TypeError, match="returns nothing"):
            @DPP
            def silent() -> Bool:
                flip(0.5)

    def test_a_variable_length_tuple_annotation_accepts_any_number_of_its_type(self):
        @DPP
        def three() -> tuple[Bool, ...]:
            return flip(0.5), flip(0.5), flip(0.5)

        assert len(three.output_types()) == 3

    def test_an_annotation_form_that_names_no_variable_types_is_not_checked(self):
        from typing import Any

        @DPP
        def loose() -> Any:
            return flip(0.5)

        @DPP
        def bare() -> tuple:
            return flip(0.5), flip(0.5)

        @DPP
        def unannotated():
            return flip(0.5)

        assert (loose.output_types(), bare.output_types(), unannotated.output_types()) == (
            (Bool,), (Bool, Bool), (Bool,))


class TestTypesForm:
    """The `types` keyword argument: a plain parameter or a variable-length one."""

    def test_a_variable_length_parameter_receives_the_inputs_one_by_one(self):
        from pylea import prob

        @DPP
        def first_not_second(*xs, types=(Bool, Bool)) -> Bool:
            assert len(xs) == 2
            return xs[0] & ~xs[1]

        @DPP
        def called() -> Bool:
            return first_not_second(flip(0.5), flip(0.25))

        assert prob(called, method="exact") == 0.375

    def test_a_plain_parameter_receives_the_inputs_as_a_tuple(self):
        @DPP
        def first(xs, types=(Bool, Bool)) -> Bool:
            assert isinstance(xs, tuple) and len(xs) == 2
            return xs[0]

        assert first.input_types() == (Bool, Bool)

    def test_a_types_form_with_a_second_parameter_is_refused(self):
        with pytest.raises(ValueError, match="takes one other parameter"):
            @DPP
            def bad(a, b, types=(Bool, Bool)) -> Bool:
                return a


class TestWidePrimitives:
    """Constants and uniform draws of any width, and tables of any size, can be recorded."""

    def test_a_constant_of_a_32_bit_type_is_built_bit_by_bit(self):
        from pylea import UInt, count, prob

        class W32(UInt):
            width = 32

        @DPP
        def three_or_more() -> Bool:
            return count((flip(0.5) for _ in range(4)), data_type=W32) >= 3

        @DPP
        def constant() -> W32:
            return W32.const(0x80000001)

        assert prob(three_or_more, method="exact") == 0.3125
        ops = constant.program().main.ops
        assert [op.callee.out_bits for op in ops] == [1] * 32
        assert [op.callee.params[0] for op in ops] == [1] + [0] * 30 + [1]

    def test_a_constant_of_at_most_eight_bits_is_one_assignment(self):
        from pylea import UInt

        class W8(UInt):
            width = 8

        @DPP
        def constant() -> W8:
            return W8.const(200)

        (op,) = constant.program().main.ops
        assert op.callee.params == (200, 8)

    def test_uniform_bits_of_a_21_bit_type_is_uniform(self):
        from pylea import UInt, prob

        class W21(UInt):
            width = 21

        @DPP
        def below_a_thousand() -> Bool:
            return W21.uniform_bits() < 1000

        assert prob(below_a_thousand, method="exact") == pytest.approx(1000 / 2**21,
                                                                      rel=1e-12)

    def test_a_table_with_21_interface_bits_can_be_called(self):
        import numpy as np
        from pylea import LikelihoodTable, UInt, prob

        class In(UInt):
            width = 10

        class Out(UInt):
            width = 11

        rng = np.random.default_rng(0)
        m = rng.random((1 << 11, 1 << 10))
        m /= m.sum(axis=0)
        table = LikelihoodTable((In,), (Out,), m)

        @DPP
        def top_bit() -> Bool:
            return table(In.uniform_bits()).bits()[0]

        assert prob(top_bit, method="exact") == pytest.approx(m[1024:].sum() / 1024,
                                                              rel=1e-9)

    def test_a_query_of_an_output_wider_than_the_output_bound_is_refused_before_allocating(
            self, monkeypatch):
        from pylea import Enum, UInt, expectation, probability_mass_function
        from pylea.syntax.method import MAX_OUTPUT_BITS

        class Wide(UInt):
            width = MAX_OUTPUT_BITS + 1

        class Many(Enum):
            elements = list(range(2**MAX_OUTPUT_BITS + 1))

        @DPP
        def wide() -> Wide:
            return Wide.const(1)

        @DPP
        def many() -> Many:
            return Many.const(1)

        def never(cls):
            raise AssertionError("decoded() was called")

        monkeypatch.setattr(Wide.encoding(), "decoded", classmethod(never))
        with pytest.raises(ValueError, match=f"2\\*\\*{MAX_OUTPUT_BITS + 1} entries.*"
                                             f"the limit is 2\\*\\*{MAX_OUTPUT_BITS}"):
            expectation(wide)
        with pytest.raises(ValueError, match="the limit is"):
            probability_mass_function(many)


class TestDiagramArguments:
    def test_two_callees_with_equal_bodies_keep_their_own_names(self):
        from pylea.draw.sketch import sketch_of  # noqa: F401  (drawing is available)

        @DPP
        def sensor_a(x: Bool) -> Bool:
            return x & flip(0.5)

        @DPP
        def sensor_b(x: Bool) -> Bool:
            return x & flip(0.5)

        @DPP
        def both() -> tuple[Bool, Bool]:
            x = flip(0.5)
            return sensor_a(x), sensor_b(x)

        svg = both.diagram().svg()
        assert ">sensor_a<" in svg and ">sensor_b<" in svg

    def test_a_keyword_argument_overrides_what_the_dpp_supplies(self):
        @DPP
        def coin() -> Bool:
            return flip(0.5)

        assert ">shown<" in coin.diagram(title="shown").svg()
        assert ">shown<" in coin.dependency_graph(title="shown").svg()


class TestPrinting:
    """What `repr` and `str` show of a DPP."""

    def test_the_repr_of_a_dpp_shows_its_signature(self):
        @DPP
        def test(has_it: Bool) -> Bool:
            return has_it & flip(0.99)

        @DPP
        def pair(xs, types=(Bool, Bool)) -> tuple[Bool, Bool]:
            return xs[1], xs[0]

        @DPP
        def nothing() -> None:
            return None

        assert repr(test) == "<DPP test(has_it: Bool) -> Bool>"
        assert repr(pair) == "<DPP pair(Bool, Bool) -> tuple[Bool, Bool]>"
        assert repr(nothing) == "<DPP nothing() -> None>"

    def test_printing_a_dpp_lists_its_assignments_by_name(self):
        from pylea import Enum
        from pylea.syntax.observe import observe

        class Weather(Enum):
            elements = ["dry", "rain", "snow"]

        @DPP
        def test(has_it: Bool) -> tuple[Bool, Bool]:
            return has_it & flip(0.95), ~has_it

        @DPP
        def screened(today: Weather) -> Bool:
            ill = flip(0.0001)
            positive, _healthy = test(ill)
            observe(positive | (today == "snow"))
            return ill

        lines = str(screened).splitlines()
        assert lines[:3] == ["@DPP", "def screened(today: Weather) -> Bool:",
                             "    w2 = flip(0.0001)"]
        assert lines[3] == "    w3, w4 = test(w2)"
        assert lines[4] == "    w5 = snow"                    # a constant by its value
        assert lines[5] == "    w7 = EQ_Weather(today, w5)"   # a variable of two bits
        assert lines[6:] == ["    w8 = w3 | w7", "    observe(w8)", "    return w2"]

    def test_a_single_bit_of_a_wider_variable_is_written_with_its_position(self):
        from pylea import UInt

        class Two(UInt):
            width = 2

        @DPP
        def both_bits(x: Two) -> Bool:
            high, low = x.bits()
            return high & low

        assert "    w2 = x[0] & x[1]" in str(both_bits).splitlines()

    def test_a_wide_constant_recorded_bit_by_bit_is_one_line(self):
        from pylea import UInt

        class Wide(UInt):
            width = 12

        @DPP
        def fixed() -> Wide:
            return Wide.const(300)

        assert str(fixed).splitlines()[2:] == ["    w0 = 300", "    return w0"]
