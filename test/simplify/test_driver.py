"""The rewriting loop, the size measure, and the budget."""
import pytest

from pylea import DPP, Bool, Enum, prob
from pylea.ir.body import Body, BodyBuilder, Op
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern
from pylea.ir.reference import semantics
from pylea.simplify import RULES, SizeDidNotDecrease, dce, fixpoint, simplify, size
from test.simplify.programs import prog, same_matrix


def _redundant() -> Program:
    b = BodyBuilder()
    (x,) = b.emit(Prim.flip(0.5), ())
    (y,) = b.emit(Prim.flip(0.25), ())
    (a,) = b.emit(Prim.and_(), (x, y))
    (c,) = b.emit(Prim.and_(), (x, y))
    (_dead,) = b.emit(Prim.flip(0.9), ())
    (r,) = b.emit(Prim.or_(), (a, c))
    return prog(b.finish((), (r,)))


def test_size_counts_one_symbol_per_op_argument_result_body_input_and_output():
    b = BodyBuilder()
    i = b.fresh()
    (x,) = b.emit(Prim.flip(0.5), ())           # 1 + 0 + 1
    (r,) = b.emit(Prim.and_(), (x, i))          # 1 + 2 + 1
    body = b.finish((i,), (r, r))               # 1 + 1 + 2
    assert size(prog(body)) == 2 + 4 + 4


def test_size_counts_a_body_called_twice_once():
    inner = BodyBuilder()
    (p,) = inner.emit(Prim.flip(0.5), ())
    callee = inner.finish((), (p,))
    outer = BodyBuilder([callee])
    (a,) = outer.emit(0, ())
    (c,) = outer.emit(0, ())
    (r,) = outer.emit(Prim.and_(), (a, c))
    assert size(prog(callee, outer.finish((), (r,)))) == (2 + 2) + (2 + 2 + 2 + 4)


def test_simplify_lowers_the_size_and_keeps_the_answer():
    p = _redundant()
    out = simplify(p)
    assert size(out) < size(p)
    assert same_matrix(semantics(out), semantics(p))


def test_simplifying_an_already_simplified_program_changes_nothing():
    once = simplify(_redundant())
    assert simplify(once) == once
    assert fixpoint(once, RULES) == once


def test_an_exhausted_budget_leaves_the_program_alone():
    p = _redundant()
    assert simplify(p, budget_ms=0.0) == intern(p)


def test_a_budget_stops_between_rules_and_not_inside_one():
    """Driven by a clock advancing a unit per reading rather than by wall clock."""
    p = _redundant()
    ticks = iter(range(100))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pylea.simplify.driver.perf_counter", lambda: float(next(ticks)))
        # Reading 0 sets the deadline at 2; readings 1 and 2 admit one rule, then stop.
        one_rule = simplify(p, budget_ms=2000.0)
    assert one_rule == intern(dce(intern(p)))


def test_a_budget_never_changes_the_answer():
    p = _redundant()
    for budget in (0.0, 1e-6, 1e6, None):
        assert same_matrix(semantics(simplify(p, budget_ms=budget)), semantics(p))


def _add_a_coin(p: Program) -> Program:
    """A rule with no certificate: it grows the program every time it runs."""
    b = p.main
    extra = Op(Prim.flip(0.5), (), (len(b.bits),))
    return Program((Body(b.bits + (1,), b.inputs, b.outputs, b.ops + (extra,)),), 0)


def test_a_rule_that_does_not_lower_the_size_is_caught_under_check():
    with pytest.raises(SizeDidNotDecrease, match="_add_a_coin"):
        fixpoint(_redundant(), (_add_a_coin,), check=True)


def test_the_check_is_off_by_default():
    p = _redundant()
    grown = fixpoint(p, (lambda q: q if q.main.ops[-1].callee == Prim.flip(0.5)
                         else _add_a_coin(q),))
    assert len(grown.main.ops) == len(p.main.ops) + 1


def test_equal_bodies_are_merged_by_the_loop():
    def coin():
        b = BodyBuilder()
        (x,) = b.emit(Prim.flip(0.5), ())
        return b.finish((), (x,))

    outer = BodyBuilder([coin(), coin()])
    (a,) = outer.emit(0, ())
    (c,) = outer.emit(1, ())
    (r,) = outer.emit(Prim.and_(), (a, c))
    p = Program((coin(), coin(), outer.finish((), (r,))), 2)
    assert len(simplify(p).bodies) < len(p.bodies)


class _Colour(Enum):
    elements = ["red", "green", "blue", "grey"]


class _Seen(Enum):
    elements = ["no", "yes"]


@DPP
def _seen() -> _Seen:
    """A distribution whose four probabilities sum to 1 + 2**-52 in the order a fold adds
    them, read by a table that is "yes" for every colour."""
    colour = _Colour.random({"red": 0.56, "green": 0.02, "blue": 0.32, "grey": 0.1})
    return _Seen.cpt([_Colour], {(c,): [0.0, 1.0] for c in ("red", "green", "blue", "grey")})(
        colour)


@DPP
def _seen_yes() -> Bool:
    return _seen() == "yes"


def test_a_model_whose_probabilities_sum_to_one_by_round_off_alone_is_simplified():
    simplified = _seen.simplified()
    assert same_matrix(semantics(simplified.program()), semantics(_seen.program()))
    assert prob(_seen_yes, simplify="always") == pytest.approx(1.0)
