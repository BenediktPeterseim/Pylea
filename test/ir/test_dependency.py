"""What the direct dependencies of an interface wire must be, whatever the program.

The relation is checked against the one full inlining gives, which is the definition it
abstracts: with every call spliced in, a dependency is a path through wires that are not
themselves part of the interface, and an interface position whose wire splicing merged
with an earlier position's is a copy of that position.  The rest of the file pins the cases
that definition is easy to get wrong on: a route through a third interface wire, a call
whose results depend on one another, a wire that is both an input and an output, and a
value handed back through definitions returning their arguments.
"""
import numpy as np
import pytest

from pylea import DPP, Bool, Enum, flip, observe
from pylea.ir.body import BodyBuilder
from pylea.ir.dependency import direct_dependencies
from pylea.ir.inline import inline_all
from pylea.ir.prim import Prim
from pylea.ir.program import Program
from pylea.examples import disease, simple, sprinkler
from test.compile.random_programs import random_program

MODELS = [simple.empty_dpp, simple.identity_dpp, simple.unused_input_dpp,
          simple.unused_output_dpp, simple.two_outputs_dpp, simple.dpp_with_observe,
          simple.two_input_output_dpp, simple.color_two_matches, simple.coin_and_color,
          simple.color_random_then_match, disease.test, disease.has_disease,
          sprinkler.sprinkler_network, sprinkler.rain_given_wet_or_damp_grass]


def by_splicing(prog):
    """The relation by interface position, read off the fully spliced body with no summary
    of any call.

    Splicing merges a call's result with the argument or result whose value it carries, so
    two interface positions of the program can name one wire of the spliced body; the later
    of them is a copy of the first.  A key for each position whose wire the program does
    not name at an earlier position.
    """
    body = inline_all(prog, prog.root, lambda _: True)
    assert all(isinstance(op.callee, Prim) for op in body.ops)

    flat = body.inputs + body.outputs
    first: dict[int, int] = {}
    for k, w in enumerate(flat):
        first.setdefault(w, k)
    into = [set() for _ in body.bits]
    for op in body.ops:
        joint = set()
        for w in op.args:
            joint |= {first[w]} if w in first else into[w]
        for w in op.outs:
            into[w] = set(joint)

    wires = prog.main.inputs + prog.main.outputs
    relation = {}
    for k, w in enumerate(wires):
        if wires.index(w) != k:
            continue
        f = first[flat[k]]
        relation[k] = (frozenset() if k < len(prog.main.inputs) else frozenset({f}) if f < k
                       else frozenset(into[flat[k]]))
    return relation


def by_position(deps, wires):
    """`deps` keyed by interface position rather than by wire, so two bodies compare."""
    at = {}
    for i, w in enumerate(wires):
        at.setdefault(w, i)
    return {at[w]: frozenset(at[v] for v in d) for w, d in deps.items()}


def agrees_with_splicing(prog) -> bool:
    return (by_position(direct_dependencies(prog), prog.main.inputs + prog.main.outputs)
            == by_splicing(prog))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name())
def test_the_summaries_agree_with_splicing_every_call_in(model):
    assert agrees_with_splicing(model.program())


def test_the_summaries_agree_with_splicing_on_random_programs():
    for seed in range(200):
        rng = np.random.default_rng(seed)
        prog = random_program(rng, n_bodies=int(rng.integers(1, 4)),
                              n_ops=int(rng.integers(1, 7)))
        assert agrees_with_splicing(prog), f"seed {seed}"


def _reordered(prog):
    """`prog` under two more bodies: one returning the results of the root in reverse, and
    one above it returning all of those but the last."""
    bodies = list(prog.bodies)
    root = prog.main
    for keep in (None, -1):
        b = BodyBuilder(bodies)
        inputs = tuple(b.fresh(root.bits[w]) for w in root.inputs)
        results = b.emit(len(bodies) - 1 if keep == -1 else prog.root, inputs)
        outputs = tuple(reversed(results)) if keep is None else tuple(results[:-1])
        bodies.append(b.finish(inputs, outputs))
        root = bodies[-1] if keep is None else root
    return Program(tuple(bodies), len(bodies) - 1)


def test_the_summaries_agree_with_splicing_when_a_caller_reorders_the_results():
    """One result of a call may be computed from a later one, which a caller's own summary
    has to order before it, and one callee wire returned twice is one value under two
    wires of the caller."""
    compared = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        prog = random_program(rng, n_bodies=int(rng.integers(1, 4)),
                              n_ops=int(rng.integers(1, 7)))
        if len(prog.main.outputs) < 2:
            continue
        prog = _reordered(prog)
        compared += 1
        assert agrees_with_splicing(prog), f"seed {seed}"
    assert compared > 50, "too few programs had two results for this to say much"


def _relayed(prog):
    """`prog` under a body that passes the root's results through a body returning its
    arguments reversed, returns all relayed results but the first, and reads the root's
    first result directly with a table of its own, which must be one bit wide."""
    bodies = list(prog.bodies)
    root = prog.main
    widths = [root.bits[w] for w in root.outputs]
    relay = BodyBuilder(bodies)
    ins = tuple(relay.fresh(width) for width in widths)
    bodies.append(relay.finish(ins, tuple(reversed(ins))))
    top = BodyBuilder(bodies)
    inputs = tuple(top.fresh(root.bits[w]) for w in root.inputs)
    results = top.emit(prog.root, inputs)
    relayed = top.emit(len(bodies) - 1, tuple(results))
    (read,) = top.emit(Prim.from_matrix([[0.3, 0.6], [0.7, 0.4]]), (results[0],))
    bodies.append(top.finish(inputs, tuple(relayed[1:]) + (read,)))
    return Program(tuple(bodies), len(bodies) - 1)


def test_the_summaries_agree_with_splicing_when_results_pass_through_a_relay():
    """A wire read directly and returned through a relay is one value under two wires."""
    compared = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        prog = random_program(rng, n_bodies=int(rng.integers(1, 4)),
                              n_ops=int(rng.integers(1, 7)))
        if len(prog.main.outputs) < 2 or prog.main.bits[prog.main.outputs[0]] != 1:
            continue
        prog = _relayed(prog)
        compared += 1
        assert agrees_with_splicing(prog), f"seed {seed}"
    assert compared > 50, "too few programs had two results for this to say much"


def test_a_result_computed_from_a_later_result_of_the_same_call_keeps_its_dependency():
    @DPP
    def inner(x: Bool) -> tuple[Bool, Bool]:
        y = x & flip(0.5)
        return y, y | flip(0.5)

    @DPP
    def swapped(x: Bool) -> tuple[Bool, Bool]:
        y, z = inner(x)
        return z, y

    @DPP
    def first(x: Bool) -> Bool:
        z, _ = swapped(x)
        return z

    body = first.program().main
    assert direct_dependencies(first.program()) == {
        body.inputs[0]: frozenset(), body.outputs[0]: frozenset({body.inputs[0]})}


def test_a_route_through_a_third_interface_wire_is_not_a_direct_dependency():
    @DPP
    def chain(x: Bool) -> tuple[Bool, Bool]:
        y = x & flip(0.5)
        z = y & flip(0.5)
        return y, z

    body = chain.program().main
    deps = direct_dependencies(chain.program())
    x, (y, z) = body.inputs[0], body.outputs
    assert deps[z] == frozenset({y}) and deps[y] == frozenset({x})


def test_a_route_bypassing_an_interface_wire_is_one():
    """`z` reads `x` itself, so the edge is there as well as the one through `y`."""
    @DPP
    def both_ways(x: Bool) -> tuple[Bool, Bool]:
        y = x & flip(0.5)
        return y, y | x

    body = both_ways.program().main
    deps = direct_dependencies(both_ways.program())
    assert deps[body.outputs[1]] == frozenset({body.inputs[0], body.outputs[0]})


def test_returning_what_a_sub_model_returned_keeps_the_sub_model_s_dependencies():
    """The relation between two results of one call lies inside the callee."""
    @DPP
    def wrapper():
        return sprinkler.sprinkler_network()

    assert (direct_dependencies(wrapper.program())
            == direct_dependencies(sprinkler.sprinkler_network.program()))


def test_an_output_naming_an_input_depends_on_nothing():
    deps = direct_dependencies(simple.identity_dpp.program())
    assert deps == {simple.identity_dpp.program().main.inputs[0]: frozenset()}


def test_an_output_that_is_a_callee_s_argument_depends_on_what_the_argument_does():
    @DPP
    def relay(x: Bool) -> Bool:
        return x

    @DPP
    def uses() -> tuple[Bool, Bool]:
        a = flip(0.5)
        return a, relay(a)

    body = uses.program().main
    deps = direct_dependencies(uses.program())
    assert deps[body.outputs[1]] == frozenset({body.outputs[0]})


def test_conditioning_adds_no_dependency():
    """An `observe` produces no wire, so it cannot carry one variable's value to another."""
    @DPP
    def conditioned() -> tuple[Bool, Bool]:
        a = flip(0.5)
        b = flip(0.5)
        observe(a | b)
        return a, b

    assert all(not d for d in direct_dependencies(conditioned.program()).values())


def test_every_wire_of_a_variable_wider_than_one_bit_is_reported():
    class Colour(Enum):
        elements = ["R", "G", "B"]

    @DPP
    def tinted(c: Colour) -> Colour:
        return c.match({"R": "G", "G": "B", "B": "R"}, data_type=Colour)

    body = tinted.program().main
    deps = direct_dependencies(tinted.program())
    assert set(deps) == set(body.inputs) | set(body.outputs)
    for w in body.outputs:
        assert deps[w] == frozenset(body.inputs)


def test_a_body_other_than_the_root_can_be_asked():
    prog = disease.has_disease.program()
    other = next(i for i in range(len(prog.bodies)) if i != prog.root)
    body = prog.bodies[other]
    deps = direct_dependencies(prog, other)
    assert set(deps) == set(body.inputs) | set(body.outputs)


@DPP
def _both(x: Bool) -> tuple[Bool, Bool]:
    return x, x


def test_an_output_built_through_argument_returning_definitions_depends_on_nothing_it_makes():
    """The value is created by the coin flip; the calls only hand it back."""
    @DPP
    def twice() -> Bool:
        _, b = _both(flip(0.5))
        _, d = _both(b)
        return d

    body = twice.program().main
    assert direct_dependencies(twice.program()) == {body.outputs[0]: frozenset()}


def test_a_value_returned_at_two_positions_makes_the_second_depend_on_the_first():
    @DPP
    def pair(x: Bool) -> tuple[Bool, Bool]:
        return x, _both(x)[0]

    @DPP
    def top() -> tuple[Bool, Bool]:
        return pair(flip(0.5))

    first, second = top.program().main.outputs
    assert direct_dependencies(top.program()) == {first: frozenset(),
                                                   second: frozenset({first})}


def test_nested_argument_returning_definitions_do_not_raise():
    @DPP
    def first(x: Bool, y: Bool) -> Bool:
        return x

    @DPP
    def mid(x: Bool) -> tuple[Bool, Bool]:
        y = first(x, x)
        return x, first(y, y)

    @DPP
    def inner() -> Bool:
        _, b = mid(flip(0.5))
        return first(b, flip(0.5))

    @DPP
    def outer() -> Bool:
        return inner()

    body = outer.program().main
    assert direct_dependencies(outer.program()) == {body.outputs[0]: frozenset()}
    assert len(outer.dependency_graph().sketch.nodes) == 1


def _handing_back(rng) -> Program:
    """A small call graph whose bodies mostly return their arguments or what a call to an
    earlier body returned, which is where one value reaches several wires."""
    bodies: list = []
    for _ in range(int(rng.integers(1, 4))):
        b = BodyBuilder(bodies)
        inputs = [b.fresh() for _ in range(int(rng.integers(0, 3)))]
        pool = list(inputs)
        for _ in range(int(rng.integers(0, 3))):
            if bodies and pool and rng.random() < 0.7:
                c = int(rng.integers(len(bodies)))
                pool += b.emit(c, [pool[int(rng.integers(len(pool)))]
                                   for _ in bodies[c].inputs])
            else:
                pool += b.emit(Prim.flip(0.5), ())
        outputs = ([pool[int(rng.integers(len(pool)))]
                    for _ in range(int(rng.integers(1, 3)))] if pool else [])
        bodies.append(b.finish(inputs, outputs))
    return Program(tuple(bodies), len(bodies) - 1)


def _is_acyclic(deps: dict[int, frozenset[int]]) -> bool:
    waiting = {w: set(d) for w, d in deps.items()}
    settled = True
    while settled:
        settled = [w for w, d in waiting.items() if not d]
        for w in settled:
            del waiting[w]
        for d in waiting.values():
            d.difference_update(settled)
    return not waiting


def test_the_relation_is_acyclic_on_random_programs():
    for seed in range(2000):
        rng = np.random.default_rng(seed)
        prog = (_handing_back(rng) if seed % 2 else
                random_program(rng, n_bodies=int(rng.integers(1, 4)),
                               n_ops=int(rng.integers(1, 7))))
        deps = direct_dependencies(prog)
        assert _is_acyclic(deps), f"seed {seed}"
        assert agrees_with_splicing(prog), f"seed {seed}"
