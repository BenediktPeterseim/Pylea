"""The `.bif` converter: modules whose variables have the distribution the file describes,
and refusals of what it does not support."""
import itertools
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

import pylea
from pylea import DPP, prob, probability_mass_function
from pylea.utility.bif_to_dpp import bif_to_python, convert_suite

ASIA = Path(__file__).resolve().parents[2] / "examples" / "asia.bif"
PYLEA_ROOT = str(Path(pylea.__file__).resolve().parents[1])

# The Asia network as in `examples/asia.bif`, written out independently of the converter:
# variable -> (parents, {parent states: probability of "yes"}).
ASIA_TABLES = {
    "asia": ([], {(): 0.01}),
    "tub": (["asia"], {("yes",): 0.05, ("no",): 0.01}),
    "smoke": ([], {(): 0.5}),
    "lung": (["smoke"], {("yes",): 0.1, ("no",): 0.01}),
    "bronc": (["smoke"], {("yes",): 0.6, ("no",): 0.3}),
    "either": (["lung", "tub"], {("yes", "yes"): 1.0, ("no", "yes"): 1.0,
                                 ("yes", "no"): 1.0, ("no", "no"): 0.0}),
    "xray": (["either"], {("yes",): 0.98, ("no",): 0.05}),
    "dysp": (["bronc", "either"], {("yes", "yes"): 0.9, ("no", "yes"): 0.7,
                                   ("yes", "no"): 0.8, ("no", "no"): 0.1}),
}


def _module(source: str) -> types.ModuleType:
    mod = types.ModuleType("converted")
    exec(compile(source, "converted", "exec"), mod.__dict__)
    return mod


def _network(source: str) -> DPP:
    """The DPP of the module `source` without inputs, the network function."""
    mod = _module(source)
    (net,) = [v for v in vars(mod).values() if isinstance(v, DPP) and not v.input_types()]
    return net


def _marginals(net: DPP) -> list[dict]:
    """The distribution of each output of `net`."""
    types_ = net.output_types()
    if len(types_) == 1:
        return [probability_mass_function(net, method="exact")]
    out = []
    for i, t in enumerate(types_):
        @DPP
        def pick() -> t:            # noqa: F821
            return net()[i]
        out.append(probability_mass_function(pick, method="exact"))
    return out


def _brute_force(states, parents, rows, order):
    """The marginal of each variable in `order`, by summing the joint over all values."""
    marg = {v: dict.fromkeys(states[v], 0.0) for v in order}
    for values in itertools.product(*(states[v] for v in order)):
        a = dict(zip(order, values))
        w = math.prod(rows[v][tuple(a[p] for p in parents[v])][states[v].index(a[v])]
                      for v in order)
        for v in order:
            marg[v][a[v]] += w
    return [marg[v] for v in order]


def _close(got: list[dict], expected: list[dict], tol: float = 1e-12) -> bool:
    return all(g.keys() == e.keys() and all(abs(g[k] - e[k]) <= tol for k in e)
               for g, e in zip(got, expected, strict=True))


def test_every_marginal_of_asia_matches_brute_force():
    order = list(ASIA_TABLES)
    states = {v: ["yes", "no"] for v in order}
    parents = {v: ASIA_TABLES[v][0] for v in order}
    rows = {v: {k: [p, 1 - p] for k, p in ASIA_TABLES[v][1].items()} for v in order}
    net = _network(bif_to_python(ASIA.read_text()))
    assert _close(_marginals(net), _brute_force(states, parents, rows, order))


def _random_network(seed: int):
    """A random network as `.bif` text, with declarations and rows shuffled, and its
    variables, states, parents and rows."""
    rng = random.Random(seed)
    order = [f"v{i}" for i in range(7)]
    states = {v: [f"s{j}" for j in range(rng.randint(2, 3))] for v in order}
    parents = {v: rng.sample(order[:i], min(i, rng.randint(0, 6))) for i, v in enumerate(order)}
    rows = {}
    for v in order:
        rows[v] = {}
        for key in itertools.product(*(states[p] for p in parents[v])):
            w = [rng.choice([0.0, rng.random(), 1.0]) for _ in states[v]]
            w[rng.randrange(len(w))] += 0.5
            rows[v][key] = [x / math.fsum(w) for x in w]
    declared = order[:]
    rng.shuffle(declared)
    text = ["network random { }"]
    text += [f"variable {v} {{ type discrete [ {len(states[v])} ] {{ {', '.join(states[v])} }}; }}"
             for v in declared]
    for v in declared:
        if not parents[v]:
            text.append(f"probability ( {v} ) {{ table {', '.join(map(repr, rows[v][()]))}; }}")
            continue
        entries = [f"  ({', '.join(k)}) {', '.join(map(repr, r))};" for k, r in rows[v].items()]
        rng.shuffle(entries)
        text += [f"probability ( {v} | {', '.join(parents[v])} ) {{", *entries, "}"]
    return "\n".join(text), order, states, parents, rows


@pytest.mark.parametrize("seed", range(5))
def test_random_networks_with_many_parents_match_brute_force(seed):
    text, order, states, parents, rows = _random_network(seed)
    net = _network(bif_to_python(text))
    expected = _brute_force(states, parents, rows, order)
    position = {v: i for i, v in enumerate(o.__name__.lower() for o in net.output_types())}
    got = _marginals(net)
    assert _close([got[position[v]] for v in order], expected)


NET = """
network n { }
variable a { type discrete [ 2 ] { t, f }; }
variable b { type discrete [ 3 ] { x, y, z }; }
probability ( a ) { table 0.3, 0.7; }
probability ( b | a ) { (t) 0.1, 0.2, 0.7; (f) 0.6, 0.3, 0.1; }
"""


def _refused(text: str, *pieces: str) -> str:
    with pytest.raises(ValueError) as info:
        bif_to_python(text)
    message = str(info.value)
    for piece in pieces:
        assert piece in message, message
    return message


def test_a_row_that_does_not_sum_to_one_is_refused_naming_the_variable_and_row():
    _refused(NET.replace("(f) 0.6, 0.3, 0.1", "(f) 0.3, 0.3, 0.1"),
             "line 6", "'b'", "('f',)", "sum to 0.7")
    _refused(NET.replace("0.3, 0.7", "0.3, 0.3"), "line 5", "'a'", "the table")


def test_a_row_within_the_tolerance_is_accepted():
    text = NET.replace("(t) 0.1, 0.2, 0.7", "(t) 0.1, 0.2, 0.6999997")
    assert _marginals(_network(bif_to_python(text)))[0] == pytest.approx({"t": 0.3, "f": 0.7})


@pytest.mark.parametrize("old, new", [("(t) 0.1, 0.2, 0.7", "(t) 0.1, 0.9"),
                                      ("(t) 0.1, 0.2, 0.7", "(t) 0.1, 0.2, 0.6, 0.1"),
                                      ("table 0.3, 0.7", "table 0.3, 0.6, 0.1")])
def test_a_row_with_the_wrong_number_of_entries_is_refused(old, new):
    _refused(NET.replace(old, new), "has", "probabilities" if "(t)" in old else "table")


def test_a_negative_or_non_numeric_probability_is_refused():
    _refused(NET.replace("0.3, 0.7", "-0.3, 1.3"), "'a'", "negative")
    _refused(NET.replace("0.3, 0.7", "0.3, x"), "line 5", "expected a probability")
    _refused(NET.replace("0.3, 0.7", "0.3, 1_0"), "line 5", "expected a probability")


def test_a_declared_state_count_that_differs_from_the_states_is_refused():
    _refused(NET.replace("[ 2 ]", "[ 3 ]"), "line 3", "'a' declares 3 states and lists 2")


def test_a_state_listed_twice_is_refused():
    _refused(NET.replace("{ x, y, z }", "{ x, y, x }"), "line 4", "'x' twice")


def test_states_keep_their_names():
    names = ["0-3_days", "12+", "<5", "2.5", "a-b", "a.b"]
    text = (f"variable age {{ type discrete [ 6 ] {{ {', '.join(names)} }}; }}\n"
            "probability ( age ) { table 0.1, 0.1, 0.1, 0.1, 0.2, 0.4; }")
    pmf = _marginals(_network(bif_to_python(text)))[0]
    assert pmf == pytest.approx(dict(zip(names, [0.1, 0.1, 0.1, 0.1, 0.2, 0.4])))


def test_states_separated_by_spaces_are_several_states():
    text = NET.replace("{ t, f }", "{ t f }")
    assert _marginals(_network(bif_to_python(text)))[0] == pytest.approx({"t": 0.3, "f": 0.7})


def test_a_block_comment_is_ignored():
    text = ("network n { }\n/* variable ghost { type discrete [ 2 ] { t, f }; }\n"
            "   probability ( ghost ) { table 0.5, 0.5; } */\n" + NET.split("\n", 2)[2]
            + "// probability ( a ) { table 0.5, 0.5; }\n")
    net = _network(bif_to_python(text))
    assert len(net.output_types()) == 2
    assert _marginals(net)[0] == pytest.approx({"t": 0.3, "f": 0.7})


def test_a_default_row_fills_every_parent_configuration_not_listed():
    text = NET.replace("(f) 0.6, 0.3, 0.1", "default 0.6, 0.3, 0.1")
    assert _marginals(_network(bif_to_python(text)))[1] == pytest.approx(
        {"x": 0.3 * 0.1 + 0.7 * 0.6, "y": 0.3 * 0.2 + 0.7 * 0.3, "z": 0.3 * 0.7 + 0.7 * 0.1})


def test_the_table_form_with_parents_follows_the_format():
    # The child's first state for every parent configuration, then its second, the last
    # parent changing fastest, as the format draft counts and pgmpy's BIFReader reads it.
    text = NET.replace("(t) 0.1, 0.2, 0.7; (f) 0.6, 0.3, 0.1;", "table 0.1, 0.6, 0.2, 0.3, 0.7, 0.1;")
    assert _marginals(_network(bif_to_python(text)))[1] == pytest.approx(
        {"x": 0.3 * 0.1 + 0.7 * 0.6, "y": 0.3 * 0.2 + 0.7 * 0.3, "z": 0.3 * 0.7 + 0.7 * 0.1})
    two = """
    variable a { type discrete [ 2 ] { t, f }; }
    variable c { type discrete [ 2 ] { u, w }; }
    variable b { type discrete [ 2 ] { y, n }; }
    probability ( a ) { table 0.5, 0.5; }
    probability ( c ) { table 0.5, 0.5; }
    probability ( b | a, c ) { table 0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6; }
    """
    source = bif_to_python(two)
    assert "('t', 'w'): [0.2, 0.8]" in source and "('f', 'u'): [0.3, 0.7]" in source
    assert _marginals(_network(source))[2] == pytest.approx({"y": 0.25, "n": 0.75})


def test_a_missing_row_without_a_default_is_refused_at_conversion():
    _refused(NET.replace(" (f) 0.6, 0.3, 0.1;", ""), "'b'", "('f',)", "no default")


def test_a_row_naming_a_state_the_parent_does_not_have_is_refused():
    _refused(NET.replace("(f) 0.6", "(g) 0.6"), "'g'", "parent 'a'")


def test_a_row_given_twice_is_refused():
    _refused(NET.replace("(f) 0.6", "(t) 0.6"), "('t',) twice")


def test_a_variable_without_a_table_is_refused_naming_it():
    _refused(NET.replace("probability ( a ) { table 0.3, 0.7; }", ""),
             "line 3", "'a' has no probability block")


def test_a_variable_with_two_tables_is_refused():
    _refused(NET + "probability ( a ) { table 0.5, 0.5; }\n", "second probability block",
             "'a'")


def test_a_table_for_an_undeclared_variable_is_refused():
    _refused(NET + "probability ( c ) { table 0.5, 0.5; }\n", "'c'", "not declared")
    _refused(NET.replace("( b | a )", "( b | c )"), "parent 'c'", "not declared")


def test_a_cycle_is_refused():
    text = NET.replace("probability ( a ) { table 0.3, 0.7; }",
                       "probability ( a | b ) { (x) 0.3, 0.7; (y) 0.3, 0.7; (z) 0.3, 0.7; }")
    _refused(text, "cycle", "'a'", "'b'")


@pytest.mark.parametrize("old, new", [("discrete", "continuous"), ("network", "netwerk"),
                                      ("(t) 0.1", "row (t) 0.1")])
def test_anything_the_converter_does_not_know_is_refused_with_its_line(old, new):
    _refused(NET.replace(old, new, 1), "line ")


def test_variables_whose_names_differ_only_in_case_or_punctuation_stay_apart():
    text = """
    network n { }
    variable p { type discrete [ 2 ] { t, f }; }
    variable x-y { type discrete [ 2 ] { t, f }; }
    variable x_y { type discrete [ 2 ] { t, f }; }
    variable X_y { type discrete [ 2 ] { t, f }; }
    probability ( p ) { table 0.5, 0.5; }
    probability ( x-y | p ) { (t) 0.9, 0.1; (f) 0.9, 0.1; }
    probability ( x_y | p ) { (t) 0.2, 0.8; (f) 0.2, 0.8; }
    probability ( X_y ) { table 0.6, 0.4; }
    """
    got = _marginals(_network(bif_to_python(text)))
    assert [m["t"] for m in got] == pytest.approx([0.5, 0.9, 0.2, 0.6])


@pytest.mark.parametrize("name", ["DPP", "Enum", "enum", "tuple", "a_given_b", "class", "n"])
def test_a_name_the_generated_code_uses_is_renamed(name):
    text = f"""
    network n {{ }}
    variable {name} {{ type discrete [ 2 ] {{ t, f }}; }}
    variable a {{ type discrete [ 2 ] {{ t, f }}; }}
    variable b {{ type discrete [ 2 ] {{ t, f }}; }}
    probability ( b ) {{ table 0.3, 0.7; }}
    probability ( a | b, {name} ) {{ default 0.9, 0.1; (f, t) 0.2, 0.8; }}
    probability ( {name} ) {{ table 0.5, 0.5; }}
    """
    got = _marginals(_network(bif_to_python(text)))
    assert [m["t"] for m in got] == pytest.approx([0.5, 0.3, 0.9 - 0.7 * 0.5 * 0.7])


@pytest.mark.parametrize("name", ["my-net", "class", "DPP", "2x", ""])
def test_an_invalid_network_name_is_refused(name):
    with pytest.raises(ValueError, match="network_name"):
        bif_to_python(NET, network_name=name)


def test_the_network_function_is_named_after_the_file_or_the_argument():
    assert "def n() ->" in bif_to_python(NET)
    assert "def my_net() ->" in bif_to_python(NET.replace("network n", "network my-net"))
    assert "def bayesian_network() ->" in bif_to_python(NET.replace("network n", "network unknown"))
    assert "def chosen() ->" in bif_to_python(NET, network_name="chosen")


def test_a_network_of_one_variable_converts_and_its_query_runs():
    text = "variable a { type discrete [ 2 ] { t, f }; }\nprobability ( a ) { table 0.3, 0.7; }"
    net = _network(bif_to_python(text))

    @DPP
    def query() -> pylea.Bool:
        return net() == "t"
    assert prob(query, method="exact") == pytest.approx(0.3)


def test_the_variables_come_out_in_the_order_of_earlier_versions():
    text = """
    variable c { type discrete [ 2 ] { t, f }; }
    variable b { type discrete [ 2 ] { t, f }; }
    variable a { type discrete [ 2 ] { t, f }; }
    probability ( c | b, a ) { default 0.5, 0.5; }
    probability ( b | a ) { default 0.5, 0.5; }
    probability ( a ) { table 0.5, 0.5; }
    """
    net = _network(bif_to_python(text))
    assert [t.__name__ for t in net.output_types()] == ["A", "B", "C"]


def test_a_table_with_parents_is_built_with_cpt_and_compiles():
    source = bif_to_python(NET)
    assert "B.cpt(" in source
    net = _network(source)
    assert net.compiled() is not None


def test_convert_suite_returns_each_failure_with_its_message(tmp_path):
    src = tmp_path / "networks"
    src.mkdir()
    (src / "good.bif").write_text(NET)
    (src / "bad.bif").write_text(NET.replace("0.3, 0.7", "0.3, 0.3"))
    converted, failed = convert_suite(src, tmp_path / "out", verbose=False)
    assert list(converted) == [src / "good.bif"]
    assert list(failed) == [src / "bad.bif"] and "sum to 0.6" in failed[src / "bad.bif"]
    assert "def good() ->" in (tmp_path / "out" / "good.py").read_text()


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=PYLEA_ROOT)
    return subprocess.run([sys.executable, "-m", "pylea.utility.bif_to_dpp", *args],
                          capture_output=True, text=True, timeout=60, cwd=cwd, env=env)


def test_the_command_line_converts_one_file_to_stdout(tmp_path):
    (tmp_path / "my-net.bif").write_text(NET)
    r = _run("--file", "my-net.bif", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout == bif_to_python(NET, network_name="my_net") + "\n"


def test_the_command_line_converts_a_folder(tmp_path):
    (tmp_path / "nets").mkdir()
    (tmp_path / "nets" / "n.bif").write_text(NET)
    r = _run("--src", "nets", "--dst", "out", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "[nets]" in r.stdout
    assert (tmp_path / "out" / "n.py").exists()


def test_the_command_line_refuses_a_folder_without_files(tmp_path):
    (tmp_path / "empty").mkdir()
    r = _run("--src", "empty", "--dst", "out", cwd=tmp_path)
    assert r.returncode == 2
    assert "no .bif files" in r.stderr
