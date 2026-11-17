"""The `.dft` converter: modules whose top event has the probability the file describes,
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
from pylea import DPP, prob
from pylea.utility.dft_to_dpp import convert_suite, dft_to_python

RBC = Path(__file__).resolve().parents[2] / "examples" / "rbc.dft"
PYLEA_ROOT = str(Path(pylea.__file__).resolve().parents[1])


def _top(source: str) -> DPP:
    """The DPP the module `source` defines."""
    mod = types.ModuleType("converted")
    exec(compile(source, "converted", "exec"), mod.__dict__)
    (fn,) = [v for v in vars(mod).values() if isinstance(v, DPP)]
    return fn


def _p(text: str, **kwargs) -> float:
    return prob(_top(dft_to_python(text, **kwargs)), method="exact")


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=PYLEA_ROOT)
    return subprocess.run([sys.executable, "-m", "pylea.utility.dft_to_dpp", *args],
                          capture_output=True, text=True, timeout=60, cwd=cwd, env=env)


BASIC = 'toplevel "T";\n"T" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n'


def test_the_rbc_tree_gives_the_probability_of_its_top_event():
    hours = 24 * 365
    q = lambda rate: -math.expm1(-rate * hours)  # noqa: E731
    ps, wan, bus, gsm, cpu, fpga = (q(r) for r in (1.8182e-5, 2.5e-6, 4.4444e-6, 5.7078e-6,
                                                     7.4074e-6, 3.003e-9))
    cpus = 3 * cpu ** 2 * (1 - cpu) + cpu ** 3
    tmr = 1 - (1 - cpus) * (1 - fpga ** 2)
    expected = 1 - (1 - ps ** 3) * (1 - wan ** 2) * (1 - bus ** 2) * (1 - gsm ** 2) * (1 - tmr)
    got = _p(RBC.read_text(), lambda_t=hours)
    assert got == pytest.approx(expected, rel=1e-12)


def test_the_rbc_module_keeps_its_assignment_order():
    source = dft_to_python(RBC.read_text(), lambda_t=8760)
    assigned = [line.split(" = ")[0].strip() for line in source.splitlines()
                if " = " in line and "flip(" not in line]
    assert assigned == ["Power", "WANinterface", "SystemBUS", "GSMRinterface", "voter",
                        "CPUcore", "TMR", "System"]


def _random_tree(seed: int):
    """A random tree as `.dft` text in shuffled statement order, and its brute-force
    probability."""
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    events = {f"e{i}": rng.choice([0.0, 1.0, rng.random(), rng.random()]) for i in range(n)}
    gates: dict[str, tuple] = {}
    for j in range(rng.randint(1, 8)):
        pool = list(events) + list(gates)
        children = rng.sample(pool, rng.randint(1, min(4, len(pool))))
        kind = rng.choice(["and", "or", "vote"])
        gates[f"g{j}"] = (rng.randint(1, len(children)) if kind == "vote" else kind, children)
    top = f"g{len(gates) - 1}"
    statements = [f'"{e}" prob={p!r};' for e, p in events.items()]
    for g, (kind, children) in gates.items():
        word = f"{kind}of{len(children)}" if isinstance(kind, int) else kind
        statements.append(f'"{g}" {word} ' + " ".join(f'"{c}"' for c in children) + ";")
    rng.shuffle(statements)
    text = f'toplevel "{top}";\n' + "\n".join(statements)

    total = 0.0
    for values in itertools.product([False, True], repeat=n):
        state = dict(zip(events, values))
        weight = math.prod(events[e] if v else 1 - events[e] for e, v in state.items())
        for g, (kind, children) in gates.items():
            hits = sum(state[c] for c in children)
            state[g] = (hits == len(children) if kind == "and" else hits >= 1 if kind == "or"
                        else hits >= kind)
        total += weight * state[top]
    return text, total


@pytest.mark.parametrize("seed", range(12))
def test_and_or_and_voting_gates_match_brute_force(seed):
    text, expected = _random_tree(seed)
    assert _p(text) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("gate", ["pand", "por", "wsp", "csp", "hsp", "spare", "seq", "fdep",
                                  "mutex", "2insp4", "xor", "pdep=0.5"])
@pytest.mark.parametrize("where", ["top", "below"])
def test_a_gate_kind_the_converter_does_not_support_is_refused_by_name(gate, where):
    if where == "top":
        text = f'toplevel "T";\n"T" {gate} "a" "b";\n"a" prob=0.1;\n"b" prob=0.2;\n'
        line, name = 2, '"T"'
    else:
        text = (f'toplevel "T";\n"T" or "X" "a";\n"X" {gate} "a" "b";\n"a" prob=0.1;\n'
                f'"b" prob=0.2;\n')
        line, name = 3, '"X"'
    keyword = gate.split("=")[0]
    with pytest.raises(ValueError) as info:
        dft_to_python(text)
    assert f"line {line}:" in str(info.value)
    assert name in str(info.value) and keyword in str(info.value)


@pytest.mark.parametrize("attribute", ["repair=0.1", "res=0.5", "cov=0.9", "interval=2",
                                       "mu=1"])
def test_an_event_attribute_the_converter_does_not_support_is_refused_by_name(attribute):
    text = BASIC.replace('"a" prob=0.5', f'"a" lambda=0.5 {attribute}')
    with pytest.raises(ValueError, match=f"line 3: .*{attribute.split('=')[0]}"):
        dft_to_python(text)


@pytest.mark.parametrize("event", ['"a" prob=1.5', '"a" prob=-0.1', '"a" lambda=-1',
                                   '"a" prob=0.5 lambda=1', '"a" dorm=0.5', '"a" prob=nan',
                                   '"a" prob=0.5 prob=0.5', '"a" prob=0.5 phases=2',
                                   '"a" lambda=1 phases=0', '"a" lambda=1 phases=1.5'])
def test_an_invalid_basic_event_is_refused(event):
    with pytest.raises(ValueError, match='line 3: .*"a"'):
        dft_to_python(BASIC.replace('"a" prob=0.5', event))


@pytest.mark.parametrize("lambda_t", [-1.0, math.inf, math.nan])
def test_a_mission_time_that_is_negative_or_not_finite_is_refused(lambda_t):
    with pytest.raises(ValueError, match="lambda_t"):
        dft_to_python(BASIC, lambda_t=lambda_t)


def test_a_toplevel_naming_no_element_uses_the_only_root_and_warns():
    text = 'toplevel "baobab3";\n"r1" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n'
    with pytest.warns(UserWarning, match='"baobab3".*"r1"'):
        source = dft_to_python(text)
    assert source.splitlines()[0] == ('# toplevel names "baobab3", which the file does not '
                                      'define; using its only root "r1"')
    assert prob(_top(source), method="exact") == pytest.approx(0.1)


def test_a_toplevel_naming_no_element_is_refused_where_the_root_is_not_unique():
    text = 'toplevel "X";\n"T" and "a" "b";\n"U" or "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n'
    with pytest.raises(ValueError, match='line 1: toplevel names "X".*"T", "U"'):
        dft_to_python(text)


def test_a_file_without_toplevel_uses_its_only_root():
    assert _p('"T" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n') == pytest.approx(0.1)


def test_a_file_without_toplevel_is_refused_where_the_root_is_not_unique():
    with pytest.raises(ValueError, match='no toplevel.*"T", "U"'):
        dft_to_python('"T" and "a" "b";\n"U" or "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n')


def test_a_toplevel_naming_a_basic_event_converts_to_that_event():
    assert _p('toplevel "a";\n"a" prob=0.5;\n') == 0.5


def test_only_the_elements_the_top_depends_on_are_defined():
    source = dft_to_python(BASIC + '"U" or "a" "c";\n"c" prob=0.3;\n')
    assert "c = " not in source and "U = " not in source
    assert prob(_top(source), method="exact") == pytest.approx(0.1)


def test_a_statement_over_several_lines_is_read_as_one():
    text = 'toplevel "T";\n"T" and "a"\n   "b";\n"S" or "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n'
    assert _p(text) == pytest.approx(0.1)


def test_two_statements_on_one_line_are_both_read():
    text = 'toplevel "T";\n"T" and "a" "b";\n"a" prob=0.5; "b" prob=0.2;\n'
    assert _p(text) == pytest.approx(0.1)


@pytest.mark.parametrize("text", [
    'toplevel "T";\n"T" and "a" "b";\n/*\n"T" or "a" "b";\n*/\n"a" prob=0.5;\n"b" prob=0.2;\n',
    'toplevel "T";\n/* old:\n"T" or "a" "b"; */ "T" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n',
    'toplevel "T"; // "T" or "a" "b";\n"T" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2;\n',
])
def test_a_block_comment_is_ignored_wherever_it_ends(text):
    assert _p(text) == pytest.approx(0.1)


def test_an_unclosed_block_comment_is_refused():
    with pytest.raises(ValueError, match="line 3: .*never closed"):
        dft_to_python(BASIC.replace('"a" prob=0.5;', '/* "a" prob=0.5;'))


def test_names_that_differ_only_in_punctuation_stay_two_events():
    text = 'toplevel "T";\n"T" and "a-b" "a_b";\n"a-b" prob=0.5;\n"a_b" prob=0.2;\n'
    source = dft_to_python(text)
    assert '#     "a-b" -> a_b_2' in source
    assert prob(_top(source), method="exact") == pytest.approx(0.1)


def test_a_quoted_name_with_spaces_is_one_name():
    text = 'toplevel "T";\n"T" or "pump a" "pump b";\n"pump a" prob=0.1;\n"pump b" prob=0.2;\n'
    assert _p(text) == pytest.approx(0.28)


@pytest.mark.parametrize("name", ["flip", "at_least", "DPP", "Bool", "class", "_x", "2nd"])
def test_a_name_the_generated_code_uses_is_renamed(name):
    text = (f'toplevel "T";\n"T" or "{name}" "G";\n"G" 2of2 "{name}" "b";\n'
            f'"{name}" prob=0.1;\n"b" prob=0.2;\n')
    assert _p(text) == pytest.approx(0.1)


def test_a_gate_named_like_a_function_of_the_module_converts():
    text = 'toplevel "T";\n"T" or "at_least" "b";\n"at_least" 2of2 "a" "b";\n' \
           '"a" prob=0.1;\n"b" prob=0.2;\n'
    assert _p(text) == pytest.approx(0.2)


def test_a_voting_gate_whose_count_differs_from_its_children_is_refused():
    with pytest.raises(ValueError, match='line 2: voting gate "T" is 2of4 but has 2 children'):
        dft_to_python(BASIC.replace('"T" and', '"T" 2of4'))


@pytest.mark.parametrize("word", ["0of2", "3of2"])
def test_a_voting_gate_needing_no_child_or_more_than_all_is_refused(word):
    with pytest.raises(ValueError, match=f'line 2: voting gate "T" is {word}'):
        dft_to_python(BASIC.replace('"T" and', f'"T" {word}'))


def test_an_element_defined_twice_is_refused():
    with pytest.raises(ValueError, match='line 4: "a" is defined twice, first on line 3'):
        dft_to_python(BASIC.replace('"b" prob=0.2;', '"a" prob=0.1;\n"b" prob=0.2;'))


def test_a_child_never_defined_is_refused_by_name():
    with pytest.raises(ValueError, match='line 2: "c" is used by "T" but never defined'):
        dft_to_python(BASIC.replace('"a" "b"', '"a" "c"'))


def test_a_cycle_is_refused_naming_its_members():
    text = 'toplevel "T";\n"T" and "G" "a";\n"G" or "T" "a";\n"a" prob=0.3;\n'
    with pytest.raises(ValueError, match="cycle.*'G', 'T'"):
        dft_to_python(text)


def test_a_rate_becomes_the_probability_of_failure_within_the_mission_time():
    source = dft_to_python('toplevel "a";\n"a" lambda=3.003e-9;\n', lambda_t=1.0)
    literal = source.split("flip(")[1].split(")")[0]
    assert float(literal) == 3.0029999954909955e-09     # 1 - exp(-x) to 1 ulp, by series
    assert _p('toplevel "a";\n"a" lambda=0.25;\n', lambda_t=2.0) == \
        pytest.approx(1 - math.exp(-0.5), rel=1e-15)


def test_a_rate_with_phases_follows_the_erlang_distribution():
    x, k = 0.5 * 3.0, 3
    expected = 1 - sum(math.exp(-x) * x ** n / math.factorial(n) for n in range(k))
    got = _p('toplevel "a";\n"a" lambda=0.5 phases=3 dorm=0;\n', lambda_t=3.0)
    assert got == pytest.approx(expected, rel=1e-12)


def test_keywords_are_read_in_any_case():
    text = 'TOPLEVEL "T";\n"T" AND "a" "G";\n"G" 1OF2 "a" "b";\n"a" PROB=0.5;\n"b" Prob=0.2;\n'
    assert _p(text) == pytest.approx(0.5)


def test_spaces_around_the_equals_sign_are_allowed():
    assert _p(BASIC.replace("prob=0.5", "prob = 0.5")) == pytest.approx(0.1)


def test_unquoted_names_tabs_and_crlf_are_read():
    text = 'toplevel\tT;\r\nT\tand\ta b;\r\na prob=0.5;\r\nb\tprob=0.2;\r\n'
    assert _p(text) == pytest.approx(0.1)


@pytest.mark.parametrize("text, line", [
    ('toplevel "T";\n\n"T" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2 extra;\n', 5),
    ('toplevel "T";\n"T" and "a" "b";\n/* a\ncomment */ "a" prob=7;\n"b" prob=0.2;\n', 4),
    ('toplevel "T";\n"T" and "a" "b";\n"a" prob=0.5;\n"b" prob=0.2\n', 4),
])
def test_an_error_names_the_line_of_the_statement(text, line):
    with pytest.raises(ValueError, match=f"^line {line}: "):
        dft_to_python(text)


def test_an_event_with_prob_and_dorm_converts():
    assert _p(BASIC.replace('"a" prob=0.5', '"a" prob=0.5 dorm=0')) == pytest.approx(0.1)


def test_the_same_file_always_gives_the_same_module(tmp_path):
    (tmp_path / "rbc.dft").write_text(RBC.read_text())
    outputs = set()
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONPATH=PYLEA_ROOT, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-m", "pylea.utility.dft_to_dpp", "--src",
                            "rbc.dft"], capture_output=True, text=True, timeout=60,
                           cwd=tmp_path, env=env)
        assert r.returncode == 0, r.stderr
        outputs.add(r.stdout)
    assert len(outputs) == 1
    assert outputs == {dft_to_python(RBC.read_text()) + "\n"}


def test_convert_suite_returns_each_failure_with_its_message(tmp_path):
    src = tmp_path / "trees"
    src.mkdir()
    (src / "good.dft").write_text(BASIC)
    (src / "bad.dft").write_text(BASIC.replace("and", "csp"))
    converted, failed = convert_suite(src, tmp_path / "out", verbose=False)
    assert list(converted) == [src / "good.dft"]
    assert list(failed) == [src / "bad.dft"] and "csp" in failed[src / "bad.dft"]


def test_the_command_line_converts_one_file_to_stdout(tmp_path):
    (tmp_path / "t.dft").write_text(BASIC)
    r = _run("--src", "t.dft", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout == dft_to_python(BASIC) + "\n"


def test_the_command_line_converts_a_folder_and_its_sub_folders(tmp_path):
    (tmp_path / "suite").mkdir()
    (tmp_path / "suite" / "t.dft").write_text(BASIC)
    r = _run("--src", ".", "--dst", "out", "--recursive", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out" / "suite" / "t.py").read_text() == dft_to_python(BASIC) + "\n"
    r = _run("--src", "suite", "--dst", "flat", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "flat" / "t.py").exists()


def test_the_command_line_refuses_a_folder_without_files(tmp_path):
    (tmp_path / "empty").mkdir()
    r = _run("--src", "empty", "--dst", "out", cwd=tmp_path)
    assert r.returncode == 2
    assert "no .dft files" in r.stderr
