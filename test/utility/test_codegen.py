"""Names, ordering and folder conversion shared by the converters."""
import random
from pathlib import Path

import pytest

from pylea.utility._codegen import Names, convert_directory, topological_order


def _order_of_repeated_passes(order, parents):
    """The order the converters' sort must reproduce: passes over `order`, each accepting
    an element once all its parents are accepted."""
    out, remaining, done = [], list(order), set()
    while remaining:
        progress = False
        for v in remaining[:]:
            if all(p in done for p in parents.get(v, [])):
                out.append(v)
                done.add(v)
                remaining.remove(v)
                progress = True
        assert progress
    return out


def test_two_names_that_sanitise_alike_get_two_identifiers():
    names = Names()
    assert names.claim("a-b") != names.claim("a_b")
    assert {names.claim("a-b"), names.claim("a_b")} == {"a_b", "a_b_2"}


def test_a_reserved_name_is_never_returned():
    names = Names(reserved=["flip", "DPP", "tuple"])
    got = [names.claim(n) for n in ["flip", "DPP", "class", "tuple"]]
    assert not {"flip", "DPP", "class", "tuple"} & set(got)
    assert all(g.isidentifier() for g in got)


def test_claiming_the_same_name_twice_gives_the_same_identifier():
    names = Names()
    first = names.claim("pump 1")
    names.claim("pump_1")
    assert names.claim("pump 1") == first


def test_a_name_that_is_already_an_identifier_keeps_it_when_claimed_together():
    assert Names().claim_all(["a-b", "a_b"]) == ["a_b_2", "a_b"]


def test_every_identifier_is_valid_and_distinct():
    rng = random.Random(0)
    alphabet = "aA_-. 0é"
    originals = {"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
                 for _ in range(300)}
    for kind in ("name", "class", "variable"):
        names = Names(reserved=["DPP"])
        got = names.claim_all(sorted(originals), kind)
        assert len(set(got)) == len(got)
        assert all(g.isidentifier() and g != "DPP" and not g.startswith("_") for g in got)


def test_only_names_written_differently_are_listed_as_renamed():
    names = Names(reserved=["flip"])
    names.claim_all(["rain", "wet-grass", "flip"])
    names.claim("Rain", "class")
    names.claim("HISTORY", "variable")
    assert names.renamed() == [("flip", "flip_2"), ("wet-grass", "wet_grass")]


def test_the_topological_order_is_the_order_of_repeated_passes():
    assert topological_order(["b", "a", "c"], {"b": ["a"]}) == ["a", "c", "b"]
    for seed in range(500):
        rng = random.Random(seed)
        names = [f"v{i}" for i in range(rng.randint(1, 25))]
        hidden = names[:]
        rng.shuffle(hidden)
        parents = {v: rng.sample(hidden[:i], min(i, rng.randint(0, 4)))
                   for i, v in enumerate(hidden)}
        assert topological_order(names, parents) == _order_of_repeated_passes(names, parents)


def test_a_cycle_is_refused_naming_its_members():
    with pytest.raises(ValueError, match="cycle") as info:
        topological_order(["a", "b", "c", "d"], {"a": ["c"], "c": ["b"], "b": ["a"],
                                                 "d": ["a"]})
    assert all(repr(v) in str(info.value) for v in "abc")
    assert "'d'" not in str(info.value)


def test_a_parent_that_is_not_defined_is_refused_by_name():
    with pytest.raises(ValueError, match="'z'"):
        topological_order(["a"], {"a": ["z"]})


def test_converting_a_folder_reports_each_failure_with_its_message(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a-b.txt").write_text("good")
    (src / "a_b.txt").write_text("good")
    (src / "c.txt").write_text("bad")

    def convert(text, name):
        if text == "bad":
            raise ValueError("line 1: bad")
        return f"NAME = {name!r}"

    converted, failed = convert_directory(src, tmp_path / "out", ".txt", convert,
                                          prefix="f_", label="x", verbose=False)
    assert {p.name: m.name for p, m in converted.items()} == {"a-b.txt": "a_b_2.py",
                                                              "a_b.txt": "a_b.py"}
    assert failed == {src / "c.txt": "line 1: bad"}
    assert (tmp_path / "out" / "__init__.py").exists()
    assert (tmp_path / "out" / "a_b_2.py").read_text() == "NAME = 'a_b_2'\n"
