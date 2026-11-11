"""The whole simplifier, and each rule alone, against exact enumeration."""
import pytest

from pylea.ir.program import intern, op_count, validate
from pylea.ir.reference import semantics
from pylea.simplify import RULES, simplify, size
from test.simplify.programs import (
    circuits, mixed_corpus, random_corpus, same_matrix,
)

CORPUS = (random_corpus() + mixed_corpus() + mixed_corpus(1, 60, near_stochastic=True)
          + circuits() + circuits(observe=True))


@pytest.mark.parametrize("prog", CORPUS, ids=range(len(CORPUS)))
def test_simplifying_a_random_program_preserves_its_matrix(prog):
    out = simplify(prog, check=True)
    validate(out)
    want, got = semantics(prog), semantics(out)
    assert got.shape == want.shape
    assert same_matrix(got, want)


def test_every_rule_alone_preserves_the_matrix():
    """A mismatch then names the rule, which the whole-phase test above cannot."""
    for prog in CORPUS:
        want = semantics(prog)
        for rule in RULES:
            out = rule(prog)
            validate(out)
            assert same_matrix(semantics(out), want), rule.__name__


def test_every_rule_lowers_the_size_whenever_it_changes_the_program():
    for prog in CORPUS:
        for rule in RULES:
            out = rule(prog)
            if out is not prog and intern(out) != prog:
                assert size(intern(out)) < size(prog), rule.__name__


def test_the_corpus_gets_simplified():
    smaller = sum(1 for p in CORPUS if op_count(simplify(p)) < op_count(p))
    assert smaller > len(CORPUS) // 2


def test_the_generators_cover_every_rule():
    fires = {r.__name__: 0 for r in RULES}
    for prog in CORPUS:
        for rule in RULES:
            if rule(prog) is not prog and intern(rule(prog)) != prog:
                fires[rule.__name__] += 1
    assert all(n > 0 for n in fires.values()), fires


def test_a_program_that_conditions_still_agrees_with_enumeration():
    conditioned = [p for p in CORPUS
                   if any(not op.callee.is_discardable
                          for b in p.bodies for op in b.ops if not isinstance(op.callee, int))]
    assert len(conditioned) > len(CORPUS) // 4
    for prog in conditioned:
        assert same_matrix(semantics(simplify(prog, check=True)), semantics(prog))
