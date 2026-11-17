"""# *Pylea:* discrete probabilistic programming in Python.

Pylea is a library for computational discrete probability theory. 
Given an arbitrary probabilistic model in which every variable can 
take at most finitely many values, Pylea can compute the probability 
of events, expected values, probability mass functions and any other quantity 
of interest.
Such a probabilistic model is written as a *discrete probabilistic program (DPP)*: 
a Python function decorated with `@DPP`, using a restricted syntax in which 
coin flips are drawn with `flip` and events are conditioned on using `observe`.

*Example.* A disease occurs in a fraction ``10**-4`` of the population.  A test for it is positive for
99% of the patients who have the disease and for 2% of those who do not.  The DPP
`has_disease` gives the probability that a patient has the disease, given one positive and
one negative test result:

    >>> from pylea import DPP, Bool, flip, observe, prob
    >>> @DPP
    ... def test(has_it: Bool) -> Bool:
    ...     positive_given_disease = flip(0.99)     # true positive rate
    ...     positive_given_no_disease = flip(0.02)  # false positive rate
    ...     true_positive = has_it & positive_given_disease
    ...     false_positive = ~has_it & positive_given_no_disease
    ...     return true_positive | false_positive
    >>> @DPP
    ... def has_disease() -> Bool:
    ...     x = flip(1e-4)
    ...     t1 = test(x)
    ...     t2 = test(x)
    ...     y = t1 & ~t2
    ...     observe(y)
    ...     return x
    >>> round(prob(has_disease), 8)
    5.051e-05

A DPP calls other DPPs as ordinary functions, as `has_disease` calls `test` twice.  The same
DPP is `pylea.examples.disease`.  The language has more constructions, shown in the
examples and the guide below.


## Examples

Each example is a notebook, shown with its output and available for download, for one
class of models.

- **Fault trees**: [This notebook](examples/fault_trees.html) writes *fault trees*, which
  reliability engineering uses for the failure probability of a system, as DPPs, and
  converts a fault tree in the `.dft` format.
- **Bayesian networks**: [This notebook](examples/bayesian_networks.html) writes a
  Bayesian network as conditional probability tables, queries it with evidence, and
  converts a network in the `.bif` format.
- **Discrete Bayesian statistics**: [This notebook](examples/bayesian_statistics.html)
  computes posterior distributions for the bias of a coin, the German tank problem and a
  bioassay.
- **Games of chance**: [This notebook](examples/games_of_chance.html) computes the
  probability of winning in the *Monty Hall problem* and in *Texas hold'em poker*.

`pylea.utility` converts Bayesian networks in the `.bif` format and fault trees in the
`.dft` format into DPPs.  More example DPPs are in `pylea.examples`.


## Writing a DPP

Python runs the body of a `@DPP` function once, when the function is defined, and records
each step.  Probabilities are computed only when the DPP is queried.  Three rules follow.

**Every variable has a type.**  The type is a subclass of `Variable`: `Bool`, or a type
declared by subclassing `Enum`, `UInt` or `Tuple`, as shown below.  It fixes the finite set
of values the variable can take.  Every parameter of a DPP is annotated with its type, and
the DPP returns one variable or a tuple of variables.

**A random variable has no value while the body runs.**  It therefore cannot be used in
Python's `if`, `while`, `and`, `or` or `not`; a `Bool` used there raises `TypeError`.
`&`, `|`, `~` and `match`, below, take their place.  Loops and `if` statements over
ordinary Python values work as usual: a `for` loop over ``range(3)`` records its steps
three times, and a Python function called in the body records the steps it makes.

**A DPP takes only variables.**  `flip`, `observe` and the other constructions below work
only inside the body of a `@DPP` function, and a parameter cannot be a number.  A number,
such as a probability, is given to a Python function that builds the DPP:

    >>> def coin(p: float) -> DPP:
    ...     @DPP
    ...     def heads() -> Bool:
    ...         return flip(p)
    ...     return heads
    >>> prob(coin(0.3))
    0.3

### Coin flips and Boolean operations

`flip(p)` is a coin flip: a `Bool` that is true with probability `p`.  Each call is a new
coin flip, independent of every other, and a variable used twice has the same value both
times.  `&`, `|`, `~` and `^` combine `Bool`s, `Bool.any` and `Bool.all` combine any number
of them, and `at_least` is true where at least a given number of its arguments are.  A
cooling system fails if both pumps fail or at least two of its three sensors fail, and a
pump fails if its motor or the shared power supply fails:

    >>> from pylea import at_least
    >>> @DPP
    ... def cooling_fails() -> Bool:
    ...     power = flip(0.001)
    ...     pump_a = power | flip(0.01)
    ...     pump_b = power | flip(0.01)
    ...     sensors = [flip(0.02) for _ in range(3)]
    ...     return (pump_a & pump_b) | at_least(2, *sensors)
    >>> round(prob(cooling_fails), 6)
    0.002283

### Calling a DPP

A DPP calls another as a function, as `has_disease` calls `test` above.  Each call makes its
own coin flips, so the two calls of `test` there are two independent tests.  A DPP with
several outputs returns a tuple:

    >>> @DPP
    ... def two_coins() -> tuple[Bool, Bool]:
    ...     return flip(0.5), flip(0.5)
    >>> @DPP
    ... def same_side() -> Bool:
    ...     a, b = two_coins()
    ...     return ~(a ^ b)
    >>> prob(same_side)
    0.5

### Conditioning

`observe(e)` conditions on the `Bool` `e` being true.  A query then gives probabilities
conditional on `e`, and so does a query of any DPP calling this one.

    >>> @DPP
    ... def first_given_either() -> Bool:
    ...     a, b = flip(0.5), flip(0.5)
    ...     observe(a | b)
    ...     return a
    >>> round(prob(first_given_either), 6)
    0.666667

``observe(e, if_true=0.9, if_false=0.3)`` multiplies the weight of each combination of
values by 0.9 where `e` is true and by 0.3 where it is false, instead of by 1 and 0.

### Enumerations: `Enum`

An `Enum` type lists its values in `elements`:

    >>> from pylea import Enum, probability_mass_function
    >>> class Weather(Enum):
    ...     elements = ["dry", "rain", "snow"]
    >>> @DPP
    ... def weather() -> Weather:
    ...     return Weather.random({"dry": 0.7, "rain": 0.25, "snow": 0.05})

`Enum.random` draws a value with the given probabilities, given as a dict from elements
to probabilities or as a list in the order of `elements`, as in
``Weather.random([0.7, 0.25, 0.05])``.  `Enum.uniform` draws every value with the same
probability, and `Enum.const` is a fixed value, as in ``Weather.const("dry")``.  A variable
compares with `==` and `!=` against another of its type or against one of the `elements`.
`<`, `<=`, `>` and `>=` follow the order of `elements`, not the order of the values
themselves.  A value that is not one of the `elements` raises `ValueError`, here and
wherever an element is expected.

`Enum.match` chooses a variable or a value for each element, and needs a key for every
element and no other key.  Where every choice is a value rather than a variable,
`data_type` names the type of the result.  `Bool.match` is the same, with the keys `True`
and `False`.

    >>> @DPP
    ... def slippery() -> Bool:
    ...     w = weather()
    ...     return w.match({"dry": flip(0.01), "rain": flip(0.2), "snow": True})
    >>> round(prob(slippery), 6)
    0.107

`Enum.cpt` builds a conditional probability table: for each tuple of values of the
parents, the probabilities of the elements, as a list in the order of `elements` or as a
dict.  Every tuple of values of the parents needs a row.  The table is called on the
parent variables, as a DPP is:

    >>> class Grass(Enum):
    ...     elements = ["dry", "damp", "wet"]
    >>> grass_given_weather = Grass.cpt([Weather], {
    ...     ("dry",):  [0.8, 0.2, 0.0],
    ...     ("rain",): [0.0, 0.3, 0.7],
    ...     ("snow",): [0.2, 0.6, 0.2]})
    >>> @DPP
    ... def wet_grass() -> Bool:
    ...     return grass_given_weather(weather()) == "wet"
    >>> round(prob(wet_grass), 6)
    0.185

`Enum.map` applies an ordinary Python function to the values of one or several `Enum`
variables.  `data_type` names the type of the result, which is the variable's own type
where it is left out.  It stores the result for every combination of values, and raises
`ValueError` where there are too many combinations.

    >>> class Die(Enum):
    ...     elements = [1, 2, 3, 4, 5, 6]
    >>> @DPP
    ... def big_product() -> Bool:
    ...     a, b = Die.uniform(), Die.uniform()
    ...     return a.map(lambda x, y: x * y > 20, b, data_type=Bool)
    >>> round(prob(big_product), 6)
    0.166667

### Integers: `UInt`

A `UInt` type holds the integers from 0 to ``2**width - 1``.  `+` adds two variables of
the type, or a variable and a number.  The sum is taken modulo ``2**width``, so the type
must be wide enough for the largest sum.  Python's `sum` adds a list of `UInt` variables,
and `count` gives the number of `Bool`s that are true.

    >>> from pylea import UInt, count
    >>> class Heads(UInt):
    ...     width = 4                    # 0 to 15
    >>> @DPP
    ... def many_heads() -> Bool:
    ...     morning = count((flip(0.5) for _ in range(5)), data_type=Heads)
    ...     evening = count((flip(0.5) for _ in range(5)), data_type=Heads)
    ...     return morning + evening >= 7
    >>> prob(many_heads)
    0.171875

### Tuples and drawing without replacement

A `Tuple` type combines several types into one: `Tuple.of` builds a value from one
variable per type, and `Tuple.components` takes it apart again.  `pick` draws one of a
list of variables, each with the same probability, and returns it with the rest of the
list; given a number `n` as well, it draws `n` in turn.  The rest is a
`pylea.syntax.choice.Remainder`, which can be indexed, sliced, iterated, and given to
`pick` again to draw more, all inside the same `@DPP` function.  The two together deal two
cards from a deck of four:

    >>> from pylea import Tuple, pick
    >>> class Rank(Enum):
    ...     elements = ["K", "A"]
    >>> class Colour(Enum):
    ...     elements = ["red", "black"]
    >>> class Card(Tuple):
    ...     types = (Rank, Colour)
    >>> @DPP
    ... def two_aces() -> Bool:
    ...     deck = [Card.of(Rank.const(r), Colour.const(c)) for r in Rank for c in Colour]
    ...     (first, second), rest = pick(deck, 2)
    ...     first_rank, _ = first.components()
    ...     second_rank, _ = second.components()
    ...     return (first_rank == "A") & (second_rank == "A")
    >>> round(prob(two_aces), 6)
    0.166667

The cost of exact inference grows quickly with the number of draws a query depends on, so
a query about the whole ordering of more than a few choices is a task for the sampling
engine, below.

## Querying a DPP

A query takes a DPP without parameters that returns what is asked about:

* `prob` gives the probability that a DPP returning one `Bool` returns true.
* `probability_mass_function` gives the probability of each value of a DPP returning one
  `Enum`, as a dict.
* `expectation` gives the expected value of a DPP returning one variable whose values are
  numbers, such as a `UInt` or an `Enum` of numbers.

Each is conditional on the DPP's observations.  To ask about one output of a DPP with
several, or about a combination of them, write a DPP that calls it and returns what is
asked about:

    >>> from pylea import expectation
    >>> @DPP
    ... def weather_given_wet_grass() -> Weather:
    ...     w = weather()
    ...     observe(grass_given_weather(w) == "wet")
    ...     return w
    >>> pmf = probability_mass_function(weather_given_wet_grass)
    >>> {w: round(p, 4) for w, p in pmf.items()}
    {'dry': 0.0, 'rain': 0.9459, 'snow': 0.0541}
    >>> @DPP
    ... def heads_in_ten() -> Heads:
    ...     return count((flip(0.5) for _ in range(10)), data_type=Heads)
    >>> round(expectation(heads_in_ten), 6)
    5.0

**An answer can be an estimate.**  The `method` argument selects how a query is computed:
``"exact"`` computes the value exactly with the exact engine, ``"smc"`` estimates it with
the sampling engine, and ``"auto"``, the default, runs whichever is predicted to be
faster.  A sampled answer is computed to relative error `rel_err`, 0.01 by default, and
`seed` makes it reproducible.  Every answer is an `Estimate`, a float with the attributes
`low` and `high`, which bound the true value with probability at least `confidence`, 0.95
by default; for an exact answer both equal the value.  Arithmetic on an `Estimate` gives a
plain float, without the bounds.

    >>> p = prob(first_given_either, method="smc", rel_err=0.05, seed=0)
    >>> bool(p.low <= 2 / 3 <= p.high)
    True

`max_time` bounds, in seconds, the time a query spends in either engine and in the choice
between them.  It does not bound the preparation of a DPP for sampling, which the first
query of the DPP that samples makes.  An exact query that runs past `max_time` raises
`TimeoutError`.  A sampled answer that has not reached its relative error by then is
returned with the interval reached, and a warning names the line of the query; under
``strict=True`` the query raises `pylea.sample.estimate.TargetNotMetError` instead, whose
`result` is that answer.

The exact engine refuses a query whose tensor network is too large to contract, with
`ExactInferenceTooLarge`, or whose contraction is estimated to take too long, with
`ExactInferenceTooSlow`.  Under ``"exact"`` the refusal is raised; under ``"auto"`` the
sampling engine answers instead where it can.

`DPP.evaluated` gives the distribution of a DPP's outputs for each value of its inputs, as
a `LikelihoodTable`, with the exact engine unless ``method="smc"`` is passed.  Printed, it
has one row per input value and one column per output value:

    >>> print(test.evaluated())
          | False | True
    --------------------
    False |  0.98 | 0.02
    True  |  0.01 | 0.99

A table is called inside a `@DPP` function as the DPP it came from is, so a DPP can be
computed once and the result used in many later DPPs.  Where the DPP observes, the
probabilities for one input value sum to the probability, on that input, that every
observation holds, rather than to 1; this is what makes calling the table the same as
calling the DPP.  Printing divides them by their sum, and `LikelihoodTable.normalised`
gives a table of the divided values, as in the
[German tank problem](examples/bayesian_statistics.html).

## Drawing a DPP

`pylea.syntax.dpp.DPP.diagram` draws a DPP as a diagram: one dot per variable, one box per
assignment, read bottom to top from the inputs to the outputs.  The Boolean gates take
their usual symbols, a coin flip is a circle carrying its probability, an observation is an
eye, and a call is a box carrying the name of the DPP it calls.  The diagram of `test`
above:

    >>> test.diagram()
    <Diagram test: 15 nodes, ... x ...>

![The diagram of test](test.svg)

The line entering at the bottom carries the name of the parameter it stands for.  A line
leaving at the top is named the same way where the DPP returned a variable; this one
returns an expression, which has no name to show.

`pylea.draw.Diagram.svg` gives the markup, `pylea.draw.Diagram.tikz` a figure for a LaTeX
document, `pylea.draw.Diagram.save` a file, and `pylea.draw.Diagram.show` opens one in a
browser; a notebook renders it where it stands.

There is no nesting: a called DPP is drawn by asking it for its own diagram.
`pylea.syntax.dpp.DPP.simplified` returns the DPP that the rewrite rules of
`pylea.simplify` make of a DPP; its diagram shows what simplification left, which helps to
find why a DPP compiles more slowly than expected.

`pylea.syntax.dpp.DPP.dependency_graph` draws the DPP's variables instead of its
assignments: one box per variable the DPP is given or returns, and an arrow to each
variable computed from it.  For a DPP whose outputs are the variables of a Bayesian
network, that is the network:

    >>> from pylea.examples.sprinkler import sprinkler_network
    >>> sprinkler_network.dependency_graph()
    <Diagram sprinkler_network: 4 nodes, ... x ...>

![The dependency graph of sprinkler_network](sprinkler_network.svg)

It reads from the top down: the cloudiness is drawn first, the sprinkler state and the rain
from it, and the wetness of the grass from those two.  A box carries the name the DPP
returned the variable as.  What is drawn is what each variable is computed from, so two
outputs sharing an internal variable the DPP does not return are drawn apart, and
conditioning is not drawn at all.


## Modules

  * `pylea.syntax`, the language: `@DPP`, variables, and the primitives beneath.
  * `pylea.semantics`, the matrices of the primitives and the semantics of a DPP.
  * `pylea.ir`, the intermediate representation in which a DPP is recorded.
  * `pylea.simplify`, the rewrite rules applied to a DPP before compilation.
  * `pylea.compile`, compilation: the tensor network and its contraction tree.
  * `pylea.eval`, the exact engine's contraction of a compiled DPP with numpy or torch.
  * `pylea.sample`, the sampling engine and the portfolio algorithm, which chooses between
    the two engines.
  * `pylea.draw`, drawing a DPP as a diagram.
  * `pylea.utility`, converters from `.bif` Bayesian networks and `.dft` fault trees.
  * `pylea.examples`, example DPPs.
"""
from pylea.compile.lower import (
    ExactInferenceInfeasible, ExactInferenceTooLarge, ExactInferenceTooSlow,
)
from pylea.semantics.likelihood_table import LikelihoodTable
from pylea.syntax.choice import pick
from pylea.syntax.dpp import DPP, Bool, at_least
from pylea.syntax.enum import Enum
from pylea.syntax.estimate import Estimate
from pylea.syntax.expectation import expectation
from pylea.syntax.flip import flip
from pylea.syntax.integer import UInt, count
from pylea.syntax.observe import observe
from pylea.syntax.pmf import probability_mass_function
from pylea.syntax.prob import prob
from pylea.syntax.tuple import Tuple
from pylea.syntax.variable import Variable

__all__ = ["DPP", "Bool", "Enum", "Tuple", "UInt", "Variable", "flip", "observe",
           "at_least", "count", "pick", "prob", "probability_mass_function",
           "expectation", "Estimate", "LikelihoodTable", "ExactInferenceInfeasible",
           "ExactInferenceTooLarge", "ExactInferenceTooSlow"]
