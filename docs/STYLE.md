# Pylea documentation style

Style for any kind of documentation: docstrings, comments, test names, and
standalone documents like this one.  

## General rules

1. Documentation serves as a technical manual to a user and must not contain any form of story telling.
2. Bottom line up front. Say directly what a thing is, do not start with anything else.
3. Be brief.
4. No emotion-carrying language. Maintain an absolutely dry and objective tone. No metaphors. Emphasis only when absolutely necessary to draw attention to something non-obvious.
5. Avoid stylistic devices of any kind (unless they accidentally arise from purely descriptive language).
6. Forbidden words:
  - "genuine", "honest" (already by point 4.)
  - "to bite" (used metaphorically)
  - "and nothing else"
  - ...
7. No em- or other dashes for parentetical remarks, better: no parenthetical remarks at all.
8. Never reference how code was created, or what its history is, be purely descriptive about what is actually in the codebase.
9. Avoid jargon at all costs. Always use the simpler, more standard word.
10. No "marketing-style" language. Never advertise anything. No value judgement anywhere (unless absolutely unavoidable).

## Where documentation lives

Module docstrings are the primary documentation.  
A module's docstring primarily says what the
module is for. 
If helpful it also describes its design, and possibly why this design was chosen (e.g. as opposed to an obvious alternative). 
It must readable on its own.  
The API reference is generated directly from docstrings
(see *Rendering* below), so the docstrings are what a user reads.

Documentation must stand alone.  Never reference planning documents, session notes, plans,
or issue threads: they are temporary, and a docstring that points at one dies with it.  
If a decision's rationale matters to the reader, state the rationale itself, in place.

## Terminology

Documentation uses the terms of the theory of discrete probabilistic programs, in the
following senses.  Code identifiers keep their names; where an identifier and a term
differ, the documentation of the identifier says which term it stands for.

* **DPP**, discrete probabilistic program: a program in which every variable takes one of
  finitely many values, built from random choice and observation, with no unbounded loops
  and no recursion.  In Pylea, a function decorated with `@DPP`, together with every DPP it
  calls.  Write "DPP", not "model" or "program", for such a function.  "Model" means only
  the probabilistic model a DPP specifies, such as a Bayesian network or a fault tree.
* **definition**: one function of a DPP under its **name**; in the IR, a `Body`.  The
  **main function** is the definition no other one calls, the root of a `Program`.  The
  **call graph** has an edge from each definition to every name it calls.
* **assignment**: one step of a definition, assigning the outputs of a primitive or of a
  called name to fresh variables; in the IR, an `Op`.
* **variable**: a value a definition is given, assigns or returns; in the IR, a wire.  A
  wire may be several bits wide, but a primitive reads and writes one wire per bit, so a
  variable of several bits written in `pylea.syntax` is several wires of a `Body`.  A definition's **input variables** and **output
  variables** are its parameters and return values.
* **primitive**: an assignment's function that is not a name, such as a gate, a flip, an
  observation or a matrix; in the IR, a `Prim`.
* **coin flip**: the random choice of one bit, `flip(p)`, true with probability `p`. Never shorten to "flip" unless you refer to the Pylea function `flip`. Correct where you see it used shortened.
* **observation**: `observe(x)`, conditioning on `x` being true.  "Conditioning" is the
  effect of observations.  "Evidence" is used only for the observed values of a Bayesian
  network.
* **semantics**: the non-negative matrix whose entry in the row of output `o` and the
  column of input `i` is the probability that, on input `i`, the DPP returns `o` and every
  observed value is true; a `LikelihoodTable`.
* **acceptance probability**: for a DPP without inputs, the sum of the entries of its
  semantics, the probability that every observed value is true.
* **inference**: computing the semantics, or a quantity derived from it, such as the
  probability of an event or an expected value.  The **exact engine** does this by
  tensor network contraction, the **sampling engine** by sequential Monte Carlo (SMC), and
  the **portfolio algorithm** chooses between them.
* **rewrite rule**: a replacement of one part of a DPP by another.  A rule is **sound** if
  it preserves the semantics, and **simplifying** if it strictly lowers the size and never
  raises the contraction cost.  **Simplification** applies the rules until none applies.
  A function applying rules is called a pass only where the code is meant.
* **size**: the number of symbols in a DPP: per assignment, one plus its numbers of input
  and output variables.
* **tensor network**: the definition read as a sum over all values of its variables of a
  product of one matrix per assignment.  A **contraction tree** is the order in which its
  matrices are multiplied, its **cost** the number of operations this takes, and the
  **contraction cost** of a definition the least cost of any contraction tree.
  **Slicing** splits a contraction into several smaller ones over fixed values of chosen
  variables.  Write "contraction tree", not "contraction order".
* **diagram**: the picture of a definition, with one dot per variable and one box per
  assignment, read from the inputs at the bottom to the outputs at the top.

## Docstrings

**The first line states what the thing is.**  A function is documented by the value it returns or the effect it has, 
as a sentence about the result; not in imperative mood, and never "This function ...":

```python
def simplify(prog, ...):
    """`prog` with simplification passes applied until none of them changes it, or the budget runs out."""

def reachable(prog):
    """The body indices reachable from the root, callees before callers."""
```

A class docstring says what an instance is (`"""A dense factor with typed inputs and
outputs."""`); a module docstring opens with a one-line summary of the module's subject
(`"""Contraction-order search."""`), then a blank line, then the prose.

**Parameters and exceptions are woven into prose.**  No `Args:` / `Returns:` / `Raises:`
sections: the signature already carries the names and types, so the docstring adds what
they *mean* — units, defaults worth knowing, and the behaviour a caller must handle:

```python
"""`budget_ms` bounds the whole phase by wall clock, measured from this call. ..."""
```

An exception is documented where the caller needs to act on it ("raises `ValueError` if
`f` has inputs"), not as a routine inventory.

**What and why, not how and not history.**  A docstring explains what the code does and why the design is the way it is, where the alternative is tempting.  
It never narrates the implementation ("first we build a dict, then ..."), never - under no circumstances - recounts what the code used to do, or how the design was reached.
Documentation is a technical manual and must not contain any form of story telling.

## Evidence and scope

**Do not mention any measurements in documentation.**: Measurements are machine-dependent and do not belong in documentation. 
Instead, mention measurements within comments, but only ever to explain hard-coded values. 
When you do, mention precisely but very briefly (in particular, no full sentences) how the value was determined/fitted. 
Never mention measurements that were exploratory, 
including in particular any single-data point ones,  
Never mention measurements that merely determined a design decision. 
In that case, and only if the decision is not already a priori reasonable, mention only (very briefly) within the documentation (or commentary in question) that this was decided empirically, but even then, mention this only if the decision is not already a priori reasonable. 

**Comments on hard-coded constants.**  
A tuned or fitted constant's comment says what it
was measured or fitted against, on what data, on what machine, and what would require re-measuring it.  


## Markdown conventions

Docstrings are Markdown, rendered as such.  In particular:

* Identifiers in single backticks: `` `Prim` ``, `` `budget_ms` ``.
* Expressions containing `*` or `_` in double backticks — ``2**out_bits`` — so that `**`
  and `_` never meet Markdown emphasis.
* A paragraph stating a rule or a finding may open with a **bold sentence**, then justify it in the rest of the paragraph.
* Preformatted blocks (indented four spaces) for anything tabular: stage ladders, grammar fragments, measured tables.
* Use bullet points, but only when appropriate.

## Citations

A citation is never written out in running text, in a docstring, a comment, a notebook or
any other document.  The text refers to a numbered entry, `[1]`, and the entries are listed
in full once, at the end: of the docstring, or of the first text cell or the last cell of a
notebook.  An entry gives the authors, the title, the venue and the year, and a DOI or a
URL where there is one.

## Comments

Comments show information outside the code, 
required for context. E.g.: an invariant, a useful fact, an
external requirement ("MPS refuses a tensor of more than 16 dimensions"). 
It never merely states what the next line does, where the code came from, or why a change was done.  
Comments are allowed to explain code, but only where this is non-obvious.
They should not paraphrase what the code is obviously doing in any line. 
On the other hand, in algorithmically non-trivial parts, 
comments should (concisely, and only if necessary) guide through the algorithm.


## Tests

**A test's name is a sentence stating the property it pins**, readable in a failure report
without opening the file: `test_reconfiguration_never_makes_the_winner_worse`,
`test_a_zero_column_stays_zero_rather_than_dividing`.  A docstring is added only when the
*why* needs more than the name.


## Doctests

Any function or method for which this is appropriate must contain a doctest demonstrating what it does and what it's for, 
while ensuring that documentation stays up to date and functional.
A sufficient condition for appropriate is "user-facing".
But it may also be appropriate in other cases:
often the behaviour of some piece of code is most clearest from what it does in some examples, so a doctest may be better than a thousand words here, and may enable keeping the words to a minimum. 


## Mechanics

* Two spaces between sentences.
* No dashes, unless unavoidable; never special unicode dashes.
* British spelling (`normalised`, `materialise`, `behaviour`), matching the API
  (`normalised()`), and British English writing style.

## Rendering

The API reference is generated by [pdoc](https://pdoc.dev) from the docstrings:

    pip install -e ".[docs]"
    python docs/make.py           # writes HTML to docs/_build/

`docs/make.py` names every module explicitly — pdoc's automatic recursion follows
`__all__`, and several packages re-export a function under its defining module's name
(`pylea.simplify.lift` the module, `lift` the function), which would silently drop the
module's page.  The generated site is static HTML: serve `docs/_build/` on GitHub Pages
(the `docs` workflow publishes it on every push to `main`) or point ReadTheDocs at the
same two commands.  After a docstring change worth publishing, rebuild and check the
affected page renders as intended — emphasis accidents from an unquoted `**` are the
common failure.

## Notebooks

`examples/` holds one notebook per application, each a `.py` file in jupytext's percent
format, with `# %%` before each code cell and `# %% [markdown]` before each text cell.
`docs/make.py` runs each one and writes it to `docs/_build/examples/` as a page and as an
`.ipynb` for download; the top-level docstring links every page.  A notebook is code first:
a heading, at most a few sentences naming the problem and the source of its numbers, and
one short line before a cell only where the code does not say what it computes.
