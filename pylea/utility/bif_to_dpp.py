"""Bayesian networks from `.bif` files.

`bif_to_python` renders one module per network; `convert_suite` does it over a folder.
The module is also a command-line tool, described by its `--help`.

## Supported syntax

    network NAME { property ...; }
    variable NAME { type discrete [ N ] { S1, S2, ... }; property ...; }
    probability ( VAR ) { table P1, P2, ...; }
    probability ( VAR | PARENT1, PARENT2, ... ) {
        ( S1, S2, ... ) P1, P2, ...;
        default P1, P2, ...;
        table P1, P2, ...;
    }
    // to the end of the line, and /* ... */, are comments

Commas and `|` separate tokens like spaces do, so ``{ t f }`` declares two states and
``( B A )`` names the child `B` and its parent `A`.  A name or state is a run of
characters other than spaces and ``{ } ( ) [ ] ; , |``, or any text in double quotes.
`property` lines are ignored.

A row ``( S1, S2, ... )`` gives the child's distribution for the parents in the states
named, one state per parent in the order of the header.  `default` gives it for every
parent configuration no row names.  `table` lists every probability of the block at once:
for a variable with parents, the child's first state for every parent configuration, then
its second state, and so on, with the parent configurations counted with the last parent
changing fastest.

Everything else is refused with a `ValueError` naming the line and the variable: another
kind of block or entry, a type other than `discrete`, a state count `N` that differs from
the states listed, a state listed twice, a row with a number of probabilities other than
the number of states, a negative probability, a row whose sum differs from 1 by more than
1e-5, a row naming a state the parent does not have or given twice, a parent configuration
with neither a row nor a `default`, a variable declared twice or without a probability
block, two probability blocks for one variable, and variables that depend on each other in
a cycle.

## What is generated

A module holding one `pylea.syntax.enum.Enum` subclass per variable, whose elements are
the variable's states as the file writes them, and one `@DPP` function returning the
variables, each after its parents, otherwise in the order of declaration.  A variable with
parents is the result of a `@DPP` function of its parents, named after the variable and
its parents, applying `pylea.syntax.enum.Enum.cpt`, so a diagram labels its box with that
name; one without is drawn by `pylea.syntax.enum.Enum.random`.  A network of one variable
returns that variable rather than a tuple.

Each variable becomes a class whose name is the variable's with a capital first letter,
and a local variable of its own name, in lower case where the name starts with a capital
letter, so that it differs from the class.  Characters other than ASCII letters, digits and
`_` become `_`, and a name that starts with a digit or `_`, or is a Python keyword, gets
the prefix `v_`.  Names that would become the same identifier, or one the module uses
itself (`DPP`, `Enum`, `tuple`, a class or a table function used in the same place), are
told apart by a suffix `_2`, `_3`, and so on; a name that is already a valid identifier
keeps it where it can.  A comment at the top of the module lists every variable written
differently.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import NamedTuple

from pylea.utility._codegen import (
    Names, Token, convert_directory, literal, module_name, number, tokenize,
    topological_order,
)

_PUNCTUATION = "{}()[];"
_SEPARATORS = ",|"
_ROW_SUM_TOLERANCE = 1e-5     # absolute; far above the rounding of published tables
_RESERVED = ("DPP", "Enum", "tuple")
_MULTILINE = 5                # more outputs than this are written one per line


class _Reader:
    """A cursor over the tokens of a file."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0
        self.last_line = tokens[-1].line if tokens else 1

    def at_end(self) -> bool:
        return self.i >= len(self.tokens)

    def next(self, expected: str) -> Token:
        """The next token, which the parser expects to be `expected`, a description."""
        if self.at_end():
            raise ValueError(f"line {self.last_line}: the file ends where {expected} is "
                             "expected")
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def peek(self) -> Token | None:
        return None if self.at_end() else self.tokens[self.i]

    def keyword(self) -> str:
        """The lower-cased text of the next token, or "" for a quoted one or the end."""
        tok = self.peek()
        return "" if tok is None or tok.quoted else tok.text.lower()

    def expect(self, char: str) -> Token:
        """The next token, which must be the punctuation character `char`."""
        tok = self.next(f"'{char}'")
        if not tok.is_mark(char):
            raise ValueError(f"line {tok.line}: expected '{char}', found {tok.shown()}")
        return tok

    def name(self, what: str) -> Token:
        tok = self.next(what)
        if not tok.is_name():
            raise ValueError(f"line {tok.line}: expected {what}, found {tok.shown()}")
        return tok

    def skip_statement(self) -> None:
        """Past the next `;`, for a property line."""
        while not (tok := self.next("';'")).is_mark(";"):
            if tok.is_mark("{") or tok.is_mark("}"):
                raise ValueError(f"line {tok.line}: a property ends without ';'")

    def values(self, what: str) -> tuple[list[float], int]:
        """The numbers up to the next `;`, and the line where they start."""
        out: list[float] = []
        nxt = self.peek()
        line = self.last_line if nxt is None else nxt.line
        while not (tok := self.next("';'")).is_mark(";"):
            value = None if tok.quoted else number(tok.text)
            if value is None:
                raise ValueError(f"line {tok.line}: {what}: expected a probability, found "
                                 f"{tok.shown()}")
            out.append(value)
        return out, line


class _Network(NamedTuple):
    name: str | None                          # as the file names it
    order: list[str]                          # variables in order of declaration
    states: dict[str, list[str]]
    parents: dict[str, list[str]]
    rows: dict[str, dict[tuple[str, ...], list[float]]]   # every parent configuration


class _Block(NamedTuple):
    parents: list[str]
    line: int
    table: tuple[list[float], int] | None
    default: tuple[list[float], int] | None
    rows: dict[tuple[str, ...], tuple[list[float], int]]


def _parse_bif(text: str) -> _Network:
    """The network `text` describes, with every check of the module docstring applied."""
    r = _Reader(tokenize(text, _PUNCTUATION, _SEPARATORS))
    network: str | None = None
    network_line = 0
    declared: dict[str, tuple[list[str], int]] = {}
    blocks: dict[str, _Block] = {}
    while not r.at_end():
        tok = r.next("a block")
        kw = "" if tok.quoted else tok.text.lower()
        if kw == "network":
            if network_line:
                raise ValueError(f"line {tok.line}: a second network block; the first is on "
                                 f"line {network_line}")
            network, network_line = r.name("the network's name").text, tok.line
            r.expect("{")
            while r.keyword() == "property":
                r.next("property")
                r.skip_statement()
            r.expect("}")
        elif kw == "variable":
            name = r.name("a variable's name")
            if name.text in declared:
                raise ValueError(f"line {name.line}: variable {name.text!r} is declared twice, "
                                 f"first on line {declared[name.text][1]}")
            declared[name.text] = (_variable_body(r, name), name.line)
        elif kw == "probability":
            child, block = _probability_block(r, tok.line)
            if child in blocks:
                raise ValueError(f"line {tok.line}: a second probability block for "
                                 f"{child!r}; the first is on line {blocks[child].line}")
            blocks[child] = block
        else:
            raise ValueError(f"line {tok.line}: expected 'network', 'variable' or "
                             f"'probability', found {tok.shown()}")

    for child, block in blocks.items():
        if child not in declared:
            raise ValueError(f"line {block.line}: probability block for {child!r}, which is "
                             "not declared as a variable")
        for p in block.parents:
            if p not in declared:
                raise ValueError(f"line {block.line}: parent {p!r} of {child!r} is not "
                                 "declared as a variable")
    order = list(declared)
    for v in order:
        if v not in blocks:
            raise ValueError(f"line {declared[v][1]}: variable {v!r} has no probability block")
    states = {v: declared[v][0] for v in order}
    rows = {v: _rows(v, blocks[v], states) for v in order}
    return _Network(network, order, states, {v: blocks[v].parents for v in order}, rows)


def _variable_body(r: _Reader, name: Token) -> list[str]:
    """The states of the variable `name`, read from ``{`` to ``}``."""
    r.expect("{")
    states: list[str] | None = None
    while True:
        tok = r.next("'type', 'property' or '}'")
        kw = "" if tok.quoted else tok.text.lower()
        if kw == "}":
            break
        if kw == "property":
            r.skip_statement()
            continue
        if kw != "type":
            raise ValueError(f"line {tok.line}: variable {name.text!r}: expected 'type', "
                             f"'property' or '}}', found {tok.shown()}")
        if states is not None:
            raise ValueError(f"line {tok.line}: variable {name.text!r} has a second type")
        kind = r.next("'discrete'")
        if kind.quoted or kind.text.lower() != "discrete":
            raise ValueError(f"line {kind.line}: variable {name.text!r} has type "
                             f"{kind.shown()}; only discrete variables are supported")
        r.expect("[")
        count = r.next("the number of states")
        if count.quoted or not count.text.isdigit():
            raise ValueError(f"line {count.line}: variable {name.text!r}: expected the number "
                             f"of states, found {count.shown()}")
        r.expect("]")
        r.expect("{")
        states = []
        while not (s := r.next("a state or '}'")).is_mark("}"):
            if not s.is_name():
                raise ValueError(f"line {s.line}: variable {name.text!r}: expected a state, "
                                 f"found {s.shown()}")
            if s.text in states:
                raise ValueError(f"line {s.line}: variable {name.text!r} lists the state "
                                 f"{s.text!r} twice")
            states.append(s.text)
        r.expect(";")
        if int(count.text) != len(states) or not states:
            raise ValueError(f"line {count.line}: variable {name.text!r} declares "
                             f"{count.text} states and lists {len(states)}")
    if states is None:
        raise ValueError(f"line {name.line}: variable {name.text!r} has no type")
    return states


def _probability_block(r: _Reader, line: int) -> tuple[str, _Block]:
    """The child a probability block is for, and the block, read from ``(`` to ``}``."""
    r.expect("(")
    child = r.name("the variable of the probability block").text
    parents: list[str] = []
    while not (tok := r.next("a parent or ')'")).is_mark(")"):
        if not tok.is_name():
            raise ValueError(f"line {tok.line}: expected a parent of {child!r}, found "
                             f"{tok.shown()}")
        if tok.text in parents:
            raise ValueError(f"line {tok.line}: {child!r} lists the parent {tok.text!r} twice")
        parents.append(tok.text)
    r.expect("{")
    table = default = None
    rows: dict[tuple[str, ...], tuple[list[float], int]] = {}
    where = f"variable {child!r}"
    while True:
        tok = r.next("an entry or '}'")
        kw = "" if tok.quoted else tok.text.lower()
        if kw == "}":
            break
        if kw == "property":
            r.skip_statement()
        elif kw == "table":
            if table is not None:
                raise ValueError(f"line {tok.line}: {where} has a second table")
            table = r.values(where)
        elif kw == "default":
            if default is not None:
                raise ValueError(f"line {tok.line}: {where} has a second default")
            default = r.values(where)
        elif kw == "(":
            key: list[str] = []
            while not (s := r.next("a state or ')'")).is_mark(")"):
                if not s.is_name():
                    raise ValueError(f"line {s.line}: {where}: expected a parent's state, "
                                     f"found {s.shown()}")
                key.append(s.text)
            if tuple(key) in rows:
                raise ValueError(f"line {tok.line}: {where} gives the row {tuple(key)} twice")
            rows[tuple(key)] = r.values(where)
        else:
            raise ValueError(f"line {tok.line}: {where}: expected 'table', 'default', a row "
                             f"or '}}', found {tok.shown()}")
    if table is not None and (rows or default is not None):
        raise ValueError(f"line {table[1]}: {where} has a table and rows; give one or the "
                         "other")
    return child, _Block(parents, line, table, default, rows)


def _rows(child: str, block: _Block, states: dict[str, list[str]]
          ) -> dict[tuple[str, ...], list[float]]:
    """The child's distribution for every parent configuration, checked.

    Rows the block lists come first, in its order, then those its `default` fills or its
    `table` gives, in counting order.
    """
    where = f"variable {child!r}"
    width = len(states[child])
    configs = list(itertools.product(*(states[p] for p in block.parents)))
    given: dict[tuple[str, ...], tuple[list[float], int]] = {}
    if block.table is not None:
        values, line = block.table
        if len(values) != width * len(configs):
            needs = (f"{width} states and {len(configs)} parent configurations, so its "
                     f"table needs {width * len(configs)} probabilities" if block.parents
                     else f"{width} states")
            raise ValueError(f"line {line}: {where} has {needs}, but its table has "
                             f"{len(values)}")
        stride = len(configs)
        for c, config in enumerate(configs):
            given[config] = (values[c::stride], line)
    for key, (values, line) in block.rows.items():
        if len(key) != len(block.parents):
            raise ValueError(f"line {line}: {where}: the row {key} names {len(key)} states "
                             f"for {len(block.parents)} parents")
        for p, s in zip(block.parents, key):
            if s not in states[p]:
                raise ValueError(f"line {line}: {where}: the row {key} names the state "
                                 f"{s!r}, which parent {p!r} does not have")
        given[key] = (values, line)
    if block.default is not None:
        for config in configs:
            given.setdefault(config, block.default)
    out: dict[tuple[str, ...], list[float]] = {}
    for config in list(given):
        values, line = given[config]
        row = f"the row {config}" if block.parents else "the table"
        if len(values) != width:
            raise ValueError(f"line {line}: {where} has {width} states, but {row} has "
                             f"{len(values)} probabilities")
        if any(v < 0.0 for v in values):
            raise ValueError(f"line {line}: {where}: {row} has a negative probability")
        total = math.fsum(values)
        if abs(total - 1.0) > _ROW_SUM_TOLERANCE:
            raise ValueError(f"line {line}: {where}: the probabilities of {row} sum to "
                             f"{total!r}, not 1")
        out[config] = values
    for config in configs:
        if config not in out:
            raise ValueError(f"line {block.line}: {where} has no row for the parent states "
                             f"{config} and no default")
    return out


def bif_to_python(bif_string: str, network_name: str | None = None) -> str:
    """The source of a standalone module for the network `bif_string` describes.

    The module defines one `pylea.syntax.enum.Enum` subclass per variable, one `@DPP`
    function per variable with parents, and a `@DPP` function returning the network's
    variables, named `network_name`, by default after the file's network block.  Raises
    `ValueError` for a `network_name` that is not a Python identifier or that the module
    uses itself, and, naming the line, for anything the module docstring lists as refused.

        >>> print(bif_to_python('''
        ...     network demo { }
        ...     variable rain { type discrete [ 2 ] { no, yes }; }
        ...     variable wet-grass { type discrete [ 2 ] { no, yes }; }
        ...     probability ( rain ) { table 0.8, 0.2; }
        ...     probability ( wet-grass | rain ) { (no) 0.9, 0.1; (yes) 0.2, 0.8; }
        ... '''))
        # Variables of the file written differently here:
        #     'wet-grass' -> class Wet_grass, variable wet_grass
        from pylea.syntax.dpp import DPP
        from pylea.syntax.enum import Enum
        <BLANKLINE>
        <BLANKLINE>
        class Rain(Enum):
            elements = ['no', 'yes']
        <BLANKLINE>
        <BLANKLINE>
        class Wet_grass(Enum):
            elements = ['no', 'yes']
        <BLANKLINE>
        <BLANKLINE>
        @DPP
        def wet_grass_given_rain(rain: Rain) -> Wet_grass:
            return Wet_grass.cpt(
                [Rain],
                {
                    ('no',): [0.9, 0.1],
                    ('yes',): [0.2, 0.8],
                },
            )(rain)
        <BLANKLINE>
        <BLANKLINE>
        @DPP
        def demo() -> tuple[Rain, Wet_grass]:
            rain = Rain.random({'no': 0.8, 'yes': 0.2})
            wet_grass = wet_grass_given_rain(rain)
            return rain, wet_grass
    """
    if network_name is not None:
        if not network_name.isidentifier() or network_name in _RESERVED or (
                Names().claim(network_name) != network_name):
            raise ValueError(f"network_name {network_name!r} is not a Python identifier, or "
                             "is one the generated module uses itself")
    net = _parse_bif(bif_string)
    order = topological_order(net.order, net.parents)

    module = Names(reserved=_RESERVED)
    if network_name is not None:
        fn_name = module.claim(network_name)
    elif net.name is None or net.name == "unknown":
        fn_name = module.claim("bayesian_network")
    else:
        fn_name = module.claim(net.name)
    classes = dict(zip(net.order, module.claim_all(net.order, kind="class")))
    # A local variable must not hide a class or table function the body uses, and a
    # table function's parameters must not hide a class.
    local = dict(zip(net.order, Names(reserved=classes.values()).claim_all(net.order,
                                                                           kind="variable")))
    table_names = Names(reserved=[*_RESERVED, fn_name, *classes.values(), *local.values()])
    tables = {v: table_names.claim(
                  f"{local[v]}_given_{'_'.join(local[p] for p in net.parents[v])}")
              for v in order if net.parents[v]}

    lines: list[str] = []
    renamed = [v for v in net.order if (classes[v] != v[:1].upper() + v[1:]
                                        or local[v] != (v.lower() if v[:1].isupper() else v))]
    if renamed:
        lines.append("# Variables of the file written differently here:")
        lines += [f"#     {literal(v)} -> class {classes[v]}, variable {local[v]}"
                  for v in renamed]
    lines += ["from pylea.syntax.dpp import DPP", "from pylea.syntax.enum import Enum", "", ""]

    for v in net.order:
        elements = ", ".join(literal(s) for s in net.states[v])
        lines += [f"class {classes[v]}(Enum):", f"    elements = [{elements}]", "", ""]

    for v, table in tables.items():
        parents = net.parents[v]
        params = {p: local[p] for p in parents}
        rows = [f"            {_tuple_literal(key)}: [{', '.join(repr(x) for x in values)}],"
                for key, values in net.rows[v].items()]
        lines += [
            "@DPP",
            f"def {table}({', '.join(f'{params[p]}: {classes[p]}' for p in parents)})"
            f" -> {classes[v]}:",
            f"    return {classes[v]}.cpt(",
            f"        [{', '.join(classes[p] for p in parents)}],",
            "        {",
            *rows,
            "        },",
            f"    )({', '.join(params[p] for p in parents)})",
            "",
            "",
        ]

    types = [classes[v] for v in order]
    if len(order) == 1:
        returned = types[0]
    elif len(order) > _MULTILINE:
        returned = "tuple[" + "".join(f"\n    {t}," for t in types) + "\n]"
    else:
        returned = f"tuple[{', '.join(types)}]"
    lines += ["@DPP", f"def {fn_name}() -> {returned}:"]
    for v in order:
        if net.parents[v]:
            args = ", ".join(local[p] for p in net.parents[v])
            lines.append(f"    {local[v]} = {tables[v]}({args})")
        else:
            dist = ", ".join(f"{literal(s)}: {x!r}"
                             for s, x in zip(net.states[v], net.rows[v][()]))
            lines.append(f"    {local[v]} = {classes[v]}.random({{{dist}}})")
    if len(order) > _MULTILINE:
        lines.append("    return (" + "".join(f"\n        {local[v]}," for v in order) + "\n    )")
    else:
        lines.append(f"    return {', '.join(local[v] for v in order)}")
    return "\n".join(lines)


def _tuple_literal(key: tuple[str, ...]) -> str:
    """`key` as a Python tuple of string literals."""
    items = ", ".join(literal(s) for s in key)
    return f"({items},)" if len(key) == 1 else f"({items})"


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_suite(
    src_dir: Path,
    dst_dir: Path,
    *,
    suite_name: str | None = None,
    verbose: bool = True,
) -> tuple[dict[Path, Path], dict[Path, str]]:
    """``(converted, failed)`` after writing one module per `.bif` file directly in
    `src_dir` into `dst_dir`, a package created where missing.

    `converted` maps each file converted to its module, `failed` each other file to the
    message saying why it was not converted.  A module and its network function are named
    after the file, made a distinct identifier with the prefix `bn_` where needed.  With
    `verbose`, a line per file is printed, labelled with `suite_name`, by default the name
    of `src_dir`.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     src, dst = Path(d, "networks"), Path(d, "modules")
        ...     src.mkdir()
        ...     _ = Path(src, "coin-1.bif").write_text(
        ...         'variable c { type discrete [ 2 ] { h, t }; } probability ( c ) { table 0.5, 0.5; }')
        ...     _ = Path(src, "coin-2.bif").write_text(
        ...         'variable c { type discrete [ 2 ] { h, t }; } probability ( c ) { table 0.5, 0.6; }')
        ...     converted, failed = convert_suite(src, dst, verbose=False)
        ...     [p.name for p in converted.values()], list(failed.values())
        (['coin_1.py'], ["line 1: variable 'c': the probabilities of the table sum to 1.1, not 1"])
    """
    return convert_directory(
        src_dir, dst_dir, ".bif", lambda text, name: bif_to_python(text, network_name=name),
        prefix="bn_", reserved=_RESERVED, label=suite_name or src_dir.name, verbose=verbose)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    """The exit status of the command line run with the arguments `argv`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pylea.utility.bif_to_dpp",
        description="Convert Bayesian networks in the .bif format to Pylea modules.  Given "
                    "--file, print its module; given --src and --dst, write one module per "
                    ".bif file of the folder --src into the folder --dst.")
    parser.add_argument("--file", help="a .bif file whose module is printed")
    parser.add_argument("--src", help="a folder of .bif files")
    parser.add_argument("--dst", help="the folder to write the modules into")
    args = parser.parse_args(argv)
    if args.file is not None:
        if args.src is not None or args.dst is not None:
            parser.error("give --file, or --src and --dst, not both")
        path = Path(args.file)
        try:
            print(bif_to_python(path.read_text(encoding="utf-8"),
                                network_name=module_name(path.stem, "bn_", _RESERVED)))
        except (ValueError, OSError) as exc:
            parser.exit(1, f"{parser.prog}: {path}: {exc}\n")
        return 0
    if args.src is None or args.dst is None:
        parser.error("give --file, or both --src and --dst")
    src = Path(args.src)
    if not src.is_dir():
        parser.error(f"{src} is not a folder")
    if not any(src.glob("*.bif")):
        parser.error(f"no .bif files in {src}")
    _, failed = convert_suite(src, Path(args.dst), verbose=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
