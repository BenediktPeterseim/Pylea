"""Fault trees from `.dft` files.

`dft_to_python` renders one module per tree; `convert_suite` and `convert_all` do it over a
folder and over the sub-folders of a folder.  The module is also a command-line tool,
described by its `--help`.

## Supported syntax

A file is a sequence of statements in the Galileo format, each ending in `;`:

    toplevel NAME;                          the element whose failure is the top event
    NAME and CHILD CHILD ...;               fails if every child fails
    NAME or CHILD CHILD ...;                fails if any child fails
    NAME KofN CHILD CHILD ...;              fails if at least K of its N children fail
    NAME prob=P;                            a basic event failing with probability P
    NAME lambda=RATE;                       a basic event failing at a constant rate
    NAME lambda=RATE phases=K;              a basic event failing after K phases at RATE
    // to the end of the line, and /* ... */, are comments

A statement may span several lines, and several statements may share one.  A name is a
run of characters other than spaces, `"`, `;` and `=` that does not start a comment, or
any text in double quotes on one line, which may contain spaces.  Spaces around `=` are
allowed.  Keywords and attribute names are read in any case, so `AND` and `1OF2` are
accepted.

A rate becomes the probability of failure within the mission time `lambda_t`: by
``1 - exp(-RATE * lambda_t)``, or for `phases=K` by the Erlang distribution of `K`
phases at that rate.  `dorm` is accepted and ignored: dormancy changes a rate only in a
spare gate, and spare gates are refused.

Everything else is refused with a `ValueError` naming the statement's line: a dynamic gate
(`pand`, `por`, `seq`, `wsp`, `csp`, `hsp`, `spare`, `fdep`, `pdep`, `mutex`, inspection
gates), any other gate type, the attributes `repair`, `res`, `cov`, `interval` and any
other, a basic event with neither or both of `prob` and `lambda`, a probability outside
``[0, 1]``, a negative rate, a voting gate whose `N` is not its number of children or whose
`K` is not between 1 and `N`, an element defined twice, a child never defined, and
elements that depend on each other in a cycle.

## The top event

The top event is the element `toplevel` names, which may be a gate or a basic event.  A
file without `toplevel` uses its only root, the one element no gate uses as a child, and is
refused where there is no single root.  A `toplevel` naming an element the file does not
define is read the same way, with a Python warning and a comment in the generated module
saying which root was used.

## What is generated

A module defining one `@DPP` function, named after the top event, that returns the top
event as a `pylea.syntax.dpp.Bool`.  Basic events become `pylea.syntax.flip.flip` calls,
`and` and `or` gates become `&` and `|`, and a voting gate becomes
`pylea.syntax.dpp.at_least`.  Only the elements the top event depends on are defined.
Basic events come first, in the order of the file; gates follow, each after its children,
otherwise and and or gates before voting gates and each in the order of the file.  The same
file therefore always gives the same module.

Each name becomes a Python identifier: characters other than ASCII letters, digits and `_`
become `_`, and a name that starts with a digit or `_`, or is a Python keyword, gets the
prefix `v_`.  Names that would become the same identifier, or one the module uses itself
(`DPP`, `Bool`, `at_least`, `flip`), are told apart by a suffix `_2`, `_3`, and so on; a
name that is already a valid identifier keeps it where it can.  A comment at the top of
the module lists every name written differently.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import NamedTuple

from pylea.utility._codegen import (
    Names, Token, convert_directory, number, tokenize, topological_order,
)

_UNSUPPORTED_GATES = {"pand", "por", "seq", "wsp", "csp", "hsp", "spare", "fdep", "pdep",
                      "mutex", "transfer"}
_INSPECTION = re.compile(r"\d*insp\d*")
_VOTING = re.compile(r"(\d+)of(\d+)")
_RESERVED = ("DPP", "Bool", "at_least", "flip")


def _q(name: str) -> str:
    """`name` in double quotes, as the file writes it."""
    return f'"{name}"'


def _statements(tokens: list[Token]) -> list[list[Token]]:
    """`tokens` split at each `;`, without empty statements."""
    statements: list[list[Token]] = []
    current: list[Token] = []
    for tok in tokens:
        if tok.is_mark(";"):
            if current:
                statements.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        raise ValueError(f"line {current[0].line}: the statement starting here does not "
                         "end with ';'")
    return statements


class _Gate(NamedTuple):
    kind: str | int             # "and", "or", or K of a voting gate
    children: list[str]
    line: int


class _Tree(NamedTuple):
    top: str
    events: dict[str, float]    # name to probability of failure, in file order
    gates: dict[str, _Gate]     # in file order
    note: str                   # what was assumed about the top event, or ""


def _event_probability(name: str, attrs: dict[str, tuple[str, int]], lambda_t: float) -> float:
    """The probability that the basic event `name` with attributes `attrs` has failed."""
    line = next(iter(attrs.values()))[1]
    where = f"line {line}: basic event {_q(name)}"
    for key in attrs:
        if key not in ("prob", "lambda", "phases", "dorm"):
            raise ValueError(f"{where}: the attribute {key!r} is not supported; supported "
                             "are prob, lambda, phases and dorm")
    values = {}
    for key, (text, _) in attrs.items():
        value = number(text)
        if value is None:
            raise ValueError(f"{where}: {key}={text} is not a finite number")
        values[key] = value
    if ("prob" in values) == ("lambda" in values):
        raise ValueError(f"{where} needs exactly one of prob and lambda")
    if "prob" in values:
        if "phases" in values:
            raise ValueError(f"{where}: phases needs lambda, not prob")
        p = values["prob"]
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{where}: prob={attrs['prob'][0]} is not between 0 and 1")
        return p
    rate = values["lambda"]
    if rate < 0.0:
        raise ValueError(f"{where}: lambda={attrs['lambda'][0]} is negative")
    phases = values.get("phases", 1.0)
    if phases < 1 or phases != int(phases):
        raise ValueError(f"{where}: phases={attrs['phases'][0]} is not a positive integer")
    x = rate * lambda_t
    if phases == 1:
        return -math.expm1(-x)
    from scipy.special import gammainc     # the Erlang distribution function
    return float(gammainc(int(phases), x))


def _parse(text: str, lambda_t: float) -> _Tree:
    """The tree `text` describes, with every check of the module docstring applied."""
    top: Token | None = None
    events: dict[str, float] = {}
    gates: dict[str, _Gate] = {}
    defined_on: dict[str, int] = {}
    for st in _statements(tokenize(text, ";=")):
        head, line = st[0], st[0].line
        if not head.quoted and head.text.lower() == "toplevel":
            if len(st) != 2 or not st[1].is_name():
                raise ValueError(f"line {line}: expected 'toplevel NAME;'")
            if top is not None:
                raise ValueError(f"line {line}: a second toplevel statement; the first is on "
                                 f"line {top.line}")
            top = st[1]
            continue
        if not head.is_name():
            raise ValueError(f"line {line}: expected a name at the start of the statement")
        name = head.text
        if name in defined_on:
            raise ValueError(f"line {line}: {_q(name)} is defined twice, first on line "
                             f"{defined_on[name]}")
        if len(st) < 2 or st[1].quoted or not st[1].is_name():
            raise ValueError(f"line {line}: expected a gate type or the attributes of a basic "
                             f"event after {_q(name)}")
        keyword = st[1].text.lower()
        if keyword in _UNSUPPORTED_GATES or _INSPECTION.fullmatch(keyword):
            raise ValueError(f"line {line}: {_q(name)} is a {st[1].text} gate, which is not "
                             "supported; supported are and, or and KofN gates")
        if len(st) > 2 and st[2].is_mark("="):
            attrs: dict[str, tuple[str, int]] = {}
            rest = st[1:]
            while rest:
                if (len(rest) < 3 or rest[0].quoted or not rest[1].is_mark("=")
                        or rest[2].quoted or not rest[2].is_name()):
                    raise ValueError(f"line {line}: expected attributes of the form "
                                     f"key=value after {_q(name)}")
                key = rest[0].text.lower()
                if key in attrs:
                    raise ValueError(f"line {line}: {_q(name)} gives {key} twice")
                attrs[key] = (rest[2].text, line)
                rest = rest[3:]
            events[name] = _event_probability(name, attrs, lambda_t)
        else:
            if any(not tok.is_name() for tok in st[2:]):
                raise ValueError(f"line {line}: expected the children of {_q(name)} after "
                                 f"{st[1].text}, found '='")
            children = [tok.text for tok in st[2:]]
            if not children:
                raise ValueError(f"line {line}: gate {_q(name)} has no children")
            voting = _VOTING.fullmatch(keyword)
            if keyword in ("and", "or"):
                gates[name] = _Gate(keyword, children, line)
            elif voting:
                k, n = int(voting.group(1)), int(voting.group(2))
                if n != len(children):
                    raise ValueError(f"line {line}: voting gate {_q(name)} is {st[1].text} but "
                                     f"has {len(children)} children")
                if not 1 <= k <= n:
                    raise ValueError(f"line {line}: voting gate {_q(name)} is {st[1].text}; K "
                                     "must be between 1 and N")
                gates[name] = _Gate(k, children, line)
            else:
                raise ValueError(f"line {line}: {_q(name)} has the unknown gate type "
                                 f"{st[1].text!r}; supported are and, or and KofN gates")
        defined_on[name] = line

    if not defined_on:
        raise ValueError("the file defines no element")
    for name, gate in gates.items():
        for child in gate.children:
            if child not in defined_on:
                raise ValueError(f"line {gate.line}: {_q(child)} is used by {_q(name)} but "
                                 "never defined")
    referenced = {c for gate in gates.values() for c in gate.children}
    roots = [name for name in defined_on if name not in referenced]
    if top is not None and top.text in defined_on:
        return _Tree(top.text, events, gates, "")
    if len(roots) != 1:
        listed = ", ".join(_q(r) for r in roots[:10]) + (", ..." if len(roots) > 10 else "")
        found = (f"{len(roots)} elements no gate uses: {listed}" if roots
                 else "every element is used by a gate")
        if top is None:
            raise ValueError(f"the file has no toplevel statement and no single element "
                             f"that no gate uses; there are {found}")
        raise ValueError(f"line {top.line}: toplevel names {_q(top.text)}, which the file "
                         f"does not define, and there is no single element that no gate "
                         f"uses to take instead; there are {found}")
    if top is None:
        return _Tree(roots[0], events, gates, "")
    note = (f"toplevel names {_q(top.text)}, which the file does not define; using its "
            f"only root {_q(roots[0])}")
    return _Tree(roots[0], events, gates, note)


def dft_to_python(dft_string: str, lambda_t: float = 1.0) -> str:
    """The source of a standalone module for the fault tree `dft_string` describes.

    The module defines one `@DPP` function returning the top event as a
    `pylea.syntax.dpp.Bool`.  `lambda_t` is the mission time at which a rate becomes a
    probability, in the time unit of the rates.  Raises `ValueError`, naming the line,
    for anything the module docstring lists as refused; a `toplevel` naming no element is
    resolved with a `UserWarning`.

        >>> print(dft_to_python('''
        ...     toplevel "pump fails";
        ...     "pump fails" or "motor" "seal";
        ...     "motor" lambda=1e-4;
        ...     "seal" prob=0.01;
        ... ''', lambda_t=100.0))
        # Names in the file written differently here:
        #     "pump fails" -> pump_fails
        from pylea.syntax.dpp import DPP, Bool, at_least
        from pylea.syntax.flip import flip
        <BLANKLINE>
        @DPP
        def pump_fails() -> Bool:
            motor = flip(0.009950166250831947)
            seal = flip(0.01)
            pump_fails = motor | seal
            return pump_fails
    """
    t = float(lambda_t)
    if not (math.isfinite(t) and t >= 0.0):
        raise ValueError(f"lambda_t must be finite and not negative, not {lambda_t!r}")
    tree = _parse(dft_string, t)
    gates = tree.gates
    voting = [g for g in gates if isinstance(gates[g].kind, int)]
    order = [g for g in gates if not isinstance(gates[g].kind, int)] + voting
    sorted_gates = topological_order(
        order, {g: [c for c in gates[g].children if c in gates] for g in order})

    needed = _cone(tree.top, gates)
    events = [e for e in tree.events if e in needed]
    emitted_gates = [g for g in sorted_gates if g in needed]

    names = Names(reserved=_RESERVED)
    ids = dict(zip(events + emitted_gates, names.claim_all(events + emitted_gates)))

    lines: list[str] = []
    if tree.note:
        warnings.warn(tree.note, stacklevel=2)
        lines.append(f"# {tree.note}")
    renamed = names.renamed()
    if renamed:
        lines.append("# Names in the file written differently here:")
        lines += [f"#     {_q(original)} -> {ident}" for original, ident in renamed]
    lines += [
        "from pylea.syntax.dpp import DPP, Bool, at_least",
        "from pylea.syntax.flip import flip",
        "",
        "@DPP",
        f"def {ids[tree.top]}() -> Bool:",
    ]
    for e in events:
        lines.append(f"    {ids[e]} = flip({tree.events[e]!r})")
    for g in emitted_gates:
        kind, children = gates[g].kind, [ids[c] for c in gates[g].children]
        if isinstance(kind, int):
            expr = f"at_least({kind}, {', '.join(children)})"
        else:
            expr = (" & " if kind == "and" else " | ").join(children)
        lines.append(f"    {ids[g]} = {expr}")
    lines.append(f"    return {ids[tree.top]}")
    return "\n".join(lines)


def _cone(top: str, gates: dict[str, _Gate]) -> set[str]:
    """`top` and every element it depends on."""
    needed = {top}
    stack = [top]
    while stack:
        gate = gates.get(stack.pop())
        if gate is None:
            continue
        for child in gate.children:
            if child not in needed:
                needed.add(child)
                stack.append(child)
    return needed


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_suite(
    src_dir: Path,
    dst_dir: Path,
    *,
    suite_name: str | None = None,
    lambda_t: float = 1.0,
    verbose: bool = True,
) -> tuple[dict[Path, Path], dict[Path, str]]:
    """``(converted, failed)`` after writing one module per `.dft` file directly in
    `src_dir` into `dst_dir`, a package created where missing.

    `converted` maps each file converted to its module, `failed` each other file to the
    message saying why it was not converted.  A module is named after its file, made a
    distinct identifier with the prefix `f_` where needed.  `lambda_t` is passed to
    `dft_to_python`.  With `verbose`, a line per file is printed, labelled with
    `suite_name`, by default the name of `src_dir`.

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     src, dst = Path(d, "trees"), Path(d, "modules")
        ...     src.mkdir()
        ...     _ = Path(src, "2of2.dft").write_text('toplevel T; T 2of2 a b; a prob=0.1; b prob=0.2;')
        ...     _ = Path(src, "pumps.dft").write_text('toplevel T; T pand a b; a prob=0.1; b prob=0.2;')
        ...     converted, failed = convert_suite(src, dst, verbose=False)
        ...     [p.name for p in converted.values()], list(failed.values())
        (['f_2of2.py'], ['line 1: "T" is a pand gate, which is not supported; supported are and, or and KofN gates'])
    """
    return convert_directory(
        src_dir, dst_dir, ".dft", lambda text, _: dft_to_python(text, lambda_t=lambda_t),
        prefix="f_", label=suite_name or src_dir.name, verbose=verbose)


def convert_all(
    benchmarks_root: Path,
    output_root: Path,
    *,
    lambda_t: float = 1.0,
    verbose: bool = True,
) -> tuple[dict[Path, Path], dict[Path, str]]:
    """``(converted, failed)`` after converting each sub-folder of `benchmarks_root` that
    holds `.dft` files into a package of the same name under `output_root`.

    The result and the arguments are those of `convert_suite`, over all sub-folders
    together.  With `verbose`, a summary line is printed at the end.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    top_init = output_root / "__init__.py"
    if not top_init.exists():
        top_init.write_text("")
    converted: dict[Path, Path] = {}
    failed: dict[Path, str] = {}
    for suite_dir in sorted(benchmarks_root.iterdir()):
        if not suite_dir.is_dir() or not any(suite_dir.glob("*.dft")):
            continue
        ok, bad = convert_suite(suite_dir, output_root / suite_dir.name,
                                suite_name=suite_dir.name, lambda_t=lambda_t, verbose=verbose)
        converted.update(ok)
        failed.update(bad)
    if verbose:
        print(f"\n{len(converted)} converted, {len(failed)} failed.")
    return converted, failed


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    """The exit status of the command line run with the arguments `argv`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pylea.utility.dft_to_dpp",
        description="Convert fault trees in the Galileo .dft format to Pylea modules.  "
                    "Given a file, print its module; given a folder, write one module per "
                    ".dft file in it into the folder --dst.")
    parser.add_argument("--src", required=True, help="a .dft file, or a folder of them")
    parser.add_argument("--dst", help="the folder to write the modules into")
    parser.add_argument("--recursive", action="store_true",
                        help="convert each sub-folder of --src into a package under --dst")
    parser.add_argument("--lambda-t", type=float, default=1.0,
                        help="the mission time at which a rate becomes a probability "
                             "(default: 1)")
    args = parser.parse_args(argv)
    src = Path(args.src)
    if src.is_file():
        if args.dst is not None or args.recursive:
            parser.error("--dst and --recursive apply only to a folder")
        try:
            print(dft_to_python(src.read_text(encoding="utf-8"), lambda_t=args.lambda_t))
        except ValueError as exc:
            parser.exit(1, f"{parser.prog}: {src}: {exc}\n")
        return 0
    if not src.is_dir():
        parser.error(f"{src} is neither a file nor a folder")
    if args.dst is None:
        parser.error("a folder needs --dst")
    if args.recursive:
        if not any(src.glob("*/*.dft")):
            parser.error(f"no .dft files in the sub-folders of {src}")
        _, failed = convert_all(src, Path(args.dst), lambda_t=args.lambda_t, verbose=True)
    else:
        if not any(src.glob("*.dft")):
            parser.error(f"no .dft files in {src}")
        _, failed = convert_suite(src, Path(args.dst), lambda_t=args.lambda_t, verbose=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
