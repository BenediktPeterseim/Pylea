"""Bayesian networks from `.bif` files.

`bif_to_python` renders one module per network; `make_query_module` renders a query against
one; `convert_suite` and `generate_benchmarks` do each over a directory.  The module is also
a command-line tool, described by its `--help`.

Supported syntax:

    network NAME { }
    variable NAME { type discrete [ N ] { s1, s2, ... }; }
    probability ( VAR ) { table p1, p2, ...; }
    probability ( VAR | P1, P2, ... ) { (v1, v2, ...) p1, p2, ...; ... }
    // line comments

Only discrete variables are supported.  A generated module holds one
`pylea.syntax.enum.Enum` subclass per variable and one `@DPP` function returning the joint
distribution in topological order.  A variable with parents becomes a `@DPP` function of
its parents applying an `pylea.syntax.enum.Enum.cpt` call, so a diagram labels its box
with that function's name; one without becomes `pylea.syntax.enum.Enum.random`.

A generated query asks ``P(leaf == first_state | evidence)``, the leaf being the variable
that is never a parent and has the most ancestors, ties broken alphabetically.  Evidence
variables are drawn at random from the rest under a seed, and their states come from one
draw from the network, which makes the evidence typical and gives it positive probability by
construction.  It is fixed as the module is written, so a query is a file to be read rather
than a function of a seed.
"""
from __future__ import annotations

import random
import re
import keyword
from pathlib import Path


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------

def _sanitize_class(name: str) -> str:
    """`name` as a valid Python class name, capitalised."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not clean or clean[0].isdigit() or keyword.iskeyword(clean):
        clean = f"V_{clean}"
    return clean[0].upper() + clean[1:] if clean else clean


def _sanitize_var(name: str) -> str:
    """`name` as a valid Python identifier, in lower case."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    if not clean or clean[0].isdigit() or keyword.iskeyword(clean):
        clean = f"v_{clean}"
    return clean


def _sanitize_state(name: str) -> str:
    """`name` as a string safe to use as a key in the generated source."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _parse_bif(bif_string: str) -> tuple[
    str,                              # network name
    list[str],                        # variable declaration order
    dict[str, list[str]],             # var_name -> [state_names] (sanitized)
    dict[str, list[str]],             # var_name -> [parent_names] (original)
    dict[str, dict[tuple, list[float]]], # var_name -> {(parent_val_tuple): [probs]}
]:
    text = _strip_comments(bif_string)

    # Network name
    net_m = re.search(r"network\s+(\S+)\s*\{", text)
    network_name = _sanitize_var(net_m.group(1)) if net_m else "bayesian_network"
    if network_name in ("unknown", ""):
        network_name = "bayesian_network"

    # Variables
    var_order: list[str] = []
    var_states: dict[str, list[str]] = {}

    for m in re.finditer(
        r"variable\s+(\S+)\s*\{[^}]*type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]+)\}",
        text,
        re.DOTALL,
    ):
        vname = m.group(1).strip()
        raw_states = [s.strip() for s in m.group(2).split(",") if s.strip()]
        states = [_sanitize_state(s) for s in raw_states]
        var_order.append(vname)
        var_states[vname] = states

    # Probability blocks
    cpt_parents: dict[str, list[str]] = {}
    cpt_rows: dict[str, dict[tuple, list[float]]] = {}

    prob_block_re = re.compile(r"probability\s*\(([^)]+)\)\s*\{([^}]*)\}", re.DOTALL)
    for m in prob_block_re.finditer(text):
        header = m.group(1).strip()
        body = m.group(2).strip()

        if "|" in header:
            parts = header.split("|", 1)
            vname = parts[0].strip()
            parents = [p.strip() for p in parts[1].split(",")]
        else:
            vname = header.strip()
            parents = []

        cpt_parents[vname] = parents

        rows: dict[tuple, list[float]] = {}

        if not parents:
            # table p1, p2, ...;
            table_m = re.search(r"table\s+([\d.,\s eE+\-]+)\s*;", body)
            if table_m:
                probs = [float(p.strip()) for p in table_m.group(1).split(",") if p.strip()]
                rows[()] = probs
        else:
            # (v1, v2, ...) p1, p2, ...;
            row_re = re.compile(r"\(([^)]+)\)\s*([\d.,\s eE+\-]+)\s*;")
            for row_m in row_re.finditer(body):
                raw_vals = [v.strip() for v in row_m.group(1).split(",")]
                key = tuple(_sanitize_state(v) for v in raw_vals)
                probs = [float(p.strip()) for p in row_m.group(2).split(",") if p.strip()]
                rows[key] = probs

        cpt_rows[vname] = rows

    return network_name, var_order, var_states, cpt_parents, cpt_rows


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def _build_random(child_class: str, child_states: list[str], probs: list[float]) -> str:
    items = ", ".join(f'"{s}": {p}' for s, p in zip(child_states, probs))
    return f"{child_class}.random({{{items}}})"


def _build_cpt_direct(
    rows: dict[tuple, list[float]],
    parent_cls_names: list[str],
    child_class: str,
    indent: int,
) -> str:
    """The source of a ``Child.cpt([ParentClass, ...], rows)`` expression.

    Applied inside a `@DPP` function named after the table, which the definition of the
    joint network calls: ``child = child_given_parent(p1, p2)``.
    """
    pad = "    " * indent
    inner_pad = pad + "    "
    row_pad = inner_pad + "    "

    parent_types_str = "[" + ", ".join(parent_cls_names) + "]"

    row_lines: list[str] = []
    for key, probs in rows.items():
        if len(key) == 1:
            key_str = f'("{key[0]}",)'
        else:
            key_str = "(" + ", ".join(f'"{s}"' for s in key) + ")"
        probs_str = "[" + ", ".join(repr(p) for p in probs) + "]"
        row_lines.append(f"{row_pad}{key_str}: {probs_str},")

    rows_block = "{\n" + "\n".join(row_lines) + "\n" + inner_pad + "}"
    return (
        f"{child_class}.cpt(\n"
        f"{inner_pad}{parent_types_str},\n"
        f"{inner_pad}{rows_block},\n"
        f"{pad})"
    )


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def _topo_sort(var_order: list[str], cpt_parents: dict[str, list[str]]) -> list[str]:
    parents_map = {v: list(cpt_parents.get(v, [])) for v in var_order}
    sorted_vars: list[str] = []
    remaining = list(var_order)
    defined: set[str] = set()

    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        progress = False
        for v in remaining[:]:
            if all(p in defined for p in parents_map[v]):
                sorted_vars.append(v)
                defined.add(v)
                remaining.remove(v)
                progress = True
        if not progress:
            raise ValueError(f"Cycle or undefined parents among: {remaining}")

    return sorted_vars


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def bif_to_python(bif_string: str, network_name: str | None = None) -> str:
    """The source of a standalone module for the network `bif_string` describes.

    The module defines one `pylea.syntax.enum.Enum` subclass per variable, one conditional
    probability table per variable with parents, and a `@DPP` function returning the joint
    distribution.  `network_name` overrides the name of that function, which otherwise comes
    from the file.

        >>> src = bif_to_python('''
        ...     network demo { }
        ...     variable rain { type discrete [ 2 ] { no, yes }; }
        ...     probability ( rain ) { table 0.8, 0.2; }
        ... ''')
        >>> "class Rain(Enum)" in src, "def demo()" in src
        (True, True)
    """
    parsed_name, var_order, var_states, cpt_parents, cpt_rows = _parse_bif(bif_string)
    fn_name = network_name or parsed_name

    sorted_vars = _topo_sort(var_order, cpt_parents)

    lines: list[str] = [
        "from pylea.syntax.dpp import DPP",
        "from pylea.syntax.enum import Enum",
        "",
        "",
    ]

    # --- Enum classes ---
    for vname in var_order:
        cls = _sanitize_class(vname)
        states = var_states[vname]
        elems = ", ".join(f'"{s}"' for s in states)
        lines += [
            f"class {cls}(Enum):",
            f"    elements = [{elems}]",
            "",
            "",
        ]

    # --- CPT objects for conditional variables (via Enum.cpt, fast path) ---
    cpt_fn_names: dict[str, str] = {}  # vname -> assigned name

    for vname in sorted_vars:
        parents = cpt_parents.get(vname, [])
        if not parents:
            continue

        child_cls = _sanitize_class(vname)
        parent_cls_names = [_sanitize_class(p) for p in parents]

        parent_part = '_'.join(_sanitize_var(p) for p in parents)
        fn_name_cpt = f"{_sanitize_var(vname)}_given_{parent_part}"
        cpt_fn_names[vname] = fn_name_cpt

        rows = cpt_rows[vname]
        expr = _build_cpt_direct(rows, parent_cls_names, child_cls, indent=1)

        params = ", ".join(f"{_sanitize_var(p)}: {c}"
                           for p, c in zip(parents, parent_cls_names))
        args = ", ".join(_sanitize_var(p) for p in parents)
        lines += [
            "@DPP",
            f"def {fn_name_cpt}({params}) -> {child_cls}:",
            f"    return {expr}({args})",
            "",
            "",
        ]

    # --- Joint distribution function ---
    MULTILINE_THRESHOLD = 5
    multiline = len(sorted_vars) > MULTILINE_THRESHOLD

    if multiline:
        type_items = "".join(f"\n    {_sanitize_class(v)}," for v in sorted_vars)
        lines += [
            "@DPP",
            f"def {fn_name}() -> tuple[{type_items}\n]:",
        ]
    else:
        ret_types = ", ".join(_sanitize_class(v) for v in sorted_vars)
        lines += [
            "@DPP",
            f"def {fn_name}() -> tuple[{ret_types}]:",
        ]

    ret_vars: list[str] = []
    for vname in sorted_vars:
        cls = _sanitize_class(vname)
        var = _sanitize_var(vname)
        if var in ret_vars:
            var = f"{var}_{sorted_vars.index(vname)}"
        ret_vars.append(var)
        parents = cpt_parents.get(vname, [])

        if not parents:
            probs = cpt_rows[vname][()]
            rhs = _build_random(cls, var_states[vname], probs)
            lines.append(f"    {var} = {rhs}")
        else:
            parent_local_vars = []
            for p in parents:
                p_var = _sanitize_var(p)
                if p_var not in ret_vars:
                    raise ValueError(f"Parent '{p}' not yet defined (check topological sort)")
                p_idx = sorted_vars.index(p)
                parent_local_vars.append(ret_vars[p_idx])
            args = ", ".join(parent_local_vars)
            fn = cpt_fn_names[vname]
            lines.append(f"    {var} = {fn}({args})")

    if multiline:
        ret_items = "".join(f"\n        {v}," for v in ret_vars)
        lines.append(f"    return ({ret_items}\n    )")
    else:
        lines.append(f"    return {', '.join(ret_vars)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def convert_suite(
    src_dir: Path,
    dst_dir: Path,
    *,
    suite_name: str | None = None,
    verbose: bool = True,
) -> tuple[int, int]:
    """``(converted, failed)`` after writing one module per `.bif` file of `src_dir`.

    `dst_dir` is created if it does not exist.  A file that fails to convert is counted and
    skipped rather than raising.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    init = dst_dir / "__init__.py"
    if not init.exists():
        init.write_text("")

    label = suite_name or src_dir.name
    ok = failed = 0

    for bif_path in sorted(src_dir.glob("*.bif")):
        stem = bif_path.stem
        mod_name = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
        if mod_name[0].isdigit():
            mod_name = f"bn_{mod_name}"
        out_path = dst_dir / f"{mod_name}.py"

        try:
            code = bif_to_python(bif_path.read_text(), network_name=mod_name)
            out_path.write_text(code + "\n")
            if verbose:
                print(f"  [{label}] {stem} → {mod_name}.py")
            ok += 1
        except Exception as exc:
            if verbose:
                print(f"  [{label}] SKIP {stem}: {exc}")
            failed += 1

    return ok, failed


# ---------------------------------------------------------------------------
# Benchmark query generation
# ---------------------------------------------------------------------------

def _ancestors(vname: str, cpt_parents: dict[str, list[str]]) -> set[str]:
    """Every ancestor of `vname`, however far up."""
    result: set[str] = set()
    stack = list(cpt_parents.get(vname, []))
    while stack:
        p = stack.pop()
        if p not in result:
            result.add(p)
            stack.extend(cpt_parents.get(p, []))
    return result


def pick_query_variable(
    var_order: list[str],
    cpt_parents: dict[str, list[str]],
) -> str:
    """The variable a generated marginal query asks about.

    The leaf with the most ancestors, a leaf being a variable that is never a parent, so
    that the query depends on as much of the network as possible.  Ties break
    alphabetically, so that the same network always gives the same query.

        >>> pick_query_variable(["rain", "sprinkler", "grass"],
        ...                     {"grass": ["rain", "sprinkler"]})
        'grass'
    """
    all_parents: set[str] = {p for ps in cpt_parents.values() for p in ps}
    leaves = [v for v in var_order if v not in all_parents]
    if not leaves:
        leaves = list(var_order)

    return max(leaves, key=lambda v: (len(_ancestors(v, cpt_parents)), v))


def query_module_name(mod_name: str, n_evidence: int) -> str:
    """The name of the query module over network `mod_name` conditioned on `n_evidence`.

    The marginal query keeps the network's own name, so generating conditioned queries
    beside an existing corpus of marginal ones leaves the latter untouched.

        >>> query_module_name("alarm", 0), query_module_name("alarm", 5)
        ('alarm', 'alarm_e5')
    """
    return mod_name if n_evidence == 0 else f"{mod_name}_e{n_evidence}"


def _sample_assignment(sorted_vars, var_states, cpt_parents, cpt_rows, rng) -> dict[str, str]:
    """One draw from the network's joint distribution, as ``{variable: state}``.

    Drawn in topological order, so a variable is drawn only once its parents have been.
    Evidence read off such a draw has positive probability by construction, which evidence
    assembled state by state does not.
    """
    drawn: dict[str, str] = {}
    for v in sorted_vars:
        key = tuple(drawn[p] for p in cpt_parents.get(v, []))
        rows = cpt_rows[v]
        if key not in rows:
            raise ValueError(f"{v}: the CPT has no row for parent states {key}")
        drawn[v] = rng.choices(var_states[v], weights=rows[key])[0]
    return drawn


def query_specification(bif_string: str, *, n_evidence: int = 0, seed: int = 0) -> tuple[
    tuple[int, str, str],                 # the query: position, variable, state
    list[tuple[int, str, str]],           # the evidence, in network order
]:
    """What the generated query asks of the network `bif_string` describes.

    A position is an index into the tuple the converted network returns, which holds the
    variables in topological order.  The query variable is `pick_query_variable`'s choice
    and the state is its first.  The evidence variables are drawn uniformly at random from
    the others under `seed`, and their states come from one draw from the network itself, so
    the evidence is typical of the model and cannot have probability zero.

    Kept apart from the module `make_query_module` renders, so that what a benchmark asks
    can be read without parsing generated source.
    """
    _, var_order, var_states, cpt_parents, cpt_rows = _parse_bif(bif_string)
    sorted_vars = _topo_sort(var_order, cpt_parents)
    rng = random.Random(seed)

    query_var = pick_query_variable(var_order, cpt_parents)
    others = [v for v in sorted_vars if v != query_var]
    chosen = rng.sample(others, min(n_evidence, len(others)))
    drawn = _sample_assignment(sorted_vars, var_states, cpt_parents, cpt_rows, rng)

    query = (sorted_vars.index(query_var), query_var, var_states[query_var][0])
    evidence = sorted((sorted_vars.index(v), v, drawn[v]) for v in chosen)
    return query, evidence


def make_query_module(
    bif_string: str,
    *,
    mod_name: str,
    pylea_import_path: str,
    n_evidence: int = 0,
    seed: int = 0,
) -> str:
    """The source of a module asking one query of the network `mod_name`, as a `@DPP`.

    The query is ``P(leaf == first_state | evidence)``, the leaf being
    `pick_query_variable`'s choice.  `n_evidence` says how many variables are conditioned
    on, and `query_specification` describes how those are chosen; at zero the query is the
    plain marginal and nothing is observed.

    The evidence is fixed here, as the module is generated, rather than at run time, so that
    a benchmark is a file which can be read and cited rather than a function of a seed.

    `pylea_import_path` is the dotted path to the converted network module, which the
    generated module imports and calls.
    """
    query, evidence = query_specification(bif_string, n_evidence=n_evidence, seed=seed)
    _, query_var, query_state = query

    given = ", ".join(f"{v} = '{s}'" for _, v, s in evidence)
    head = f"# Query: P({query_var} == '{query_state}'"
    lines = [
        "from pylea import DPP, Bool" + (", observe" if evidence else ""),
        f"from {pylea_import_path} import {mod_name} as _network",
        "",
        head + (f" | {given})" if evidence else ")"),
    ]
    if evidence:
        lines += [
            "# Evidence variables drawn uniformly at random, their states drawn from the",
            f"# network's own joint distribution, seed {seed}.",
        ]
    lines += [
        "@DPP",
        f"def {query_module_name(mod_name, n_evidence)}() -> Bool:",
        "    xs = _network()",
    ]
    lines += [f'    observe(xs[{i}] == "{s}")   # {v}' for i, v, s in evidence]
    lines.append(f'    return xs[{query[0]}] == "{query_state}"   # {query_var}')
    return "\n".join(lines)


def generate_benchmarks(
    bif_dir: Path,
    pylea_dir: Path,
    dst_dir: Path,
    *,
    package: str,
    n_evidence: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[int, int]:
    """``(written, failed)`` after writing one query module per `.bif` file of `bif_dir`.

    Each module asks `make_query_module`'s query of the corresponding network, conditioned
    on `n_evidence` variables under `seed`.  `pylea_dir` holds the networks `convert_suite`
    converted, and `package` is the dotted path under which the generated modules import
    them, such as ``"networks"`` for a folder `networks` on the import path.  A network
    without a module in `pylea_dir` is counted as failed.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    init = dst_dir / "__init__.py"
    if not init.exists():
        init.write_text("")

    ok = failed = 0
    for bif_path in sorted(bif_dir.glob("*.bif")):
        stem = bif_path.stem
        mod_name = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
        if mod_name[0].isdigit():
            mod_name = f"bn_{mod_name}"

        pylea_mod_path = pylea_dir / f"{mod_name}.py"
        if not pylea_mod_path.exists():
            if verbose:
                print(f"  SKIP {stem}: no Pylea module at {pylea_mod_path}")
            failed += 1
            continue

        import_path = f"{package}.{mod_name}"
        out_path = dst_dir / f"{query_module_name(mod_name, n_evidence)}.py"

        try:
            code = make_query_module(
                bif_path.read_text(),
                mod_name=mod_name,
                pylea_import_path=import_path,
                n_evidence=n_evidence,
                seed=seed,
            )
            out_path.write_text(code + "\n")
            if verbose:
                print(f"  {stem} → {out_path.name}")
            ok += 1
        except Exception as exc:
            if verbose:
                print(f"  SKIP {stem}: {exc}")
            failed += 1

    return ok, failed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert .bif Bayesian network files to Pylea DPP Python modules."
    )
    parser.add_argument(
        "--src", default=None,
        help="Directory containing .bif files.",
    )
    parser.add_argument(
        "--dst", default=None,
        help="Output directory for generated Python modules.",
    )
    parser.add_argument(
        "--file", default=None,
        help="Convert a single .bif file and print to stdout.",
    )
    parser.add_argument(
        "--queries", default=None,
        help="Also write one query module per network into this directory.",
    )
    parser.add_argument(
        "--package", default=None,
        help="The dotted path under which the query modules import the networks of --dst.",
    )
    parser.add_argument(
        "--evidence", type=int, default=0,
        help="How many variables each generated query conditions on.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed choosing the evidence variables and their states.",
    )
    args = parser.parse_args()
    if not args.file and (args.src is None or args.dst is None):
        parser.error("give --file, or both --src and --dst")
    if args.queries and args.package is None:
        parser.error("--queries needs --package")

    if args.file:
        bif_path = Path(args.file)
        print(bif_to_python(bif_path.read_text(), network_name=bif_path.stem))
    else:
        convert_suite(
            Path(args.src),
            Path(args.dst),
            suite_name="bayesian_networks",
            verbose=True,
        )
        if args.queries:
            generate_benchmarks(
                Path(args.src),
                Path(args.dst),
                Path(args.queries),
                package=args.package,
                n_evidence=args.evidence,
                seed=args.seed,
                verbose=True,
            )
