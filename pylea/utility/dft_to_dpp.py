"""Fault trees from `.dft` files.

`dft_to_python` renders one module per tree; `convert_suite` and `convert_all` do it over a
directory and over a tree of directories.  The module is also a command-line tool, described
by its `--help`.

## Supported syntax

    toplevel NAME;                       the root gate
    NAME prob=<p>;                       a basic event, by probability
    NAME lambda=<rate>;                  a basic event, by rate
    NAME lambda=<rate> dorm=<d>;         the dormancy factor is ignored
    NAME and  child1 child2 ...;
    NAME or   child1 child2 ...;
    NAME KofN child1 child2 ...;         at least K of the N children, such as 2of4
    // line comments

Names and children may be quoted or bare.  A rate becomes a probability by
``1 - exp(-rate * lambda_t)``, the mission time defaulting to 1.  Only static gates are
supported; a dynamic one is refused rather than approximated.

## What is generated

A module defining one `@DPP` function that returns the top-level failure event.  Basic
events become `pylea.syntax.flip.flip` calls, `and` and `or` gates become `&` and `|`, and a
voting gate becomes `pylea.syntax.dpp.at_least`.

The root is the gate the file names as its top level.  A file may hold several gates that
are never a child, so the gate nothing references does not identify the root; where the file
names none, the first in sorted order is taken, so that a file always converts to the same
tree.  Gates are emitted in the order the file declares them for the same reason: iterating
a set instead would make the emitted order depend on string hashing, and a different
order of assignments is a different tensor network.
"""
from __future__ import annotations

import math
import re
import keyword
from pathlib import Path


# ---------------------------------------------------------------------------
# Name sanitisation
# ---------------------------------------------------------------------------

def sanitize(name: str) -> str:
    """`name` as a valid Python identifier.

        >>> sanitize("pump-1.a"), sanitize("2nd")
        ('pump_1_a', 'v_2nd')
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not clean or clean[0].isdigit() or keyword.iskeyword(clean):
        clean = f"v_{clean}"
    return clean


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# A name is either a double-quoted string or a bare token of letters, digits, `_`, `.` or `-`.
_NAME_PAT = r'(?:"([^"]+)"|([A-Za-z0-9_.\\-]+))'

def _extract_name(m_group1: str | None, m_group2: str | None) -> str:
    """Whichever of the two capture groups matched the node name."""
    return (m_group1 or m_group2 or "").strip()


def _parse_children(raw: str) -> list[str]:
    """The child names in `raw`, which is a space-separated list, quoted or not."""
    tokens = raw.split()
    children = []
    for tok in tokens:
        tok = tok.strip('"').strip("'")
        if tok:
            children.append(sanitize(tok))
    return children


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def dft_to_python(dft_string: str, lambda_t: float = 1.0) -> str:
    """The source of a standalone module for the fault tree `dft_string` describes.

    The module defines one `@DPP` function returning the top-level failure event as a
    `pylea.syntax.dpp.Bool`.  `lambda_t` is the mission time used to turn a rate into a
    probability.  Raises `ValueError` for a dynamic gate, for a file with no gates, and for
    children that are cyclic or never defined.

        >>> src = dft_to_python('''
        ...     toplevel "top";
        ...     "top" or "a" "b";
        ...     "a" prob=0.1;
        ...     "b" prob=0.2;
        ... ''')
        >>> "def top() -> Bool:" in src, "a | b" in src
        (True, True)
    """
    toplevel_re = re.compile(
        r"toplevel\s+" + _NAME_PAT + r"\s*;",
        re.IGNORECASE,
    )
    prob_re = re.compile(
        _NAME_PAT + r"\s+prob=([0-9.eE+\-]+)\s*;",
    )
    lambda_re = re.compile(
        _NAME_PAT + r"\s+lambda=([0-9.eE+\-]+)(?:\s+dorm=[^\s;]+)?\s*;",
    )
    # Voting gate: KofN  e.g. "2of4", "3of8"
    vote_re = re.compile(
        _NAME_PAT + r"\s+(\d+)of(\d+)\s+(.+?)\s*;",
    )
    gate_re = re.compile(
        _NAME_PAT + r"\s+(and|or)\s+(.+?)\s*;",
        re.IGNORECASE,
    )

    system_name = "fault_tree"
    declared_top = ""                              # the gate named by `toplevel`, if any
    gates: dict[str, tuple[str, list[str]]] = {}   # name → (type, children)
    votes: dict[str, tuple[int, list[str]]] = {}   # name → (k, children)
    probs: dict[str, str] = {}                     # name → prob literal
    all_children: set[str] = set()

    for raw_line in dft_string.splitlines():
        line = raw_line.strip()
        # Skip blank lines and comments
        if not line or line.startswith("//") or line.startswith("/*"):
            continue

        if m := toplevel_re.match(line):
            system_name = declared_top = sanitize(_extract_name(m.group(1), m.group(2)))

        elif m := vote_re.match(line):
            node_id = sanitize(_extract_name(m.group(1), m.group(2)))
            k = int(m.group(3))
            children = _parse_children(m.group(5))
            votes[node_id] = (k, children)
            all_children.update(children)

        elif m := gate_re.match(line):
            node_id = sanitize(_extract_name(m.group(1), m.group(2)))
            gate_type = m.group(3).lower()
            children = _parse_children(m.group(4))
            gates[node_id] = (gate_type, children)
            all_children.update(children)

        elif m := prob_re.match(line):
            node_id = sanitize(_extract_name(m.group(1), m.group(2)))
            probs[node_id] = m.group(3)

        elif m := lambda_re.match(line):
            node_id = sanitize(_extract_name(m.group(1), m.group(2)))
            lam = float(m.group(3))
            p = 1.0 - math.exp(-lam * lambda_t)
            probs[node_id] = repr(p)

        elif ";" in line:
            _check_unsupported(line)        # a statement none of the patterns matched

    # The root is the gate the file declares as its top level.  A file may hold several gates
    # that are never a child, so "the one nothing references" does not identify it and
    # picking among them would silently convert a different tree.  That fallback is used only
    # where the file declares nothing, and then in sorted order, so that at least the same
    # file always converts to the same tree.
    all_gate_names = set(gates) | set(votes)
    if declared_top in all_gate_names:
        root_gate = declared_top
    elif orphans := sorted(g for g in all_gate_names if g not in all_children):
        root_gate = orphans[0]
    elif all_gate_names:
        root_gate = sorted(all_gate_names)[-1]
    else:
        raise ValueError("No gates found in .dft file.")

    # Build output
    lines: list[str] = [
        "from pylea.syntax.dpp import DPP, Bool, at_least",
        "from pylea.syntax.flip import flip",
        "",
        "@DPP",
        f"def {system_name}() -> Bool:",
    ]

    defined: set[str] = set()

    # Emit basic events first
    for node_id, p_str in probs.items():
        lines.append(f"    {node_id} = flip({p_str})")
        defined.add(node_id)

    # Sorted topologically, taking nodes in the order the file declares them.  Iterating a
    # set instead would make the emitted order depend on string hashing, so the same file
    # would convert to a different though equivalent module in every process, and a different
    # order of assignments means a different tensor network and different timings.
    remaining = list(gates) + [v for v in votes if v not in gates]
    max_passes = len(remaining) + 1
    passes = 0
    while remaining and passes < max_passes:
        passes += 1
        progress = False
        for node_id in remaining[:]:
            if node_id in votes:
                k, children = votes[node_id]
            else:
                _, children = gates[node_id]
            if not all(c in defined for c in children):
                continue
            if node_id in votes:
                k, children = votes[node_id]
                args = ", ".join(children)
                lines.append(f"    {node_id} = at_least({k}, {args})")
            else:
                g_type, children = gates[node_id]
                op = " & " if g_type == "and" else " | "
                expr = op.join(children)
                lines.append(f"    {node_id} = {expr}")
            defined.add(node_id)
            remaining.remove(node_id)
            progress = True
        if not progress:
            missing = {
                c for node_id in remaining
                for c in (votes[node_id][1] if node_id in votes else gates[node_id][1])
                if c not in defined
            }
            raise ValueError(
                f"Cannot resolve definitions for: {remaining}. "
                f"Undefined/cyclic children: {missing}"
            )

    lines.append(f"    return {root_gate}")
    return "\n".join(lines)


def _check_unsupported(line: str) -> None:
    """Nothing, or `ValueError` where `line` declares a dynamic gate."""
    unsupported = ("pand", "seq", "spare", "fdep", "transfer")
    lower = line.lower()
    for gate in unsupported:
        if f" {gate} " in lower or f'"{gate}"' in lower:
            raise ValueError(
                f"Unsupported dynamic gate '{gate}' in line: {line!r}\n"
                "Only static gates (and, or, KofN) are supported."
            )


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
) -> tuple[int, int]:
    """``(converted, failed)`` after writing one module per `.dft` file of `src_dir`.

    `dst_dir` is created if it does not exist.  A file that fails to convert is counted and
    skipped rather than raising.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    init = dst_dir / "__init__.py"
    if not init.exists():
        init.write_text("")

    label = suite_name or src_dir.name
    ok = failed = 0

    for dft_path in sorted(src_dir.glob("*.dft")):
        stem = dft_path.stem
        mod_name = re.sub(r"[^a-zA-Z0-9_]", "_", stem)
        if mod_name[0].isdigit():
            mod_name = f"f_{mod_name}"
        out_path = dst_dir / f"{mod_name}.py"

        try:
            code = dft_to_python(dft_path.read_text(), lambda_t=lambda_t)
            out_path.write_text(code + "\n")
            if verbose:
                print(f"  [{label}] {stem} → {mod_name}.py")
            ok += 1
        except Exception as exc:
            if verbose:
                print(f"  [{label}] SKIP {stem}: {exc}")
            failed += 1

    return ok, failed


def convert_all(
    benchmarks_root: Path,
    output_root: Path,
    *,
    lambda_t: float = 1.0,
    verbose: bool = True,
) -> None:
    """Nothing, having converted every suite under `benchmarks_root` into `output_root`.

    A suite is a subdirectory holding `.dft` files, and each becomes a package of modules.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    top_init = output_root / "__init__.py"
    if not top_init.exists():
        top_init.write_text("")

    total_ok = total_failed = 0
    for suite_dir in sorted(benchmarks_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        dft_files = list(suite_dir.glob("*.dft"))
        if not dft_files:
            continue
        dst = output_root / suite_dir.name
        ok, failed = convert_suite(
            suite_dir, dst,
            suite_name=suite_dir.name,
            lambda_t=lambda_t,
            verbose=verbose,
        )
        total_ok += ok
        total_failed += failed

    print(f"\nConversion complete: {total_ok} succeeded, {total_failed} skipped/failed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Galileo/SCRAM .dft files to Pylea DPP Python modules."
    )
    parser.add_argument(
        "--src", required=True,
        help="Root directory containing suite sub-folders of .dft files.",
    )
    parser.add_argument(
        "--dst", required=True,
        help="Output root directory for generated Python packages.",
    )
    parser.add_argument(
        "--lambda-t", type=float, default=1.0,
        help="Mission time for lambda→probability conversion (default: 1.0).",
    )
    args = parser.parse_args()

    convert_all(
        Path(args.src),
        Path(args.dst),
        lambda_t=args.lambda_t,
        verbose=True,
    )
