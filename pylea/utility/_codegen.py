"""Code generation shared by the `.bif` and `.dft` converters.

Both converters split a file into tokens, read names from it and write them into Python
source, sort elements so that each is defined after what it depends on, and convert every
file of a folder.

`tokenize` splits a file into names, double-quoted strings and punctuation, without the
comments both formats allow: ``//`` to the end of the line and ``/* ... */``.

`Names` turns the names of a file into identifiers of one Python namespace.  Two names
that become the same identifier, or a name that becomes an identifier the generated code
itself uses, get distinct identifiers by a numbered suffix, and `Names.renamed` lists
every name that is not written as it is, for a comment in the generated module.

`topological_order` orders elements by their dependencies.  Among the orders that put
every element after its parents, it returns the one that takes the elements in repeated
passes over the given order, accepting an element as soon as all of its parents have been
accepted.  The order matters to a user: it fixes the positions of the tuple a converted
Bayesian network returns.

`convert_directory` converts every file of a folder into a package of modules and reports
each failure with its message.
"""
from __future__ import annotations

import keyword
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

_NOT_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class Token(NamedTuple):
    """One token of a file: a name, a quoted string, or a punctuation character."""
    text: str
    quoted: bool
    punctuation: bool
    line: int

    def is_name(self) -> bool:
        return not self.punctuation

    def is_mark(self, char: str) -> bool:
        """Whether the token is the punctuation character `char`."""
        return self.punctuation and self.text == char

    def shown(self) -> str:
        """The token as the file writes it, for a message."""
        return f'"{self.text}"' if self.quoted else repr(self.text)


def tokenize(text: str, punctuation: str, separators: str = "") -> list[Token]:
    """The tokens of `text`, without comments, whitespace and the characters `separators`.

    A token is a character of `punctuation`, a double-quoted string on one line, or a run
    of other characters.  Raises `ValueError` naming the line of a comment or a quoted
    string that is not closed.

        >>> [t.text for t in tokenize('"pump a" prob=0.1; // a comment', ";=")]
        ['pump a', 'prob', '=', '0.1', ';']
    """
    p, s = re.escape(punctuation), re.escape(separators)
    space = rf"[\s{s}]+" if separators else r"\s+"
    pattern = re.compile(
        rf'(?P<sp>{space})|(?P<lc>//[^\n]*)|(?P<bc>/\*.*?\*/)|(?P<open>/\*)'
        rf'|"(?P<q>[^"\n]*)"|(?P<unclosed>")|(?P<p>[{p}])'
        rf'|(?P<w>(?:[^\s"{p}{s}/]|/(?![/*]))+)', re.S)
    tokens: list[Token] = []
    line = 1
    for m in pattern.finditer(text):
        kind = m.lastgroup
        if kind == "sp" or kind == "bc":
            line += m.group().count("\n")
        elif kind == "q":
            tokens.append(Token(m.group("q"), True, False, line))
        elif kind == "p":
            tokens.append(Token(m.group(), False, True, line))
        elif kind == "w":
            tokens.append(Token(m.group(), False, False, line))
        elif kind == "open":
            raise ValueError(f"line {line}: a comment opened with /* is never closed")
        elif kind == "unclosed":
            raise ValueError(f'line {line}: a name opened with " is not closed on its line')
    return tokens


class Names:
    """Identifiers of one Python namespace, claimed for the names a file uses.

    `reserved` holds the identifiers the generated code itself uses in that namespace;
    no name is given one of them.  `prefix` is put in front of a name that would
    otherwise be empty, start with a digit or an underscore, or be a keyword.

        >>> names = Names(reserved=["flip"])
        >>> [names.claim(n) for n in ["a-b", "a_b", "flip", "class", "2nd", "a-b"]]
        ['a_b', 'a_b_2', 'flip_2', 'v_class', 'v_2nd', 'a_b']
        >>> names.renamed()
        [('a-b', 'a_b'), ('a_b', 'a_b_2'), ('flip', 'flip_2'), ('class', 'v_class'), ('2nd', 'v_2nd')]
    """

    def __init__(self, reserved: Iterable[str] = (), prefix: str = "v_") -> None:
        self._taken: set[str] = set(reserved)
        self._prefix = prefix
        self._claimed: dict[tuple[str, str], str] = {}

    def claim(self, original: str, kind: str = "name") -> str:
        """The identifier for `original`, the same on every call with the same arguments.

        Characters other than ASCII letters, digits and ``_`` become ``_``.  `kind` is
        ``"name"``, ``"class"`` or ``"variable"``.  A class name starts with an upper-case
        letter.  A variable starting with an upper-case letter is written in lower case,
        so that it differs from the class of the same name.  An identifier that is
        reserved or already given to another name gets the suffix ``_2``, ``_3``, and so
        on, in the order of claiming.

            >>> Names().claim("HISTORY", "class"), Names().claim("HISTORY", "variable")
            ('HISTORY', 'history')
        """
        if kind not in ("name", "class", "variable"):
            raise ValueError(f"kind must be 'name', 'class' or 'variable', not {kind!r}")
        key = (original, kind)
        if key in self._claimed:
            return self._claimed[key]
        base = _plain(original, kind, self._prefix)
        ident, n = base, 1
        while ident in self._taken:
            n += 1
            ident = f"{base}_{n}"
        self._taken.add(ident)
        self._claimed[key] = ident
        return ident

    def claim_all(self, originals: Iterable[str], kind: str = "name") -> list[str]:
        """The identifiers for `originals`, claimed so that a name keeps its spelling
        wherever it is already a valid identifier that nothing reserved or claimed before
        takes.

            >>> Names().claim_all(["a-b", "a_b"])
            ['a_b_2', 'a_b']
        """
        originals = list(originals)
        unchanged = [o for o in originals if _plain(o, kind, self._prefix) == _as_kind(o, kind)]
        for o in unchanged:
            self.claim(o, kind)
        return [self.claim(o, kind) for o in originals]

    def renamed(self) -> list[tuple[str, str]]:
        """``(original, identifier)`` for every name claimed whose identifier is not the
        name itself, apart from the case `kind` sets."""
        return [(original, ident) for (original, kind), ident in self._claimed.items()
                if ident != _as_kind(original, kind)]


def _as_kind(original: str, kind: str) -> str:
    """`original` in the case `kind` sets."""
    if kind == "class":
        return original[:1].upper() + original[1:]
    if kind == "variable" and original[:1].isupper():
        return original.lower()
    return original


def _plain(original: str, kind: str, prefix: str) -> str:
    """The identifier `Names.claim` gives `original` where nothing is taken."""
    ident = _NOT_IDENTIFIER.sub("_", original)
    if not ident or ident[0].isdigit() or ident[0] == "_" or keyword.iskeyword(ident):
        ident = prefix + ident
    return _as_kind(ident, kind)


def topological_order(order: Sequence[str], parents: Mapping[str, Sequence[str]]) -> list[str]:
    """`order` rearranged so that every element comes after each of its `parents`.

    The result is the order in which repeated passes over `order` accept the elements, a
    pass accepting an element once all its parents are accepted, in an earlier pass or
    earlier in the same one.  Raises `ValueError` naming a parent that is not in
    `order`, or the elements of a cycle.

        >>> topological_order(["b", "a", "c"], {"b": ["a"]})
        ['a', 'c', 'b']

    The pass in which an element is accepted is
    ``p(v) = max(1, max over parents u of p(u) + (1 if u comes after v in order else 0))``,
    and the passes accept elements in the order given, so the result is `order` sorted by
    ``(p(v), position of v)``.  `p` is computed in one sweep of Kahn's algorithm.
    """
    index = {v: i for i, v in enumerate(order)}
    if len(index) != len(order):
        raise ValueError("an element is listed twice")
    children: dict[str, list[str]] = {v: [] for v in order}
    waiting = dict.fromkeys(order, 0)
    for v in order:
        for u in parents.get(v, ()):
            if u not in index:
                raise ValueError(f"{v!r} depends on {u!r}, which is not defined")
            children[u].append(v)
            waiting[v] += 1
    p = dict.fromkeys(order, 1)
    ready = [v for v in order if waiting[v] == 0]
    done = 0
    while ready:
        u = ready.pop()
        done += 1
        for v in children[u]:
            p[v] = max(p[v], p[u] + (index[u] > index[v]))
            waiting[v] -= 1
            if waiting[v] == 0:
                ready.append(v)
    if done < len(order):
        raise ValueError("these depend on each other in a cycle: "
                         + ", ".join(repr(v) for v in _a_cycle(order, parents, waiting)))
    return sorted(order, key=lambda v: (p[v], index[v]))


def _a_cycle(order: Sequence[str], parents: Mapping[str, Sequence[str]],
             waiting: Mapping[str, int]) -> list[str]:
    """One cycle among the elements Kahn's algorithm left waiting, in dependency order.

    Every element left waiting has a parent left waiting, so following such parents
    from any of them repeats an element, which closes a cycle.
    """
    start = next(v for v in order if waiting[v] > 0)
    path: list[str] = []
    seen: dict[str, int] = {}
    v = start
    while v not in seen:
        seen[v] = len(path)
        path.append(v)
        v = next(u for u in parents[v] if waiting[u] > 0)
    return path[seen[v]:][::-1]


def literal(text: str) -> str:
    """`text` as a Python string literal, the one way a name of the file enters the source.

        >>> print(literal('0-3 "days"'))
        '0-3 "days"'
    """
    return repr(text)


def number(text: str) -> float | None:
    """`text` as a finite float where it is a decimal number, otherwise None.

        >>> number("5.0e-6"), number("1_0"), number("inf"), number("1e999")
        (5e-06, None, None, None)
    """
    if not _NUMBER.fullmatch(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def module_name(stem: str, prefix: str, reserved: Iterable[str] = ()) -> str:
    """The module name for a file named `stem`, `prefix` taking the place of `Names`'s
    ``v_``, and avoiding the identifiers `reserved`.

        >>> module_name("my-net", "bn_"), module_name("2of3", "f_")
        ('my_net', 'f_2of3')
    """
    return Names(reserved, prefix=prefix).claim(stem)


def convert_directory(
    src_dir: Path,
    dst_dir: Path,
    suffix: str,
    convert: Callable[[str, str], str],
    *,
    prefix: str,
    reserved: Iterable[str] = (),
    label: str,
    verbose: bool,
) -> tuple[dict[Path, Path], dict[Path, str]]:
    """``(converted, failed)`` after writing one module per file ending in `suffix`
    directly in `src_dir`.

    `convert(text, module_name)` returns the source of the module for a file's text.
    `dst_dir` is created with an `__init__.py` where missing.  `converted` maps each file
    converted to the module written, `failed` each file that could not be read or
    converted to the message saying why.  Module names come from the file names, each
    distinct and none of `reserved`, with `prefix` where a file name does not start with
    a letter.  With
    `verbose`, a line per file is printed, starting with `label`.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    init = dst_dir / "__init__.py"
    if not init.exists():
        init.write_text("")
    paths = sorted(src_dir.glob(f"*{suffix}"))
    names = Names(reserved, prefix=prefix)
    modules = names.claim_all(p.stem for p in paths)
    converted: dict[Path, Path] = {}
    failed: dict[Path, str] = {}
    for path, mod in zip(paths, modules):
        out = dst_dir / f"{mod}.py"
        try:
            source = convert(path.read_text(encoding="utf-8"), mod)
        except (ValueError, OSError) as exc:
            failed[path] = str(exc)
            if verbose:
                print(f"  [{label}] failed {path.name}: {exc}")
            continue
        out.write_text(source + "\n", encoding="utf-8")
        converted[path] = out
        if verbose:
            print(f"  [{label}] {path.name} -> {out.name}")
    return converted, failed
