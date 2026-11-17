"""Discrete probabilistic programs.

The `@DPP` decorator runs its function once, at definition time, over symbolic arguments.
Each parameter becomes a typed input `Variable`, each call to a `ProbabilisticCallable`
records one assignment, and the variables returned are the outputs of the definition.
Together with the DPPs it calls, this gives the `pylea.ir.program.Program` that
`DPP.program` returns.  Tracing does no arithmetic and no compilation, and its time grows
with the length of the definition traced, not with what the DPPs it calls contain, so
defining a DPP is cheap.

    >>> from pylea import flip, observe, prob
    >>> @DPP
    ... def burglary_given_alarm() -> Bool:
    ...     burglary = flip(0.001)
    ...     earthquake = flip(0.002)
    ...     observe(burglary | earthquake)
    ...     return burglary
    >>> round(prob(burglary_given_alarm), 6)
    0.333556

Three kinds of thing are callable inside a definition.  Another `DPP` composes by ordinary
function call.  The caller keeps the DPPs it calls, and its program, made the first time
it is asked for, holds the definitions of all of them with equal ones merged, so a helper
called many times is stored once.  A `Primitive` records a single assignment, carrying
either a matrix or a `source` DPP whose circuit is kept because its matrix would be
exponentially large.  A `pylea.semantics.likelihood_table.LikelihoodTable` is the third.

`Bool` is the variable of one bit; every other `pylea.syntax.variable.Variable` is made of
`Bool`s, one per bit.  `DPP.evaluated` gives the semantics as a table and `DPP.compiled`
the DPP compiled for the exact engine.  `DPP.diagram` draws the traced definition,
`DPP.dependency_graph` the variables of its interface rather than its assignments, and
`DPP.simplified` returns the result of simplification as a DPP in its own right.  Printing a
DPP lists its main definition, one line per assignment, and its `repr` gives its signature.

Tracing also records the name each call was made through: that of every DPP, table and
other matrix called, and the value of every constant.  A diagram labels its boxes with these, and the
listing writes them.  They are held here rather than on `pylea.ir.program.Program`
because a name is no part of the semantics and no compilation step carries one.

The variables at the interface are named too: an input from the parameter it stands for,
and an output from the source of the return statement, where what was returned was a
variable rather than an expression.

A variable may be of any width.  A constant or a uniform draw wider than 8 bits is
recorded one bit at a time, since one assignment of ``w`` bits is a tensor of ``2**w``
entries that every contraction through it carries, and a table of any size can be called.
A primitive that would need a matrix of more than ``2**20`` entries and has no form in
narrower primitives is refused when it is recorded.
"""
from __future__ import annotations
from typing import (Any, Callable, Iterable, Literal, NamedTuple, TypeVar, cast, get_args,
                    get_origin, overload)
from abc import ABC, abstractmethod
from collections.abc import Iterable as _IterableABC
import ast
import contextvars
import inspect
import numbers
import sys
import textwrap
from inspect import Parameter, getsource
from pylea.syntax.variable import DataType, Variable, as_variable, is_bool, reads_as
from pylea.syntax.compiled import Compiled, DEFAULT_KAPPA, check_compile_options
from pylea.syntax.encoding import BitEncoded, Encoding
from pylea.ir.body import Body, BodyBuilder, Op
from pylea.ir.prim import Prim as _Prim, Tag as _Tag
from pylea.ir.program import (Program, intern_with_remap as _intern_with_remap,
                              reachable as _reachable)

_T = TypeVar('_T', bound=Variable)

# Above this, a primitive other than a constant or a uniform draw is recorded as several
# narrower ones where it has such a form, and refused where it has none: an if-then-else
# over 7 bits would be a 128 x 32768 matrix.  A `MATRIX` is never refused, its array
# already existing.
_MAX_FACTOR_ENTRIES = 1 << 20

# A constant or a uniform draw wider than this is recorded one bit at a time.  Decided
# empirically: a dense constant of `w` bits is a tensor of ``2**w`` entries that every
# contraction through it carries, and from 11 bits on it made compilation and evaluation
# take seconds, where per bit they stayed below a second up to 16 bits.  At 8 bits or
# fewer the primitive stays one assignment, so an `Enum` constant of up to 256 elements
# is one box in a diagram.
_MAX_CONSTANT_BITS = 8


if sys.version_info >= (3, 14):
    import annotationlib as _annotationlib

    def _signature_and_annotations(function) -> tuple[inspect.Signature, dict]:
        """The signature of `function` and its annotations, without evaluating a name that
        is not defined: such an annotation comes back as a `ForwardRef`."""
        form = _annotationlib.Format.FORWARDREF
        return (inspect.signature(function, annotation_format=form),
                _annotationlib.get_annotations(function, format=form))

    def _annotation_text(annotation) -> str | None:
        """The source text of an annotation left unevaluated, or `None` for a value."""
        if isinstance(annotation, str):
            return annotation
        if isinstance(annotation, _annotationlib.ForwardRef):
            return annotation.__forward_arg__
        return None
else:
    def _signature_and_annotations(function) -> tuple[inspect.Signature, dict]:
        """The signature of `function` and its annotations, those written under
        ``from __future__ import annotations`` left as strings."""
        return inspect.signature(function), inspect.get_annotations(function)

    def _annotation_text(annotation) -> str | None:
        """The source text of an annotation left unevaluated, or `None` for a value."""
        return annotation if isinstance(annotation, str) else None


class _Unresolved:
    """An annotation that names something not visible where it is evaluated."""

    def __init__(self, text: str, error: Exception) -> None:
        self.text = text
        self.error = error


def _resolved(annotation, function, scope) -> object:
    """`annotation` evaluated, or an `_Unresolved` where it names something not visible.

    An annotation left as text is evaluated in the globals of `function` and the local
    variables `scope` of the place the decorator was applied, which is where a type
    defined inside a function, under ``from __future__ import annotations``, is visible.
    """
    text = _annotation_text(annotation)
    if text is None:
        return annotation
    try:
        return eval(text, function.__globals__, scope)
    except Exception as exc:                    # noqa: BLE001, reported to the caller
        return _Unresolved(text, exc)


def _is_variable_type(value) -> bool:
    """Whether `value` is a `Variable` subclass."""
    return isinstance(value, type) and issubclass(value, Variable)


def _declared_outputs(annotation) -> tuple | None:
    """The output types a return annotation declares, or `None` for a form that names no
    variable types, which is not checked.

    ``tuple[T, ...]`` gives ``(T, Ellipsis)``, any number of outputs of type `T`.

        >>> _declared_outputs(Bool), _declared_outputs(None), _declared_outputs(tuple[()])
        ((<class 'pylea.syntax.dpp.Bool'>,), (), ())
        >>> _declared_outputs(tuple[Bool, ...])
        (<class 'pylea.syntax.dpp.Bool'>, Ellipsis)
        >>> _declared_outputs(int) is None, _declared_outputs(tuple) is None
        (True, True)
    """
    if annotation is None or annotation is type(None):
        return ()
    if _is_variable_type(annotation):
        return (annotation,)
    if get_origin(annotation) is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return (args[0], Ellipsis) if _is_variable_type(args[0]) else None
        if all(_is_variable_type(t) for t in args):
            return args
    return None


def _written(annotation) -> str:
    """`annotation` as it would be written in the source, for a message.

        >>> _written(int), _written(tuple[Bool, Bool]), _written(list[Bool])
        ('int', 'tuple[Bool, Bool]', 'list[Bool]')
    """
    origin = get_origin(annotation)
    if origin is not None and get_args(annotation):
        return (f"{_written(origin)}["
                f"{', '.join(_written(a) for a in get_args(annotation))}]")
    if annotation is Ellipsis:
        return "..."
    return getattr(annotation, "__name__", None) or repr(annotation)


def _described(types) -> str:
    """`types` in words, for a message: one type by name, several as a tuple."""
    if len(types) == 2 and types[1] is Ellipsis:
        return f"any number of {types[0].__name__}"
    if not types:
        return "nothing"
    if len(types) == 1:
        return types[0].__name__
    return "(" + ", ".join(t.__name__ for t in types) + ")"


# How a message describes a parameter that cannot be an input.
_KINDS = {Parameter.VAR_POSITIONAL: "variable-length",
          Parameter.KEYWORD_ONLY: "keyword-only",
          Parameter.VAR_KEYWORD: "a collection of keyword arguments"}
_STARS = {Parameter.VAR_POSITIONAL: "*", Parameter.VAR_KEYWORD: "**"}


def _shown(f: ProbabilisticCallable) -> str:
    """What a message calls `f`: what it is written as, where that differs from its name."""
    return getattr(f, "label", None) or f.name()


def _returned_names(function, count: int) -> tuple[str | None, ...]:
    """The identifier each of `count` returned values was written as, where it was one.

    Read from the source of `function`.  A returned expression has no name to show, and a
    function with more than one return statement has no single statement to read, since
    which of them ran is not a property of the source; either way the position is left
    unnamed.  Nothing is named where the source cannot be reached, as for a function
    defined in a shell.

        >>> def one(a):
        ...     b = a
        ...     return b
        >>> def two(a):
        ...     return a & a
        >>> _returned_names(one, 1), _returned_names(two, 1)
        (('b',), (None,))
    """
    blank: tuple[str | None, ...] = (None,) * count
    try:
        tree = ast.parse(textwrap.dedent(getsource(function)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return blank
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return blank

    returns = [node for node in _own_returns(tree.body[0]) if node.value is not None]
    if len(returns) != 1:
        return blank
    value = returns[0].value
    parts = value.elts if isinstance(value, ast.Tuple) else [value]
    if len(parts) != count:
        return blank
    return tuple(p.id if isinstance(p, ast.Name) else None for p in parts)


def _own_returns(function_node) -> list:
    """The return statements of `function_node` itself, not those of a function inside it."""
    found = []

    def visit(node) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                  ast.ClassDef)):
                continue
            if isinstance(child, ast.Return):
                found.append(child)
            visit(child)

    visit(function_node)
    return found


class _Assembly(NamedTuple):
    """A DPP's program and what a diagram labels it with, each table keyed by body index.

    `body_names` names a called body, `op_names` the ops of a body by position,
    `op_groups` the ops one call recorded where it recorded several, and `wire_groups` the
    wires of each variable of more than one bit.
    """

    program: Program
    body_names: dict[int, str]
    op_names: dict[int, dict[int, str]]
    op_groups: dict[int, tuple[tuple[int, ...], ...]]
    wire_groups: dict[int, tuple[tuple[int, ...], ...]]


def _carried(table: dict, remap: dict[int, int]) -> dict:
    """`table`, keyed by body index, moved onto the indices `remap` gives.

    `remap` runs callees before callers, so where several bodies merged into one the entry
    of the first of them survives.
    """
    moved: dict = {}
    for old, new in remap.items():
        if old in table:
            moved.setdefault(new, table[old])
    return moved


def _overridden(supplied: dict, given: dict) -> dict:
    """`supplied` with every argument of `given` taking precedence, the labels by position
    merged, so that a caller naming one position keeps the labels of the others.

        >>> _overridden({"title": "f", "output_names": {0: "a", 1: "b"}},
        ...             {"title": "g", "output_names": {1: "c"}})
        {'title': 'g', 'output_names': {0: 'a', 1: 'c'}}
    """
    merged = {**supplied, **given}
    for key in ("input_names", "output_names"):
        if key in supplied and key in given:
            merged[key] = {**supplied[key], **given[key]}
    return merged


def _interface_labels(names, variables) -> dict[int, str]:
    """A label for each position of `variables`, taken from the matching name in `names`.

    Keyed by position among the variables, a variable of several bits being one position.
    A position whose name is missing or empty is left out.
    """
    return {i: name for i, (name, _) in enumerate(zip(names, variables)) if name}


def _variable_labels(names, variables) -> dict[int, str]:
    """A label for every position of `variables`: its name in `names`, or else its type.

    Unlike `_interface_labels`, which leaves an unnamed position to be drawn without a
    label, every position gets one here, a node of a dependency graph being nothing but
    its label.  A DPP declaring its input types through the `types` argument has one
    parameter for any number of inputs, so it offers no name for any of them.
    """
    names = tuple(names)[:len(variables)]
    names += (None,) * (len(variables) - len(names))
    return {i: name or type(x).__name__
            for i, (name, x) in enumerate(zip(names, variables))}


class _VariableNames:
    """How `DPP.__str__` writes the variables of one body.

    An input is written by its parameter name where it has one; any other variable as
    ``w`` and its first wire.  A variable of several bits is a group of wires, taken from
    the inputs and then from `groups`, the first group holding a wire keeping it; a run of
    wires equal to a group is written as one name, and a single wire of it as the name
    and its position.
    """

    def __init__(self, groups, inputs, labels: dict[int, str]) -> None:
        self._group_of: dict[int, tuple[int, ...]] = {}
        self._name: dict[tuple[int, ...], str] = {}
        self._bit: dict[int, str] = {}
        for i, x in enumerate(inputs):
            self._add(tuple(b.wire for b in x.bits()), labels.get(i))
        for g in groups:
            self._add(tuple(g), None)

    def _add(self, wires: tuple[int, ...], name: str | None) -> None:
        if (not wires or len(set(wires)) != len(wires)
                or any(w in self._group_of or w in self._bit for w in wires)):
            return
        name = name or f"w{wires[0]}"
        if len(wires) == 1:
            self._bit[wires[0]] = name
            return
        self._name[wires] = name
        for k, w in enumerate(wires):
            self._group_of[w] = wires
            self._bit[w] = f"{name}[{k}]"

    def written(self, wires) -> list[str]:
        """`wires` as names, a run equal to a group being one name."""
        out: list[str] = []
        wires = tuple(wires)
        i = 0
        while i < len(wires):
            g = self._group_of.get(wires[i])
            if g is not None and wires[i:i + len(g)] == g:
                out.append(self._name[g])
                i += len(g)
                continue
            out.append(self._bit.get(wires[i], f"w{wires[i]}"))
            i += 1
        return out


def _assignment_text(op: Op, named: _VariableNames, op_name: str | None,
                     body_name: str | None, callees: tuple) -> str:
    """The right-hand side of the line `DPP.__str__` writes for `op`.

    `op_name` is the name the call was recorded under, `body_name` that of a called body,
    and `callees` the primitives of every op the line stands for, where it stands for
    several constants or uniform draws recorded one bit at a time.
    """
    args = named.written(op.args)
    bits = [named.written((w,))[0] for w in op.args]      # one name per wire
    callee = op.callee
    if not isinstance(callee, _Prim):
        name = op_name or body_name or f"f{callee}"
        return f"{name.strip('_') or name}({', '.join(args)})"
    tag = callee.tag
    if tag is _Tag.AND:
        return f"{bits[0]} & {bits[1]}"
    if tag is _Tag.OR:
        return f"{bits[0]} | {bits[1]}"
    if tag is _Tag.NOT:
        return f"~{bits[0]}"
    if tag is _Tag.FLIP:
        return f"flip({callee.params[0]!r})"
    if tag is _Tag.OBSERVE:
        return f"observe({bits[0]})"
    if tag is _Tag.CONST:
        if op_name is not None:
            return op_name
        value = 0
        for c in callees:                   # the first is the most significant
            bit_value, bits = c.params
            value = (value << bits) | bit_value
        return str(value)
    if tag is _Tag.UNIFORM:
        return "uniform_bits()"
    if tag is _Tag.ITE:
        m = callee.params[0]
        then_, else_ = (named.written(op.args[1:1 + m]),
                        named.written(op.args[1 + m:]))
        shown = [", ".join(part) if len(part) == 1 else f"({', '.join(part)})"
                 for part in (then_, else_)]
        return f"{shown[0]} if {bits[0]} else {shown[1]}"
    if tag is _Tag.MATRIX:
        name = op_name or "table"
        return f"{name.strip('_') or name}({', '.join(args)})"
    return f"{tag.name.lower()}({', '.join(args)})"


class UnconvertibleCallee(TypeError):
    """Raised for a callee that is neither a `DPP` nor carries `prim` or `source`."""


# The DPP whose definition is being traced in the current thread or task.  A
# `ContextVar` keeps definitions in separate threads, and in separate asyncio tasks, apart,
# and `reset` restores the outer definition exactly for nested ones.  The class-level
# caches of types and primitives may build an equal object twice when filled from several
# threads at once, which is harmless.
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("pylea_active_definition",
                                                         default=None)


def active_definition() -> DPP | None:
    """The DPP whose definition is being traced in the current thread or task, or `None`.

        >>> from pylea import DPP, Bool, flip
        >>> active_definition() is None
        True
        >>> @DPP
        ... def inside() -> Bool:
        ...     assert active_definition() is not None
        ...     return flip(0.5)
    """
    return _ACTIVE.get()


class ProbabilisticCallable[**P, Q: tuple[Variable, ...] | Variable | None](ABC):
    """Something callable inside a `DPP` definition.

    Calling an instance inside an active definition records the call as one assignment
    of the enclosing definition and returns fresh output `Variable`s.  The concrete kinds
    are `DPP`, `Primitive` and `pylea.semantics.likelihood_table.LikelihoodTable`.
    """

    @abstractmethod
    def input_types(self) -> tuple[DataType, ...]:
        """The type of each input, in order."""

    @abstractmethod
    def output_types(self) -> tuple[DataType, ...]:
        """The type of each output, in order."""

    @abstractmethod
    def compiled(self) -> Compiled | Primitive:
        """The form a query or a call uses: a `Compiled` for a `DPP`, a `Primitive`
        otherwise."""

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Whether this callable draws no randomness: no coin flip, no uniform draw and no
        table other than a function of its inputs."""

    @property
    @abstractmethod
    def is_discardable(self) -> bool:
        """Whether this callable conditions on nothing: no `observe`, and every table it
        calls sums to 1 in each column, so that a call whose outputs are never read
        changes nothing."""

    def name(self: ProbabilisticCallable) -> str:
        """What messages and diagrams call this callable."""
        return f"f_{id(self)}"

    def parameter_names(self) -> tuple[str, ...]:
        """The name of each input, in order, where the inputs have names; `()` otherwise.

        A `DPP` traced from annotated parameters has them; a `DPP` declaring its inputs
        through `types`, a `Primitive` and a table do not.
        """
        return ()

    def n_inputs(self) -> int:
        """The number of inputs."""
        return len(self.input_types())

    def n_outputs(self) -> int:
        """The number of outputs."""
        return len(self.output_types())

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Q:
        """The output variables of this call, which is now recorded as an assignment.

        Each argument must be a variable of the declared input type, or a value of that
        type, which becomes a constant.  Keyword arguments bind to the parameter names, as
        in Python:

            >>> from pylea import flip, prob
            >>> @DPP
            ... def first_only(a: Bool, b: Bool) -> Bool:
            ...     return a & ~b
            >>> @DPP
            ... def called() -> Bool:
            ...     return first_only(b=False, a=flip(0.25))
            >>> prob(called)
            0.25

        Raises `TypeError` for a wrong number of arguments, a keyword that names no
        parameter, a keyword given to a callable without parameter names, and a variable
        of another type; `RuntimeError` outside a DPP definition.
        """
        bound = self._bound(args, kwargs) if kwargs else args
        return DPP._call(self, *bound)  # type: ignore

    def _bound(self, args: tuple, kwargs: dict) -> tuple:
        """`args` followed by the keyword arguments `kwargs`, in parameter order."""
        shown = _shown(self)
        names = self.parameter_names()
        if not names:
            if self.n_inputs() == 0:
                raise TypeError(f"{shown} takes no arguments; got keyword argument(s) "
                                f"{', '.join(map(repr, kwargs))}")
            raise TypeError(f"{shown} takes positional arguments only; got keyword "
                            f"argument(s) {', '.join(map(repr, kwargs))}")
        for name in kwargs:
            if name not in names:
                raise TypeError(f"{shown}() got an unexpected keyword argument {name!r}")
            if names.index(name) < len(args):
                raise TypeError(f"{shown}() got multiple values for argument {name!r}")
        missing = [name for name in names[len(args):] if name not in kwargs]
        if missing:
            raise TypeError(f"{shown}() missing required argument(s) "
                            f"{', '.join(map(repr, missing))}")
        return tuple(args) + tuple(kwargs[name] for name in names[len(args):])


class DPP[**P, Q: tuple[Variable, ...] | Variable | None](ProbabilisticCallable[P, Q]):
    """A discrete probabilistic program.

    Normally built by applying the `@DPP` decorator to a Python function whose parameters
    are annotated with `Variable` subclasses, or whose types come from a `types` keyword
    argument.  Running the decorated function records each of its `ProbabilisticCallable`
    calls as one assignment of its definition, giving the `pylea.ir.program.Program` that
    `compiled` compiles for the exact engine.

    A DPP with inputs is a conditional distribution and composes as a function:

        >>> from pylea import flip, prob
        >>> @DPP
        ... def noisy(x: Bool) -> Bool:
        ...     return x & flip(0.9) | ~x & flip(0.1)
        >>> @DPP
        ... def relayed() -> Bool:
        ...     return noisy(noisy(flip(0.5)))
        >>> prob(relayed)
        0.5
    """

    def __init__(
            self,
            defining_function: Callable[..., Any],
            name: str | None = None,
        ):
        """The DPP traced from `defining_function`, called `name`.

        `name` defaults to the function's own, which is what the `@DPP` decorator gives.
        It is what a diagram labels a call to this DPP with, so a DPP built inside a
        helper, where the function's name is an implementation detail, passes one.

        Input types come from the annotation on each parameter, one input per parameter:

            >>> from pylea import flip, prob
            >>> @DPP
            ... def both(a: Bool, b: Bool) -> Bool:
            ...     return a & b
            >>> [t.__name__ for t in both.input_types()]
            ['Bool', 'Bool']

        A DPP over a number of inputs fixed at definition time instead declares their
        types in a `types` keyword argument with a tuple as its default, beside one other
        parameter.  A plain parameter receives the inputs as one tuple, and a
        variable-length parameter receives them one by one:

            >>> @DPP
            ... def all_of(xs, types=(Bool,) * 3) -> Bool:
            ...     return xs[0] & xs[1] & xs[2]
            >>> @DPP
            ... def any_of(*xs, types=(Bool,) * 3) -> Bool:
            ...     return xs[0] | xs[1] | xs[2]
            >>> len(all_of.input_types()), len(any_of.input_types())
            (3, 3)

        An annotation written as text, as under ``from __future__ import annotations``, is
        looked up in the module of `defining_function` and in the local variables of the
        place the decorator is applied, so a type defined in the same function as the DPP
        can be named.

        A return annotation that names variable types is checked against what the
        function returns: one type, ``tuple[T1, T2]``, ``tuple[T, ...]`` or `None`,
        each returned variable being of the type at its position.  Any other return
        annotation, such as `Any`, a bare `tuple` or one that cannot be looked up, is not
        checked.

        Raises `ValueError` where neither form is used or the `types` form is malformed.
        Raises `TypeError` where an annotation cannot be looked up or is not a `Variable`
        subclass, where a parameter is keyword-only or variable-length in the annotated
        form, and where the function returns something other than `Variable`s or
        variables that disagree with its return annotation.

        A definition nested inside another is allowed, as happens inside a factory such as
        `pylea.syntax.enum.Enum.uniform`: the outer definition is suspended for the duration
        and restored afterwards.  The inner DPP may not use variables belonging to the outer
        one, which is checked here for the outputs and in `DPP._call` for every argument.
        """
        self.__name = defining_function.__name__ if name is None else name

        # The local variables of the place the decorator is applied, for annotations left
        # as text.
        caller = sys._getframe(1)
        token = _ACTIVE.set(self)
        try:
            sig, annotations = _signature_and_annotations(defining_function)
            params = sig.parameters
            if "types" in params:
                input_types, unpacked = self._types_form(params)
            else:
                input_types = self._annotated_form(params, annotations, defining_function,
                                                   caller.f_locals)
                unpacked = True
            returned = _resolved(annotations.get("return", Parameter.empty),
                                 defining_function, caller.f_locals)
            # Tracing builds the IR body directly: every `Bool` names a wire and every
            # recorded call is an `Op`.  A call of a DPP names a slot of `callee_index`, one
            # per distinct callee, whose root body is all the builder reads; `_assemble`
            # puts the program together from the callees when it is first asked for.
            self.__root: Body | None = None
            self.__slot_bodies: list[Body] = []
            self.__builder = BodyBuilder(self.__slot_bodies)
            self.__callee_index: dict[DPP, int] = {}
            # What a diagram labels a box with.  Names are not part of the semantics and no
            # pass carries them, so they are held here rather than on `Program`.
            self.__slot_names: dict[int, str] = {}
            self.__own_op_names: dict[int, str] = {}
            # The ops each call recorded, where it recorded several, which a diagram may
            # draw as one box.
            self.__own_op_groups: list[tuple[int, ...]] = []
            # The wires of each variable of more than one bit, which a diagram draws as
            # one dot.  Like the names, no part of the semantics.
            self.__own_groups: list[tuple[int, ...]] = []
            self.__parameter_names: tuple[str, ...] = (
                () if "types" in params else tuple(params))

            self.__input_variables = tuple(
                t.make(*(
                    Bool(self, wire=self.__builder.fresh(1))  # type: ignore[arg-type]
                    for _ in range(t.nbits())
                ))
                for t in input_types
            )
            self.__input_wires = tuple(b.wire for v in self.__input_variables
                                       for b in v.bits())
            for v in self.__input_variables:
                self._group(v)

            # The tracer passes `Variable`s whatever the declared parameter types are, so
            # the call deliberately escapes the signature.
            trace = cast(Callable[..., Any], defining_function)
            if unpacked:
                output_variables = trace(*self.__input_variables)
            else:
                output_variables = trace(self.__input_variables)

            if isinstance(output_variables, tuple):
                for i, x in enumerate(output_variables):
                    if not isinstance(x, Variable):
                        raise TypeError(
                            f"{self.__name} returned a {type(x).__name__} at position "
                            f"{i + 1}; a DPP returns variables, one variable or None.")
                self.__output_variables = output_variables
            elif isinstance(output_variables, Variable):
                self.__output_variables = (output_variables,)
            elif output_variables is None:
                self.__output_variables = tuple()
            else:
                raise TypeError(
                    f"{self.__name} returned a {type(output_variables).__name__}; a DPP "
                    "returns a tuple of variables, one variable or None.")

            for i, x in enumerate(self.__output_variables):
                if x.to_be_used_within() is not self:
                    raise ValueError(
                        f"Output variable at position {i} belongs to DPP "
                        f"'{x.to_be_used_within().name()}', not '{self.__name}'. "
                        "Returning variables from an enclosing DPP is not allowed."
                    )
            self._check_returned(returned)

            for v in self.__output_variables:
                self._group(v)
            output_wires = tuple(b.wire for v in self.__output_variables
                                 for b in v.bits())
            self.__root = self.__builder.finish(self.__input_wires, output_wires)
            # The callee of each slot, in the order of their first calls.
            self.__callees: tuple[DPP, ...] = tuple(self.__callee_index)
            self.__op_groups = tuple(self.__own_op_groups)
            self.__wire_groups = tuple(dict.fromkeys(self.__own_groups))
            del (self.__builder, self.__slot_bodies, self.__callee_index,
                 self.__own_op_groups, self.__own_groups)
            self.__traced = True
            self.__assembly: _Assembly | None = None

            # Read from the source only if a diagram asks for it, since most DPPs are
            # never drawn.
            self.__defining_function = defining_function
            self.__output_names: tuple[str | None, ...] | None = None
            self.__compiled: dict = {}
            self.__simplified: DPP | None = None
            self.__is_discardable: bool | None = None

        finally:
            _ACTIVE.reset(token)                    # even if tracing raised
            del caller

    def _types_form(self, params) -> tuple[tuple[DataType, ...], bool]:
        """The input types a `types` keyword argument declares, and whether the other
        parameter receives the inputs one by one rather than as a tuple."""
        types_param = params["types"]
        if types_param.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            raise ValueError(f"{self.__name}: 'types' must be a keyword argument whose "
                             "default is the tuple of input types")
        types = types_param.default
        if types is Parameter.empty:
            raise ValueError(f"{self.__name}: no default value provided for argument "
                             "'types', which must be the tuple of input types")
        if not isinstance(types, tuple):
            raise ValueError(f"{self.__name}: the default value of 'types' must be a "
                             f"tuple of input types; got {types!r}")
        others = [p for p in params.values() if p.name != "types"]
        if len(others) != 1 or others[0].kind not in (Parameter.POSITIONAL_OR_KEYWORD,
                                                      Parameter.VAR_POSITIONAL):
            raise ValueError(
                f"{self.__name}: with 'types', the function takes one other parameter, "
                "either plain, receiving the inputs as a tuple, or variable-length, "
                "receiving them one by one; got "
                f"{', '.join(repr(str(p)) for p in others) or 'none'}")
        for i, t in enumerate(types):
            if not _is_variable_type(t):
                raise TypeError(f"{self.__name}: input type {i + 1} in 'types' is "
                                f"{_written(t)}, which is not a subclass of Variable")
        return types, others[0].kind is Parameter.VAR_POSITIONAL

    def _annotated_form(self, params, annotations: dict, function,
                        scope) -> tuple[DataType, ...]:
        """The input types the annotations of `params` declare, one per parameter."""
        types = []
        for name, param in params.items():
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.KEYWORD_ONLY,
                              Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"{self.__name}: parameter "
                    f"{_STARS.get(param.kind, '') + name!r} is "
                    f"{_KINDS[param.kind]}, but each input of a DPP is one plain "
                    "parameter; a number of inputs fixed at definition time is declared "
                    "with a 'types' keyword argument")
            if name not in annotations:
                raise ValueError(
                    f"{self.__name}: parameter {name!r} has no type annotation.  "
                    "Annotate every parameter with a variable type, such as 'x: Bool', or "
                    "declare the input types with a 'types' keyword argument.")
            t = _resolved(annotations[name], function, scope)
            if isinstance(t, _Unresolved):
                raise TypeError(
                    f"{self.__name}: the annotation {t.text!r} of parameter {name!r} "
                    f"cannot be looked up ({t.error}).  An annotation written as text is "
                    "looked up in the module and where the decorator is applied; define "
                    "the type there, or declare the input types with a 'types' keyword "
                    "argument.")
            if not _is_variable_type(t):
                hint = ("; a parameter stands for one variable, and a number of inputs "
                        "fixed at definition time is declared with a 'types' keyword "
                        "argument" if get_origin(t) is not None else "")
                raise TypeError(f"{self.__name}: parameter {name!r} is annotated "
                                f"{_written(t)}, which is not a subclass of "
                                f"Variable{hint}")
            types.append(t)
        return tuple(types)

    def _check_returned(self, annotation) -> None:
        """Nothing, where the outputs agree with the return annotation `annotation` or it
        names no variable types; `TypeError` otherwise."""
        declared = _declared_outputs(annotation)
        if declared is None:
            return
        found = tuple(type(x) for x in self.__output_variables)
        if len(declared) == 2 and declared[1] is Ellipsis:
            agrees = all(reads_as(t, declared[0]) for t in found)
        else:
            agrees = (len(found) == len(declared)
                      and all(reads_as(t, d) for t, d in zip(found, declared)))
        if not agrees:
            raise TypeError(f"{self.__name} returns {_described(found)}, but its return "
                            f"annotation says {_described(declared)}")

    @classmethod
    def _from_program(cls, program: Program, input_types: tuple[DataType, ...],
                      output_types: tuple[DataType, ...], name: str,
                      parameter_names: tuple[str, ...] = (),
                      defining_function=None) -> DPP:
        """A `DPP` standing for `program`, with the given interface and name.

        For a `Program` that was not traced from a function, which is what `simplified`
        returns.  The root body must carry one wire per bit the types declare, and
        `ValueError` says so where it does not.  The outputs are named from the source of
        `defining_function`, as those of a traced DPP are; without one, none is named.

        The variables are built before `__output_variables` is set, since `Bool` refuses a
        definition that is already complete.  None of the state tracing needs is set up: a
        DPP built this way is already finished, so it can be called and compiled but not
        traced into.  Its program is `program` as given, and a DPP calling it takes in all
        of `program`.
        """
        body = program.main
        for label, types, wires in (("input", input_types, body.inputs),
                                    ("output", output_types, body.outputs)):
            declared = sum(t.nbits() for t in types)
            if declared != len(wires):
                raise ValueError(f"{name}: the types declare {declared} {label} bits, but "
                                 f"the main definition of the program has {len(wires)}")

        self = cls.__new__(cls)
        self.__name = name
        self.__parameter_names = parameter_names

        def variables(types, wires):
            at, made = 0, []
            for t in types:
                made.append(t.make(*(Bool(self, wire=w)
                                     for w in wires[at:at + t.nbits()])))
                at += t.nbits()
            return tuple(made)

        self.__input_variables = variables(input_types, body.inputs)
        self.__input_wires = tuple(body.inputs)
        self.__output_variables = variables(output_types, body.outputs)

        self.__root = body
        self.__callees = ()
        self.__traced = False
        self.__assembly = _Assembly(program, {}, {}, {}, {program.root: tuple(
            tuple(b.wire for b in x.bits())
            for x in self.__input_variables + self.__output_variables
            if len(x.bits()) > 1)})
        self.__defining_function = defining_function
        self.__output_names = None
        self.__compiled = {}
        self.__simplified = None
        self.__is_discardable = None
        return self

    def name(self: DPP) -> str:
        """The name of this DPP: that of the decorated function, unless another was given.

        Diagrams, listings and messages call the DPP by it.

            >>> from pylea import flip
            >>> @DPP
            ... def test(has_it: Bool) -> tuple[Bool, Bool]:
            ...     return has_it & flip(0.99), ~has_it & flip(0.05)
            >>> test.name(), test.parameter_names()
            ('test', ('has_it',))
            >>> [t.__name__ for t in test.input_types()], [t.__name__ for t in test.output_types()]
            (['Bool'], ['Bool', 'Bool'])
            >>> len(test.input_variables()), len(test.output_variables())
            (1, 2)
        """
        return self.__name

    def parameter_names(self: DPP) -> tuple[str, ...]:
        """The name of each input, in order; `()` for a DPP declaring its inputs through
        the `types` argument."""
        return self.__parameter_names

    def input_variables(self: DPP) -> tuple[Variable, ...]:
        """The variables the definition was traced with, one per input.

        They belong to this DPP's definition, so they can be used in no other one.
        """
        return self.__input_variables

    def output_variables(self: DPP) -> tuple[Variable, ...]:
        """The variables the definition returned, one per output, in order.

        They belong to this DPP's definition, so they can be used in no other one.
        """
        return self.__output_variables

    def program(self: DPP) -> Program:
        """The `pylea.ir.program.Program` traced from this DPP, whose main function is
        this DPP's definition.

        It holds this DPP's definition and the definition of each DPP it calls, directly or
        through others, structurally equal ones merged, so a helper called twice is stored
        once:

            >>> from pylea import flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.5)
            >>> @DPP
            ... def both() -> Bool:
            ...     return coin() & coin()
            >>> len(both.program().bodies)
            2

        The program is made the first time it is asked for, and kept.  That takes time in
        proportion to its size, which counts each distinct definition reached once.  In a
        chain of DPPs each calling the one before, every DPP's program holds all those
        below it, so asking each of ``n`` of them for its program takes time growing as
        ``n**2``, where defining them takes time growing as ``n``.

        Definitions are numbered callees first, in the order of their calls.  Where two
        calls lead to equal definitions, the one kept is labelled in a diagram with the name
        of the call reached first in that order.
        """
        return self._assembled().program

    def _assembled(self: DPP) -> _Assembly:
        """This DPP's program and labels, made by `_assemble` on first use.

        Two threads asking at once may both make it; the results are equal.
        """
        if self.__assembly is None:
            self.__assembly = self._assemble()
        return self.__assembly

    def _assemble(self: DPP) -> _Assembly:
        """The program and labels of this DPP, put together from its callees.

        Visits each distinct DPP reachable from this one once, without recursion, callees
        before callers in the order of their first calls, and appends its root body with
        each call redirected from the caller's slot to the callee's body.  A DPP made by
        `_from_program` contributes its whole program.  Interning the table then merges
        equal bodies and numbers them as `pylea.ir.program.reachable` lists them.

        The name of a body is that of the call through which the visit first reached its
        DPP; where several bodies merge, the first of them that has a name keeps it, which
        is `_carried`'s rule for every table.
        """
        bodies: list[Body] = []
        index: dict[DPP, int] = {}          # the position of each DPP's root body
        named: dict[DPP, str] = {}
        body_names: dict[int, str] = {}
        op_names: dict[int, dict[int, str]] = {}
        op_groups: dict[int, tuple[tuple[int, ...], ...]] = {}
        wire_groups: dict[int, tuple[tuple[int, ...], ...]] = {}

        stack = [(self, iter(enumerate(self.__callees)))]
        while stack:
            d, pending = stack[-1]
            step = next(pending, None)
            if step is None:
                stack.pop()
                at = index[d] = len(bodies)
                slots, root = d.__callees, d.__root
                assert root is not None
                bodies.append(Body(root.bits, root.inputs, root.outputs, tuple(
                    op if isinstance(op.callee, _Prim)
                    else Op(index[slots[op.callee]], op.args, op.outs)
                    for op in root.ops)))
                op_names[at] = d.__own_op_names
                op_groups[at] = d.__op_groups
                wire_groups[at] = d.__wire_groups
                continue
            slot, g = step
            named.setdefault(g, d.__slot_names[slot])
            if g in index:
                continue
            if g.__traced:
                stack.append((g, iter(enumerate(g.__callees))))
                continue
            given = g._assembled()
            offset = len(bodies)
            bodies.extend(Body(b.bits, b.inputs, b.outputs, tuple(
                op if isinstance(op.callee, _Prim)
                else Op(op.callee + offset, op.args, op.outs)
                for op in b.ops)) for b in given.program.bodies)
            index[g] = offset + given.program.root
            body_names.update((offset + i, e) for i, e in given.body_names.items())
            op_names.update((offset + i, e) for i, e in given.op_names.items())
            op_groups.update((offset + i, e) for i, e in given.op_groups.items())
            wire_groups.update((offset + i, e) for i, e in given.wire_groups.items())

        body_names.update((index[g], name) for g, name in named.items())
        program, remap = _intern_with_remap(Program(tuple(bodies), index[self]))
        return _Assembly(program, _carried(body_names, remap), _carried(op_names, remap),
                         _carried(op_groups, remap), _carried(wire_groups, remap))

    # -- tracing into the IR ----------------------------------------------------

    def _resolve(self: DPP, f: ProbabilisticCallable) -> _Prim | int:
        """`f` as a primitive, or as a slot of this DPP's table of callees.

        The name of `f` is recorded against the slot it resolves to, so that a diagram can
        label the call.  It is the name of the callable at the call site rather than the
        one the body was traced under, which is what makes a retyped comparison show as
        `_EQ_Colour_` rather than as the `_eq` circuit beneath it.

        Raises `UnconvertibleCallee` for a callee with neither form.
        """
        if isinstance(f, DPP):
            return self._named(self._slot(f), f.name())
        prim = getattr(f, "prim", None)
        if prim is not None:
            return prim
        source = getattr(f, "source", None)
        if source is not None:
            return self._named(self._slot(source), f.name())
        raise UnconvertibleCallee(
            f"{type(f).__name__} {f.name()!r} has no IR form: it is neither a DPP nor a "
            f"primitive.  Give its construction site a `prim=` (if its matrix is small) "
            f"or a `source=` (if it is a merged view of a DPP)."
        )

    def _named(self: DPP, slot: int, name: str) -> int:
        """`slot`, with `name` recorded for it unless it is named already."""
        self.__slot_names.setdefault(slot, name)
        return slot

    def _group(self: DPP, x: Variable) -> None:
        """The wires of `x` recorded as one variable of this body, where it has several.

        What `DPP.diagram` draws as a single dot; the semantics is the same without it.
        """
        if len(x.bits()) > 1:
            self.__own_groups.append(tuple(b.wire for b in x.bits()))

    def _slot(self: DPP, g: DPP) -> int:
        """The slot of `g` in this DPP's table of callees, added on its first call.

        The slot holds `g`'s root body, whose interface is all the builder reads; its
        callees are reached through `g` when the program is assembled.  Raises
        `ValueError` where `g` is a DPP whose definition is still being traced, which is a
        DPP calling itself, directly or from a DPP defined inside it.
        """
        # Keyed on the callee object, never on `id(g)`.  A DPP built inside a helper becomes
        # unreachable as soon as the call returns and its address may then be reused, at
        # which point an id-keyed entry would answer for an unrelated callee.  A dict keyed
        # by the object holds a reference, so the address cannot be recycled.
        slot = self.__callee_index.get(g)
        if slot is not None:
            return slot
        root = g.__root
        if root is None:
            raise ValueError(f"{g.name()!r} is called inside its own definition; a DPP "
                             "cannot call itself, directly or from a DPP defined inside it")
        slot = self.__callee_index[g] = len(self.__slot_bodies)
        self.__slot_bodies.append(root)
        return slot

    def _emit(self: DPP, callee: _Prim | int, args: tuple[int, ...]) -> tuple[int, ...]:
        """The result wires of one call, which is now recorded.

        A constant or a uniform draw of more than `_MAX_CONSTANT_BITS` bits is recorded as
        one of each bit, most significant first, which keeps a type of any width able to
        hold a constant.  An if-then-else whose matrix would be larger than
        `_MAX_FACTOR_ENTRIES` is recorded as one per bit, sharing the condition.  A
        `MATRIX` is recorded as it is, whatever its size.  Any other primitive above that
        bound raises `UnconvertibleCallee`, having no form in narrower primitives.
        """
        if isinstance(callee, _Prim):
            if callee.tag is _Tag.CONST and callee.out_bits > _MAX_CONSTANT_BITS:
                value, n = callee.params
                return tuple(self.__builder.emit(_Prim.const((value >> (n - 1 - i)) & 1, 1),
                                                 ())[0]
                             for i in range(n))
            if callee.tag is _Tag.UNIFORM and callee.out_bits > _MAX_CONSTANT_BITS:
                one = _Prim.uniform(1)
                return tuple(self.__builder.emit(one, ())[0] for _ in range(callee.out_bits))
            too_wide = _MAX_FACTOR_ENTRIES.bit_length() - 1
            if callee.tag is not _Tag.MATRIX and callee.in_bits + callee.out_bits > too_wide:
                if callee.tag is _Tag.ITE:
                    # An m-bit if-then-else is m one-bit ones sharing the condition, far
                    # smaller than its ``2**m x 2**(2m + 1)`` matrix.
                    m = callee.params[0]
                    cond, then_, else_ = args[0], args[1:1 + m], args[1 + m:]
                    one = _Prim.ite(1)
                    return tuple(self.__builder.emit(one, (cond, t, e))[0]
                                 for t, e in zip(then_, else_))
                n = callee.in_bits + callee.out_bits
                raise UnconvertibleCallee(
                    f"the {callee.tag.name} primitive over {n} bits would need a matrix of "
                    f"2**{n} entries, more than 2**{too_wide}, and has no form in narrower "
                    "primitives")
        return self.__builder.emit(callee, args)

    def input_types(self) -> tuple[DataType, ...]:
        """The type of each input, in order."""
        return tuple(type(x) for x in self.input_variables())

    def output_types(self) -> tuple[DataType, ...]:
        """The type of each output, in order."""
        return tuple(type(x) for x in self.output_variables())

    @property
    def is_deterministic(self) -> bool:
        """Whether this DPP draws no randomness: it calls no `flip`, draws nothing
        uniformly and calls no table other than a function of its inputs.

            >>> from pylea import flip
            >>> @DPP
            ... def gate(a: Bool, b: Bool) -> Bool:
            ...     return a & b
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.5)
            >>> gate.is_deterministic, coin.is_deterministic
            (True, False)
        """
        program = self.program()
        return all(op.callee.is_deterministic
                   for i in _reachable(program)
                   for op in program.bodies[i].ops
                   if isinstance(op.callee, _Prim))

    @property
    def is_discardable(self) -> bool:
        """Whether this DPP conditions on nothing: it calls no `observe`, and every table
        it calls sums to 1 in each column.  A call of such a DPP whose outputs are never
        read changes nothing.  A comparison such as ``x == y`` conditions on nothing.

            >>> from pylea import flip, observe
            >>> @DPP
            ... def conditioned() -> Bool:
            ...     x = flip(0.5)
            ...     observe(x)
            ...     return x
            >>> @DPP
            ... def compared() -> Bool:
            ...     return flip(0.5) == flip(0.5)
            >>> conditioned.is_discardable, compared.is_discardable
            (False, True)
        """
        if self.__is_discardable is None:
            program = self.program()
            self.__is_discardable = all(
                op.callee.is_discardable
                for i in _reachable(program)
                for op in program.bodies[i].ops
                if isinstance(op.callee, _Prim))
        return self.__is_discardable

    def __repr__(self: DPP) -> str:
        """The name of this DPP and its signature, as the definition would be written.

            >>> from pylea import flip
            >>> @DPP
            ... def test(has_it: Bool) -> tuple[Bool, Bool]:
            ...     return has_it & flip(0.99), ~has_it & flip(0.05)
            >>> test
            <DPP test(has_it: Bool) -> tuple[Bool, Bool]>

        An input declared through the `types` argument has no parameter name, and is shown
        by its type alone.
        """
        return f"<DPP {self.__name}{self._signature()}>"

    def _signature(self: DPP) -> str:
        """``(parameters) -> outputs``, each parameter with its type."""
        labels = self._input_labels()
        parameters = ", ".join(f"{labels[i]}: {t.__name__}" if i in labels else t.__name__
                               for i, t in enumerate(self.input_types()))
        outputs = [t.__name__ for t in self.output_types()]
        returned = ("None" if not outputs else outputs[0] if len(outputs) == 1
                    else f"tuple[{', '.join(outputs)}]")
        return f"({parameters}) -> {returned}"

    def __str__(self: DPP) -> str:
        """The main definition of this DPP as a listing, one line per assignment.

        An input is written by its parameter name and any other variable as ``w`` followed
        by a number.  A call of another DPP or of a table is written with the name it was
        called by, and a constant by its value.  A variable of several bits is one name,
        and a single bit of it is written with its position, as in ``w4[0]``.

            >>> from pylea import flip, observe
            >>> @DPP
            ... def test(has_it: Bool) -> Bool:
            ...     return has_it & flip(0.99)
            >>> @DPP
            ... def screened() -> Bool:
            ...     ill = flip(0.01)
            ...     observe(test(ill))
            ...     return ill
            >>> print(test)
            @DPP
            def test(has_it: Bool) -> Bool:
                w1 = flip(0.99)
                w2 = has_it & w1
                return w2
            >>> print(screened)
            @DPP
            def screened() -> Bool:
                w0 = flip(0.01)
                w1 = test(w0)
                observe(w1)
                return w0

        The listing is of the definition as it was traced; `simplified` gives the DPP
        that simplification makes of it, which prints the same way.
        """
        assembly = self._assembled()
        program = assembly.program
        body = program.main
        op_names = assembly.op_names.get(program.root, {})
        named = _VariableNames(assembly.wire_groups.get(program.root, ()),
                               self.__input_variables, self._input_labels())

        # Constants and uniform draws recorded one bit at a time are one line.
        joined: dict[int, tuple[int, ...]] = {}
        for ops in assembly.op_groups.get(program.root, ()):
            tags = {c.tag if isinstance(c, _Prim) and c.in_bits == 0 else None
                    for c in (body.ops[j].callee for j in ops)}
            if (len(ops) > 1 and len(tags) == 1 and tags <= {_Tag.CONST, _Tag.UNIFORM}
                    and not any(j in joined for j in ops)):
                for j in ops:
                    joined[j] = ops

        lines = ["@DPP", f"def {self.__name}{self._signature()}:"]
        for j, op in enumerate(body.ops):
            ops = joined.get(j, (j,))
            if j != ops[0]:
                continue
            outs = named.written(tuple(w for k in ops for w in body.ops[k].outs))
            rhs = _assignment_text(op, named, op_names.get(j),
                                   assembly.body_names.get(op.callee)
                                   if not isinstance(op.callee, _Prim) else None,
                                   tuple(body.ops[k].callee for k in ops))
            lines.append(f"    {', '.join(outs)} = {rhs}" if outs else f"    {rhs}")
        if body.outputs:
            lines.append(f"    return {', '.join(named.written(body.outputs))}")
        elif not body.ops:
            lines.append("    pass")
        return "\n".join(lines)

    @staticmethod
    def _call(
            f: ProbabilisticCallable,
            *args: Variable,
        ) -> tuple[Variable, ...] | Variable | None:
        """The output variables of a call to `f`, which is now recorded as an `Op`.

        A single `Variable` where `f` has one output, a tuple where it has several, and
        `None` where it has none.  Each argument goes through
        `pylea.syntax.variable.as_variable`: a variable of the declared type is used as it
        is, a value of that type becomes a constant, a variable of another type raises
        `TypeError` and a value the type cannot hold `ValueError`, each naming `f` and
        the position.  Raises `RuntimeError` outside a DPP definition,
        `TypeError` if the argument count is wrong, and `ValueError` if an argument
        belongs to a different definition or `f` is a DPP still being defined, which a
        DPP calling itself would be.
        """
        shown = _shown(f)
        active = _ACTIVE.get()
        if active is None:
            raise RuntimeError(
                f"{shown} was called outside a DPP definition: flip, observe, operations "
                "on variables and calls of DPPs work only inside a function decorated "
                "with @DPP.")
        declared = f.input_types()
        if len(args) != len(declared):
            raise TypeError(f"{shown} takes {len(declared)} "
                            f"argument{'' if len(declared) == 1 else 's'}; got {len(args)}")
        names = f.parameter_names()
        args = tuple(
            as_variable(arg, t, f"{shown}: argument {i + 1}"
                        + (f" ({names[i]})" if i < len(names) else ""))
            for i, (arg, t) in enumerate(zip(args, declared)))
        for i, arg in enumerate(args):
            owner = arg.to_be_used_within()
            if owner is not active:
                raise ValueError(
                    f"{shown}: argument {i + 1} is a variable of the DPP {owner.name()!r}, "
                    f"used in the definition of {active.name()!r}; a variable can be used "
                    "only in the definition that made it.")

        callee = active._resolve(f)
        first = len(active.__builder.ops)
        wires = active._emit(callee, tuple(b.wire for x in args for b in x.bits()))
        if not isinstance(callee, _Prim) or callee.tag in (_Tag.MATRIX, _Tag.CONST):
            # A matrix is opaque, and a constant holds the bit pattern rather than the value
            # of the type it belongs to.  A called body may be shared by callables of other
            # names, since equal bodies merge, so the name at the call site is the one to
            # show.  One call can emit several ops, which is why the range is taken.
            written = getattr(f, "label", None) or f.name()
            for op in range(first, len(active.__builder.ops)):
                active.__own_op_names[op] = written
        if len(active.__builder.ops) - first > 1:
            active.__own_op_groups.append(tuple(range(first, len(active.__builder.ops))))
        outputs: tuple[Variable, ...] = ()
        at = 0
        for t in f.output_types():
            outputs += (t.make(*(Bool(active, wire=w)
                                 for w in wires[at:at + t.nbits()])),)
            at += t.nbits()
        if at != len(wires):
            raise ValueError(
                f"{f.name()} declares {at} output bits but produced {len(wires)}")
        for x in outputs:
            active._group(x)

        if len(outputs) > 1:
            return outputs
        elif len(outputs) == 1:
            return outputs[0]

    def compiled(self: DPP, order_finder=None, kappa: int | None = DEFAULT_KAPPA,
                 recompile: bool = False, simplify: bool | Literal["always"] = True,
                 device: str | None = None) -> Compiled:
        """This DPP compiled for the exact engine: a tensor network per definition, with a
        contraction tree for each.

        `order_finder` chooses the contraction tree; by default compilation tries several
        contraction tree finders, each where its own cost is likely to pay
        (`pylea.compile.planner`).  `kappa` is the size limit slicing aims for,
        ``2**kappa`` entries per intermediate tensor; compiling raises
        `pylea.compile.lower.ExactInferenceTooLarge` above ``max(2**29, 2**kappa)``
        entries.  `simplify` applies the rewrite rules
        to the DPP first, wherever that is likely to pay against one evaluation;
        ``"always"`` applies them regardless, which is what a caller who will evaluate
        many times wants.  `device` names where to evaluate, such as ``"numpy"`` or
        ``"mps"``; left unset, the size of the contraction decides.

        An option of the wrong type raises `TypeError` and one out of range `ValueError`,
        such as a negative `kappa` or a finder given by its name rather than as an object.

        The result is cached per ``(order_finder, kappa, simplify, device)``, and
        `recompile` discards that cache, so repeated queries of one DPP compile once:

            >>> from pylea import flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.25)
            >>> coin.compiled() is coin.compiled()
            True
            >>> coin.compiled() is coin.compiled(recompile=True)
            False
        """
        kappa, recompile, simplify = check_compile_options(order_finder, kappa, recompile,
                                                           simplify, device)
        # Keyed on the finder object rather than on `id(...)`, for the reason given in
        # `_slot`.
        key = (order_finder, kappa, simplify, device)
        if recompile:
            self.__compiled.clear()
        hit = self.__compiled.get(key)
        if hit is None:
            hit = self.__compiled[key] = Compiled(self, order_finder, kappa, simplify,
                                                  device)
        return hit

    def evaluated(self: DPP, *, rel_err: float | None = None,
                  floor: float | None = None, confidence: float | None = None,
                  max_time: float | None = None, method: str | None = None,
                  seed: int | None = None, strict: bool = False, **kwargs):
        """The semantics of this DPP, as a typed
        `pylea.semantics.likelihood_table.LikelihoodTable`.

        The table holds unnormalised likelihoods, and its `probabilities` view is
        P(output | input).  A table may itself be called inside another ``@DPP``
        definition, so a DPP evaluated once can be reused as one assignment of its matrix.
        Keyword arguments go to `compiled`, whose result is cached, so a second call does
        not compile again; it does evaluate again.

            >>> from pylea import flip
            >>> @DPP
            ... def biased() -> Bool:
            ...     return flip(0.25)
            >>> print(biased.evaluated())
               | False | True
            -----------------
            () |  0.75 | 0.25

        `method`, `rel_err`, `confidence`, `max_time`, `seed` and `strict` behave as on
        `pylea.syntax.prob.prob`.  A sampled table is an
        `pylea.sample.estimate.EstimatedLikelihoodTable`, whose matrix is the point
        estimate and whose entrywise bounds hold jointly at `confidence`: each entry is
        estimated to relative error `rel_err` or certified below `floor` on the
        normalised scale, the default floor being ``min(rel_err, 1 / (4 n))`` for `n`
        entries.  Certifying every entry separately prices a wide table by its entry
        count, so sampling requires at most 20 output bits and a DPP with no inputs;
        with inputs, each of the ``2**in_bits`` columns would cost a full run of its
        own, and ``method="smc"`` raises.  ``method="auto"`` runs the exact engine here;
        `pylea.syntax.pmf.probability_mass_function` chooses between the engines for a
        distribution.
        """
        from pylea.syntax.method import resolve

        contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                           max_time=max_time, seed=seed, strict=strict, what="evaluated")
        if contract.method == "smc":
            if self.input_types() != tuple():
                columns = 1 << sum(t.nbits() for t in self.input_types())
                raise ValueError(
                    f"a sampled table needs a zero-input DPP: this one has {columns} "
                    "input columns, and each would cost a full certified run; evaluate "
                    "exactly, or wrap the input you care about into a zero-input DPP")
            from pylea.sample.estimate import EstimatedLikelihoodTable
            from pylea.sample.query import certified_table
            from pylea.syntax.method import report_missed_target

            outcome = certified_table(
                self, rel_err=contract.rel_err, confidence=contract.confidence,
                floor=contract.floor, max_time=contract.max_time, seed=contract.seed,
                normalised_contract=False, what="evaluated")
            table = EstimatedLikelihoodTable.from_outcome(
                self.input_types(), self.output_types(), outcome,
                confidence=contract.confidence)
            report_missed_target(table, contract, "evaluated", outcome.target_met,
                                 "an interval for each entry")
            return table
        if contract.max_time is not None:
            from pylea.sample.exact_lane import exact_table_within

            return exact_table_within(self, contract.max_time, kwargs)
        return self.compiled(**kwargs).evaluated()

    def matrix(self: DPP, **kwargs):
        """The ``(2**out_bits, 2**in_bits)`` matrix of this DPP, and its log magnitude.

        The true value is ``array * exp(log_scale)``.  This is what `evaluated` cannot
        return when the magnitude does not fit a float64, which is the case for a DPP
        conditioned on a conjunction of rare events:

            >>> from pylea import flip, observe
            >>> @DPP
            ... def all_twenty_hold() -> Bool:
            ...     coins = [flip(1e-30) for _ in range(20)]
            ...     for coin in coins:
            ...         observe(coin)
            ...     return coins[0]
            >>> array, log_scale = all_twenty_hold.matrix()
            >>> array.tolist(), round(log_scale, 2)
            ([[0.0], [1.0]], -1381.55)

        The acceptance probability here is ``e**-1381.55``, far below the smallest float, so
        folding the scale back in would turn it into zero and `evaluated` refuses:

            >>> all_twenty_hold.evaluated()
            Traceback (most recent call last):
              ...
            ValueError: the matrix's magnitude (e**-1381.6) does not fit float64 to full precision; use matrix(), which returns the array and its log scale apart.
        """
        return self.compiled(**kwargs).matrix()

    def simplified(self: DPP) -> DPP:
        """This DPP with `pylea.simplify` applied to it, as a DPP in its own right.

        The two mean the same thing and have the same interface, so one stands in for the
        other anywhere: in a query, in a later definition, or in a diagram.  Simplification
        preserves the interface of the root because dead code elimination holds every
        output of the root live and no rewrite rule narrows an input, so an input nothing
        reads survives.

            >>> from pylea import flip, prob
            >>> from pylea.ir.program import op_count
            >>> @DPP
            ... def redundant() -> Bool:
            ...     x = flip(0.3)
            ...     return (x & x) | (x & x)
            >>> op_count(redundant.program()), op_count(redundant.simplified().program())
            (4, 1)
            >>> prob(redundant), prob(redundant.simplified())
            (0.3, 0.3)

        This applies the rules until none applies, where `compiled` decides whether they
        are worth their own cost against one evaluation and bounds them by wall clock, so
        the two need not agree on the DPP they compile.

        The result carries this DPP's name and the names of its parameters, and no other
        names: the rules merge, inline and delete definitions, so a name recorded while
        tracing need not describe any definition afterwards.  A diagram of it therefore
        labels a call by its index.
        """
        if self.__simplified is None:
            from pylea.simplify import simplify

            self.__simplified = DPP._from_program(
                simplify(self.program()), self.input_types(), self.output_types(),
                self.__name, self.__parameter_names, self.__defining_function)
        return self.__simplified

    def diagram(self: DPP, **kwargs):
        """The diagram of this DPP's main definition, as a `pylea.draw.Diagram`.

        Every variable is a dot and every assignment is a box, read bottom to top from
        the inputs to the outputs.  A variable of several bits is one dot wherever its
        bits appear only together, and separate dots where an assignment reads or produces
        one of them alone, as the selections of a `match` do.  A call is one box carrying
        the name of the DPP it calls; there is no nesting, so a called DPP is drawn by
        asking it for its own diagram.  Keyword arguments go to `pylea.draw.diagram_of`.

            >>> from pylea import flip, observe
            >>> @DPP
            ... def test(carrier: Bool) -> Bool:
            ...     return carrier & flip(0.99)
            >>> @DPP
            ... def screened() -> Bool:
            ...     ill = flip(0.01)
            ...     observe(test(ill))
            ...     return ill
            >>> print(screened.diagram())
            <Diagram screened: 6 nodes, ... x ...>

        The boxes are labelled with what was called, and the inputs with the parameters
        they stand for:

            >>> "test" in screened.diagram().svg()
            True
            >>> "carrier" in test.diagram().svg()
            True
        """
        from pylea.draw import diagram_of

        assembly = self._assembled()
        root = assembly.program.root
        supplied = dict(body_names=assembly.body_names,
                        op_names=assembly.op_names.get(root, {}),
                        input_names=self._input_labels(),
                        output_names=self._output_labels(),
                        input_widths=tuple(t.nbits() for t in self.input_types()),
                        output_widths=tuple(t.nbits() for t in self.output_types()),
                        wire_groups=assembly.wire_groups.get(root, ()),
                        op_groups=assembly.op_groups.get(root, ()),
                        title=self.__name)
        return diagram_of(assembly.program, **_overridden(supplied, kwargs))

    def dependency_graph(self: DPP, **kwargs):
        """This DPP as a `pylea.draw.Diagram` of the dependencies between its variables.

        One box per variable the DPP takes in or returns, and an arrow from a variable
        to each variable computed from it, the picture reading from the top down.  A
        variable the DPP is given carries a second outline inside the first.  Where the
        outputs are the variables of a network, this is the network:

            >>> from pylea.examples.sprinkler import sprinkler_network
            >>> print(sprinkler_network.dependency_graph())
            <Diagram sprinkler_network: 4 nodes, ... x ...>
            >>> svg = sprinkler_network.dependency_graph().svg()
            >>> all(f">{name}<" in svg
            ...     for name in ("cloudiness", "sprinkler_state", "rain", "grass_wetness"))
            True

        A box carries the name the DPP returned the variable as, or took it in as, and
        the name of its type where the source offers neither.  Keyword arguments go to
        `pylea.draw.dependency_graph_of`.

        What is drawn is what each variable is computed from.  An `observe` computes no
        variable of the interface and so adds no arrow, and two outputs sharing an internal
        variable the DPP does not return are correlated but are drawn apart.  See
        `pylea.draw.dependency`.
        """
        from pylea.draw import dependency_graph_of

        supplied = dict(
            input_names=_variable_labels(self.__parameter_names, self.__input_variables),
            output_names=_variable_labels(self._returned(), self.__output_variables),
            input_widths=tuple(t.nbits() for t in self.input_types()),
            output_widths=tuple(t.nbits() for t in self.output_types()),
            title=self.__name)
        return dependency_graph_of(self.program(), **_overridden(supplied, kwargs))

    def _returned(self: DPP) -> tuple[str | None, ...]:
        """What each returned value was written as, read from the source the first time."""
        if self.__output_names is None:
            self.__output_names = _returned_names(self.__defining_function,
                                                  len(self.__output_variables))
        return self.__output_names

    def _output_labels(self: DPP) -> dict[int, str]:
        """A name for each output position, where the DPP returned a variable rather
        than an expression.

        Keyed by position among the output variables, a value of several bits being one
        position.  An output returning the same variable twice is two positions and so
        two labels.
        """
        return _interface_labels(self._returned(), self.__output_variables)

    def _input_labels(self: DPP) -> dict[int, str]:
        """A name for each input position, taken from the parameter whose value it
        carries.

        Keyed by position among the parameters, a parameter of several bits being one
        position.  A DPP declaring its input types through the `types` argument has no
        parameter names to take, so it labels nothing.
        """
        return _interface_labels(self.__parameter_names, self.__input_variables)

    def distribution(self: DPP, **kwargs):
        """The normalised distribution over this DPP's outputs, as a flat array.

        The DPP must take no inputs.  Index `i` is the outputs read as one bit pattern, the
        first output variable most significant.

            >>> from pylea import flip
            >>> @DPP
            ... def two() -> tuple[Bool, Bool]:
            ...     return flip(0.5), flip(0.25)
            >>> two.distribution().round(4).tolist()
            [0.375, 0.125, 0.375, 0.125]
        """
        return self.compiled(**kwargs).distribution()


class Primitive(ProbabilisticCallable):
    """A callable standing for one primitive, or for a `DPP` with its interface retyped.

        >>> from pylea.ir.prim import Prim as _P
        >>> biased = Primitive(input_types=(), output_types=(Bool,),
        ...                    name="_BIASED_", prim=_P.from_matrix([[0.75], [0.25]]))
        >>> from pylea import prob
        >>> @DPP
        ... def draw() -> Bool:
        ...     return biased()
        >>> prob(draw)
        0.25

    Exactly one of `prim` and `source` is given, and `ValueError` is raised otherwise.
    `prim` is the matrix this callable stands for.  `source` is a DPP whose interface is
    retyped while its circuit is kept, as for the comparators of
    `pylea.syntax.comparison`.  `label` is what a diagram writes in the box, where the name
    is not what a reader wants to see.
    """

    def __init__(
            self,
            input_types: tuple[DataType, ...],
            output_types: tuple[DataType, ...],
            name: str,
            prim: _Prim | None = None,
            source: DPP | None = None,
            label: str | None = None,
        ) -> None:
        if (prim is None) == (source is None):
            raise ValueError(f"{name}: give exactly one of `prim` and `source`")
        if prim is not None:
            declared_in = sum(t.nbits() for t in input_types)
            declared_out = sum(t.nbits() for t in output_types)
            if (declared_in, declared_out) != (prim.in_bits, prim.out_bits):
                raise ValueError(
                    f"{name}: types declare {declared_in} in / {declared_out} out bits, "
                    f"but the primitive has {prim.in_bits} / {prim.out_bits}")
        self.__input_types = input_types
        self.__output_types = output_types
        self.__name = name
        self.__label = label
        self.__prim = prim
        self.__source = source

    @property
    def label(self) -> str | None:
        """What a diagram writes in the box for this callable, where its name will not do.

        A constant is the case: the primitive carries the bit pattern it produces, which for
        a type whose values are not the integers 0, 1, 2 is not the value the caller wrote.
        """
        return self.__label

    @property
    def prim(self) -> _Prim | None:
        """The primitive this callable stands for, or `None` where it has a `source`."""
        return self.__prim

    @property
    def source(self) -> DPP | None:
        """The DPP this callable retypes, or `None` where it has a `prim`."""
        return self.__source

    def input_types(self) -> tuple[DataType, ...]:
        return self.__input_types

    def output_types(self) -> tuple[DataType, ...]:
        return self.__output_types

    def name(self) -> str:
        return self.__name

    @property
    def is_deterministic(self) -> bool:
        if self.__prim is not None:
            return self.__prim.is_deterministic
        assert self.__source is not None
        return self.__source.is_deterministic

    @property
    def is_discardable(self) -> bool:
        if self.__prim is not None:
            return self.__prim.is_discardable
        assert self.__source is not None
        return self.__source.is_discardable

    def compiled(self) -> Primitive:
        return self

    def __str__(self) -> str:
        return f"<Primitive '{self.name()}'>"


class EncodedBool(BitEncoded):
    """The one-bit encoding of Python's built-in `bool`, which `Bool` is built from.

    `True` and `False` are its values, and so are numpy's; the integers 0 and 1 are not,
    being more often a mistake than meant as truth values:

        >>> import numpy as np
        >>> EncodedBool(np.True_).to_bits()
        (True,)
        >>> EncodedBool(1)
        Traceback (most recent call last):
          ...
        ValueError: a Bool holds True or False; got 1 (int)
    """

    def __init__(self, x: bool) -> None:
        if not is_bool(x):
            raise ValueError(f"a Bool holds True or False; got {x!r} ({type(x).__name__})")
        self._value = bool(x)

    @property
    def value(self) -> bool:
        return self._value

    @classmethod
    def nbits(cls) -> int:
        return 1

    @classmethod
    def base_type(cls) -> type:
        return bool

    @classmethod
    def decoded(cls) -> dict[int, object]:
        return {0: False, 1: True}

    @classmethod
    def from_bits(cls, *bits: bool) -> EncodedBool:
        if len(bits) == 1:
            return EncodedBool(bits[0])
        else:
            raise ValueError(f"Expected single bit; got {bits}.")

    def to_bits(self: EncodedBool) -> tuple[bool, ...]:
        return (self._value,)


class Bool(Variable):
    """A Boolean random variable: a variable of one bit of the enclosing definition.

    The primitive `Variable` type, from which every other one is built.  `&`, `|`, `~` and
    `^` record the corresponding gates inside an active `DPP` definition.

        >>> from pylea import flip, prob
        >>> @DPP
        ... def either() -> Bool:
        ...     return flip(0.5) | flip(0.5)
        >>> prob(either)
        0.75
    """

    def __init__(
            self,
            to_be_used_within: DPP,
            name: str | None = None,
            wire: int = -1,
        ) -> None:

        if not isinstance(to_be_used_within, DPP):
            raise ValueError(f"Expected DPP; got {type(to_be_used_within)}")
        if hasattr(to_be_used_within, "_DPP__output_variables"):
            raise ValueError(
                "Cannot create Boolean variable to be used within "
                f"DPP \"{to_be_used_within.name()}\": definition is already complete."
            )
        self.__to_be_used_within: DPP = to_be_used_within
        self.name = name
        # A `Bool` is the only variable naming a wire; every other one is a view over `Bool`s.
        self.wire = wire

    @classmethod
    def make(cls: DataType, *xs: Bool, name: str | None = None) -> Bool:
        if len(xs) != 1:
            raise ValueError(f"Expected single Boolean; got {len(xs)}.")
        xs[0].name = name
        return xs[0]

    @classmethod
    def encoding(cls) -> Encoding:
        return EncodedBool

    def to_be_used_within(self) -> DPP:
        return self.__to_be_used_within

    def bits(self) -> tuple[Bool, ...]:
        return (self,)

    def __and__(self: Bool, other: Bool | bool) -> Bool:
        return _AND_(self, self._coerced(other))

    def __or__(self, other: Bool | bool) -> Bool:
        return _OR_(self, self._coerced(other))

    def __invert__(self: Bool) -> Bool:
        return _NOT_(self)

    def __xor__(self: Bool, other: Bool | bool) -> Bool:
        other = self._coerced(other)
        return (self & ~other) | (~self & other)

    # A plain Python `bool` does not know how to combine with a `Bool`, so `True & x`
    # reaches here through Python's reflected-operator fallback; `&`, `|` and `^` are all
    # commutative, so the reflected form is the same operation with the operands swapped
    # back.
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    @staticmethod
    def any(xs: Iterable[Bool]) -> Bool:
        """The event that at least one of `xs` holds, as a chain of ORs.

        `False` where `xs` is empty, the identity element of disjunction.  Must be called
        inside an active DPP definition.

            >>> from pylea import DPP, Bool, flip, prob
            >>> @DPP
            ... def one_or_more() -> Bool:
            ...     return Bool.any(flip(0.1) for _ in range(5))
            >>> round(prob(one_or_more), 6)
            0.40951
        """
        result = Bool.const(False)
        for x in xs:
            result = result | x
        return result

    @staticmethod
    def all(xs: Iterable[Bool]) -> Bool:
        """The event that every one of `xs` holds, as a chain of ANDs.

        `True` where `xs` is empty, the identity element of conjunction.  Must be called
        inside an active DPP definition.

            >>> from pylea import DPP, Bool, flip, prob
            >>> @DPP
            ... def every_one() -> Bool:
            ...     return Bool.all(flip(0.9) for _ in range(5))
            >>> round(prob(every_one), 6)
            0.59049
        """
        result = Bool.const(True)
        for x in xs:
            result = result & x
        return result

    @overload
    def match(self, match_dict: dict[bool, _T], data_type: None = None) -> _T: ...
    @overload
    def match(self, match_dict: dict, data_type: type[_T]) -> _T: ...
    def match(self, match_dict: dict[bool, Variable],
              data_type: type[Variable] | None = None) -> Variable:
        """The value `match_dict` gives for this `Bool`, selected inside the DPP.

        `match_dict` has exactly the keys `True` and `False`, each mapping to the variable
        to select in that case or to a value of the output type, which becomes a constant.
        The output type is `data_type` where that is given, and otherwise the type of the
        first variable among the values.

            >>> from pylea import flip, prob
            >>> @DPP
            ... def noisy() -> Bool:
            ...     c = flip(0.25)
            ...     return c.match({True: flip(0.9), False: flip(0.1)})
            >>> round(prob(noisy), 6)
            0.3

        Raises `ValueError` for a missing or other key, where the output type cannot be
        determined, and for a value the output type cannot hold; `TypeError` where
        `data_type` is not a `Variable` subclass or a value is a variable of another type.

        Where both values are plain constants the selection is wired rather than
        recorded as an assignment: each output bit is a constant where the two constants
        agree there, and this `Bool` or its negation where they differ.  A long chain of
        such selections, as a sum of amounts each contributed or not, therefore stays a
        circuit of the width of one amount.
        """
        if data_type is not None and not _is_variable_type(data_type):
            raise TypeError(f"match: data_type must be a subclass of Variable, such as "
                            f"Bool; got {data_type!r}")
        others = [key for key in match_dict if not is_bool(key)]
        if others:
            raise ValueError(f"match on a Bool takes the keys True and False only; got "
                             f"{', '.join(map(repr, others))}")
        if True not in match_dict or False not in match_dict:
            raise ValueError("match_dict must contain both True and False keys.")

        then_val = match_dict[True]
        else_val = match_dict[False]

        output_type = data_type
        if output_type is None:
            for v in [then_val, else_val]:
                if isinstance(v, Variable):
                    output_type = type(v)
                    break
        if output_type is None:
            raise ValueError(
                "Cannot infer output type from match values; "
                "please provide data_type explicitly."
            )

        if not isinstance(then_val, Variable) and not isinstance(else_val, Variable):
            def encoded(key, value):
                try:
                    return output_type.encoding()(value).to_bits()
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"match value for {key!r}: {exc}") from None

            then_bits = encoded(True, then_val)
            else_bits = encoded(False, else_val)
            wired = tuple(
                Bool.const(t) if t == e else (self if t else ~self)
                for t, e in zip(then_bits, else_bits))
            return output_type.make(*wired)

        then_var = as_variable(then_val, output_type, "match value for True")
        else_var = as_variable(else_val, output_type, "match value for False")
        m = output_type.nbits()

        # An m-bit if-then-else is m independent one-bit ones sharing a condition, and
        # `DPP._emit` expands it that way rather than building the wide matrix.
        ite = Primitive(
            input_types=(Bool, output_type, output_type),
            output_types=(output_type,),
            name=f"_ITE_{output_type.__name__}_",
            prim=_Prim.ite(m),
            label="match",
        )
        return ite(self, then_var, else_var)  # type: ignore[return-value]

    def __bool__(self) -> bool:
        raise TypeError(
            "a Bool is a random variable and has no value True or False while a DPP is "
            "defined.  Python asks for one in 'if', 'while', 'and', 'or', 'not', in "
            "membership tests such as 'x in [a, b]' and in chained comparisons such as "
            "'a <= x <= b'.  Write '&', '|', '~' and 'match' instead, "
            "'Bool.any(x == v for v in values)' for membership and "
            "'(a <= x) & (x <= b)' for a chained comparison.")


def at_least(k: int, *xs: Bool | bool | Iterable[Bool | bool]) -> Bool:
    """The event that at least `k` of the events are true, as a voting gate.

    The events are given as one iterable, as to `Bool.any`, or as separate arguments; a
    Python `True` or `False` among them is a constant:

        >>> from pylea import flip, prob
        >>> @DPP
        ... def two_of_three() -> Bool:
        ...     return at_least(2, [flip(0.5), flip(0.5), flip(0.5)])
        >>> @DPP
        ... def two_of_two_and_true() -> Bool:
        ...     return at_least(2, flip(0.5), flip(0.5), True)
        >>> prob(two_of_three), prob(two_of_two_and_true)
        (0.5, 0.75)

    Built as a carry-ripple counter with one Boolean per threshold level, which takes
    ``O(n * k)`` binary operations where writing the disjunction of all k-subsets out
    would take ``C(n, k)`` terms.  `k` must be an integer; `TypeError` otherwise.  Must be
    called inside an active `DPP` definition.

    Each call builds a fresh inner DPP.  Structurally equal gates merge in the program of
    the enclosing DPP, so repeated calls do not grow it.
    """
    if not isinstance(k, numbers.Integral) or is_bool(k):
        raise TypeError(f"at_least: k must be an integer; got {k!r}")
    k = int(k)
    if (len(xs) == 1 and not isinstance(xs[0], Variable) and not is_bool(xs[0])
            and isinstance(xs[0], _IterableABC)):
        xs = tuple(xs[0])
    n = len(xs)

    def _gate(args, types=(Bool,) * n):
        if k <= 0:
            return Bool.const(True)         # every input already meets the threshold
        if k > n:
            return Bool.const(False)        # no input ever can

        # `cs[j]` is the event that at least `j + 1` of the inputs seen so far are true,
        # so all of them start false.  Each input raises every level it can reach.
        cs = [Bool.const(False) for _ in range(k)]
        for x in args:
            new_cs = [cs[0] | x]
            for j in range(1, k):
                new_cs.append(cs[j] | (x & cs[j - 1]))
            cs = new_cs
        return cs[k - 1]

    return DPP(_gate, name=f"at_least({k})")(*xs)


_AND_ = Primitive(
    input_types=(Bool, Bool),
    output_types=(Bool,),
    name="_AND_",
    label="&",
    prim=_Prim.and_(),
)

_OR_ = Primitive(
    input_types=(Bool, Bool),
    output_types=(Bool,),
    name="_OR_",
    label="|",
    prim=_Prim.or_(),
)

_NOT_ = Primitive(
    input_types=(Bool,),
    output_types=(Bool,),
    name="_NOT_",
    label="~",
    prim=_Prim.not_(),
)
