"""Discrete probabilistic programs.

The `@DPP` decorator runs its function once, at definition time, over symbolic arguments.
Each parameter becomes a typed input `Variable`, each call to a `ProbabilisticCallable`
records one `pylea.ir.body.Op`, and the variables returned name the body's outputs.  The
result is a `pylea.ir.program.Program` the `DPP` carries.  Tracing does no arithmetic and no
compilation, so defining a DPP is cheap.

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
function call, its bodies copied into the caller and equal ones merged, so a helper called
many times is stored once.  A `Primitive` records a single assignment, carrying either a
matrix or a `source` DPP whose circuit is kept because its matrix would be exponentially
large.  A `pylea.semantics.likelihood_table.LikelihoodTable` is the third.

`Bool` is the one variable naming a wire; every other `pylea.syntax.variable.Variable` is a
view over `Bool`s.  Inference entry points live on `DPP`: `DPP.evaluated` for the semantics
as a table, `DPP.compiled` for the plan and the settings behind it.  `DPP.diagram` draws the
traced definition, `DPP.dependency_graph` the variables of its interface rather than its
assignments, and `DPP.simplified` returns the result of simplification as a DPP in its own
right.

Tracing also records what each call was made through: the name of the callee against the
body it resolves to, and, for an opaque matrix alone, the name against the op.  A diagram
labels its boxes with these.  They are held here rather than on `pylea.ir.program.Program`
because a name is no part of the semantics and no compilation pass carries one.

The wires at the interface are named too: an input from the parameter it stands for, and an
output from the source of the return statement, where what was returned was a variable
rather than an expression.
"""
from __future__ import annotations
from typing import Any, Callable, Iterable, Literal, TypeVar, cast, get_type_hints, overload
from abc import ABC, abstractmethod
import ast
import textwrap
from inspect import getsource, signature, _empty
from pylea.syntax.variable import DataType, Variable
from pylea.syntax.compiled import Compiled, DEFAULT_KAPPA
from pylea.syntax.encoding import BitEncoded, Encoding
from pylea.ir.body import Body, BodyBuilder, Op
from pylea.ir.prim import Prim as _Prim, Tag as _Tag
from pylea.ir.program import (Program, intern_with_remap as _intern_with_remap,
                              reachable as _reachable)

_T = TypeVar('_T', bound=Variable)

# Above this, a primitive is expanded into a body of narrower ones rather than materialised:
# an if-then-else over 7 bits would be a 128 x 32768 matrix.
_MAX_FACTOR_ENTRIES = 1 << 20


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


class UnconvertibleCallee(TypeError):
    """Raised for a callee that is neither a `DPP` nor carries `prim` or `source`."""


class ProbabilisticCallable[**P, Q: tuple[Variable, ...] | Variable | None](ABC):
    """Something callable inside a `DPP` definition.

    Calling an instance inside an active definition records the call as an `Op` of the
    enclosing DPP's body and returns fresh output `Variable`s.  The concrete kinds are
    `DPP`, `Primitive` and `pylea.semantics.likelihood_table.LikelihoodTable`.
    """

    @abstractmethod
    def input_types(self) -> tuple[DataType, ...]: ...

    @abstractmethod
    def output_types(self) -> tuple[DataType, ...]: ...

    @abstractmethod
    def compiled(self) -> Compiled | Primitive:
        """The executable form: a `Compiled` plan for a `DPP`, the `Primitive` itself
        otherwise."""

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Whether this callable draws no randomness, so no `flip` and no `uniform`."""

    @property
    @abstractmethod
    def is_discardable(self) -> bool:
        """Whether this callable conditions on nothing, so no `observe` and no equality."""

    def name(self: ProbabilisticCallable) -> str:
        return f"f_{id(self)}"

    def n_inputs(self) -> int:
        return len(self.input_types())

    def n_outputs(self) -> int:
        return len(self.output_types())

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Q:
        """The output variables of this call, which is now recorded as an `Op`.

        Raises `RuntimeError` outside an active `DPP` definition.
        """
        return DPP._call(self, *args)  # type: ignore


class DPP[**P, Q: tuple[Variable, ...] | Variable | None](ProbabilisticCallable[P, Q]):
    """A discrete probabilistic program.

    Normally built by applying the `@DPP` decorator to a Python function whose parameters
    are annotated with `Variable` subclasses, or whose types come from a `types` keyword
    argument.  Running the decorated function records its `ProbabilisticCallable` calls as
    the `Op`s of an IR `Body`, giving a `pylea.ir.program.Program` that `compiled` lowers to
    an executable plan.

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

    __currently_processing: DPP | None = None

    def __init__(
            self,
            defining_function: Callable[P, Q],
            name: str | None = None,
        ):
        """The DPP traced from `defining_function`, called `name`.

        `name` defaults to the function's own, which is what the `@DPP` decorator gives.
        It is what a diagram labels a call to this DPP with, so a DPP built inside a
        helper, where the function's name is an implementation detail, passes one.

        Input types come from the annotation on each positional parameter:

            >>> from pylea import flip, prob
            >>> @DPP
            ... def both(a: Bool, b: Bool) -> Bool:
            ...     return a & b
            >>> [t.__name__ for t in both.input_types()]
            ['Bool', 'Bool']

        A function with no annotations must instead take a single variable-length parameter
        and declare a `types` keyword argument, which is how a DPP over a number of
        inputs fixed at definition time is written:

            >>> @DPP
            ... def all_of(xs, types=(Bool,) * 3) -> Bool:
            ...     return xs[0] & xs[1] & xs[2]
            >>> len(all_of.input_types())
            3

        Raises `ValueError` where neither form is available, and `TypeError` where a
        declared input type is not a `Variable` subclass or the function returns something
        other than `Variable`s.

        A definition nested inside another is allowed, as happens inside a factory such as
        `pylea.syntax.enum.Enum.uniform`: the outer definition is suspended for the duration
        and restored afterwards.  The inner DPP may not use variables belonging to the outer
        one, which is checked here for the outputs and in `DPP._call` for every argument.
        """
        self.__name = defining_function.__name__ if name is None else name

        _outer = DPP.__currently_processing
        DPP.__currently_processing = self
        try:
            sig = signature(defining_function)
            params = sig.parameters

            if any(param.annotation is _empty for param in params.values()):
                # No annotations, so the types come from the `types` keyword argument.
                using_types_kwarg = True
                try:
                    types_param = sig.parameters['types']
                except KeyError:
                    raise ValueError(
                        "Neither type annotations nor explicit tuple of types provided.\n"
                        "To create a DPP, either type annotations for every "
                        "positional argument, or the keyword argument 'types' must be "
                        "provided.\nIn the latter case, only a single variable-length "
                        "positional argument, and no positional-only argument should "
                        "be provided."
                    )
                types = types_param.default
                if types is types_param.empty:
                    raise ValueError(
                        "No default value provided for argument 'types'."
                    )
                if not isinstance(types, tuple):
                    raise ValueError(
                        f"Default value of keyword argument 'types' must be a tuple; "
                        f"got {types}."
                    )
                positional_only_arguments = [
                    param for param in params.values()
                    if param.kind == param.POSITIONAL_ONLY
                ]
                if positional_only_arguments:
                    raise ValueError(
                        f"Keyword argument 'types' was provided alongside "
                        f"positional-only arguments {positional_only_arguments}.\n"
                        "If 'types' is provided, only a single variable-length "
                        "positional argument, and no positional-only argument should "
                        "be provided."
                    )
                input_types: tuple[type[Variable]] = types
            else:
                # `get_type_hints` is what resolves annotations left as strings.
                using_types_kwarg = False
                try:
                    hints = get_type_hints(defining_function)
                except Exception:
                    hints = {}
                input_types: tuple[type[Variable]] = tuple(
                    hints.get(name, param.annotation)
                    for name, param in params.items()
                )
            for i, t in enumerate(input_types):
                if not issubclass(t, Variable):
                    raise TypeError(
                        f"Input type no. {i} is not a subclass of Variable, "
                        "as required for all input types."
                    )
            # Tracing builds the IR body directly: every `Bool` names a wire and every
            # recorded call is an `Op`.  `bodies` accumulates the callees' bodies, so a DPP
            # carries every body it needs and nothing is held globally.
            self.__bodies: list[Body] = []
            self.__builder = BodyBuilder(self.__bodies)
            self.__callee_index: dict[DPP, int] = {}
            # What a diagram labels a box with.  Names are not part of the semantics and no
            # pass carries them, so they are held here rather than on `Program`.
            self.__body_names: dict[int, str] = {}
            self.__op_names: dict[int, dict[int, str]] = {}
            self.__own_op_names: dict[int, str] = {}
            # The wires of each variable of more than one bit, which a diagram draws as
            # one dot.  Like the names, no part of the semantics.
            self.__own_groups: list[tuple[int, ...]] = []
            self.__wire_groups: dict[int, tuple[tuple[int, ...], ...]] = {}
            self.__parameter_names: tuple[str, ...] = (
                () if using_types_kwarg else tuple(params))

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
            if not using_types_kwarg:
                output_variables = trace(*self.__input_variables)
            else:
                output_variables = trace(self.__input_variables)

            if isinstance(output_variables, tuple):
                for i, x in enumerate(output_variables):
                    if not isinstance(x, Variable):
                        raise TypeError(
                            f"Applying defining_function returned output of type "
                            f"{type(x).__name__} at position {i}; expected Variable.")
                self.__output_variables = output_variables
            elif isinstance(output_variables, Variable):
                self.__output_variables = (output_variables,)
            elif output_variables is None:
                self.__output_variables = tuple()
            else:
                raise TypeError(
                    f"Defining function returned output of type "
                    f"{type(output_variables).__name__}; "
                    "expected tuple[Variable, ...], Variable or None.")

            for i, x in enumerate(self.__output_variables):
                if x.to_be_used_within() is not self:
                    raise ValueError(
                        f"Output variable at position {i} belongs to DPP "
                        f"'{x.to_be_used_within().name()}', not '{self.__name}'. "
                        "Returning variables from an enclosing DPP is not allowed."
                    )

            for v in self.__output_variables:
                self._group(v)
            output_wires = tuple(b.wire for v in self.__output_variables
                                 for b in v.bits())
            body = self.__builder.finish(self.__input_wires, output_wires)
            self.__bodies.append(body)
            self.__op_names[len(self.__bodies) - 1] = self.__own_op_names
            self.__wire_groups[len(self.__bodies) - 1] = tuple(
                dict.fromkeys(self.__own_groups))
            self.__program, remap = _intern_with_remap(
                Program(tuple(self.__bodies), len(self.__bodies) - 1))
            self.__body_names = _carried(self.__body_names, remap)
            self.__op_names = _carried(self.__op_names, remap)
            self.__wire_groups = _carried(self.__wire_groups, remap)

            # Read from the source only if a diagram asks for it, since most DPPs are
            # never drawn.
            self.__defining_function = defining_function
            self.__output_names: tuple[str | None, ...] | None = None
            self.__compiled: dict = {}
            self.__simplified: DPP | None = None
            self.__is_discardable: bool | None = None

        finally:
            DPP.__currently_processing = _outer      # even if tracing raised

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
        traced into.
        """
        body = program.main
        for label, types, wires in (("input", input_types, body.inputs),
                                    ("output", output_types, body.outputs)):
            declared = sum(t.nbits() for t in types)
            if declared != len(wires):
                raise ValueError(f"{name}: the types declare {declared} {label} bits, but "
                                 f"the root body has {len(wires)}")

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

        self.__program = program
        self.__defining_function = defining_function
        self.__output_names = None
        self.__body_names = {}
        self.__op_names = {}
        self.__wire_groups = {program.root: tuple(
            tuple(b.wire for b in x.bits())
            for x in self.__input_variables + self.__output_variables
            if len(x.bits()) > 1)}
        self.__compiled = {}
        self.__simplified = None
        self.__is_discardable = None
        return self

    def name(self: DPP) -> str:
        return self.__name

    def input_variables(self: DPP) -> tuple[Variable, ...]:
        return self.__input_variables

    def output_variables(self: DPP) -> tuple[Variable, ...]:
        return self.__output_variables

    def program(self: DPP) -> Program:
        """The `pylea.ir.program.Program` traced from this DPP, rooted at its own body.

        A callee's bodies are copied in, and structurally equal ones merge, so a helper
        called twice is stored once:

            >>> from pylea import flip
            >>> @DPP
            ... def coin() -> Bool:
            ...     return flip(0.5)
            >>> @DPP
            ... def both() -> Bool:
            ...     return coin() & coin()
            >>> len(both.program().bodies)
            2
        """
        return self.__program

    # -- tracing into the IR ----------------------------------------------------

    def _resolve(self: DPP, f: ProbabilisticCallable) -> _Prim | int:
        """`f` as a primitive, or as an index into this DPP's body table.

        The name of `f` is recorded against the body it resolves to, so that a diagram can
        label the call.  It is the name of the callable at the call site rather than the
        one the body was traced under, which is what makes a retyped comparison show as
        `_EQ_Colour_` rather than as the `_eq` circuit beneath it.

        Raises `UnconvertibleCallee` for a callee with neither form.
        """
        if isinstance(f, DPP):
            return self._named(self._absorb(f), f.name())
        prim = getattr(f, "prim", None)
        if prim is not None:
            return prim
        source = getattr(f, "source", None)
        if source is not None:
            return self._named(self._absorb(source), f.name())
        raise UnconvertibleCallee(
            f"{type(f).__name__} {f.name()!r} has no IR form: it is neither a DPP nor a "
            f"primitive.  Give its construction site a `prim=` (if its matrix is small) "
            f"or a `source=` (if it is a merged view of a DPP)."
        )

    def _named(self: DPP, body: int, name: str) -> int:
        """`body`, with `name` recorded for it unless it is named already."""
        self.__body_names.setdefault(body, name)
        return body

    def _group(self: DPP, x: Variable) -> None:
        """The wires of `x` recorded as one variable of this body, where it has several.

        What `DPP.diagram` draws as a single dot; the semantics is the same without it.
        """
        if len(x.bits()) > 1:
            self.__own_groups.append(tuple(b.wire for b in x.bits()))

    def _absorb(self: DPP, g: DPP) -> int:
        """The index of `g`'s own body, after copying `g`'s bodies into this DPP's table."""
        # The cache is keyed on the callee object, never on `id(g)`.  A DPP built inside a
        # helper becomes unreachable as soon as the call returns and its address may then be
        # reused, at which point an id-keyed entry would answer for an unrelated callee.  A
        # dict keyed by the object holds a reference, so the address cannot be recycled.
        cached = self.__callee_index.get(g)
        if cached is not None:
            return cached
        offset = len(self.__bodies)
        for b in g.program().bodies:
            self.__bodies.append(Body(b.bits, b.inputs, b.outputs, tuple(
                Op(op.callee if isinstance(op.callee, _Prim) else op.callee + offset,
                   op.args, op.outs)
                for op in b.ops)))
        for i, name in g.__body_names.items():
            self.__body_names.setdefault(offset + i, name)
        for i, named_ops in g.__op_names.items():
            self.__op_names.setdefault(offset + i, named_ops)
        # Wires are numbered within a body and bodies are copied verbatim, so the groups
        # carry over unchanged.
        for i, groups in g.__wire_groups.items():
            self.__wire_groups.setdefault(offset + i, groups)

        index = offset + g.program().root
        self.__callee_index[g] = index
        return index

    def _emit(self: DPP, callee: _Prim | int, args: tuple[int, ...]) -> tuple[int, ...]:
        """The result wires of one call, which is now recorded.

        A primitive whose matrix would be larger than `_MAX_FACTOR_ENTRIES` is expanded
        into narrower ops where it has such a form, and raises `UnconvertibleCallee`
        where it does not.
        """
        too_wide = _MAX_FACTOR_ENTRIES.bit_length() - 1
        if isinstance(callee, _Prim) and callee.in_bits + callee.out_bits > too_wide:
            if callee.tag is _Tag.ITE:
                # An m-bit if-then-else is m one-bit ones sharing the condition, far smaller
                # than its ``2**m x 2**(2m + 1)`` matrix.
                m = callee.params[0]
                cond, then_, else_ = args[0], args[1:1 + m], args[1 + m:]
                one = _Prim.ite(1)
                return tuple(self.__builder.emit(one, (cond, t, e))[0]
                             for t, e in zip(then_, else_))
            raise UnconvertibleCallee(
                f"{callee.tag.name} with a {1 << callee.out_bits} x {1 << callee.in_bits} "
                f"matrix is too large to build and has no structural expansion."
            )
        return self.__builder.emit(callee, args)

    def input_types(self) -> tuple[DataType, ...]:
        return tuple(type(x) for x in self.input_variables())

    def output_types(self) -> tuple[DataType, ...]:
        return tuple(type(x) for x in self.output_variables())

    @property
    def is_deterministic(self) -> bool:
        """Whether this DPP draws no randomness, so no `flip` and no `uniform`.

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
        return all(op.callee.is_deterministic
                   for i in _reachable(self.__program)
                   for op in self.__program.bodies[i].ops
                   if isinstance(op.callee, _Prim))

    @property
    def is_discardable(self) -> bool:
        """Whether this DPP conditions on nothing, so no `observe` and no equality.

            >>> from pylea import flip, observe
            >>> @DPP
            ... def conditioned() -> Bool:
            ...     x = flip(0.5)
            ...     observe(x)
            ...     return x
            >>> conditioned.is_discardable
            False
        """
        if self.__is_discardable is None:
            self.__is_discardable = all(
                op.callee.is_discardable
                for i in _reachable(self.__program)
                for op in self.__program.bodies[i].ops
                if isinstance(op.callee, _Prim))
        return self.__is_discardable

    def __str__(self: DPP) -> str:
        parameters = ', '.join(f"{x}: {type(x).__name__}"
                               for x in self.input_variables())
        output_types = ', '.join(f"{type(x).__name__}" for x in self.output_variables())
        match len(self.output_variables()):
            case 0:
                output_type = "None"
            case 1:
                output_type = output_types
            case _:
                output_type = f"tuple[{output_types}]"
        header = f"@DPP\ndef {self.name()}({parameters}) -> {output_type}:"
        body = "\n".join(f"    {op}" for op in self.__program.main.ops)
        outs = ', '.join(str(y) for y in self.output_variables())
        return f"{header}\n{body}\n    return {outs}"

    @staticmethod
    def _call(
            f: ProbabilisticCallable,
            *args: Variable,
        ) -> tuple[Variable, ...] | Variable | None:
        """The output variables of a call to `f`, which is now recorded as an `Op`.

        A single `Variable` where `f` has one output, a tuple where it has several, and
        `None` where it has none.  Raises `RuntimeError` outside an active `DPP`
        definition, and `ValueError` if the argument count is wrong or an argument
        belongs to a different DPP.
        """
        if len(args) != f.n_inputs():
            raise ValueError(
                f"ProbabilisticCallable \"{f.name()}\" "
                f"takes {f.n_inputs()} arguments; got {len(args)}."
            )
        if DPP.__currently_processing is None:
            raise RuntimeError(
                f"No active DPP definition in call to ProbabilisticCallable \"{f.name()}\".\n"
                "Any call to a ProbabilisticCallable must be made within a DPP "
                "definition (via @DPP)."
            )
        for i, arg in enumerate(args):
            if not arg.to_be_used_within() is DPP.__currently_processing:
                raise ValueError(
                    f"Context mismatch: argument at position {i} is to be used within DPP "
                    f"\"{arg.to_be_used_within().name()}\", but was attempted to be used "
                    "within "
                    f"\"{DPP.__currently_processing.name()}\"."
                )
            for bit in arg.bits():
                if not 0 <= bit.wire < len(DPP.__currently_processing.__builder.bits):
                    raise ValueError(
                        f"Argument at position {i} does not name a wire of DPP "
                        f"\"{DPP.__currently_processing.name()}\"."
                    )

        active = DPP.__currently_processing
        callee = active._resolve(f)
        first = len(active.__builder.ops)
        wires = active._emit(callee, tuple(b.wire for x in args for b in x.bits()))
        if isinstance(callee, _Prim) and callee.tag in (_Tag.MATRIX, _Tag.CONST):
            # Neither of these carries what a reader wants to see: a matrix is opaque, and a
            # constant holds the bit pattern rather than the value of the type it belongs
            # to.  One call can emit several ops, which is why the range is taken.
            written = getattr(f, "label", None) or f.name()
            for op in range(first, len(active.__builder.ops)):
                active.__own_op_names[op] = written
        outputs: tuple[Variable, ...] = ()
        at = 0
        for t in f.output_types():
            outputs += (t.make(*(Bool(active, wire=w)
                                 for w in wires[at:at + t.nbits()])),)
            at += t.nbits()
        if at != len(wires):
            raise ValueError(
                f"{f.name()} declares {at} output bits but produced {len(wires)} wires")
        for x in outputs:
            active._group(x)

        if len(outputs) > 1:
            return outputs
        elif len(outputs) == 1:
            return outputs[0]

    def compiled(self: DPP, order_finder=None, kappa: int | None = DEFAULT_KAPPA,
                 recompile: bool = False, simplify: bool | Literal["always"] = True,
                 device: str | None = None) -> Compiled:
        """This DPP as an executable plan.

        `order_finder` chooses the contraction tree, the default being the ladder in
        `pylea.compile.planner`.  `kappa` bounds a single intermediate to ``2**kappa``
        entries by slicing.  `simplify` applies the rewrite rules to the DPP first,
        wherever the planner judges them worth their own cost against one evaluation;
        ``"always"`` applies them regardless, which is what a caller who will evaluate
        many times wants.  `device` names where to evaluate, such as ``"numpy"``
        or ``"mps"``; left unset, the plan's shape decides.

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
        # Keyed on the finder object rather than on `id(...)`, for the reason given in
        # `_absorb`.
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

        This is the main inference entry point.  The table holds unnormalised likelihoods,
        and its `probabilities` view is P(output | input).  A table may itself be called
        inside another ``@DPP`` definition, so a DPP evaluated once can be reused as one
        assignment of its matrix.  Keyword arguments go to `compiled`, whose cache makes
        repeated evaluation free.

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
        own, and ``method="smc"`` raises.  ``method="auto"`` runs the exact engine for
        tables; the scalar cost comparison does not price a table's entries.
        """
        from pylea.syntax.method import resolve

        contract = resolve(method, rel_err, kwargs, floor=floor, confidence=confidence,
                           max_time=max_time, seed=seed, strict=strict)
        if contract.method == "smc":
            if self.input_types() != tuple():
                columns = 1 << sum(t.nbits() for t in self.input_types())
                raise ValueError(
                    f"a sampled table needs a zero-input DPP: this one has {columns} "
                    "input columns, and each would cost a full certified run; evaluate "
                    "exactly, or wrap the input you care about into a zero-input DPP")
            from pylea.sample.estimate import EstimatedLikelihoodTable
            from pylea.sample.query import certified_table

            outcome = certified_table(
                self, rel_err=contract.rel_err, confidence=contract.confidence,
                floor=contract.floor, max_time=max_time, seed=contract.seed,
                strict=contract.strict, normalised_contract=False, what="evaluated")
            return EstimatedLikelihoodTable.from_outcome(
                self.input_types(), self.output_types(), outcome,
                confidence=contract.confidence)
        if max_time is not None:
            from pylea.sample.portfolio import exact_table_within

            return exact_table_within(self, max_time, kwargs)
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
            ValueError: the matrix's magnitude (e**-1381.6) does not fit float64; use matrix(), which returns the array and its log scale apart.
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

        This applies the rules until none applies, where `compiled` leaves the planner to
        decide whether they are worth their own cost against one evaluation and to bound
        them by wall clock, so the two need not agree on the DPP they lower.

        The result carries this DPP's name and the names of its parameters, and no other
        names: the rules merge, inline and delete bodies, so a name recorded while tracing
        need not describe any body afterwards.  A diagram of it therefore labels a call by
        its index.
        """
        if self.__simplified is None:
            from pylea.simplify import simplify

            self.__simplified = DPP._from_program(
                simplify(self.__program), self.input_types(), self.output_types(),
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

        The boxes are labelled with what was called, and the wires into the picture with
        the parameters they stand for:

            >>> "test" in screened.diagram().svg()
            True
            >>> "carrier" in test.diagram().svg()
            True
        """
        from pylea.draw import diagram_of

        return diagram_of(self.__program, body_names=self.__body_names,
                          op_names=self.__op_names.get(self.__program.root, {}),
                          input_names=self._input_labels(),
                          output_names=self._output_labels(),
                          input_widths=tuple(t.nbits() for t in self.input_types()),
                          output_widths=tuple(t.nbits() for t in self.output_types()),
                          wire_groups=self.__wire_groups.get(self.__program.root, ()),
                          title=self.__name, **kwargs)

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

        What is drawn is what each variable is computed from.  An `observe` reads no
        variable of the interface and so adds no wire, and two outputs sharing an internal
        variable the DPP does not return are correlated but are drawn apart.  See
        `pylea.draw.dependency`.
        """
        from pylea.draw import dependency_graph_of

        return dependency_graph_of(
            self.__program,
            input_names=_variable_labels(self.__parameter_names, self.__input_variables),
            output_names=_variable_labels(self._returned(), self.__output_variables),
            input_widths=tuple(t.nbits() for t in self.input_types()),
            output_widths=tuple(t.nbits() for t in self.output_types()),
            title=self.__name, **kwargs)

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
    """A callable standing for one IR primitive, or for a retyped view of a `DPP`.

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
    being retyped while its circuit is kept, which is what an n-bit comparator needs: its
    matrix would be ``2 x 2**(2n)`` where its circuit is linear in n.  `label` is what a
    diagram writes in the box, where the name is not what a reader wants to see.
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
        return self.__prim

    @property
    def source(self) -> DPP | None:
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
    """The one-bit encoding of Python's built-in `bool`, which `Bool` is built from."""

    def __init__(self, x: bool) -> None:
        self._value = x

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
    """A Boolean random variable, standing for one wire of the enclosing DPP's body.

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
        """The event that at least one of `xs` holds, as a chain of ``n - 1`` ORs.

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
        """The event that every one of `xs` holds, as a chain of ``n - 1`` ANDs.

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

        `match_dict` needs both a `True` and a `False` key, each mapping to the output
        `Variable` to select in that case, or to a constant of the output type's base type,
        which is wrapped with `Variable.const`.  The output type is taken from the first
        `Variable` among the values, or from `data_type` where that is given.  Raises
        `ValueError` if a key is missing or the output type cannot be determined.

            >>> from pylea import flip, prob
            >>> @DPP
            ... def noisy() -> Bool:
            ...     c = flip(0.25)
            ...     return c.match({True: flip(0.9), False: flip(0.1)})
            >>> round(prob(noisy), 6)
            0.3

        Where both values are plain constants the selection is wired rather than
        recorded as an assignment: each output bit is a constant where the two constants
        agree there, and this `Bool` or its negation where they differ.  A long chain of
        such selections, as a sum of amounts each contributed or not, therefore stays a
        circuit of the width of one amount.
        """
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
            then_bits = output_type.encoding()(then_val).to_bits()
            else_bits = output_type.encoding()(else_val).to_bits()
            wired = tuple(
                Bool.const(t) if t == e else (self if t else ~self)
                for t, e in zip(then_bits, else_bits))
            return output_type.make(*wired)

        def to_variable(val) -> Variable:
            if isinstance(val, Variable):
                return val
            return output_type.const(val)  # type: ignore[union-attr]

        then_var = to_variable(then_val)
        else_var = to_variable(else_val)
        m = output_type.nbits()

        # An m-bit if-then-else is m independent one-bit ones sharing a condition, and
        # `DPP._emit` expands it that way rather than building the wide matrix.
        ite = Primitive(
            input_types=(Bool, output_type, output_type),
            output_types=(output_type,),
            name=f"_ITE_{output_type.__name__}_",
            prim=_Prim.ite(m),
        )
        return ite(self, then_var, else_var)  # type: ignore[return-value]

    def __bool__(self) -> bool:
        raise TypeError("Cannot cast boolean random variable (`Bool`) to Python `bool`.")


def at_least(k: int, *xs: Bool) -> Bool:
    """The event that at least `k` of `xs` are true, as a voting gate.

    Built as a carry-ripple counter with one Boolean per threshold level, which takes
    ``O(n * k)`` binary operations where writing the disjunction of all k-subsets out
    would take ``C(n, k)`` terms.  Must be called inside an active `DPP` definition.

    Each call builds a fresh inner DPP.  Structurally equal gates merge when the enclosing
    DPP is interned, so repeated calls do not grow it.

        >>> from pylea import flip, prob
        >>> @DPP
        ... def two_of_three() -> Bool:
        ...     return at_least(2, flip(0.5), flip(0.5), flip(0.5))
        >>> prob(two_of_three)
        0.5
    """
    from pylea.syntax.flip import flip as _flip

    n = len(xs)

    def _gate(args, types=(Bool,) * n):
        if k <= 0:
            return _flip(1.0)               # every input already meets the threshold
        if k > n:
            return _flip(0.0)               # no input ever can

        # `cs[j]` is the event that at least `j + 1` of the inputs seen so far are true,
        # so all of them start false.  Each input raises every level it can reach.
        cs = [_flip(0.0) for _ in range(k)]
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
    prim=_Prim.and_(),
)

_OR_ = Primitive(
    input_types=(Bool, Bool),
    output_types=(Bool,),
    name="_OR_",
    prim=_Prim.or_(),
)

_NOT_ = Primitive(
    input_types=(Bool,),
    output_types=(Bool,),
    name="_NOT_",
    prim=_Prim.not_(),
)
