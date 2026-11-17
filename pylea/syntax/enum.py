"""Finite enumerated types.

An `Enum` subclass declares its values in `elements`, a list, tuple, range or
one-dimensional numpy array of distinct, hashable values.  Their order fixes the encoding,
consecutive bit patterns MSB first in as few bits as hold them all, and the order `<`
compares by.  Where the count is not a power of two, or is 1, some patterns decode to no
element.  `elements` is checked when the class is created and stored as a tuple.

    >>> from pylea import DPP, probability_mass_function
    >>> class Colour(Enum):
    ...     elements = ["R", "G", "B"]
    >>> @DPP
    ... def draw() -> Colour:
    ...     return Colour.random({"R": 0.5, "G": 0.25, "B": 0.25})
    >>> probability_mass_function(draw)
    {'R': 0.5, 'G': 0.25, 'B': 0.25}

A value is looked up among the elements by Python's equality and hashing, so `1.0` and
`True` both find the element `1`.

`Enum.random` draws from a distribution over the elements, as one column of weights, and
gives a pattern decoding to no element weight zero.  `Enum.uniform` is that over all
elements equally.

`Enum.match` selects a value per element.  Where every value is a constant and the table
fits the bound below, it is one table from this variable to the result; otherwise it is a
balanced tree of one-bit selections,
``2**k - 1`` of them over `m` bits each, in place of the ``2**m x 2**(k + 2**k * m)`` matrix
a single primitive over the same interface would carry.  A pattern decoding to no element
selects the first value; a variable puts no weight there unless `uniform_bits` or a table
puts it there.

`Enum.cpt` builds a conditional probability table as one dense matrix, a column per bit
pattern of the parents.  A parent pattern that encodes no value takes the row of that
parent's first value, which agrees with what `Enum.match` selects there.

`Enum.map` and `Enum.cpt` hold a table densely, so its inputs and output together may span
at most 20 bits, ``2**20`` entries; `ValueError` says where they would span more.

Probabilities are held as floats throughout, so anything a double represents survives.  A
distribution normalising to zero at a positive entry is refused rather than accepted, since
that would turn an improbable outcome into an impossible one.
"""
from __future__ import annotations
import itertools
import math
import numbers
import numpy as np
from abc import ABCMeta
from pylea.syntax.dpp import Bool
from pylea.syntax.encoding import Encoding, BitEncoded
from pylea.syntax.variable import (
    Variable, as_variable, reads_as, require_definition, require_variable_type, _prefixed,
)
from pylea.ir.prim import Prim as _Prim
from typing import (
    TYPE_CHECKING, Any, ClassVar, Iterator, Mapping, Sequence, Self, TypeVar, overload,
)

if TYPE_CHECKING:
    from pylea.syntax.dpp import Primitive

_T = TypeVar('_T', bound=Variable)

# The most bits, inputs and output together, of a table that `map`, `cpt` or a constant
# `match` builds densely: ``2**20`` float64 entries are 8 MiB.
_TABLE_BITS = 20


def _listed(items, limit: int = 5) -> str:
    """`items` as a comma-separated list of reprs, at most `limit` of them."""
    items = list(items)
    shown = ", ".join(repr(x) for x in items[:limit])
    if len(items) > limit:
        shown += f" and {len(items) - limit} more"
    return shown


def _lookup(index: Mapping, value) -> int | None:
    """`index[value]`, or `None` where `value` is not a key, hashable or not."""
    try:
        return index.get(value)
    except TypeError:
        return None


def _validated_elements(name: str, raw) -> tuple[tuple, dict]:
    """`raw` as a tuple of elements, with the position of each; raises where `raw` is not
    an ordered sequence of distinct, hashable values."""
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1:
            raise ValueError(f"{name}.elements must be one-dimensional; got an array of "
                             f"shape {raw.shape}")
        values = raw.tolist()
    elif isinstance(raw, (str, bytes)):
        raise TypeError(f"{name}.elements is a string, whose characters would become the "
                        f"elements; give a list such as {list(raw)[:3]!r}")
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        raise TypeError(f"{name}.elements must be a list, tuple or range, since its order "
                        f"fixes the encoding and the order `<` compares by; got a "
                        f"{type(raw).__name__}")
    if not values:
        raise ValueError(f"{name}.elements is empty; an Enum needs at least one element")
    index: dict = {}
    for i, e in enumerate(values):
        try:
            hash(e)
        except TypeError:
            raise TypeError(f"{name}.elements[{i}] = {e!r} cannot be hashed; elements must "
                            "be hashable, such as strings, numbers or tuples") from None
        if e != e:
            raise ValueError(f"{name}.elements[{i}] = {e!r} is not equal to itself, so it "
                             "could never be looked up")
        j = index.setdefault(e, i)
        if j != i:
            first = values[j]
            note = (f", since {first!r} == {e!r} in Python"
                    if type(first) is not type(e) else "")
            raise ValueError(f"{name}.elements[{j}] = {first!r} and {name}.elements[{i}] = "
                             f"{e!r} are the same element{note}; elements must be distinct")
    return tuple(values), index


def _require_elements(cls, what: str) -> None:
    """Nothing, or `NotImplementedError` where `cls` declares no `elements`."""
    if cls.elements is None:
        raise NotImplementedError(
            f"{what}: {cls.__name__} declares no elements; set its class variable "
            "`elements` in a subclass")


def _weight(w, where: str, element) -> float:
    """`w` as a float, or an error naming `element` where it is no finite, non-negative
    real number."""
    if not isinstance(w, numbers.Real):
        raise TypeError(f"{where}: the weight of {element!r} must be a number; got {w!r}")
    x = float(w)
    if not math.isfinite(x):
        raise ValueError(f"{where}: the weight of {element!r} is {w!r}, which is not finite")
    if x < 0.0:
        raise ValueError(f"{where}: the weight of {element!r} is {w!r}, which is negative")
    return x


def _weights(spec, cls: type[Enum], where: str) -> np.ndarray:
    """The distribution over `cls.elements`, in element order, that `spec` gives.

    `spec` is a mapping from element to weight, an omitted element weighing zero, or a
    sequence of one weight per element in element order.  The weights are normalised to
    sum to 1, so counts serve as well as probabilities.  Any probability a double can
    represent survives, so 1e-300 stays 1e-300, as it does in `pylea.syntax.flip.flip`.

    Raises `ValueError` for a key that is not an element, a sequence of the wrong length,
    a weight that is negative or not finite, weights that are all zero, and a positive
    weight that normalising would turn into zero, since that would make an improbable
    outcome an impossible one with no warning; `TypeError` where `spec` or a weight is
    not of a kind described here.  Every message starts with `where`.
    """
    elements = cls.elements
    assert elements is not None
    n = len(elements)
    if isinstance(spec, Mapping):
        raw = [0.0] * n
        given = [False] * n
        unknown = []
        for key, w in spec.items():
            i = _lookup(cls._index, key)
            if i is None:
                unknown.append(key)
                continue
            if given[i]:
                raise ValueError(f"{where}: {key!r} is given twice, as the element "
                                 f"{elements[i]!r}")
            given[i] = True
            raw[i] = _weight(w, where, elements[i])
        if unknown:
            raise ValueError(f"{where}: {_listed(unknown)} "
                             f"{'is not an element' if len(unknown) == 1 else 'are not elements'}"
                             f" of {cls.__name__}, whose elements are {_listed(elements)}")
    elif isinstance(spec, (Sequence, np.ndarray)) and not isinstance(spec, (str, bytes)):
        if isinstance(spec, np.ndarray) and spec.ndim != 1:
            raise ValueError(f"{where}: weights must be one-dimensional; got an array of "
                             f"shape {spec.shape}")
        entries = spec.tolist() if isinstance(spec, np.ndarray) else list(spec)
        if len(entries) != n:
            raise ValueError(f"{where}: a list of weights needs {n} entries, one per element "
                             f"of {cls.__name__} in order; got {len(entries)}")
        raw = [_weight(w, where, e) for w, e in zip(entries, elements)]
    else:
        raise TypeError(f"{where}: weights must be a dict from element to weight, or a list "
                        f"of {n} weights in the order of the elements; got a "
                        f"{type(spec).__name__}")
    total = math.fsum(raw)
    if total <= 0.0:
        raise ValueError(f"{where}: every weight is zero, so there is no distribution")
    out = np.array(raw) / total
    for e, p, q in zip(elements, raw, out):
        if p > 0.0 and q == 0.0:
            raise ValueError(
                f"{where}: the weight {p!r} of {e!r} underflows to 0 once normalised by "
                f"{total!r}, and would be indistinguishable from an impossible outcome")
    return out


def _value_rows(types: Sequence[type[Variable]]) -> tuple[np.ndarray, list[dict]]:
    """For each bit pattern of `types` packed MSB first, the position of its values in
    the product of the types' values; and per type, a dict from value to position.

    The product is ordered as `itertools.product` orders it, the first type varying
    slowest.  A pattern of a type that encodes no value counts as that type's first value.
    """
    row_of_column = np.zeros(1, dtype=np.int64)
    positions: list[dict] = []
    for t in types:
        decoded = t.encoding().decoded()
        ordinal = np.zeros(1 << t.nbits(), dtype=np.int64)
        ordinal[np.fromiter(decoded.keys(), dtype=np.int64, count=len(decoded))] = (
            np.arange(len(decoded)))
        row_of_column = (row_of_column[:, None] * len(decoded) + ordinal[None, :]).ravel()
        positions.append({v: i for i, v in enumerate(decoded.values())})
    return row_of_column, positions


def _deterministic_table(input_types, output_patterns: np.ndarray,
                         out_bits: int) -> np.ndarray:
    """The 0/1 matrix sending each input pattern to `output_patterns[row]`, `row` being
    its position in the product of the input types' values (see `_value_rows`)."""
    row_of_column, _ = _value_rows(input_types)
    M = np.zeros((1 << out_bits, len(row_of_column)))
    M[output_patterns[row_of_column], np.arange(len(row_of_column))] = 1.0
    return M


def _table_fits(in_bits: int, out_bits: int) -> bool:
    """Whether a table over `in_bits` input and `out_bits` output bits may be built."""
    return in_bits + out_bits <= _TABLE_BITS


class EnumMeta(ABCMeta):
    """The metaclass of `Enum`, through which an `Enum` class iterates over its elements
    and has their number as its length.

        >>> class Colour(Enum):
        ...     elements = ["R", "G", "B"]
        >>> list(Colour), len(Colour)
        (['R', 'G', 'B'], 3)
    """
    elements: tuple[Any, ...] | None

    def __iter__(cls) -> Iterator:
        _require_elements(cls, "iteration")
        return iter(cls.elements)  # type: ignore[arg-type]

    def __len__(cls) -> int:
        return len(cls.elements) if cls.elements is not None else 0


def _select(bits, values, output_type):
    """`values[i]`, where `i` is `bits` read as an unsigned integer, MSB first.

    Built as a balanced tree of one-bit selections, the leading bit choosing between the
    halves of `values` and each half being selected the same way.  The tree holds
    ``2**k - 1`` selections of `m` bits each, against the ``2**m x 2**(k + 2**k * m)``
    matrix a single primitive over the same interface would carry.  A pair of constants
    reaches `pylea.syntax.dpp.Bool.match` as constants, which it selects without recording
    an assignment.
    """
    if not bits:
        return values[0]
    half = len(values) // 2
    return bits[0].match(
        {False: _select(bits[1:], values[:half], output_type),
         True: _select(bits[1:], values[half:], output_type)},
        data_type=output_type,
    )


class Enum(Variable, metaclass=EnumMeta):
    """A variable over a finite, ordered sequence of values.

    A subclass declares its values by setting the class variable `elements` to a list,
    tuple, range or one-dimensional numpy array of distinct, hashable values.  The
    encoding follows: values map to consecutive unsigned integers, held in
    ``max(1, ceil(log2(len(elements))))`` bits, MSB first.  A subclass that does not set
    `elements` inherits its parent's.

        >>> class Colour(Enum):
        ...     elements = ["R", "G", "B"]
        >>> Colour.nbits(), len(Colour), Colour.elements
        (2, 3, ('R', 'G', 'B'))

    `elements` is checked when the class is created.  A set or a generator is refused,
    since their order is not fixed, and so is an empty sequence and an unhashable element.
    Two elements equal in Python are refused, as they would be one value:

        >>> class Mixed(Enum):
        ...     elements = [1, True, 2]
        Traceback (most recent call last):
          ...
        ValueError: Mixed.elements[0] = 1 and Mixed.elements[1] = True are the same element, since 1 == True in Python; elements must be distinct

    Where the number of elements is not a power of two, or is 1, some patterns decode to
    no value; `Enum.random` gives those weight zero.  Using a subclass that has not set
    `elements` raises `NotImplementedError`.
    """

    elements: ClassVar[tuple[Any, ...] | None] = None
    # The position of each element, set together with `elements`.
    _index: ClassVar[dict | None] = None
    # The built encoding, stored on the class that built it and read from its own
    # namespace only, so a subclass declaring its own parameters never inherits it.
    _encoding: ClassVar[Encoding | None] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "elements" not in vars(cls):
            return
        if cls.elements is None:
            cls._index = None
            return
        cls.elements, cls._index = _validated_elements(cls.__name__, vars(cls)["elements"])

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        _require_elements(type(self), f"Cannot create a {type(self).__name__} variable")
        super().__init__(*xs, name=name)

    @classmethod
    def _value_table(cls) -> object:
        return cls.elements

    @classmethod
    def encoding(cls) -> Encoding:
        cached = vars(cls).get("_encoding")
        if cached is not None:
            return cached
        _require_elements(cls, f"{cls.__name__}.encoding")
        elements: tuple = cls.elements  # type: ignore[assignment]
        index: dict = cls._index  # type: ignore[assignment]
        name = cls.__name__
        n = len(elements)
        nbits = max(1, (n - 1).bit_length())
        kinds = {type(e) for e in elements}
        base = kinds.pop() if len(kinds) == 1 else object

        class EncodedEnum(BitEncoded):
            """An element of an `Enum`, encoded as its position, MSB first."""

            def __init__(self, v) -> None:
                position = _lookup(index, v)
                if position is None:
                    raise ValueError(f"{v!r} is not an element of {name}, whose elements "
                                     f"are {_listed(elements)}")
                self._position = position

            @property
            def value(self):
                return elements[self._position]

            @classmethod
            def nbits(cls) -> int:
                return nbits

            @classmethod
            def n_values(cls) -> int:
                return n

            @classmethod
            def base_type(cls) -> type:
                return base

            @classmethod
            def decoded(cls) -> dict[int, object]:
                return dict(enumerate(elements))

            @classmethod
            def from_bits(cls, *bits: bool) -> "EncodedEnum":
                if len(bits) != nbits:
                    raise ValueError(f"Expected {nbits} bits; got {len(bits)}.")
                position = sum(int(bit) << (nbits - 1 - i) for i, bit in enumerate(bits))
                if position >= n:
                    raise ValueError(
                        f"Bit pattern {position} encodes no element of {name}, which has "
                        f"{n}.")
                return EncodedEnum(elements[position])

            def to_bits(self) -> tuple[bool, ...]:
                return tuple(bool((self._position >> (nbits - 1 - i)) & 1)
                             for i in range(nbits))

        cls._encoding = EncodedEnum
        return EncodedEnum

    @classmethod
    def random(cls, probs: Mapping[object, float] | Sequence[float]) -> Self:
        """A random variable over this type, distributed according to `probs`.

        `probs` maps an element to its weight, an omitted element weighing zero, or lists
        one weight per element in the order of `elements`.  Weights that do not sum to one
        are normalised, so a table of counts may be given directly:

            >>> from pylea import DPP, probability_mass_function
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> @DPP
            ... def from_counts() -> Colour:
            ...     return Colour.random({"R": 20, "G": 10, "B": 10})
            >>> probability_mass_function(from_counts)
            {'R': 0.5, 'G': 0.25, 'B': 0.25}
            >>> @DPP
            ... def from_list() -> Colour:
            ...     return Colour.random([20, 10, 10])
            >>> probability_mass_function(from_list)
            {'R': 0.5, 'G': 0.25, 'B': 0.25}

        Raises `ValueError` for a key that is not an element, a list whose length is not
        the number of elements, and a weight that is negative or not finite, since each of
        these would otherwise change the distribution without a word.  Must be called
        inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        where = f"{cls.__name__}.random"
        _require_elements(cls, where)
        weights = _weights(probs, cls, where)
        require_definition(where)
        column = np.zeros((1 << cls.nbits(), 1))
        column[:len(weights), 0] = weights
        return Primitive(
            input_types=tuple(),
            output_types=(cls,),
            name="_RAND_" + cls.__name__,
            # A distribution over k bits is one column of ``2**k`` weights, small enough to
            # hold densely rather than build as a circuit.
            prim=_Prim.from_matrix(column),
        )()

    @classmethod
    def uniform(cls) -> Self:
        """A random variable drawn uniformly from this type's elements.

        The same as ``cls.random([1] * len(cls))``.
        `pylea.syntax.variable.Variable.uniform_bits` draws over bit patterns instead, which
        differs where the element count is not a power of two, or is 1.

            >>> from pylea import DPP, probability_mass_function
            >>> class Die(Enum):
            ...     elements = [1, 2, 3, 4, 5, 6]
            >>> @DPP
            ... def roll() -> Die:
            ...     return Die.uniform()
            >>> sorted(round(p, 4) for p in probability_mass_function(roll).values())
            [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]

        Must be called inside an active DPP definition.
        """
        _require_elements(cls, f"{cls.__name__}.uniform")
        require_definition(f"{cls.__name__}.uniform()")
        return cls.random([1.0] * len(cls))

    @overload
    def map(self, f, *others: Enum, data_type: None = None) -> Self: ...
    @overload
    def map(self, f, *others: Enum, data_type: type[_T]) -> _T: ...
    def map(self, f, *others: Enum,
            data_type: type[Variable] | None = None) -> Variable:
        """`f` applied to this variable, and to `others` alongside it, inside the DPP.

        `f` is an ordinary Python function of one value per variable, called once per
        combination of elements while the DPP is built; nothing about it is symbolic.
        The result type is `data_type`, which may be any `pylea.syntax.variable.Variable`
        subclass, or this variable's own type where left out; every value `f` returns
        must be one the result type holds, and `ValueError` names the offender otherwise.

            >>> from pylea import DPP, Bool, prob
            >>> class Die(Enum):
            ...     elements = [1, 2, 3, 4, 5, 6]
            >>> @DPP
            ... def big_product() -> Bool:
            ...     a, b = Die.uniform(), Die.uniform()
            ...     return a.map(lambda x, y: x * y > 20, b, data_type=Bool)
            >>> round(prob(big_product), 6)
            0.166667

        With no `others` and no `data_type` it transforms a variable within its own type:

            >>> from pylea import probability_mass_function
            >>> @DPP
            ... def raised() -> Die:
            ...     return Die.uniform().map(lambda v: min(v + 2, 6))
            >>> probability_mass_function(raised)[6]
            0.5

        The call is recorded as one table over all the inputs.  Every input must be an
        `Enum`, and the inputs and the output together may span at most 20 bits;
        `TypeError` and `ValueError` say otherwise.  A bit pattern of an input that encodes
        no element is mapped as the first element, which `Enum.match` also does.  Must be
        called inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        inputs: tuple[Enum, ...] = (self,) + others
        for i, x in enumerate(inputs):
            if not isinstance(x, Enum):
                raise TypeError(
                    f"map: every input must be an Enum variable; input {i} is a "
                    f"{type(x).__name__}.")
        if data_type is not None:
            require_variable_type(data_type, "map: data_type")
        output_type: type[Variable] = data_type if data_type is not None else type(self)

        input_types = tuple(type(x) for x in inputs)
        total_in = sum(t.nbits() for t in input_types)
        out_bits = output_type.nbits()
        if not _table_fits(total_in, out_bits):
            raise ValueError(
                f"map: the inputs and the output span {total_in} + {out_bits} bits, so its "
                f"table would have 2**{total_in + out_bits} entries, more than the "
                f"2**{_TABLE_BITS} a table may have.")

        def pattern_of(value) -> int:
            try:
                return output_type._pattern(output_type.encoding()(value))
            except (TypeError, ValueError):
                raise ValueError(
                    f"map: f returned {value!r}, which {output_type.__name__} cannot "
                    "hold.") from None

        values = [t.elements for t in input_types]
        patterns = np.fromiter(
            (pattern_of(f(*arguments)) for arguments in itertools.product(*values)),
            dtype=np.int64)
        written = getattr(f, "__name__", None)
        made = Primitive(
            input_types=input_types,
            output_types=(output_type,),
            name=f"_MAP_{output_type.__name__}_",
            prim=_Prim.from_matrix(_deterministic_table(input_types, patterns, out_bits)),
            label=None if written in (None, "<lambda>") else written,
        )
        return made(*inputs)  # type: ignore[return-value]

    @overload
    def match(self, match_dict: dict[Any, _T], data_type: None = None) -> _T: ...
    @overload
    def match(self, match_dict: dict, data_type: type[_T]) -> _T: ...
    def match(self, match_dict: dict, data_type: type[Variable] | None = None) -> Variable:
        """The value `match_dict` gives for this variable, selected inside the DPP.

        `match_dict` has one key for every element of this type, mapping it to a variable
        of the output type or to a value of that type, which becomes a constant.  This is
        how a conditional distribution is written by hand, each branch drawing its own
        value:

            >>> from pylea import DPP, Bool, prob
            >>> class Weather(Enum):
            ...     elements = ["dry", "wet"]
            >>> @DPP
            ... def grass_is_damp() -> Bool:
            ...     from pylea import flip
            ...     weather = Weather.random({"dry": 0.8, "wet": 0.2})
            ...     return weather.match({"dry": flip(0.1), "wet": flip(0.9)})
            >>> round(prob(grass_is_damp), 6)
            0.26

        The output type is `data_type` where given, and otherwise the type of the variables
        among the values, which must then all be of one type.  Raises `ValueError` for a
        missing key, for a key that is not an element, and where every value is a constant
        and `data_type` is left out; `TypeError` for a variable of a type other than the
        output type, since reading its bits as the output type would mean something else.
        Must be called inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        t = type(self)
        where = f"{t.__name__}.match"
        _require_elements(t, where)
        if not isinstance(match_dict, Mapping):
            raise TypeError(f"{where}: match_dict must be a dict from element to value; got "
                            f"a {type(match_dict).__name__}")
        elements: tuple = t.elements  # type: ignore[assignment]
        missing = object()
        chosen: list = [missing] * len(elements)
        unknown = []
        for key, value in match_dict.items():
            i = _lookup(t._index, key)  # type: ignore[arg-type]
            if i is None:
                unknown.append(key)
            elif chosen[i] is not missing:
                raise ValueError(f"{where}: {key!r} is given twice, as the element "
                                 f"{elements[i]!r}")
            else:
                chosen[i] = value
        if unknown:
            raise ValueError(f"{where}: {_listed(unknown)} "
                             f"{'is not an element' if len(unknown) == 1 else 'are not elements'}"
                             f" of {t.__name__}, whose elements are {_listed(elements)}")
        absent = [e for e, v in zip(elements, chosen) if v is missing]
        if absent:
            raise ValueError(f"{where}: no value for {_listed(absent)}; give one for every "
                             f"element of {t.__name__}")

        if data_type is not None:
            require_variable_type(data_type, f"{where}: data_type")
            output_type: type[Variable] = data_type
        else:
            found = list(dict.fromkeys(type(v) for v in chosen if isinstance(v, Variable)))
            if not found:
                raise ValueError(f"{where}: every value is a constant, so give data_type to "
                                 "name the type of the result")
            if not all(reads_as(c, found[0]) for c in found[1:]):
                raise TypeError(f"{where}: the values are variables of several types, "
                                f"{', '.join(c.__name__ for c in found)}; give data_type to "
                                "name the type of the result")
            output_type = found[0]

        leaves: list = []
        patterns: list[int] = []
        for element, value in zip(elements, chosen):
            what = f"{where}: the value for {element!r}"
            if isinstance(value, Variable):
                leaves.append(as_variable(value, output_type, what))
                continue
            try:
                encoded = output_type.encoding()(value)
            except (TypeError, ValueError) as exc:
                raise _prefixed(exc, what) from None
            leaves.append(encoded.value)
            patterns.append(output_type._pattern(encoded))

        if len(patterns) == len(leaves) and _table_fits(t.nbits(), output_type.nbits()):
            table = Primitive(
                input_types=(t,),
                output_types=(output_type,),
                name=f"_MATCH_{output_type.__name__}_",
                prim=_Prim.from_matrix(_deterministic_table(
                    (t,), np.asarray(patterns, dtype=np.int64), output_type.nbits())),
            )
            return table(self)  # type: ignore[return-value]
        # Patterns past the last element encode none, so what they select cannot be
        # observed; they take the first value.
        leaves += [leaves[0]] * ((1 << t.nbits()) - len(elements))
        return _select(self.bits(), leaves, output_type)

    @classmethod
    def cpt(
        cls,
        parent_types: Sequence[type[Variable]],
        rows: Mapping[tuple, Mapping | Sequence[float]],
    ) -> Primitive:
        """A conditional probability table as a single dense matrix.

        `parent_types` is the list of the parents' types, which may be any variable types.
        `rows` maps a tuple of parent values, one per parent in that order, to the child's
        distribution, given as `Enum.random` takes it: a dict from element to weight, or a
        list of weights in the order of `elements`.  A `Bool` parent is keyed by `True` and
        `False`, a `pylea.syntax.integer.UInt` parent by its integers, and a
        `pylea.syntax.tuple.Tuple` parent by plain tuples.  Each row is normalised.

        The result is a `pylea.syntax.dpp.Primitive` taking the parents and returning a
        variable of this type, so a definition calls it exactly as it would a `match`-based
        table.

            >>> class Weather(Enum):
            ...     elements = ["dry", "wet"]
            >>> class Grass(Enum):
            ...     elements = ["dry", "damp"]
            >>> grass_given_weather = Grass.cpt(
            ...     [Weather], {("dry",): [0.9, 0.1], ("wet",): [0.2, 0.8]})
            >>> grass_given_weather.prim.matrix.tolist()
            [[0.9, 0.2], [0.1, 0.8]]

        The matrix has a column for every bit pattern of the parents, and a pattern
        encoding no value of a parent takes the row of that parent's first value.  The
        parents and the child together may span at most 20 bits, ``2**20`` entries.
        Raises `TypeError` where `parent_types` is not a list or tuple of types, and
        `ValueError` for a row missing, a row for parent values that do not exist, a table
        too large, and every fault `Enum.random` refuses in a row.
        """
        from pylea.syntax.dpp import Primitive

        where = f"{cls.__name__}.cpt"
        _require_elements(cls, where)
        if isinstance(parent_types, type) or not isinstance(parent_types, (list, tuple)):
            shown = (parent_types.__name__ if isinstance(parent_types, type)
                     else type(parent_types).__name__)
            raise TypeError(f"{where}: parent_types must be a list of types, such as "
                            f"[{shown}]; got {shown}")
        parents = tuple(parent_types)
        for i, p in enumerate(parents):
            require_variable_type(p, f"{where}: parent_types[{i}]")
        in_bits = sum(p.nbits() for p in parents)
        out_bits = cls.nbits()
        if not _table_fits(in_bits, out_bits):
            raise ValueError(
                f"{where}: the parents and the child span {in_bits} + {out_bits} bits, so "
                f"the table would have 2**{in_bits + out_bits} entries, more than the "
                f"2**{_TABLE_BITS} a table may have.")
        if not isinstance(rows, Mapping):
            raise TypeError(f"{where}: rows must be a dict from a tuple of parent values to "
                            f"a distribution; got a {type(rows).__name__}")

        row_of_column, positions = _value_rows(parents)
        n_rows = math.prod(len(p) for p in positions)
        R = np.zeros((1 << out_bits, n_rows))
        filled = np.zeros(n_rows, dtype=bool)
        unknown = []
        for key, spec in rows.items():
            row = 0
            if isinstance(key, tuple) and len(key) == len(parents):
                for value, position in zip(key, positions):
                    j = _lookup(position, value)
                    if j is None:
                        break
                    row = row * len(position) + j
                else:
                    if filled[row]:
                        raise ValueError(f"{where}: two rows, one of them {key!r}, name the "
                                         "same parent values")
                    R[:len(cls), row] = _weights(spec, cls, f"{where}: row {key!r}")
                    filled[row] = True
                    continue
            unknown.append(key)
        if unknown:
            hint = ""
            if len(parents) == 1 and not isinstance(unknown[0], tuple):
                hint = f"; key each row by a tuple of one value, such as ({unknown[0]!r},)"
            raise ValueError(f"{where}: the rows for {_listed(unknown)} name no values of "
                             f"the parents {', '.join(p.__name__ for p in parents)}{hint}")
        if not filled.all():
            combos = itertools.product(*(list(p) for p in positions))
            absent = [c for c, f in zip(combos, filled) if not f]
            raise ValueError(f"{where}: no row for {_listed(absent)}; give one for every "
                             "combination of parent values")
        return Primitive(
            input_types=parents,
            output_types=(cls,),
            name=f"_CPT_{cls.__name__}_",
            prim=_Prim.from_matrix(R[:, row_of_column]),
        )
