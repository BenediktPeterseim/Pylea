"""Finite enumerated types.

An `Enum` subclass declares an ordered sequence of `elements`, which map to consecutive bit
patterns, MSB first, in as few bits as hold them all.  Where the count is not a power of
two, some patterns decode to no element.

    >>> from pylea import DPP, probability_mass_function
    >>> class Colour(Enum):
    ...     elements = ["R", "G", "B"]
    >>> @DPP
    ... def draw() -> Colour:
    ...     return Colour.random({"R": 0.5, "G": 0.25, "B": 0.25})
    >>> probability_mass_function(draw)
    {'R': 0.5, 'G': 0.25, 'B': 0.25}

`Enum.random` draws from a distribution over the elements, as one column of weights, and
gives a pattern decoding to no element weight zero.  `Enum.uniform` is that over all
elements equally.

`Enum.match` selects a value per element.  It builds a balanced tree of one-bit selections,
``2**k - 1`` of them over `m` bits each, in place of the ``2**m x 2**(k + 2**k * m)`` matrix
a single primitive over the same interface would carry.  The tree needs one value per bit
pattern, so a pattern decoding to no element repeats the first value; nothing can select it.

`Enum.cpt` builds a conditional probability table as one dense matrix, a column per parent
configuration, with equal rows sharing a column.  A parent state that no bit pattern encodes
repeats the nearest row that one does, which adds no column and agrees with what
`Enum.match` selects there.

Probabilities are held as floats throughout, so anything a double represents survives.  A
distribution normalising to zero at a positive entry is refused rather than accepted, since
that would turn an improbable outcome into an impossible one.
"""
from __future__ import annotations
import math
import numpy as np
from abc import ABCMeta
from pylea.syntax.dpp import Bool
from pylea.syntax.encoding import Encoding, BitEncoded
from pylea.syntax.variable import Variable
from pylea.ir.prim import Prim as _Prim
from typing import (
    TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Mapping, Sequence, Self, TypeVar,
    overload,
)


def _distribution(raw: Sequence[float], where: str) -> list[float]:
    """`raw` normalised to sum to 1, as floats.

    Any probability a double can represent survives, so 1e-300 stays 1e-300, as it does in
    `pylea.syntax.flip.flip`.  Raises `ValueError` for a negative entry, for an all-zero
    input, and for a positive probability that normalising would turn into zero, since
    that would make an improbable outcome an impossible one with no warning.
    """
    for i, p in enumerate(raw):
        if p < 0:
            raise ValueError(f"{where}: probability {p!r} at position {i} is negative.")
    total = math.fsum(raw)
    if total <= 0.0:
        raise ValueError(f"{where}: all probabilities are zero, so there is nothing to scale.")
    out = [p / total for p in raw]
    for p, q in zip(raw, out):
        if p > 0.0 and q == 0.0:
            raise ValueError(
                f"{where}: probability {p!r} underflows to 0 once normalised by {total!r}, "
                f"and would be indistinguishable from an impossible outcome.")
    return out

if TYPE_CHECKING:
    from pylea.syntax.dpp import Primitive

_T = TypeVar('_T', bound=Variable)


class EnumMeta(ABCMeta):
    elements: Sequence[Any] | None

    def __iter__(cls) -> Iterator:
        if cls.elements is None:
            raise TypeError(f"Cannot iterate over abstract Enum class {cls.__name__!r}")
        return iter(cls.elements)

    def __len__(cls) -> int:
        return len(cls.elements) if cls.elements is not None else 0


def _select(bits, values, output_type):
    """`values[i]`, where `i` is `bits` read as an unsigned integer, MSB first.

    Built as a balanced tree of one-bit selections, the leading bit choosing between the
    halves of `values` and each half being selected the same way.  The tree holds
    ``2**k - 1`` selections of `m` bits each, against the ``2**m x 2**(k + 2**k * m)``
    matrix a single primitive over the same interface would carry.
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

    A subclass declares its values by setting the class variable `elements` to an ordered
    sequence of distinct values.  The encoding follows: values map to consecutive unsigned
    integers, held in ``ceil(log2(len(elements)))`` bits, MSB first.

        >>> class Colour(Enum):
        ...     elements = ["R", "G", "B"]
        >>> Colour.nbits(), len(Colour)
        (2, 3)

    Where the number of elements is not a power of two, some patterns decode to no value;
    `Enum.random` gives those weight zero.  Instantiating a subclass that has not set
    `elements` raises `NotImplementedError`.
    """

    elements: ClassVar[Sequence[Any] | None] = None
    # The built encoding, stored on the class that built it and read from its own
    # namespace only, so a subclass declaring its own parameters never inherits it.
    _encoding: ClassVar[Encoding | None] = None

    def __init__(self, *xs: Bool, name: str | None = None) -> None:
        if type(self).elements is None:
            raise NotImplementedError(
                f"Cannot instantiate subclass \"{type(self).__name__}\" of Enum; "
                "class variable `elements` unspecified."
            )
        super().__init__(*xs, name=name)

    @classmethod
    def encoding(cls) -> Encoding:
        cached = vars(cls).get("_encoding")
        if cached is not None:
            return cached
        if cls.elements is None:
            raise NotImplementedError(
                f"Cannot get encoding for \"{cls.__name__}\"; "
                "class variable `elements` unspecified."
            )
        if not isinstance(cls.elements, Iterable):
            raise ValueError(
                f"Class variable `elements` of \"{cls.__name__}\" must be Iterable."
            )

        elements = list(cls.elements)
        n = len(elements)
        if n == 0:
            raise ValueError("Enum `elements` must be non-empty.")
        nbits = max(1, math.ceil(math.log2(n))) if n > 1 else 1
        base = type(elements[0])

        class EncodedEnum(BitEncoded):

            def __init__(self, v) -> None:
                if v not in elements:
                    raise ValueError(f"Value {v!r} is not in elements.")
                self._value = v

            @property
            def value(self):
                return self._value

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
                index = sum(int(bit) << (nbits - 1 - i) for i, bit in enumerate(bits))
                if index >= n:
                    raise ValueError(
                        f"Bit pattern encodes index {index}, out of range for {n} elements."
                    )
                return EncodedEnum(elements[index])

            def to_bits(self) -> tuple[bool, ...]:
                index = elements.index(self._value)
                return tuple(bool((index >> (nbits - 1 - i)) & 1) for i in range(nbits))

        cls._encoding = EncodedEnum
        return EncodedEnum

    @classmethod
    def random(cls, probs: Mapping[object, float]) -> Self:
        """A random variable over this type, distributed according to `probs`.

        `probs` maps an element to its probability.  An omitted element has probability
        zero, and probabilities that do not sum to one are normalised, so a table of counts
        may be given directly:

            >>> from pylea import DPP, probability_mass_function
            >>> class Colour(Enum):
            ...     elements = ["R", "G", "B"]
            >>> @DPP
            ... def from_counts() -> Colour:
            ...     return Colour.random({"R": 20, "G": 10, "B": 10})
            >>> probability_mass_function(from_counts)
            {'R': 0.5, 'G': 0.25, 'B': 0.25}

        Must be called inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive

        if cls.elements is None:
            raise NotImplementedError(
                f"Cannot get encoding for \"{cls.__name__}\"; "
                "class variable `elements` unspecified."
            )
        if not isinstance(cls.elements, Iterable):
            raise ValueError(
                f"Class variable `elements` of \"{cls.__name__}\" must be Iterable."
            )

        elements = list(cls.elements)
        n = len(elements)

        scaled = _distribution([float(probs.get(elem, 0.0)) for elem in elements],
                               f"{cls.__name__}.random")

        nbits = cls.nbits()
        size = 2 ** nbits  # may be larger than n if n is not a power of 2

        # Map each bit-pattern index to its probability; unused patterns get 0
        index_probs = [scaled[i] if i < n else 0.0 for i in range(size)]

        compiled = Primitive(
            input_types=tuple(),
            output_types=(cls,),
            name="_RAND_" + cls.__name__,
            # A distribution over k bits is one column of ``2**k`` weights, small enough to
            # hold densely rather than build as a circuit.
            prim=_Prim.from_matrix(np.asarray([[float(p)] for p in index_probs])),
        )
        return compiled()

    @classmethod
    def uniform(cls) -> Self:
        """A random variable drawn uniformly from this type's elements.

        The same as ``cls.random({e: 1.0 for e in cls.elements})``.
        `pylea.syntax.variable.Variable.uniform_bits` draws over bit patterns instead, which
        differs where the element count is not a power of two.

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
        if cls.elements is None:
            raise NotImplementedError(
                f"Cannot create uniform variable for \"{cls.__name__}\"; "
                "class variable `elements` unspecified."
            )
        elements = list(cls.elements)
        return cls.random({e: 1.0 for e in elements})

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

        The call is recorded as one dense 0/1 matrix over all the inputs, so the
        combined width is bounded: wide inputs, such as a `pylea.syntax.integer.UInt` of
        many bits, need a circuit written by hand instead, and `ValueError` says so.
        Each input must be an `Enum`; a bit pattern of an input that encodes no element
        is mapped as the first element, which `Enum.match` also does.  Must be called
        inside an active DPP definition.
        """
        from pylea.syntax.dpp import Primitive, _MAX_FACTOR_ENTRIES

        inputs: tuple[Enum, ...] = (self,) + others
        for i, x in enumerate(inputs):
            if not isinstance(x, Enum):
                raise TypeError(
                    f"map: every input must be an Enum variable; input {i} is "
                    f"{type(x).__name__}.")
        output_type: type[Variable] = data_type if data_type is not None else type(self)

        in_bits = [type(x).nbits() for x in inputs]
        total_in = sum(in_bits)
        out_bits = output_type.nbits()
        if (1 << (total_in + out_bits)) > _MAX_FACTOR_ENTRIES:
            raise ValueError(
                f"map: the inputs and output span {total_in} + {out_bits} bits, whose "
                f"dense matrix would exceed {_MAX_FACTOR_ENTRIES} entries; write the "
                "function as a circuit instead.")

        values = [list(type(x).elements or ()) for x in inputs]
        encode = output_type.encoding()

        def pattern_of(value) -> int:
            try:
                bits = encode(value).to_bits()  # type: ignore[call-arg]
            except (TypeError, ValueError):
                raise ValueError(
                    f"map: f returned {value!r}, which {output_type.__name__} cannot "
                    "hold.")
            return sum(int(b) << (len(bits) - 1 - i) for i, b in enumerate(bits))

        M = np.zeros((1 << out_bits, 1 << total_in))
        for column in range(1 << total_in):
            at, arguments = column, []
            for k, elems in zip(reversed(in_bits), reversed(values)):
                index = at & ((1 << k) - 1)
                at >>= k
                # A pattern encoding no element selects the first, as `match` does.
                arguments.append(elems[index] if index < len(elems) else elems[0])
            arguments.reverse()
            M[pattern_of(f(*arguments)), column] = 1.0

        written = getattr(f, "__name__", None)
        made = Primitive(
            input_types=tuple(type(x) for x in inputs),
            output_types=(output_type,),
            name=f"_MAP_{output_type.__name__}_",
            prim=_Prim.from_matrix(M),
            label=None if written in (None, "<lambda>") else written,
        )
        return made(*inputs)  # type: ignore[return-value]

    @overload
    def match(self, match_dict: dict[Any, _T], data_type: None = None) -> _T: ...
    @overload
    def match(self, match_dict: dict, data_type: type[_T]) -> _T: ...
    def match(self, match_dict: dict, data_type: type[Variable] | None = None) -> Variable:
        """The value `match_dict` gives for this variable, selected inside the DPP.

        `match_dict` needs a key for every element of this type, mapping it to an output
        `Variable` or to a constant of the output type's base type, which is wrapped with
        `pylea.syntax.variable.Variable.const`.  This is how a conditional distribution is
        written by hand, each branch drawing its own value:

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

        The output type comes from the first `Variable` among the values, or from
        `data_type` where given.  Raises `ValueError` for a missing key, or where the output
        type cannot be determined.  Must be called inside an active DPP definition.
        """
        elems = type(self).elements
        if elems is None:
            raise NotImplementedError(
                f"Cannot get encoding for \"{type(self).__name__}\"; "
                "class variable `elements` unspecified."
            )
        if not isinstance(elems, Iterable):
            raise ValueError(
                f"Class variable `elements` of \"{type(self).__name__}\" must be Iterable."
            )

        elements = list(elems)
        n = len(elements)
        k = type(self).nbits()
        size = 2 ** k

        output_type = data_type
        if output_type is None:
            for v in match_dict.values():
                if isinstance(v, Variable):
                    output_type = type(v)
                    break
        if output_type is None:
            raise ValueError(
                "Cannot infer output type from match values; "
                "please provide data_type explicitly."
            )

        def to_variable(val) -> Variable:
            if isinstance(val, Variable):
                return val
            return output_type.const(val)

        # Indices below `n` are the elements.  The rest are patterns no element encodes to,
        # so what they select cannot be observed; they take the first value.
        value_vars: list[Variable] = []
        for i in range(n):
            elem = elements[i]
            if elem not in match_dict:
                raise ValueError(f"Missing key {elem!r} in match dict.")
            value_vars.append(to_variable(match_dict[elem]))
        for _ in range(size - n):
            value_vars.append(value_vars[0])

        return _select(self.bits(), value_vars, output_type)

    @classmethod
    def cpt(
        cls,
        parent_types: Sequence[type[Enum]],
        rows: Mapping[tuple, Mapping | list],
    ) -> Primitive:
        """A conditional probability table as a single dense matrix.

        `parent_types` is the ordered sequence of parent `Enum` subclasses.  `rows` maps a
        tuple of parent states, one per parent in that order, to the child distribution,
        given either as a dict from child state to probability, missing states counting as
        zero, or as a list of probabilities indexed by position in `cls.elements`.

        Each row is normalised to sum to 1.  Rows carrying the same distribution share one
        column, so a table with many parent configurations but few distinct rows stays
        cheap.  A parent state that no bit pattern encodes repeats the nearest row that one
        does, adding no column and matching what `Enum.match` does there.  Raises `KeyError`
        for a missing row.

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
        """
        from pylea.syntax.dpp import Primitive

        assert cls.elements is not None
        elements = list(cls.elements)
        n = len(elements)
        m = cls.nbits()
        size_child = 2 ** m

        parent_types = tuple(parent_types)
        parent_bits = [p.nbits() for p in parent_types]
        parent_n_values = [p.encoding().n_values() for p in parent_types]
        parent_elements: list[list] = []
        for p in parent_types:
            assert p.elements is not None
            parent_elements.append(list(p.elements))
        K = sum(parent_bits)

        def _probs_key(prob_spec) -> tuple:
            """One row as a normalised tuple of floats, padded to the child's bit-width.

            Also the cache key below, so two rows with the same distribution share a column.
            """
            if isinstance(prob_spec, list):
                raw = {elements[i]: prob_spec[i] for i in range(min(len(prob_spec), n))}
            else:
                raw = dict(prob_spec)
            scaled = _distribution([float(raw.get(elem, 0.0)) for elem in elements],
                                   f"{cls.__name__}.cpt")
            return tuple(scaled) + (0.0,) * (size_child - n)

        # One column per parent configuration, built once per distinct distribution.
        filler_key: tuple | None = None
        M = np.empty((size_child, 1 << K))
        _col_cache: dict[tuple, np.ndarray] = {}

        def _column(pk: tuple) -> np.ndarray:
            v = _col_cache.get(pk)
            if v is None:
                # `_probs_key` has already normalised, so this is the key as an array.
                v = _col_cache[pk] = np.asarray(pk, dtype=float)
            return v

        for i in range(1 << K):
            # Decode each parent's state index out of the MSB-first index `i`.
            state_indices = []
            remaining = i
            for k_p in reversed(parent_bits):
                state_indices.append(remaining & ((1 << k_p) - 1))
                remaining >>= k_p
            state_indices.reverse()  # now state_indices[j] for parent j

            # A state index at or above a parent's element count is a pattern no value
            # encodes to.
            reachable = all(
                si < nv for si, nv in zip(state_indices, parent_n_values)
            )

            if reachable:
                parent_key = tuple(
                    parent_elements[j][state_indices[j]]
                    for j in range(len(parent_types))
                )
                row = rows.get(parent_key)
                if row is None:
                    raise KeyError(
                        f"Enum.cpt: missing row for parent config {parent_key!r} "
                        f"in CPT for {cls.__name__}."
                    )
                key = _probs_key(row)
            else:
                # Replace each out-of-range state index with 0 to find the nearest row a
                # value does encode to, which is what `Enum.match` selects there.  Reusing
                # an existing row also keeps the column count down.
                corrected = [
                    si if si < nv else 0
                    for si, nv in zip(state_indices, parent_n_values)
                ]
                corrected_key = tuple(
                    parent_elements[j][corrected[j]]
                    for j in range(len(parent_types))
                )
                fallback = rows.get(corrected_key)
                if fallback is None:
                    if filler_key is None:
                        filler_key = _probs_key(next(iter(rows.values())))
                    key = filler_key
                else:
                    key = _probs_key(fallback)

            M[:, i] = _column(key)

        return Primitive(
            input_types=parent_types,
            output_types=(cls,),
            name=f"_CPT_{cls.__name__}_",
            prim=_Prim.from_matrix(M),
        )
