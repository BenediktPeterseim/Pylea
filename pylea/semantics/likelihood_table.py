"""The semantics of a DPP, as a typed matrix.

A `LikelihoodTable` is a non-negative matrix of shape ``(2**out_bits, 2**in_bits)`` together
with the types of its inputs and outputs.  ``matrix[o, i]`` is the weight of output pattern
`o` given input pattern `i`, packed MSB first as everywhere in the IR.

The weights are unnormalised.  Where the DPP conditions, a column sums to the probability,
on that input, that every observed value is true, rather than to 1; for a DPP without inputs
this is the acceptance probability.  `probabilities` divides each column by its total, and
`normalised` returns that as a table in its own right.

A table arises either from evaluating a DPP, through `pylea.syntax.dpp.DPP.evaluated`, or
by wrapping a matrix the caller already has, such as a noisy channel or a table read from a
file.  Either way it is a `pylea.syntax.dpp.ProbabilisticCallable`, so calling it inside a
``@DPP`` definition records one assignment, whose primitive is the table's matrix.  A DPP
can therefore be evaluated once and the result reused in any number of later DPPs.

Printing a table shows the normalised weights with the bit patterns decoded back to the
values of the declared types:

    >>> from pylea import DPP, Bool, flip
    >>> @DPP
    ... def noisy_channel(x: Bool) -> Bool:
    ...     return x & flip(0.9) | ~x & flip(0.1)
    >>> print(noisy_channel.evaluated())
          | False | True
    --------------------
    False |   0.9 |  0.1
    True  |   0.1 |  0.9
"""
from __future__ import annotations

import numpy as np

from pylea.ir.prim import Prim, is_indicator, is_stochastic
from pylea.syntax.dpp import Primitive, ProbabilisticCallable
from pylea.syntax.variable import DataType

def normalised_distribution(array: np.ndarray) -> np.ndarray:
    """The flat normalised distribution of an evaluator result's array.

    Raises `ZeroDivisionError` where the total mass is zero, which for a conditioned
    DPP means its acceptance probability is zero.

        >>> normalised_distribution(np.array([[1.0], [3.0]])).tolist()
        [0.25, 0.75]
    """
    flat = np.asarray(array, dtype=float).reshape(-1)
    total = flat.sum()
    if total <= 0.0:
        raise ZeroDivisionError(
            "the distribution has total mass 0, so it cannot be normalised; the "
            "acceptance probability of this DPP is 0.")
    return flat / total


def scaled_matrix(array: np.ndarray, log_scale: float) -> np.ndarray:
    """``array * exp(log_scale)`` as a float64 array.

    Raises `ValueError` where the product does not fit float64: where the factor itself
    overflows or underflows, or where a positive entry would become zero or infinite.
    Every consumer that folds an evaluator's log scale back into an array goes through
    this one criterion.

        >>> scaled_matrix(np.array([[0.5]]), 0.0).tolist()
        [[0.5]]
        >>> scaled_matrix(np.array([[1.0]]), -1000.0)
        Traceback (most recent call last):
          ...
        ValueError: the matrix's magnitude (e**-1000.0) does not fit float64; use matrix(), which returns the array and its log scale apart.
    """
    matrix = np.asarray(array, dtype=float)
    if log_scale == 0.0:
        return matrix
    factor = np.exp(log_scale)
    scaled = matrix * factor
    if (not 0.0 < factor < np.inf or np.isinf(scaled).any()
            or bool(((matrix > 0.0) & (scaled == 0.0)).any())):
        raise ValueError(
            f"the matrix's magnitude (e**{log_scale:.1f}) does not fit float64; "
            "use matrix(), which returns the array and its log scale apart.")
    return scaled


def scaled_table(input_types, output_types, array: np.ndarray,
                 log_scale: float) -> "LikelihoodTable":
    """A table from an evaluator result, the log magnitude folded back in.

    Raises `ValueError` where ``array * exp(log_scale)`` does not fit float64, in which
    case the array and its scale must be consumed apart; the criterion is
    `scaled_matrix`'s.

        >>> scaled_table((), (), np.array([[0.5]]), 0.0).matrix.tolist()
        [[0.5]]
    """
    return LikelihoodTable(input_types, output_types, scaled_matrix(array, log_scale))


_next_table_key: int = 0


def _fresh_table_key() -> int:
    global _next_table_key
    _next_table_key += 1
    return _next_table_key - 1


class LikelihoodTable(ProbabilisticCallable):
    """The semantics of a DPP, as a non-negative matrix with typed input and output
    dimensions.

    Wrapping a matrix the caller already has is how data enters a DPP without being
    written as a DPP itself:

        >>> import numpy as np
        >>> from pylea import DPP, Bool, flip, prob
        >>> channel = LikelihoodTable((Bool,), (Bool,), np.array([[0.9, 0.2],
        ...                                                      [0.1, 0.8]]))
        >>> @DPP
        ... def received() -> Bool:
        ...     return channel(flip(0.5))
        >>> round(prob(received), 6)
        0.45

    Raises `ValueError` where `matrix` is negative anywhere, or where its shape disagrees
    with the bit widths the declared types imply.
    """

    def __init__(
            self,
            input_types: tuple[DataType, ...],
            output_types: tuple[DataType, ...],
            matrix: np.ndarray,
        ) -> None:
        matrix = np.ascontiguousarray(matrix, dtype=float)
        in_bits = sum(t.nbits() for t in input_types)
        out_bits = sum(t.nbits() for t in output_types)
        if matrix.shape != (1 << out_bits, 1 << in_bits):
            raise ValueError(
                f"types declare {in_bits} input and {out_bits} output bits, so the matrix "
                f"must be {1 << out_bits} x {1 << in_bits}; got {matrix.shape}")
        if matrix.size and matrix.min() < 0.0:
            raise ValueError("a likelihood table must be non-negative")
        matrix.flags.writeable = False
        self._input_types = tuple(input_types)
        self._output_types = tuple(output_types)
        self.matrix = matrix
        self._compiled: Primitive | None = None

    def input_types(self) -> tuple[DataType, ...]:
        return self._input_types

    def output_types(self) -> tuple[DataType, ...]:
        return self._output_types

    @property
    def probabilities(self) -> np.ndarray:
        """The matrix with each column divided by its total.

            >>> from pylea import DPP, Bool, flip
            >>> @DPP
            ... def channel(x: Bool) -> Bool:
            ...     return x & flip(0.9) | ~x & flip(0.1)
            >>> channel.evaluated().probabilities.round(4).tolist()
            [[0.9, 0.1], [0.1, 0.9]]

        A column summing to zero is left as it is rather than divided by, that being an
        input under which the observed values cannot all be true.
        """
        totals = self.matrix.sum(axis=0)
        return np.divide(self.matrix, totals, out=np.zeros_like(self.matrix),
                         where=totals > 0.0)

    def normalised(self) -> LikelihoodTable:
        """`probabilities` as a table of its own.

            >>> from pylea import DPP, Bool, flip, observe
            >>> @DPP
            ... def survivor() -> Bool:
            ...     x = flip(0.5)
            ...     observe(x | flip(0.5))
            ...     return x
            >>> t = survivor.evaluated()
            >>> round(float(t.matrix.sum()), 4), round(float(t.normalised().matrix.sum()), 4)
            (0.75, 1.0)


        The difference tells when the table is called inside another DPP.  This table
        treats the DPP it came from as an independent draw from its conditional, where the
        unnormalised one additionally weights each input by the probability, on that input,
        that every observed value is true.  Normalising
        turns "draw uniformly from 1 to 40, then keep only draws at most `n`" into "draw
        uniformly from 1 to `n`".
        """
        return LikelihoodTable(self._input_types, self._output_types, self.probabilities)

    @property
    def is_deterministic(self) -> bool:
        return is_indicator(self.matrix)

    @property
    def is_discardable(self) -> bool:
        return is_stochastic(self.matrix)

    def compiled(self) -> Primitive:
        """This table as a `pylea.syntax.dpp.Primitive`, built once and then reused.

        Calling the table inside a definition goes through this, so a table evaluated once
        can stand in for its DPP wherever that DPP would be called:

            >>> from pylea import DPP, Bool, flip, prob
            >>> @DPP
            ... def channel(x: Bool) -> Bool:
            ...     return x & flip(0.9) | ~x & flip(0.1)
            >>> table = channel.evaluated()
            >>> @DPP
            ... def sent_twice() -> Bool:
            ...     return table(table(flip(0.5)))
            >>> prob(sent_twice)
            0.5
        """
        if self._compiled is None:
            self._compiled = Primitive(
                input_types=self._input_types,
                output_types=self._output_types,
                name=f"_TABLE_{_fresh_table_key()}_",
                prim=Prim.from_matrix(self.matrix),
            )
        return self._compiled

    def __call__(self, *args):
        return self.compiled()(*args)

    def __str__(self) -> str:
        """The normalised weights, with bit patterns decoded to the declared types\' values.

        Rows are inputs and columns are outputs.  A pattern decoding to no value is left
        out, which happens for a type whose element count is not a power of two.
        """
        probabilities = self.probabilities

        def _label(idx: int, types: tuple[DataType, ...]) -> str | None:
            if not types:
                return "()"
            total_bits = sum(t.nbits() for t in types)
            bits = [bool((idx >> (total_bits - 1 - i)) & 1) for i in range(total_bits)]
            parts: list[str] = []
            offset = 0
            for t in types:
                n = t.nbits()
                try:
                    encoded = t.encoding().from_bits(*bits[offset:offset + n])
                except (ValueError, IndexError):
                    return None
                parts.append(str(encoded.value))
                offset += n
            return ", ".join(parts)

        rows, columns = self.matrix.shape
        valid_in = [(i, label) for i in range(columns)
                    if (label := _label(i, self._input_types)) is not None]
        valid_out = [(o, label) for o in range(rows)
                     if (label := _label(o, self._output_types)) is not None]

        def cell(o: int, i: int) -> str:
            return f"{probabilities[o, i]:.4g}"

        widths = [max(len(label), max((len(cell(o, i)) for i, _ in valid_in), default=0))
                  for o, label in valid_out]
        row_width = max((len(label) for _, label in valid_in), default=2)

        header = " " * row_width + " | " + " | ".join(
            label.rjust(widths[j]) for j, (_, label) in enumerate(valid_out))
        lines = [header, "-" * len(header)]
        for i, row_label in valid_in:
            cells = " | ".join(cell(o, i).rjust(widths[j]) for j, (o, _) in enumerate(valid_out))
            lines.append(row_label.ljust(row_width) + " | " + cells)
        return "\n".join(lines)
