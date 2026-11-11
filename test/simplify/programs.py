"""Random programs, random Boolean circuits, and a stand-in clock, shared by the
simplification tests."""
import numpy as np

from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program, intern
from pylea.ir.reference import TooLarge, semantics
from test.compile.random_programs import random_program


def prog(*bodies) -> Program:
    return Program(tuple(bodies), len(bodies) - 1)


def random_corpus(seed: int = 0, n: int = 150) -> list[Program]:
    """Small random programs over every primitive, each enumerable by the reference."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = random_program(rng, n_bodies=int(rng.integers(1, 4)), n_ops=int(rng.integers(1, 7)))
        try:
            semantics(p)
        except TooLarge:
            continue
        out.append(intern(p))
    return out


def random_circuit(rng, n_leaves: int, n_gates: int, observe: bool = False) -> Program:
    """A random circuit of gates over coins, with negations, and perhaps one observation."""
    b = BodyBuilder()
    pool = [b.emit(Prim.flip(round(rng.uniform(0.1, 0.9), 3)), ())[0] for _ in range(n_leaves)]
    for _ in range(n_gates):
        gate = Prim.and_() if rng.random() < 0.5 else Prim.or_()
        w = pool[int(rng.integers(len(pool)))]
        if rng.random() < 0.2:
            (w,) = b.emit(Prim.not_(), (w,))
        for _ in range(int(rng.integers(1, 3))):
            (w,) = b.emit(gate, (w, pool[int(rng.integers(len(pool)))]))
        pool.append(w)
    if observe:
        b.emit(Prim.observe(), (pool[int(rng.integers(len(pool)))],))
    return Program((b.finish((), (pool[-1],)),), 0)


def circuits(n: int = 60, observe: bool = False) -> list[Program]:
    return [random_circuit(np.random.default_rng(s), 4, 5, observe) for s in range(n)]


def random_mixed(rng, near_stochastic: bool = False) -> Program:
    """A random body over gates, selections of one or two results, observations, and tables.

    Its sources include written constants, flips of exactly 0 or 1, and tables reading
    nothing whose column does not sum to 1, which are the cases the rules read off a
    matrix rather than a tag, and tables reading nothing with two results, which no rule
    may treat as a flip.  Its effects include soft observations, weighting the patterns
    of one or two wires.  With `near_stochastic`, every source is a table whose column
    sums to 1 only nearly: up to round-off, above or below, as a sum of several outcomes
    can, or within 1e-6, as a table of rounded probabilities does.
    """
    b = BodyBuilder()
    pool = [b.fresh() for _ in range(int(rng.integers(0, 3)))]
    inputs = tuple(pool)

    def pick() -> int:
        return pool[int(rng.integers(len(pool)))]

    for _ in range(int(rng.integers(1, 4))):
        if near_stochastic:
            pool.append(b.emit(_near_stochastic_source(rng), ())[0])
            continue
        r = rng.random()
        if r < 0.15:
            source = Prim.const(int(rng.integers(2)), 1)
        elif r < 0.25:
            source = Prim.flip(float(rng.choice([0.0, 1.0, 0.5])))
        elif r < 0.4:
            column = rng.uniform(0.05, 1, size=(4, 1))
            pool.extend(b.emit(Prim.from_matrix(column / column.sum()), ()))
            continue
        else:
            source = Prim.flip(round(float(rng.uniform(0.1, 0.9)), 2))
        pool.append(b.emit(source, ())[0])
    for _ in range(int(rng.integers(2, 9))):
        r = rng.random()
        if r < 0.3:
            pool.append(b.emit(Prim.and_(), (pick(), pick()))[0])
        elif r < 0.6:
            pool.append(b.emit(Prim.or_(), (pick(), pick()))[0])
        elif r < 0.72:
            pool.append(b.emit(Prim.not_(), (pick(),))[0])
        elif r < 0.8:
            k = int(rng.integers(1, 3))
            pool.extend(b.emit(Prim.ite(k), tuple(pick() for _ in range(1 + 2 * k))))
        elif r < 0.84:
            b.emit(Prim.observe(), (pick(),))
        elif r < 0.89:
            k = int(rng.integers(1, 3))
            b.emit(Prim.from_matrix(rng.uniform(0.05, 1, size=(1, 1 << k))),
                   tuple(pick() for _ in range(k)))
        elif r < 0.94:
            m = rng.uniform(0.05, 1, size=(2, 2))
            pool.append(b.emit(Prim.from_matrix(m / m.sum(0)), (pick(),))[0])
        else:
            pool.append(b.emit(Prim.from_matrix(rng.uniform(0.0, 1, size=(2, 1))), ())[0])
    outputs = tuple(pool[-1 - int(rng.integers(min(len(pool), 4)))]
                    for _ in range(int(rng.integers(1, 4))))
    return Program((b.finish(inputs, outputs),), 0)


def _near_stochastic_source(rng) -> Prim:
    """A table reading nothing whose column sums to 1 up to round-off, above or below, or
    within 1e-6."""
    p = round(float(rng.uniform(0.05, 0.95)), 6)
    r = rng.random()
    if r < 0.25:
        return Prim.from_matrix([[0.0], [1.0 + 2.0 ** -52]])
    off = float(rng.choice([-2.0 ** -52, 2.0 ** -52] if r < 0.6 else [-1e-6, 1e-6]))
    return Prim.from_matrix([[1.0 - p + off], [p]])


def mixed_corpus(seed: int = 0, n: int = 150, near_stochastic: bool = False) -> list[Program]:
    """`n` programs of `random_mixed`, each enumerable by the reference."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = random_mixed(rng, near_stochastic)
        try:
            semantics(p)
        except TooLarge:
            continue
        out.append(intern(p))
    return out


def same_matrix(got: np.ndarray, want: np.ndarray) -> bool:
    """Whether two matrices have one shape and agree to round-off.

    The relative tolerance is far below that of `np.allclose` by default, 1e-5, which a
    rule changing a table's entries in their sixth digit would pass.
    """
    return got.shape == want.shape and np.allclose(got, want, rtol=1e-12, atol=1e-15)


class Clock:
    """A clock reading 0 for its first `allowed` readings and 1 after them."""

    def __init__(self, allowed: float) -> None:
        self.allowed = allowed
        self.readings = 0

    def __call__(self) -> float:
        self.readings += 1
        return 0.0 if self.readings <= self.allowed else 1.0
