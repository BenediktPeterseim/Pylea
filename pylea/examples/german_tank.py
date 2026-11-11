"""The German tank problem: a population size estimated from serial numbers.

Captured equipment carries sequential serial numbers, and treating the unknown fleet size
as a random variable gives an estimate of it from those numbers alone.  The example builds
that estimate in six steps:

  1. one `pylea.syntax.enum.Enum` type for the integers 1 to `N_MAX`, serving both as the
     fleet sizes and as the serial numbers;
  2. `uniform_serial`, the DPP of one capture: draw a serial uniformly, then condition on
     it being at most the fleet size;
  3. `prior`, a uniform draw over the fleet sizes;
  4. `update`, which wraps a belief in one more layer of conditioning on an observed serial;
  5. `fit`, which applies `update` once per observation, threading the posterior through;
  6. `probability_mass_function` on the result.

Each capture is modelled as an independent uniform draw, so P(S = s | N = n) is 1/n.  The
exact likelihood without replacement differs by a factor that approaches 1 as the number of
captures falls below the fleet size, and the two give the same estimate.

    >>> posterior = fit([15, 14, 3])
    >>> pmf = probability_mass_function(posterior)
    >>> max(pmf, key=pmf.get)
    15
    >>> round(sum(p for n, p in pmf.items() if n > 30), 3)
    0.112
"""
import numpy as np

from pylea import DPP, Enum, observe, probability_mass_function

N_MAX = 40      # the largest fleet size the prior admits


class N(Enum):
    """The integers from 1 to `N_MAX`, serving as fleet sizes and as serial numbers."""
    elements = list(range(1, N_MAX + 1))

@DPP
def uniform_serial(n: N) -> N:
    """One serial number, drawn uniformly from a fleet of size `n`."""
    s = N.uniform()
    observe(s <= n)
    return s


# One capture is an independent draw from P(S | N = n), which is `uniform_serial` given its
# observation: `normalised` turns "uniform up to N_MAX, conditioned on being at most n" into
# "uniform up to n".  A table is callable inside a DPP, so every update below reuses this
# one evaluated table rather than re-deriving it.
serial_given_fleet = uniform_serial.evaluated().normalised()


@DPP
def prior() -> N:
    """A flat prior, every fleet size up to `N_MAX` being equally probable."""
    return N.uniform()


def update(belief, s: int):
    """`belief` conditioned on having observed the serial number `s`."""
    @DPP
    def posterior() -> N:
        n = belief()                    # the fleet size, under the current belief
        obs = serial_given_fleet(n)     # one capture from a fleet of that size
        observe(obs == s)
        return n

    return posterior


def fit(observed_serials: list[int]):
    """The posterior over the fleet size after observing `observed_serials`.

    After k observations the result is proportional to ``P(N) / N**k`` for every fleet size
    at least as large as the largest serial seen, and zero below it.
    """
    belief = prior
    for s in observed_serials:
        belief = update(belief, s)
    return belief


def summarise(belief) -> None:
    """Nothing, having printed the most probable fleet size, the median, P(N > 30) and a
    bar chart of the posterior."""
    pmf    = probability_mass_function(belief)
    ns     = np.array(list(pmf.keys()))
    ps     = np.array([float(p) for p in pmf.values()])

    map_n  = ns[np.argmax(ps)]
    median = ns[np.searchsorted(np.cumsum(ps), 0.5)]
    p_gt_30 = ps[ns > 30].sum()

    print(f"  Most probable     N = {map_n}")
    print(f"  Posterior median  N = {median}")
    print(f"  P(N > 30)           = {p_gt_30:.3f}")
    print()

    for n, p in zip(ns, ps):
        if p >= 0.001:
            bar = "█" * round(40 * p / ps.max())
            print(f"  {n:3d}  {p:.4f}  {bar}")


def example():
    """Nothing, having fitted and printed the posterior for two sets of serial numbers.

    Both are drawn from a fleet of 20.
    """
    data_small = [15, 14, 3]
    print("=" * 55)
    print(f"k = {len(data_small)} captures,  serials = {data_small}")
    print(f"(True fleet size: N = 20)")
    print("=" * 55)
    posterior_small = fit(data_small)
    summarise(posterior_small)

    # The standard minimum-variance unbiased estimate, for comparison.
    m, k = max(data_small), len(data_small)
    print(f"\n  Unbiased point estimate:  N = {m + m // k - 1}")

    data_large = [15, 14, 3, 6, 2, 10, 5, 16, 8, 1, 4, 19]
    print()
    print("=" * 55)
    print(f"k = {len(data_large)} captures,  serials = {data_large}")
    print(f"(True fleet size: N = 20)")
    print("=" * 55)
    posterior_large = fit(data_large)
    summarise(posterior_large)


if __name__ == "__main__":
    example()
