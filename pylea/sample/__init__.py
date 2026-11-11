"""The sampling engine: sequential Monte Carlo estimation with certified error bounds.

This package estimates a DPP's semantics by running particles through the unfolded
DPP.  Every answer it returns is certified: the reported interval contains the true
value with probability at least the requested confidence, unconditionally, for any
particle count, schedule and stopping time.  Efficiency depends on the DPP; validity
never does.  All certificates are anytime-valid, built from confidence sequences, so
stopping at an arbitrary moment, as a `max_time` deadline does, and reporting the current
interval is sound.

Two execution modes, chosen per query by `pylea.sample.pilot`:

  * **i.i.d. mode** (no resampling): every particle is an independent draw, and
    certification happens at the particle level.  With 0/1 weights the draws are Bernoulli
    and the certificate is a Clopper-Pearson sequence; with soft weights they are values
    in [0, 1] and the certificate is empirical-Bernstein.  Certifying such a DPP at
    the sweep level instead would lose a factor of the sweep size, which is why the
    certifier dispatches on the mode.
  * **sweep mode** (resampling at low effective sample size): required once some
    conditioning stage would starve the i.i.d. mode of survivors.  Sweeps are i.i.d.
    values in [0, 1], certified by a betting confidence sequence at the sweep level.
    Any [0, 1] certifier has a range floor: a mass of `rel_err * p` cannot be excluded on
    fewer than about ``L / (rel_err * p)`` samples, `L` the certificate's log factor, so
    certification below the floor is impossible in this mode however small the variance.
    Sweep mode's place is survival and the anytime bounds that come with it: on every
    query both modes can run, the mode arithmetic prices the i.i.d. mode lower, a
    conclusion reached empirically across the constructed and corpus instances.

Cost, in order-of-magnitude terms with `q` the smallest mass the query certifies: the
i.i.d. mode needs about ``L / (rel_err**2 * q)`` particles, the sweep mode about
``R * L / rel_err**2`` particles plus the floor above, where `R` sums ``(1 - p_k) / p_k``
over the conditioning stages.  Consequences a caller should know:

  * The supported regime is moderate probabilities at any width.  The exact engine's
    contract, relative precision at any magnitude, is not achievable by sampling at
    bounded cost; a probability far below `rel_err` should be answered exactly, and the
    portfolio algorithm routes it there.
  * For a conditioned DPP the certified quantity is a quotient of two masses estimated
    from the same samples; the interval quotient with a union bound over the two
    certificates makes no independence assumption.
  * A relative-error guarantee over every entry of a table is priced by the rarest entry,
    which can be doubly exponentially small in the size of the DPP, so the table contract is
    per-entry: each entry is estimated to `rel_err` relative error or certified to lie
    below `floor`, jointly at the requested confidence.
  * The mean of a value with a rare large contribution is expensive to certify: the cost
    is proportional to the second moment over the squared mean of the value rescaled to
    [0, 1].  This is a property of the estimation problem, not of the implementation.

Modules: `pylea.sample.sweep` runs particles on byte planes and `pylea.sample.packed` at
64 per machine word, `pylea.sample.certify` holds the confidence sequences and the
failure-budget ledger, `pylea.sample.schedule` rewrites the DPP for lower variance,
`pylea.sample.collapse` splices exactly evaluated narrow bodies in as tables,
`pylea.sample.pilot` prices the modes and picks one, `pylea.sample.run` drives a query to
certification, `pylea.sample.query` connects the public entry points to all of it,
`pylea.sample.portfolio`, the portfolio algorithm, decides between this engine and the
exact engine, `pylea.sample.estimate` holds the missed-target warning and the sampled
table, and `pylea.sample.cost_model` keeps every tunable constant in one place.
"""
