# %% [markdown]
# # Fault trees
#
# [Fault tree analysis](https://en.wikipedia.org/wiki/Fault_tree_analysis) computes the
# probability that a system fails, the top-level event, from the probabilities that its
# components fail, the basic events. Events are related by logic gates: 
# AND (fails only if both child event fail), OR (fails if at least one child event failts), 
# and voting (fail when at least `k` of their child events fail) gates.
#
# In Pylea, a fault tree is a DPP that returns the top-level event as a `Bool`.  Each basic
# event is a coin flip, `flip(p)`, and each gate is `&`, `|` or `at_least`.
#
# ## A cooling system
#
# The failure of a cooling system is modelled hierarchically using fault tree analysis as follows.
# The cooling system fails if both pumps fail or the controller fails.  A pump fails if its 
# motor or the power supply fails, and both pumps share one power supply.  The controller 
# fails if at least two of its three sensors fail.  The probabilities are for illustration.

# %%
from pylea import DPP, Bool, at_least, flip, prob


@DPP
def cooling_fails() -> Bool:
    # basic events
    power = flip(0.001)
    motor_a, motor_b = flip(0.01), flip(0.01)
    sensors = [flip(0.02) for _ in range(3)]
    # gates
    pump_a = motor_a | power
    pump_b = motor_b | power
    controller = at_least(2, *sensors)
    return (pump_a & pump_b) | controller


prob(cooling_fails)

# %% [markdown]
# The diagram of the DPP is the fault tree.  The power supply is one basic event read by
# both pumps, so the two pumps do not fail independently.

# %%
cooling_fails.diagram()

# %% [markdown]
# ## The Radio Block Centre
#
# The Radio Block Centre (RBC) of the European Train Control System computes the movement
# authorities of trains.  The fault tree and the failure rates are from [1], as published
# in the FFORT benchmark suite [2]; repair is left out.  A component with failure rate `λ`
# has failed after `t` hours with probability `1 - exp(-λ t)`.  `rbc` returns the failure
# events of the five subsystems rather than only the top-level event, so that the same DPP
# also answers which subsystem caused a failure.

# %%
import math

import matplotlib.pyplot as plt

from pylea import observe

RATES = {  # failures per hour
    "power supply": 1.8182e-5, "WAN card": 2.5e-6, "bus": 4.4444e-6,
    "GSM-R card": 5.7078e-6, "CPU board": 7.4074e-6, "FPGA": 3.003e-9,
}
SUBSYSTEMS = ["power", "WAN interface", "system bus", "GSM-R interface", "processing"]


def rbc(hours: float) -> DPP:
    """Whether each subsystem of the RBC has failed after `hours` of operation."""

    def failed(component: str) -> Bool:
        return flip(1 - math.exp(-RATES[component] * hours))

    @DPP
    def subsystems() -> tuple[Bool, Bool, Bool, Bool, Bool]:
        power = failed("power supply") & failed("power supply") & failed("power supply")
        wan = failed("WAN card") & failed("WAN card")
        bus = failed("bus") & failed("bus")
        gsmr = failed("GSM-R card") & failed("GSM-R card")
        cpus = at_least(2, failed("CPU board"), failed("CPU board"), failed("CPU board"))
        voter = failed("FPGA") & failed("FPGA")
        return power, wan, bus, gsmr, cpus | voter

    return subsystems


def rbc_fails(hours: float) -> DPP:
    """The event that the RBC has failed after `hours`."""
    subsystems = rbc(hours)

    @DPP
    def fails() -> Bool:
        return Bool.any(subsystems())

    return fails


YEAR = 24 * 365
prob(rbc_fails(YEAR))

# %%
years = [k / 4 for k in range(41)]
plt.plot(years, [prob(rbc_fails(y * YEAR)) for y in years])
plt.xlabel("years without repair")
plt.ylabel("P(RBC failed)")
plt.show()

# %% [markdown]
# Which subsystem has failed, given that the RBC failed within a year; `observe`
# conditions on the top-level event:


# %%
def cause(name: str, hours: float) -> DPP:
    subsystems = rbc(hours)

    @DPP
    def failed() -> Bool:
        state = subsystems()
        observe(Bool.any(state))
        return state[SUBSYSTEMS.index(name)]

    return failed


for name in SUBSYSTEMS:
    print(f"{name:<16} {prob(cause(name, YEAR)):.4f}")

# %%
rbc(YEAR).diagram()

# %% [markdown]
# ## Fault trees in the `.dft` format
#
# `dft_to_python` turns a fault tree in the Galileo `.dft` format, here the RBC as the
# file [rbc.dft](rbc.dft), into the source of a DPP of the form above.  Failure rates
# become probabilities at the mission time `lambda_t`, here one year in hours.

# %%
from pathlib import Path

from pylea.utility.dft_to_dpp import dft_to_python

print(dft_to_python(Path("rbc.dft").read_text(), lambda_t=YEAR))

# %% [markdown]
# The result is ordinary Pylea code, so it can be copied and edited like any other DPP:
# to change the tree, or to return the subsystem events instead of `System` and then
# condition on the top-level event, as `cause` does above.
#
# ## References
#
# [1] F. Flammini, N. Mazzocca, M. Iacono, S. Marrone.  Using repairable fault trees for
# the evaluation of design choices for critical repairable systems.  *HASE*, 2005.
#
# [2] FFORT, <https://dftbenchmarks.utwente.nl/ffort>, model `rbc`.
