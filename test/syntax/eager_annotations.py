"""The DPPs of `postponed_annotations.py`, without postponed annotations.

Before Python 3.14 an annotation here is evaluated where the function is defined, so a
name that does not exist fails there; from 3.14 on it is evaluated when first read.
"""

from pylea import DPP, Bool, Enum, flip


def local_parameter_type():
    """A DPP whose parameter has a type defined in the same function."""
    class Weather(Enum):
        elements = ["dry", "rain"]

    @DPP
    def wet(w: Weather) -> Bool:
        return w == "rain"

    @DPP
    def rainy() -> Bool:
        return wet(Weather.random({"dry": 0.75, "rain": 0.25}))

    return rainy


def local_return_type():
    """A DPP whose return annotation is a type defined in the same function."""
    class Payout(Enum):
        elements = [0, 10]

    @DPP
    def payout() -> Payout:
        return flip(0.5).match({True: 10, False: 0}, data_type=Payout)

    return payout


def unresolvable_return():
    """A DPP whose return annotation names nothing that exists."""
    @DPP
    def coin() -> Missing:  # type: ignore[name-defined]  # noqa: F821
        return flip(0.5)

    return coin


def unresolvable_parameter():
    """A DPP whose parameter annotation names nothing that exists."""
    @DPP
    def coin(x: Missing) -> Bool:  # type: ignore[name-defined]  # noqa: F821
        return flip(0.5)

    return coin

