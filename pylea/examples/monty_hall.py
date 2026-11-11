"""The Monty Hall problem: switching wins the car with probability 2/3.

A car is behind one of three doors, each equally likely.  The player picks door 1.  The
host opens a door that is neither the player's nor the car's, choosing uniformly where two
qualify, and is seen to open door 3.  `car_behind_door_2` is the event that the car is
behind door 2, the door the player can switch to:

    >>> from pylea import prob
    >>> round(prob(car_behind_door_2), 6)
    0.666667

`host_opens` is the host's rule, with one row per position of the car:

    >>> print(host_opens.evaluated())
      | 1 |   2 |   3
    -----------------
    1 | 0 | 0.5 | 0.5
    2 | 0 |   0 |   1
    3 | 0 |   1 |   0
"""
from pylea import DPP, Bool, Enum, observe


class Door(Enum):
    elements = [1, 2, 3]


PLAYERS_DOOR = 1


@DPP
def host_opens(car: Door) -> Door:
    """A door that is neither the player's nor the car's, uniformly at random."""
    return car.match({
        cars_door: Door.random({door: 1 for door in Door
                                if door not in (PLAYERS_DOOR, cars_door)})
        for cars_door in Door
    })


@DPP
def car_behind_door_2() -> Bool:
    """The event that the car is behind door 2, given that the host opens door 3."""
    car = Door.uniform()
    observe(host_opens(car) == 3)
    return car == 2
