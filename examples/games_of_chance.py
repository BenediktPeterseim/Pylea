# %% [markdown]
# # Games of chance
#
# ## The Monty Hall problem
#
# A car is behind one of three doors, each equally likely. You initially pick door 1. The
# host opens a door that is neither your choice nor the one with the car, choosing uniformly
# where two qualify, and is seen to open door 3. What is the probability that if you switch
# your choice to door 2, you win the car?

# %%
from pylea import DPP, Bool, Enum, observe, prob


# There are three doors, labelled 1 to 3.
class Door(Enum):
    elements = [1, 2, 3]


# Your chosen door, initially door 1.
chosen_door = 1


@DPP
def door_that_host_opens(door_with_car: Door) -> Door:
    """A door that is neither the player's nor the door with the car behind it,
    uniformly at random."""
    return door_with_car.match({
        door: Door.random({
            d: 1 for d in Door              # Give equal weight to all doors that are neither
            if d not in (chosen_door, door) # the chosen one nor the one with the car.
        })
        for door in Door
    })


@DPP
def win_car() -> Bool:
    # Prior to the show, a car (the prize) is placed behind an unknown door.
    door_with_car = Door.uniform()
    # Using his knowledge of where the car is placed, the host reveals a door
    # that is neither your choice, nor does it have the car behind it.
    opened_door = door_that_host_opens(door_with_car)
    # You observe that the host has opened door 3.
    observe(opened_door == 3)
    # You switch our choice from door 1 to the only remaining choice, door 2.
    chosen_door = 2
    # The event that you win, i.e. your chosen door is the door with the car:
    return door_with_car == chosen_door


prob(win_car)

# %% [markdown]
# ## Texas hold'em
#
# The probability that a hand beats one opponent at showdown, every unseen card dealt at
# random.  The rules of the game are written as a DPP in `pylea.examples.holdem`.  The
# answer is an estimate with confidence intervals; `low` and `high` bound it at confidence
# 0.95.

# %%
from pylea.examples.holdem import hero_wins

for hand, board in [(("Ah", "Ad"), ()),
                    (("7c", "2d"), ()),
                    (("Kh", "Qh"), ("Jh", "Th", "2c"))]:
    p = prob(hero_wins(hand, board=board), rel_err=0.05, seed=0)
    print(f"{' '.join(hand)} on [{' '.join(board)}]:  {p:.3f}  ({p.low:.3f} to {p.high:.3f})")
