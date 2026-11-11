# Pylea

Discrete probabilistic programming in Python.

Pylea computes probabilities, expected values and probability mass functions in
probabilistic models in which every variable takes only finitely many values.  A model is
written as a discrete probabilistic program (DPP): a Python function decorated with
`@DPP`, in which coin flips are drawn with `flip` and events are conditioned on with
`observe`.

## Installation

```
pip install pylea
```

Pylea requires Python 3.12 or later.  `pip install "pylea[gpu]"` also installs PyTorch,
for evaluation on a GPU, and `pip install "pylea[search]"` installs kahypar, optuna and
cmaes, which cotengra uses where installed when it searches for a contraction order for the
costliest programs (macOS and Linux only).

## Example

A disease occurs in a fraction 10⁻⁴ of the population.  A test for it is positive for 99%
of the patients who have the disease and for 2% of those who do not.  The probability that
a patient has the disease, given one positive and one negative test result:

```python
from pylea import DPP, Bool, flip, observe, prob


@DPP
def test(has_it: Bool) -> Bool:
    positive_given_disease = flip(0.99)     # true positive rate
    positive_given_no_disease = flip(0.02)  # false positive rate
    return (has_it & positive_given_disease) | (~has_it & positive_given_no_disease)


@DPP
def has_disease() -> Bool:
    x = flip(1e-4)                          # prevalence
    observe(test(x) & ~test(x))             # one positive, one negative result
    return x


print(prob(has_disease))                    # 5.0512703945042225e-05
```

## Documentation

<https://benediktpeterseim.github.io/Pylea/>

## Licence

MIT; see [LICENSE](https://github.com/BenediktPeterseim/Pylea/blob/main/LICENSE).
