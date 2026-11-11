"""A rare disease and two noisy tests.

The disease occurs in a fraction ``10**-4`` of the population.  A test is positive for 99% of
the patients who have the disease, its true positive rate, and for 2% of those who do not,
its false positive rate.  `has_disease` is the event that a patient has the disease, given
one positive and one negative test result:

    >>> from pylea import prob
    >>> round(prob(has_disease), 8)
    5.051e-05

The semantics of `test`, printed with one row per value of its input, whether the patient
has the disease, and one column per value of its output, whether the test is positive:

    >>> print(test.evaluated())
          | False | True
    --------------------
    False |  0.98 | 0.02
    True  |  0.01 | 0.99
"""
from pylea import DPP, Bool, flip, observe


@DPP
def test(has_it: Bool) -> Bool:
    """The event that one test is positive, given whether the patient has the disease."""
    positive_given_disease = flip(0.99)  # true positive rate
    positive_given_no_disease = flip(0.02)  # false positive rate
    true_positive = has_it & positive_given_disease
    false_positive = ~has_it & positive_given_no_disease
    return true_positive | false_positive


@DPP
def has_disease() -> Bool:
    """The event that a patient has the disease, given one positive and one negative test.

    The two results are independent given whether the patient has the disease, so that
    variable is drawn once and passed to both tests.
    """
    x = flip(1e-4)
    t1 = test(x)
    t2 = test(x)
    y = t1 & ~t2
    observe(y)
    return x
