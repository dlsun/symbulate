from .index_sets import Naturals
from .random_variables import RV
from .result import TimeFunction, is_scalar


class RandomProcess(RV):
    """Defines a random process.

    A random process defines a random variable X(t)
    for each time t in an collection of times
    (called an index set).

    Args:
      prob_space (ProbabilitySpace): the underlying
        probability space for the random process.
      index_set (IndexSet): the index set for the
        random process. (By default, the index set
        is the natural numbers 0, 1, 2, 3, ....)
      func: a function that takes in an outcome from
        the probability space and a time from the
        index set and returns the value of the
        random process at that time. (By default,
        func is the canonical function. That is,
        we assume that every outcome x from the
        probability space is a function of time and
        the value of the process is simply x(t).)
    """

    def __init__(self, prob_space, index_set=Naturals(),
                 func=lambda outcome, t: outcome[t]):
        self.index_set = index_set
        # This dict stores random variables at specific times.
        self.rvs = {}

        # Define the function for the RV.
        def _func(outcome):
            def x(t):
                # First, check if the time is in self.rvs.
                if t in self.rvs:
                    return self.rvs[t].func(outcome)
                return func(outcome, t)
            return TimeFunction.from_index_set(self.index_set, x)

        super().__init__(prob_space, _func)

    def __setitem__(self, t, value):
        if t not in self.index_set:
            raise KeyError(
                "Time %s is not in the index set for this "
                "random process." % str(t)
            )
        # If value is a RV, store it in self.rvs.
        if isinstance(value, RV):
            self.rvs[t] = value
        # If value is a scalar, create and store a constant random variable
        elif is_scalar(value):
            self.rvs[t] = RV(self.prob_space, lambda outcome: value)

    def __getitem__(self, t):
        # First, check if the time is in self.rvs.
        if t in self.rvs:
            return self.rvs[t]
        return super().__getitem__(t)

    def __call__(self, t):
        return RV(self.prob_space, lambda outcome: self.func(outcome)(t))
