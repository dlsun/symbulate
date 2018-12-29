from .index_sets import Naturals
from .random_variables import RV
from .result import TimeFunction, is_scalar
from .results import RVResults


class RandomProcess(RV):

    def __init__(self, probSpace,
                 index_set=Naturals(),
                 func=lambda outcome, t: outcome[t]):
        self.probSpace = probSpace
        self.index_set = index_set
        # This dict stores a mapping between
        # times and random variables. When the user
        # asks for the random process at a time t,
        # it looks for the random variable in self.rvs
        # first, and only if it is not there does it
        # define a random variable using self.func.
        self.rvs = {}

        # Define the function for the RV
        def func_(outcome):
            def x(t):
                if t in self.rvs:
                    return self.rvs[t].func(outcome)
                else:
                    return func(outcome, t)
            return TimeFunction.from_index_set(self.index_set, x)
        self.func = func_

    def __setitem__(self, t, value):
        if t not in self.index_set:
            raise KeyError(
                "Time %s is not in the index set for this "
                "random process." % str(t)
            )
        # If value is a RV, store it in self.rvs
        if isinstance(value, RV):
            self.rvs[t] = value
        # If value is a scalar, create and store a constant random variable
        elif is_scalar(value):
            self.rvs[t] = RV(self.probSpace, lambda outcome: value)

    def __getitem__(self, t):
        if t in self.rvs:
            return self.rvs[t]
        else:
            return super().__getitem__(t)

    def __call__(self, t):
        return RV(self.probSpace, lambda outcome: self.func(outcome)(t))

