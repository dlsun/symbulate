import numpy as np
import matplotlib.pyplot as plt

from numbers import Number
from copy import deepcopy as copy

from .results import RVResults

class RV:

    def __init__(self, probSpace, fun=lambda x: x):
        self.probSpace = probSpace
        # convert vector outputs to Numpy arrays
        # (to support vector addition, etc.)
        def fun1(outcome):
            out = fun(outcome)
            if hasattr(out, "__len__"):
                out = np.array(out)
            return out
        self.fun = fun1

    def draw(self):
        return self.fun(self.probSpace.draw())

    def sim(self, n):
        return RVResults(self.draw() for _ in range(n))

    def check_same_probSpace(self, other):
        if self.probSpace != other.probSpace:
            raise Exception("Random variables must be defined on same probability space.")

    def apply(self, function):
        def f_new(outcome):
            return function(self.fun(outcome))
        return RV(self.probSpace, copy(f_new))

    def component(self, i):
        return self.apply(lambda x: x[i])

    def __add__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: x + other)
        self.check_same_probSpace(other)
        def fun(outcome):
            return self.fun(outcome) + other.fun(outcome)
        return RV(self.probSpace, copy(fun))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: x - other)
        self.check_same_probSpace(other)
        def fun(outcome):
            return self.fun(outcome) - other.fun(outcome)
        return RV(self.probSpace, fun)

    def __sub__(self, other):
        return -1 * self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: x * other)
        self.check_same_probSpace(other)
        def fun(outcome):
            return self.fun(outcome) * other.fun(outcome)
        return RV(self.probSpace, fun)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: x / other)
        self.check_same_probSpace(other)
        def fun(outcome):
            return self.fun(outcome) / other.fun(outcome)
        return RV(self.probSpace, fun)

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: other / x)
        self.check_same_probSpace(other)
        def fun(outcome):
            return other.fun(outcome) / self.fun(outcome)
        return RV(self.probSpace, fun)

    def __pow__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: x ** other)
        self.check_same_probSpace(other)
        def fun(outcome):
            return self.fun(outcome) ** other.fun(outcome)
        return RV(self.probSpace, fun)

    def __rpow__(self, other):
        if isinstance(other, Number):
            return self.apply(lambda x: other ** x)
        self.check_same_probSpace(other)
        def fun(outcome):
            return other.fun(outcome) ** self.fun(outcome)
        return RV(self.probSpace, fun)
                
    def __and__(self, other):
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            a = tuple(a) if hasattr(a, "__len__") else (a, )
            b = other.fun(outcome)
            b = tuple(b) if hasattr(b, "__len__") else (b, )
            return a + b
        return RV(self.probSpace, fun)
