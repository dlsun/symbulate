import matplotlib.pyplot as plt

from numbers import Number
from copy import deepcopy as copy

from .results import RVResults

def is_scalar(x):
    if isinstance(x, Number):
        return True
    else:
        return False

def is_vector(x):
    if hasattr(x, "__iter__") and all(is_scalar(i) for i in x):
        return True
    else:
        return False

class RV:

    def __init__(self, probSpace, fun=lambda x: x):
        self.probSpace = probSpace
        self.fun = fun

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
        if is_scalar(other):
            return self.apply(lambda x: x + other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i + j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a + b
            else:
                raise Exception("I don't know how to add those two random variables.")
        return RV(self.probSpace, fun)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x - other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i - j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a - b
            else:
                raise Exception("I don't know how to subtract those two random variables.")
        return RV(self.probSpace, fun)

    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    def __mul__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x * other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i * j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a * b
            else:
                raise Exception("I don't know how to multiply those two random variables.")
        return RV(self.probSpace, fun)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x / other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i / j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a / b
            else:
                raise Exception("I don't know how to divide those two random variables.")
        return RV(self.probSpace, fun)

    def __rtruediv__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: other / x)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(j / i for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return b / a
            else:
                raise Exception("I don't know how to divide those two random variables.")
        return RV(self.probSpace, fun)

    def __pow__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x ** other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i ** j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a ** b
            else:
                raise Exception("I don't know how to raise that random variable to that power.")
        return RV(self.probSpace, fun)

    def __rpow__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: other ** x)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(j ** i for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return b ** a
            else:
                raise Exception("I don't know how to raise that random variable to that power.")
        return RV(self.probSpace, fun)
                
    def __and__(self, other):
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            a = tuple(a) if is_vector(a) else (a, )
            b = tuple(b) if is_vector(b) else (b, )
            return a + b
        return RV(self.probSpace, fun)
