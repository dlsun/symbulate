import numpy as np

from math import floor
from copy import deepcopy

from .probability_space import ArbitrarySpace
from .random_variables import RV
from .results import RandomProcessResults
from .seed import get_seed
from .sequences import TimeFunction
from .time_index import TimeIndex
from .utils import is_scalar, is_vector, get_dimension

class RandomProcess:

    def __init__(self, probSpace, timeIndex=TimeIndex(fs=1), fun=lambda x, t: None):
        self.probSpace = probSpace
        self.timeIndex = timeIndex
        self.fun = fun

    def draw(self):
        outcome = self.probSpace.draw()
        def f(t):
            return self.fun(outcome, t)
        return TimeFunction(f, self.timeIndex)

    def sim(self, n):
        return RandomProcessResults([self.draw() for _ in range(n)], self.timeIndex)

    def __getitem__(self, t):
        fun_copy = deepcopy(self.fun)
        if is_scalar(t):
            return RV(self.probSpace, lambda x: fun_copy(x, t))
        elif isinstance(t, RV):
            return RV(self.probSpace, lambda x: fun_copy(x, t.fun(x)))
    
    def __setitem__(self, t, value):
        # copy existing function to use inside redefined function
        fun_copy = deepcopy(self.fun)
        if is_scalar(value):
            def fun_new(x, s):
                if s == t:
                    return value
                else:
                    return fun_copy(x, s)
        elif isinstance(value, RV):
            def fun_new(x, s):
                if s == t:
                    return value.fun(x)
                else:
                    return fun_copy(x, s)
        else:
            raise Exception("The value of the process at any time t must be a RV.")
        self.fun = fun_new

    def apply(self, function):
        def fun(x, t):
            return function(self.fun(x, t))
        return RandomProcess(self.probSpace, self.timeIndex, fun)

    def check_same_probSpace(self, other):
        if is_scalar(other):
            return
        else:
            self.probSpace.check_same(other.probSpace)

    def check_same_timeIndex(self, other):
        if is_scalar(other) or isinstance(other, RV):
            return
        elif isinstance(other, RandomProcess):
            self.timeIndex.check_same(other.timeIndex)
        else:
            raise Exception("Cannot add object to random process.")

    # e.g., abs(X)
    def __abs__(self):
        return self.apply(lambda x: abs(x))

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            self.check_same_probSpace(other)
            self.check_same_timeIndex(other)
            if is_scalar(other):
                def fun(x, t):
                    return op(self.fun(x, t), other)
            elif isinstance(other, RV):
                def fun(x, t):
                    return op(self.fun(x, t), other.fun(x))
            elif isinstance(other, RandomProcess):
                def fun(x, t):
                    return op(self.fun(x, t), other.fun(x, t))
            return RandomProcess(self.probSpace, self.timeIndex, fun)

        return op_fun

    # e.g., X(t) + Y(t) or X(t) + Y or X(t) + 3
    def __add__(self, other):
        op_fun = self._operation_factory(lambda x, y: x + y)
        return op_fun(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    # e.g., X(t) - Y(t) or X(t) - Y or X(t) - 3
    def __sub__(self, other):
        op_fun = self._operation_factory(lambda x, y: x - y)
        return op_fun(self, other)

    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    def __neg__(self):
        return -1 * self

    # e.g., X(t) * Y(t) or X(t) * Y or X * 2
    def __mul__(self, other):
        op_fun = self._operation_factory(lambda x, y: x * y)
        return op_fun(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., X(t) / Y(t) or X(t) / Y or X / 2
    def __truediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: x / y)
        return op_fun(self, other)

    def __rtruediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: y / x)
        return op_fun(self, other)

    # e.g., X(t) ** Y(t) or X(t) ** Y or X(t) ** 2
    def __pow__(self, other):
        op_fun = self._operation_factory(lambda x, y: x ** y)
        return op_fun(self, other)

    def __rpow__(self, other):
        op_fun = self._operation_factory(lambda x, y: y ** x)
        return op_fun(self, other)

    # Alternative notation for powers: e.g., X ^ 2
    def __xor__(self, other):
        return self.__pow__(other)
    
    # Alternative notation for powers: e.g., 2 ^ X
    def __rxor__(self, other):
        return self.__rpow__(other)

    # Define a joint distribution of two random processes
    def __and__(self, other):
        self.check_same_probSpace(other)
        self.check_same_timeIndex(other)

        if isinstance(other, RandomProcess):
            def fun(x, t):
                a = self.fun(x, t)
                b = other.fun(x, t)
                a = tuple(a) if is_vector(a) else (a, )
                b = tuple(b) if is_vector(b) else (b, )
                return a + b
            return RandomProcess(self.probSpace, self.timeIndex, fun)
        else:
            raise Exception("Joint distributions are only defined for random processes.")

