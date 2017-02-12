import numpy as np

from .probability_space import ArbitrarySpace
from .random_variables import RV
from .results import RandomProcessResults
from .utils import is_scalar, is_vector, get_dimension

class RandomProcess:

    def __init__(self, probSpace, time_fun):
        self.probSpace = probSpace
        self.time_fun = time_fun

    def draw(self):
        seed = np.random.randint(1e9)
        def f(t):
            np.random.seed(seed)
            return self[t].draw()
        return f

    def sim(self, n):
        return RandomProcessResults(self.draw() for _ in range(n))

    def __getitem__(self, t):
        return self.time_fun(t)
    
    def apply(self, function):
        def time_fun(t):
            return function(self.time_fun(t))
        return RandomProcess(self.probSpace, time_fun)

    def check_same_probSpace(self, other):
        if is_scalar(other):
            return
        else:
            self.probSpace.check_same(other.probSpace)

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            self.check_same_probSpace(other)
            if is_scalar(other):
                def time_fun(t):
                    return op(self.time_fun(t), other)
            elif isinstance(other, RV):
                def time_fun(t):
                    return op(self.time_fun(t), other)
            elif isinstance(other, RandomProcess):
                def time_fun(t):
                    return op(self.time_fun(t), other.time_fun(t))
            return RandomProcess(self.probSpace, time_fun)

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

TimeIndex = RandomProcess(ArbitrarySpace(), lambda t: t)

