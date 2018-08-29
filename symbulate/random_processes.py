from copy import deepcopy

from .index_sets import Reals
from .random_variables import RV
from .result import TimeFunction
from .results import RandomProcessResults
from .utils import is_scalar, is_vector

    
class RandomProcess:

    def __init__(self, probSpace,
                 fun=lambda outcome, t: outcome[t],
                 index_set=Reals()):
        self.probSpace = probSpace
        self.fun = fun
        self.index_set = index_set
        # This dict stores a mapping between
        # times and random variables. It is only
        # set if manually defined by the user.
        self.rvs = {}

    def draw(self):
        """Draws one realization from the random process.

        Returns:
          A TimeFunction representing one realization of the
          random process. This TimeFunction can be evaluated 
          at any time in the index set of the process.
        """
        outcome = self.probSpace.draw()
        def fn(t):
            if t in self.rvs:
                return self.rvs[t].fun(outcome)
            else:
                return self.fun(outcome, t)
        return TimeFunction.from_index_set(self.index_set, fn)

    def sim(self, n):
        """Simulates n realizations of the random process.

        Args:
          n: A positive integer specifying the number of draws.
        
        Returns:
          A RandomProcessResults object of length n consisting
          of the n realizations. Each realization is a
          TimeFunction that can be evaluated at any time in
          the index set.
        """
        return RandomProcessResults(
            [self.draw() for _ in range(n)],
            self.index_set                        
        )

    def __getitem__(self, t):
        if t not in self.index_set:
            raise KeyError(
                "Time %s is not in the index set for this "
                "random process." % str(t)
            )
        fun_copy = deepcopy(self.fun)
        if t in self.rvs:
            return self.rvs[t]
        elif isinstance(t, RV):
            def fn(outcome):
                time = t.fun(outcome)
                if time in self.rvs:
                    return self.rvs[time].fun(outcome)
                else:
                    return fun_copy(outcome, time)
            return RV(self.probSpace, fn)
        else:
            def fn(outcome):
                return fun_copy(outcome, t)
            return RV(self.probSpace, fn)
    
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

    def apply(self, function):
        """Defines a new RandomProcess Y(t), which is f(X(t)).

        Args:
          function: A function that maps values of the RandomProcess
            to new values.

        Returns:
          A new RandomProcess defined on the same probability space
          and with the same index set.
        """
        def fn(outcome, t):
            return function(self[t].fun(outcome))
        return RandomProcess(self.probSpace, fn, self.index_set)

    def check_same_probSpace(self, other):
        if is_scalar(other):
            return
        else:
            self.probSpace.check_same(other.probSpace)

    def check_same_index_set(self, other):
        if is_scalar(other) or isinstance(other, RV):
            return
        elif isinstance(other, RandomProcess):
            self.index_set.check_same(other.index_set)
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
            self.check_same_index_set(other)
            if is_scalar(other):
                def fn(x, t):
                    return op(self[t].fun(x), other)
            elif isinstance(other, RV):
                def fn(x, t):
                    return op(self[t].fun(x), other.fun(x))
            elif isinstance(other, RandomProcess):
                def fn(x, t):
                    return op(self[t].fun(x), other[t].fun(x))
            return RandomProcess(self.probSpace, fn, self.index_set)

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
        self.check_same_index_set(other)

        if isinstance(other, RandomProcess):
            def fn(x, t):
                a = self[t].fun(x)
                b = other[t].fun(x)
                a = tuple(a) if is_vector(a) else (a, )
                b = tuple(b) if is_vector(b) else (b, )
                return a + b
            return RandomProcess(self.probSpace, fn, self.index_set)
        else:
            raise Exception("Joint distributions are only defined "
                            "for two random processes.")

