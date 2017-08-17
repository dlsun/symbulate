import numpy as np
import matplotlib.pyplot as plt
import collections

from .time_index import TimeIndex
from .utils import is_scalar

class InfiniteSequence:

    def __init__(self, fun):
        self.fun = fun

    def __getitem__(self, n):
        if n == int(n):
            return self.fun(n)
        else:
            raise Exception("Index to a sequence must be an integer.")

    def __call__(self, n):
        return self.__getitem__(n)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(tuple([self.fun(n) for n in range(10)] + ["..."]))

class TimeFunction:

    def __init__(self, fun, timeIndex):
        self.fun = fun
        self.timeIndex = timeIndex

    def __getitem__(self, t):
        return self.fun(t)

    def __call__(self, t):
        return self.fun(t)

    def __str__(self):
        if self.timeIndex.fs == float("inf"):
            return "(continuous-time function)"
        else:
            return str(tuple([self.fun(self.timeIndex[n]) for n in range(10)] + ["..."]))

    def plot(self, *args, **kwargs):
        axes = plt.gca()
        tmin, tmax = axes.get_xlim()
        if self.timeIndex.fs == float("inf"):
            ts = np.linspace(tmin, tmax, 200)
        else:
            nmin = int(np.floor(tmin * self.timeIndex.fs))
            nmax = int(np.ceil(tmax * self.timeIndex.fs))
            ts = [self.timeIndex[n] for n in range(nmin, nmax + 1)]
        y = [self[t] for t in ts]
        plt.plot(ts, y, *args, **kwargs)

    def _operation_factory(self, op):
        def op_fun(self, other):
            if is_scalar(other):
                return TimeFunction(lambda t: op(self[t], other), self.timeIndex)
            elif isinstance(other, TimeFunction):
                self.timeIndex.check_same(other.timeIndex)
                return TimeFunction(lambda t: op(self[t], other[t]), self.timeIndex)
            else:
                return NotImplemented

        return op_fun

    def __add__(self, other):
        op_fun = self._operation_factory(lambda x, y: x + y)
        return op_fun(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        op_fun = self._operation_factory(lambda x, y: x - y)
        return op_fun(self, other)

    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        op_fun = self._operation_factory(lambda x, y: x * y)
        return op_fun(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: x / y)
        return op_fun(self, other)

    def __rtruediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: y / x)
        return op_fun(self, other)

    def __pow__(self, other):
        op_fun = self._operation_factory(lambda x, y: x ** y)
        return op_fun(self, other)

    def __rpow__(self, other):
        op_fun = self._operation_factory(lambda x, y: y ** x)
        return op_fun(self, other)


class LazyFunction:

    def __init__(self, fun):
        self.fun = fun
        self.cached_args = []
        self.cached_vals = []

    def __call__(self, arg):
        for a, v in zip(self.cached_args, self.cached_vals):
            if arg == a:
                return v

        val = self.fun(arg,
                       self.cached_args, self.cached_vals)
        self.cached_args.append(arg)
        self.cached_vals.append(val)
        return val
