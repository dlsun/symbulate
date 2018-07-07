import numpy as np
import matplotlib.pyplot as plt

from .index_sets import (
    DiscreteTimeSequence,
    Reals,
    Naturals
)


class Scalar:

    def __new__(cls, value, *args, **kwargs):
        if isinstance(value, int):
            return Int(value)
        elif isinstance(value, float):
            return Float(value)
    
    def join(self, other):
        if isinstance(other, Scalar):
            return Vector([self, other])
        elif isinstance(other, Vector):
            return Vector(np.insert(other.array, 0, [self]))
        else:
            raise NotImplementedError()


class Int(int, Scalar):
    
    def __new__(cls, value, *args, **kwargs):
        return super(Int, cls).__new__(cls, value)
    

class Float(float, Scalar):
    
    def __new__(cls, value, *args, **kwargs):
        return super(Float, cls).__new__(cls, value)
        

class Vector:

    def __init__(self, values):
        self.array = np.asarray(values)
        
    def join(self, other):
        if isinstance(other, Scalar):
            return Vector(np.append(self.array, [other]))
        elif isinstance(other, Vector):
            return Vector(np.append(self.array, other.array))
        else:
            raise NotImplementedError

    def __getitem__(self, key):
        return self.array[key]

    def __len__(self):
        return self.array.size
    
    def sum(self):
        return self.array.sum()

    def mean(self):
        return self.array.mean()

    def cumsum(self):
        total = 0
        sums = [total]
        for i in self:
            total += i
            sums.append(total)
        return Vector(sums)

    def median(self):
        return self.array.median()
    
    def sd(self):
        return self.std()

    def std(self):
        return self.array.std()

    def var(self):
        return self.array.var()

    def max(self):
        return self.array.max()

    def min(self):
        return self.array.min()

    def count_eq(self, x):
        return np.count_nonzero(self.array == x)
    
    def __str__(self):
        if len(self) <= 6:
            return "(" + ", ".join(str(x) for x in self.array) + ")"
        else:
            first_few = [str(x) for x in self.array[:5]]
            return "(" + ", ".join(first_few) + ", ..., " + str(self.array[-1]) + ")"

    def __repr__(self):
        return self.__str__()


class TimeFunction:

    @classmethod
    def from_index_set(cls, index_set, fn=None):
        if isinstance(index_set, DiscreteTimeSequence):
            return DiscreteTimeFunction(fn, index_set=index_set)
        elif isinstance(index_set, Reals):
            return ContinuousTimeFunction(fn)
        elif isinstance(index_set, Naturals):
            return InfiniteVector(fn)

    
class InfiniteVector(TimeFunction):

    def __init__(self, fn=None):
        """Initializes a (lazy) data structure for an infinite vector.

        Args:
          fn: A function of n that returns the value in position n.
              n is assumed to be a natural number (integer >= 0).
              This function can be defined at initialization time,
              or later. By default, it is not set at initialization.
        """
        if fn is not None:
            self.fn = fn
        self.values = []

    def __getitem__(self, n):
        m = len(self.values)
        # Add necessary elements to self.values
        n0 = None
        if isinstance(n, slice) and n.stop >= m:
            n0 = n.stop
        elif isinstance(n, int) and n >= m:
            n0 = n
        if n0 is not None:
            for i in range(m, n0 + 1):
                self.values.append(self.fn(i))
        # Return the corresponding value(s)
        return self.values[n]

    def __str__(self):
        first_few = [str(self[i]) for i in range(6)]
        return "(" + ", ".join(first_few) + ", ...)"

    def __repr__(self):
        return self.__str__()

    def cumsum(self):
        result = InfiniteVector()
        def fn(n):
            return sum(self[i] for i in range(n + 1))
        result.fn = fn
        
        return result


class DiscreteTimeFunction(TimeFunction):

    def __init__(self, fn=None, fs=1, index_set=None):
        """Initializes a data structure for a discrete-time function.

        Args:
          fn: A function of n that returns the value at time n / fs.
              n is assumed to be any integer (postive or negative).
              This function can be defined at initialization time,
              or later. By default, it is not set at initialization.
          fs: The sampling rate for the function.
          index_set: An IndexSet that specifies the index set of
                     the discrete-time function. (fs is ignored if
                     this is specified.)
        """
        if fn is not None:
            self.fn = fn
        if index_set is None:
            self.index_set = DiscreteTimeSequence(fs)
        else:
            self.index_set = index_set
        self.array_pos = [] # stores values for t >= 0
        self.array_neg = [] # stores values for t < 0

    def __getitem__(self, n):
        if not isinstance(n, int):
            raise Exception(
                "With a DiscreteTimeFunction f, "
                "f[n] returns the nth time sample, "
                "so n must be an integer. If you "
                "intended to get the value at time t, "
                "call f(t) instead."
            )

        # Get the nth time sample
        if n >= 0:
            m = len(self.array_pos)
            if n >= m:
                for i in range(m, n+1):
                    self.array_pos.append(self.fn(i))
            return self.array_pos[n]
        else:
            m = len(self.array_neg)
            if -n > m:
                for i in range(-m - 1, n - 1, -1):
                    self.array_neg.append(self.fn(i))
            return self.array_neg[-n - 1]

    def __call__(self, t):
        fs = self.index_set.fs
        if not t in self.index_set:
            raise KeyError((
                "No value at time %.2f for a process sampled"
                "at a rate of %d Hz.") % (t, fs))
        n = int(t * fs)
        return self[n]

    def plot(self, tmin=0, tmax=10, **kwargs):
        nmin = int(np.floor(tmin * self.index_set.fs))
        nmax = int(np.ceil(tmax * self.index_set.fs))
        ts = [self.index_set[n] for n in range(nmin, nmax)]
        ys = [self[n] for n in range(nmin, nmax)]
        plt.plot(ts, ys, ".--", **kwargs)


class ContinuousTimeFunction(TimeFunction):

    def __init__(self, fn):
        """Initializes a data structure for a discrete-time function.

        Args:
          fn: A function of n that returns the value in position n.
              n is assumed to be any integer (postive or negative).
              This function can be defined at initialization time,
              or later. By default, it is not set at initialization.
        """
        if fn is not None:
            self.fn = fn

    def __call__(self, t):
        return self.fn(t)

    def __getitem__(self, t):
        return self(t)

    def plot(self, tmin=0, tmax=10, **kwargs):
        ts = np.linspace(tmin, tmax, 200)
        ys = [self(t) for t in ts]
        plt.plot(ts, ys, "-", **kwargs)
        

class DiscreteValued:

    def get_states(self):
        if not hasattr(self, "states"):
            raise NameError("States not defined for "
                            "function.")
        return self.states

    def get_interarrival_times(self):
        if not hasattr(self, "interarrival_times"):
            raise NameError("Interarrival times not "
                            "defined for function.")
        return self.interarrival_times

    def get_arrival_times(self):
        if not hasattr(self, "interarrival_times"):
            raise NameError("Interarrival times not "
                            "defined for function.")
        return self.interarrival_times.cumsum()
