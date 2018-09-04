import numbers
import numpy as np
import matplotlib.pyplot as plt

from .index_sets import (
    DiscreteTimeSequence,
    Reals,
    Naturals
)


class Scalar(numbers.Number):

    def __new__(cls, value, *args, **kwargs):
        if isinstance(value, (int, np.int64)):
            return Int(value)
        elif isinstance(value, float):
            return Float(value)
        else:
            raise Exception("Scalar type not understood.")

class Int(int, Scalar):
    
    def __new__(cls, value, *args, **kwargs):
        return super(Int, cls).__new__(cls, value)
    

class Float(float, Scalar):
    
    def __new__(cls, value, *args, **kwargs):
        return super(Float, cls).__new__(cls, value)
        

class Vector(object):

    def __init__(self, values):
        if is_scalar(values):
            self.values = (values, )
        elif hasattr(values, "__iter__"):
            # convert vectors of scalars to Numpy arrays
            if all(isinstance(value, numbers.Number)
                   for value in values):
                self.values = np.asarray(values)
            else:
                self.values = tuple(values)
        else:
            raise Exception(
                "Vectors can only be created for "
                "iterable data types."
            )
            
    def __getitem__(self, key):
        return self.values[key]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for x in self.values:
            yield x

    def __hash__(self):
        return hash(self.values)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        elif (isinstance(self.values, np.ndarray) or
              isinstance(other.values, np.ndarray)):
            return (self.values == other.values).all()
        else:
            return all(
                a == b for a, b in zip(self, other)
            )

    def apply(self, function):
        """Apply function to every element of an Vector.

        Args:
          function: function to apply to the Vector
        
        Example:
          x = Vector([1, 2, 3])
          y = x.apply(log)

        Note: For most standard functions, you can apply the function to
          the Vector directly. For example, in the example above,
          y = log(x) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def log_squared(n):
            return log(n) ** 2
          y = x.apply(log_squared)
        """
        return Vector(function(x) for x in self.values)
        
    def sum(self):
        return self.values.sum()

    def mean(self):
        return self.values.mean()

    def cumsum(self):
        return Vector(self.values.cumsum())

    def median(self):
        return self.values.median()
    
    def sd(self):
        return self.std()

    def std(self):
        return self.values.std()

    def var(self):
        return self.values.var()

    def max(self):
        return self.values.max()

    def min(self):
        return self.values.min()

    def count_eq(self, x):
        if isinstance(self.values, np.ndarray):
            return np.count_nonzero(self.values == x)
        else:
            return sum(1 for value in self.values if value == x)

    # e.g., abs(X)
    def __abs__(self):
        return self.apply(abs)

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            if len(self) != len(other):
                raise Exception(
                    "Operations can only be performed between "
                    "two Vectors of the same length."
                )
            if isinstance(other, numbers.Number):
                return Vector(op(value, other) for value in self.values)
            elif isinstance(other, Vector):
                return Vector(
                    op(a, b) for a, b in zip(self.values, other.values)
                )
            else:
                return NotImplemented
    
    # e.g., f + g or f + 3
    def __add__(self, other):
        op_fun = self._operation_factory(lambda x, y: x + y)
        return op_fun(self, other)

    # e.g., 3 + f
    def __radd__(self, other):
        return self.__add__(other)

    # e.g., f - g or f - 3
    def __sub__(self, other):
        op_fun = self._operation_factory(lambda x, y: x - y)
        return op_fun(self, other)

    # e.g., 3 - f
    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    # e.g., -f
    def __neg__(self):
        return -1 * self

    # e.g., f * g or f * 2
    def __mul__(self, other):
        op_fun = self._operation_factory(lambda x, y: x * y)
        return op_fun(self, other)
            
    # e.g., 2 * f
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., f / g or f / 2
    def __truediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: x / y)
        return op_fun(self, other)

    # e.g., 2 / f
    def __rtruediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: y / x)
        return op_fun(self, other)

    # e.g., f ** 2
    def __pow__(self, other):
        op_fun = self._operation_factory(lambda x, y: x ** y)
        return op_fun(self, other)

    # e.g., 2 ** f
    def __rpow__(self, other):
        op_fun = self._operation_factory(lambda x, y: y ** x)
        return op_fun(self, other)

    # Alternative notation for powers: e.g., f ^ 2
    def __xor__(self, other):
        return self.__pow__(other)
    
    # Alternative notation for powers: e.g., 2 ^ f
    def __rxor__(self, other):
        return self.__rpow__(other)
        
    def __str__(self):
        if len(self) <= 6:
            return "(" + ", ".join(str(x) for x in self.values) + ")"
        else:
            first_few = ", ".join(str(x) for x in self.values[:5])
            last = str(self.values[-1])
            return "(" + first_few + ", ..., " + last + ")"

    def __repr__(self):
        return self.__str__()

    def plot(self, **kwargs):
        plt.plot(range(len(self)), self.values, '.--', **kwargs)


class TimeFunction(object):

    @classmethod
    def from_index_set(cls, index_set, fn=None):
        if isinstance(index_set, DiscreteTimeSequence):
            return DiscreteTimeFunction(fn, index_set=index_set)
        elif isinstance(index_set, Reals):
            return ContinuousTimeFunction(fn)
        elif isinstance(index_set, Naturals):
            return InfiniteVector(fn)

    def check_same_index_set(self, other):
        if isinstance(other, numbers.Number):
            return
        if isinstance(other, TimeFunction):
            if self.index_set != other.index_set:
                raise Exception(
                    "Operations can only be performed on "
                    "TimeFunctions with the same index set."
                )
        else:
            raise Exception("Cannot add object to random process.")
    
    # e.g., abs(X)
    def __abs__(self):
        return self.apply(abs)

    # e.g., f + g or f + 3
    def __add__(self, other):
        op_fun = self._operation_factory(lambda x, y: x + y)
        return op_fun(self, other)

    # e.g., 3 + f
    def __radd__(self, other):
        return self.__add__(other)

    # e.g., f - g or f - 3
    def __sub__(self, other):
        op_fun = self._operation_factory(lambda x, y: x - y)
        return op_fun(self, other)

    # e.g., 3 - f
    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    # e.g., -f
    def __neg__(self):
        return -1 * self

    # e.g., f * g or f * 2
    def __mul__(self, other):
        op_fun = self._operation_factory(lambda x, y: x * y)
        return op_fun(self, other)
            
    # e.g., 2 * f
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., f / g or f / 2
    def __truediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: x / y)
        return op_fun(self, other)

    # e.g., 2 / f
    def __rtruediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: y / x)
        return op_fun(self, other)

    # e.g., f ** 2
    def __pow__(self, other):
        op_fun = self._operation_factory(lambda x, y: x ** y)
        return op_fun(self, other)

    # e.g., 2 ** f
    def __rpow__(self, other):
        op_fun = self._operation_factory(lambda x, y: y ** x)
        return op_fun(self, other)

    # Alternative notation for powers: e.g., f ^ 2
    def __xor__(self, other):
        return self.__pow__(other)
    
    # Alternative notation for powers: e.g., 2 ^ f
    def __rxor__(self, other):
        return self.__rpow__(other)

    
class InfiniteVector(TimeFunction):

    def __init__(self, fn=lambda n: n):
        """Initializes a (lazy) data structure for an infinite vector.

        Args:
          fn: A function of n that returns the value in position n.
              n is assumed to be a natural number (integer >= 0).
              This function can be defined at initialization time,
              or later. By default, it is not set at initialization.
        """
        if fn is not None:
            self.fn = fn
        self.index_set = Naturals()
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

    def __call__(self, n):
        return self[n]
    
    def __str__(self):
        first_few = [str(self[i]) for i in range(6)]
        return "(" + ", ".join(first_few) + ", ...)"

    def __repr__(self):
        return self.__str__()

    def apply(self, function):
        """Apply function to every element of an InfiniteVector.

        Args:
          function: function to apply to the InfiniteVector
        
        Example:
          x = InfiniteVector(lambda n: n)
          y = x.apply(log)

        Note: For most standard functions, you can apply the function to
          the InfiniteVector directly. For example, in the example above,
          y = log(x) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def log_squared(n):
            return log(n) ** 2
          y = x.apply(log_squared)
        """
        return InfiniteVector(lambda n: function(self[n]))

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            self.check_same_index_set(other)
            if isinstance(other, numbers.Number):
                return InfiniteVector(lambda n: op(self[n], other))
            elif isinstance(other, InfiniteVector):
                return InfiniteVector(lambda n: op(self[n], other[n]))
            else:
                return NotImplemented

        return op_fun

    def cumsum(self):
        result = InfiniteVector()
        def fn(n):
            return sum(self[i] for i in range(n + 1))
        result.fn = fn
        
        return result

    def plot(self, nmin=0, nmax=10, **kwargs):
        xs = range(nmin, nmax)
        ys = [self[n] for n in range(nmin, nmax)]
        plt.plot(xs, ys, '.--', **kwargs)



class DiscreteTimeFunction(TimeFunction):

    def __init__(self, fn=lambda n: n, fs=1, index_set=None):
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

    def apply(self, function):
        """Compose function with the TimeFunction.

        Args:
          function: function to compose with the TimeFunction
        
        Example:
          f = DiscreteTimeFunction(lambda t: t, fs=1)
          g = f.apply(log)

        Note: For most standard functions, you can apply the function to
          the TimeFunction directly. For example, in the example above,
          g = log(f) would have been equivalent and more readable.

        User-defined functions can also be applied.

        Example:
          def log_squared(f):
            return log(f) ** 2
          g = f.apply(log_squared)
        """
        return DiscreteTimeFunction(lambda n: function(self[n]),
                                    index_set=self.index_set)

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            self.check_same_index_set(other)
            if isinstance(other, numbers.Number):
                return DiscreteTimeFunction(
                    lambda n: op(self[n], other),
                    index_set=self.index_set
                )
            elif isinstance(other, DiscreteTimeFunction):
                return DiscreteTimeFunction(
                    lambda n: op(self[n], other[n]),
                    index_set=self.index_set
                )
            else:
                return NotImplemented

        return op_fun

    def __str__(self):
        first_few = ", ".join(str(self[n]) for n in range(-2, 3))
        return "(..., " + first_few + ", ...)"

    def __repr__(self):
        return self.__str__()

    def plot(self, tmin=0, tmax=10, **kwargs):
        nmin = int(np.floor(tmin * self.index_set.fs))
        nmax = int(np.ceil(tmax * self.index_set.fs))
        ts = [self.index_set[n] for n in range(nmin, nmax)]
        ys = [self[n] for n in range(nmin, nmax)]
        plt.plot(ts, ys, ".--", **kwargs)


class ContinuousTimeFunction(TimeFunction):

    def __init__(self, fn=lambda t: t):
        """Initializes a data structure for a discrete-time function.

        Args:
          fn: A function of n that returns the value in position n.
              n is assumed to be any integer (postive or negative).
              This function can be defined at initialization time,
              or later. By default, it is not set at initialization.
        """
        self.index_set = Reals()
        if fn is not None:
            self.fn = fn

    def __call__(self, t):
        return self.fn(t)

    def __getitem__(self, t):
        return self(t)

    def apply(self, function):
        """Compose function with the TimeFunction.

        Args:
          function: function to compose with the TimeFunction
        
        Example:
          f = ContinuousTimeFunction(lambda t: t)
          g = f.apply(log)

        Note: For most standard functions, you can apply the function to
          the TimeFunction directly. For example, in the example above,
          g = log(f) would have been equivalent and more readable.

        User-defined functions can also be applied.

        Example:
          def log_squared(f):
            return log(f) ** 2
          g = f.apply(log_squared)
        """
        return ContinuousTimeFunction(lambda t: function(self(t)))

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            self.check_same_index_set(other)
            if isinstance(other, numbers.Number):
                return ContinuousTimeFunction(
                    lambda t: op(self(t), other)
                )
            elif isinstance(other, ContinuousTimeFunction):
                return ContinuousTimeFunction(
                    lambda t: op(self(t), other(t))
                )
            else:
                return NotImplemented

        return op_fun

    def __str__(self):
        return "(continuous-time function)"

    def __repr__(self):
        return self.__str__()

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


def join(*args):
    """Joins Result objects into a single Result object.
    """
    # convert scalars to Vectors
    results = [Vector([result]) if is_scalar(result) else result
               for result in args]

    # If all results are Vectors, concatenate into one giant Vector.
    if all(isinstance(result, Vector) for result in results):
        return Vector(np.concatenate(results))
    
    # Otherwise, return a tuple of the original results.
    return tuple(args)
    

def is_scalar(x):
    return isinstance(x, numbers.Number) or isinstance(x, str)


def is_vector(x):
    if hasattr(x, "__len__") and all(is_scalar(i) for i in x):
        return True
    else:
        return False


def is_time_function(x):
    return isinstance(x, TimeFunction)
