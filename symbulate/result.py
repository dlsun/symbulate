import numbers
import numpy as np
import matplotlib.pyplot as plt

import symbulate
from .base import Arithmetic, Transformable, Statistical, Filterable
from .index_sets import DiscreteTimeSequence, Reals, Naturals


class Scalar(numbers.Number):

    def __new__(cls, value, *args, **kwargs):
        if isinstance(value, numbers.Integral):
            return Int(value)
        elif isinstance(value, (float, np.floating)):
            return Float(value)
        else:
            raise Exception("Scalar type not understood.")


class Int(int, Scalar):

    def __new__(cls, value, *args, **kwargs):
        return super(Int, cls).__new__(cls, value)


class Float(float, Scalar):

    def __new__(cls, value, *args, **kwargs):
        return super(Float, cls).__new__(cls, value)


class Tuple(Arithmetic, Transformable, Statistical, Filterable):
    """A collapsible data structure.
    """

    def __init__(self, values):
        if is_scalar(values):
            self.values = (values, )
        elif hasattr(values, "__len__") or hasattr(values, "__next__"):
            self.values = tuple(values)
        else:
            raise Exception(
                "Tuples can only be created from "
                "finite iterable data."
            )

    def __getitem__(self, n):
        # if n is a numeric array, return a Tuple of those values
        if is_numeric_vector(n):
            return type(self)(self.values[i] for i in n)
        # otherwise, return the value at n
        return self.values[n]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for value in self.values:
            yield value

    def __hash__(self):
        return hash(tuple(self.values))

    # Define comparison operators to handle sorting.
    def __eq__(self, other):
        if not hasattr(other, "__len__"):
            return False
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

    def __lt__(self, other):
        return tuple(self.values) < tuple(other.values)

    def apply(self, func):
        """Apply function to every element of a Tuple.

        Args:
          func: function to apply to the Tuple

        Example:
          x = Tuple([1, 2, 3])
          y = x.apply(log)

        Note: For most standard functions, you can apply the function to
          the Tuple directly. For example, in the example above,
          y = log(x) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def log_squared(n):
            return log(n) ** 2
          y = x.apply(log_squared)
        """
        return type(self)(func(e) for e in self)

    # The Filterable superclass will use this to define all of the
    # .filter_*() and .count_*() methods.
    def filter(self, filt):
        """Get only the elements that satisfy the given criterion.

        Args:
          filt: A function that takes in an element and returns
            a boolean.

        Returns:
          Tuple: Another Tuple containing only those elements e
          where filt(e) is True.
        """
        return type(self)(e for e in self if filt(e))

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
            if is_number(other):
                return type(self)(op(value, other) for value in self)
            elif is_vector(other):
                # check that other is the same length as the Tuple
                if len(self) != len(other):
                    raise Exception(
                        "Arithmetic operations between a %s and a %s "
                        "are only valid if they are the same length. "
                        "You attempted to combine a %s of length %d "
                        "with a %s of length %d." % (
                            type(self).__name__,
                            type(other).__name__,
                            type(self).__name__,
                            len(self),
                            type(other).__name__,
                            len(other)
                        ))
                # return a new Tuple/Vector of the same length
                return type(self)(op(a, b) for a, b in zip(self, other))
            else:
                return NotImplemented

        return _op_func

    # The Statistical superclass will use this to define all of the
    # usual statistical functions (e.g., mean, var, etc.)
    def _statistic_factory(self, op):
        def _op_func(self):
            return op(self.values)
        return _op_func

    def cumsum(self):
        return type(self)(np.cumsum(self.values))

    def plot(self, **kwargs):
        plt.plot(range(len(self)), self.values, '.--', **kwargs)

    def __str__(self):
        if len(self) <= 6:
            return "(" + ", ".join(str(x) for x in self) + ")"
        else:
            first_few = ", ".join(str(x) for x in self[:5])
            last = str(self[-1])
            return "(" + first_few + ", ..., " + last + ")"

    def __repr__(self):
        return self.__str__()


class Vector(Tuple):
    """A data structure like a Tuple, except it does not collapse.
    """
    pass


class TimeFunction(Arithmetic):

    @classmethod
    def from_index_set(cls, index_set, func=None):
        if isinstance(index_set, DiscreteTimeSequence):
            return DiscreteTimeFunction(func, index_set=index_set)
        elif isinstance(index_set, Reals):
            return ContinuousTimeFunction(func)
        elif isinstance(index_set, Naturals):
            return InfiniteVector(func)

    def check_same_index_set(self, other):
        if isinstance(other, (numbers.Number, symbulate.RV)):
            return
        elif isinstance(other, TimeFunction):
            if self.index_set != other.index_set:
                raise Exception(
                    "Operations can only be performed on "
                    "TimeFunctions with the same index set."
                )
        else:
            raise Exception("Cannot combine %s with %s." % (
                type(self).__name__, type(other).__name__
            ))


class InfiniteTuple(TimeFunction):

    def __init__(self, func=lambda n: n):
        """Initializes a (lazy) data structure for an infinite vector.

        Args:
          func: A function of n that returns the value in position n.
                n is assumed to be a natural number (integer >= 0).
                This function can be defined at initialization time,
                or later. By default, it is not set at initialization.
        """
        if func is not None:
            self.func = func
        self.index_set = Naturals()
        self.values = []

    def __getitem__(self, n):
        m = len(self.values)
        # Add necessary elements to self.values
        n0 = None
        # handle the case where n is a slice
        if isinstance(n, slice):
            if n.stop is None:
                if n.start is None:
                    return self
                else:
                    return type(self)(lambda i: self[i + n.start])
            if n.stop >= m:
                n0 = n.stop
        elif isinstance(n, numbers.Integral) and n >= m:
            n0 = n
        if n0 is not None:
            for i in range(m, n0 + 1):
                self.values.append(self.func(i))
        # Return the corresponding value(s)
        return self.values[n]

    def __call__(self, n):
        return self[n]

    def __str__(self):
        first_few = [str(self[i]) for i in range(6)]
        return "(" + ", ".join(first_few) + ", ...)"

    def __repr__(self):
        return self.__str__()

    def apply(self, func):
        """Apply function to every element of an InfiniteTuple.

        Args:
          func: function to apply to the InfiniteTuple

        Example:
          x = InfiniteTuple(lambda n: n)
          y = x.apply(log)

        Note: For most standard functions, you can apply the function to
          the InfiniteTuple directly. For example, in the example above,
          y = log(x) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def log_squared(n):
            return log(n) ** 2
          y = x.apply(log_squared)
        """
        return type(self)(lambda n: func(self[n]))

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
            self.check_same_index_set(other)
            if is_number(other):
                return type(self)(lambda n: op(self[n], other))
            elif isinstance(other, InfiniteTuple):
                return type(self)(lambda n: op(self[n], other[n]))
            else:
                return NotImplemented

        return _op_func


class InfiniteVector(InfiniteTuple):

    def cumsum(self):
        def _func(n):
            return sum(self[i] for i in range(n + 1))
        return InfiniteVector(_func)

    def plot(self, tmin=0, tmax=10, **kwargs):
        xs = range(tmin, tmax)
        ys = [self[t] for t in range(tmin, tmax)]
        plt.plot(xs, ys, '.--', **kwargs)


class DiscreteTimeFunction(TimeFunction):

    def __init__(self, func=None, fs=1, index_set=None):
        """Initializes a data structure for a discrete-time function.

        Args:
          func: A function of n that returns the value at time n / fs.
            n is assumed to be any integer (postive or negative).
            By default, it is set to the identity function f[n] = n / fs.
          fs (int): The sampling rate of the function, in Hertz (samples
            per second).
          index_set (IndexSet): The index set of the discrete-time function
            (fs is ignored if this is specified.)
        """
        if func is not None:
            self.func = func
        else:
            self.func = lambda n: n / fs
        if index_set is None:
            self.index_set = DiscreteTimeSequence(fs)
        else:
            self.index_set = index_set
        self.array_pos = [] # stores values for t >= 0
        self.array_neg = [] # stores values for t < 0

    def _get_value_at_index(self, n):
        if not isinstance(n, numbers.Integral):
            raise KeyError(
                "For a DiscreteTimeFunction f, f[n] returns the "
                "the nth time sample, so n must be an integer. "
                "If you want the value at time t, try f(t) instead.")

        if n >= 0:
            m = len(self.array_pos)
            if n >= m:
                for i in range(m, n + 1):
                    self.array_pos.append(self.func(i))
            return self.array_pos[n]
        else:
            m = len(self.array_neg)
            if -n > m:
                for i in range(-m - 1, n - 1, -1):
                    self.array_neg.append(self.func(i))
            return self.array_neg[-n - 1]

    def _get_value_at_time(self, t):
        fs = self.index_set.fs
        if not t in self.index_set:
            raise KeyError((
                "No value at time %.2f for a function with "
                "a sampling rate of %d Hz.") % (t, fs))
        return self._get_value_at_index(int(t * fs))

    def __getitem__(self, n):
        if is_number(n):
            return self._get_value_at_index(n)
        elif is_numeric_vector(n):
            return Vector(self._get_value_at_index(e) for e in n)
        elif isinstance(n, slice):
            return Vector(self._get_value_at_index(e) for e in
                          range(n.start, n.stop, n.step or 1))
        else:
            raise TypeError("Cannot evaluate DiscreteTimeFunction at "
                            "index %s (type %s)." % (n, type(n).__name__))

    def __call__(self, t):
        if is_number(t):
            return self._get_value_at_time(t)
        elif is_numeric_vector(t):
            return Vector(self._get_value_at_time(e) for e in t)
        elif isinstance(t, DiscreteTimeFunction):
            self.check_same_index_set(t)
            return DiscreteTimeFunction(func=lambda n: self(t[n]),
                                        index_set=self.index_set)
        else:
            raise TypeError("Cannot evaluate DiscreteTimeFunction at "
                            "time %s (type %s)." % (t, type(t).__name__))

    def apply(self, func):
        """Compose function with the TimeFunction.

        Args:
          func: function to compose with the TimeFunction

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
        return DiscreteTimeFunction(lambda n: func(self[n]),
                                    index_set=self.index_set)

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
            self.check_same_index_set(other)
            if is_number(other):
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

        return _op_func

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

    def __init__(self, func=lambda t: t):
        """Initializes a data structure for a discrete-time function.

        Args:
          func: A function of n that returns the value in position n.
                n is assumed to be any integer (postive or negative).
                This function can be defined at initialization time,
                or later. By default, it is not set at initialization.
        """
        self.index_set = Reals()
        if func is not None:
            self.func = func

    def __call__(self, t):
        if is_number(t):
            return self.func(t)
        elif is_numeric_vector(t):
            try:
                # Use vectorized function if it exists
                return Vector(self.vfunc(t))
            except:
                return Vector(self.func(e) for e in t)
        elif isinstance(t, ContinuousTimeFunction):
            return ContinuousTimeFunction(func=lambda s: self(t(s)))
        else:
            raise TypeError("Cannot evaluate ContinuousTimeFunction at "
                            "time %s (type %s)." % (t, type(t).__name__))

    def __getitem__(self, t):
        return self(t)

    def apply(self, func):
        """Compose function with the TimeFunction.

        Args:
          func: function to compose with the TimeFunction


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
        return ContinuousTimeFunction(lambda t: func(self(t)))

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
            self.check_same_index_set(other)
            if is_number(other):
                return ContinuousTimeFunction(
                    lambda t: op(self(t), other)
                )
            elif isinstance(other, ContinuousTimeFunction):
                return ContinuousTimeFunction(
                    lambda t: op(self(t), other(t))
                )
            else:
                return NotImplemented

        return _op_func

    def __str__(self):
        return "[continuous-time function]"

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


def join(result1, result2):
    """Joins two result objects into a single result object.

    Args:
      result1: The first result.
      result2: The second result.
    """

    a = tuple(result1.values) if type(result1) == Tuple else (result1, )
    b = tuple(result2.values) if type(result2) == Tuple else (result2, )

    return Tuple(a + b)


def concat(*args):
    """Concatenates scalars and vectors into one data structure.

    Args:
      *args: Any number of scalar or vector objects. The last
          argument can be an InfiniteTuple.

    Returns:
      A Vector or an InfiniteTuple, depending on whether the
      last argument is an InfiniteTuple.
    """
    values = []
    for i, arg in enumerate(args):
        if is_scalar(arg):
            values.append(arg)
        elif is_vector(arg):
            values.extend(arg)
        elif isinstance(arg, InfiniteTuple):
            # check that InfiniteTuple is the last arg
            if i == len(args) - 1:
                # define concatenated InfiniteTuple
                def _func(n):
                    if n < len(values):
                        return values[n]
                    else:
                        return arg[n - len(values)]
                return type(arg)(_func)

            raise Exception("InfiniteTuple must be the last "
                            "argument to concat().")
        else:
            raise TypeError("Every argument to concat() must be either "
                            "a scalar, a vector, or an InfiniteTuple.")

    return Vector(values)


def is_scalar(x):
    return isinstance(x, (numbers.Number, str))


def is_vector(x):
    return hasattr(x, "__len__")


def is_number(x):
    return isinstance(x, numbers.Number)


def is_numeric_vector(x):
    return hasattr(x, "__len__") and all(is_number(i) for i in x)
