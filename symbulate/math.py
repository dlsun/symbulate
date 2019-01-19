import math
import numbers
import operator as op

import numpy as np
import scipy.stats as stats

from .random_variables import RV
from .result import (
    Tuple,
    TimeFunction,
    ContinuousTimeFunction,
    DiscreteValued
)
from .results import Results

pi = math.pi
e = math.e
inf = float("inf")

floor = math.floor
ceil = math.ceil

def operation_factory(operation):

    def _op_func(x):
        if isinstance(x, (RV, Tuple, TimeFunction)):
            # recursively call op_fun until x is a scalar
            return x.apply(_op_func)
        elif isinstance(x, Results):
            return x.apply(_op_func)
        else:
            return operation(x)

    return _op_func

sqrt = operation_factory(math.sqrt)
exp = operation_factory(math.exp)
sin = operation_factory(math.sin)
cos = operation_factory(math.cos)
tan = operation_factory(math.tan)
factorial = operation_factory(math.factorial)

def log(value, base=e):
    return operation_factory(lambda x: math.log(x, base))(value)

def mean(x):
    if isinstance(x, numbers.Real):
        raise Exception("Taking the mean with one value is unnecessary.")
    else:
        return sum(x) / len(x)

def cumsum(x):
    return x.cumsum()

def var(x):
    return mean([(i - mean(x)) ** 2 for i in x])

def sd(x):
    return math.sqrt(var(x))

def median(x):
    if isinstance(x, numbers.Real):
        raise Exception("Taking the median of one value is unnecessary.")
    else:
        return np.median(x)

def min_max_diff(x):
    if isinstance(x, numbers.Real):
        raise Exception("Taking the range of one value is unnecessary.")
    else:
        return max(x) - min(x)

def med_abs_dev(x):
    return median(list(abs(i-median(x)) for i in x))

def quantile(q):
    return lambda x: np.percentile(x, q * 100)

def iqr(x):
    if isinstance(x, numbers.Real):
        raise Exception("Taking the iqr of one value is unnecessary.")
    else:
        q75, q25 = np.percentile(x, [75, 25])
        return q75 - q25

def orderstatistics(n):
    if n <= 0:
        raise Exception("Out of bounds. Lowest order is 1.")
    else:
        return lambda x: np.partition(x, n - 1)[n - 1]

def skewness(x):
    if isinstance(x, numbers.Real):
        raise Exception("Finding the skenewss of one value is unnecessary,")
    else:
        return stats.skew(x)

def kurtosis(x):
    if isinstance(x, numbers.Real):
        raise Exception("Finding the kurtosis of one value is unnecessary.")
    else:
        return stats.kurtosis(x)

def moment(k):
    return lambda x: stats.moment(x, k)

def trimmed_mean(alpha):
    return lambda x: stats.trim_mean(x, alpha)

def comparefun(x, compare, value):
    count = 0
    for i in x:
        if compare(i, value):
            count += 1
    return count

def count(func=lambda x: True):
    def _func(x):
        val = 0
        for i in x:
            if func(i):
                val += 1
        return val
    return _func

def count_eq(value):
    def func(x):
        return comparefun(x, op.eq, value)
    return func

def count_neq(value):
    def func(x):
        return comparefun(x, op.ne, value)
    return func

def count_lt(value):
    def func(x):
        return comparefun(x, op.lt, value)
    return func

def count_gt(value):
    def func(x):
        return comparefun(x, op.gt, value)
    return func

def count_geq(value):
    def func(x):
        return comparefun(x, op.ge, value)
    return func

def count_leq(value):
    def func(x):
        return comparefun(x, op.le, value)
    return func

def interarrival_times(continuous_time_function):
    """Given a realization of a continuous-time,
       discrete-state process, returns the interarrival
       times (i.e., the times between each state change).

    Args:
      continuous_time_function: A ContinuousTimeFunction
        object, such as ContinuousTimeMarkovChainResult or
        PoissonProcessResult.
    """
    if not (isinstance(continuous_time_function,
                       ContinuousTimeFunction) and
            isinstance(continuous_time_function,
                       DiscreteValued)):
        raise TypeError(
            "Interarrival times are only defined for "
            "continuous-time, discrete-valued functions."
        )
    return continuous_time_function.get_interarrival_times()

def arrival_times(continuous_time_function):
    """Given a realization of a continuous-time,
       discrete-state process, returns the arrival
       times (i.e., the times when the state changes).

    Args:
      continuous_time_function: A ContinuousTimeFunction
        object, such as ContinuousTimeMarkovChainResult or
        PoissonProcessResult.
    """
    if not (isinstance(continuous_time_function,
                       ContinuousTimeFunction) and
            isinstance(continuous_time_function,
                       DiscreteValued)):
        raise TypeError(
            "Interarrival times are only defined for "
            "continuous-time, discrete-valued functions."
        )
    return continuous_time_function.get_arrival_times()

def states(discrete_valued_function):
    """Given a realization of a discrete-valued function,
       returns an InfiniteVector of the sequence of
       values (or states).

    Args:
      discrete_valued_function: A DiscreteValued object
                                (e.g., MarkovChainResult or
                                       PoissonProcessResult)
    """
    if not isinstance(discrete_valued_function, DiscreteValued):
        raise TypeError(
            "States are only defined for discrete-valued "
            "functions."
        )
    return discrete_valued_function.get_states()
