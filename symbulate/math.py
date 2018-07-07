import math
import numpy as np
import operator as op

from .random_variables import RV
from .random_processes import RandomProcess
from .result import ContinuousTimeFunction, DiscreteValued
from .results import *

pi = math.pi
e = math.e
inf = float("inf")

def operation_factory(op):

    def op_fun(x):
        if isinstance(x, RandomProcess):
            return x.apply(op_fun)
        elif isinstance(x, RV):
            return x.apply(op)
        else:
            return op(x)

    return op_fun

sqrt = operation_factory(math.sqrt)
exp = operation_factory(math.exp)
sin = operation_factory(math.sin)
cos = operation_factory(math.cos)
tan = operation_factory(math.tan)
factorial = operation_factory(math.factorial)
def log(x, base=e):
    if isinstance(x, RVResults):
        with np.errstate(all='raise'):
            try:
                return RVResults(np.log(x) / np.log(base))
            except FloatingPointError as e:
                raise type(e)("I can't take the log of these values.")
    else:
        try: 
            return operation_factory(lambda y: math.log(y, base))(x)
        except ValueError as e:
            raise type(e)("I can't take the log of these values.")

def mean(x):
    return sum(x) / len(x)

def cumsum(x):
    return x.cumsum()

def var(x):
    return mean([(i - mean(x)) ** 2 for i in x])

def sd(x):
    return math.sqrt(var(x))

def median(x):
    return np.median(x)

def quantile(q):
    return lambda x: np.percentile(x, q*100)    

def comparefun(x, compare, value):
    count = 0
    for i in x:
        if compare(i, value):
            count += 1
    return count
    
def count_eq(value):
    def fun(x):
        return comparefun(x, op.eq, value)
    return fun    

def count_neq(value):
    def fun(x):
        return comparefun(x, op.ne, value)
    return fun

def count_lt(value):
    def fun(x):
        return comparefun(x, op.lt, value)
    return fun 

def count_gt(value):
    def fun(x):
        return comparefun(x, op.gt, value)
    return fun 

def count_geq(value):
    def fun(x):
        return comparefun(x, op.ge, value)
    return fun 

def count_leq(value):
    def fun(x):
        return comparefun(x, op.le, value)
    return fun

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
                       ContinuousTimeFunction) or
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
                       ContinuousTimeFunction) or
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
