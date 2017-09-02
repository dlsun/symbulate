import math
import numpy as np
import operator as op

from .random_variables import RV
from .random_processes import RandomProcess
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
    total = 0
    sums = [total]
    for i in x:
        total += i
        sums.append(total)
    return tuple(sums)

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
