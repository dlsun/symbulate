import math

from .random_variables import RV
from .random_processes import RandomProcess

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
    return operation_factory(lambda y: math.log(y, base))(x)

def mean(x):
    return sum(x) / len(x)

def cumsum(x):
    total = 0
    sums = [total]
    for i in x:
        total += i
        sums.append(total)
    return tuple(sums)
