import math

from .random_variables import RV

pi = math.pi
e = math.e
inf = float("inf")

def exp(x):
    if isinstance(x, RV):
        return x.apply(math.exp)
    else:
        return math.exp(x)

def log(x, base=e):
    if isinstance(x, RV):
        return x.apply(lambda y: math.log(y, base))
    else:
        return math.log(x, base)

def factorial(x):
    if isinstance(x, RV):
        return x.apply(math.factorial)

def sqrt(x):
    if isinstance(x, RV):
        return x.apply(math.sqrt)
    else:
        return math.sqrt(x)

def sin(x):
    if isinstance(x, RV):
        return x.apply(math.sin)
    else:
        return math.sin(x)

def cos(x):
    if isinstance(x, RV):
        return x.apply(math.cos)
    else:
        return math.cos(x)

def tan(x):
    if isinstance(x, RV):
        return x.apply(math.tan)
    else:
        return math.tan(x)

def mean(x):
    return sum(x) / len(x)

def cumsum(x):
    sums = []
    total = 0
    for i in x:
        total += i
        sums.append(total)
    return tuple(sums)
