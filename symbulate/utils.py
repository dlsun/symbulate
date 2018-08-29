from numbers import Number

from .result import Scalar

def is_scalar(x):
    return isinstance(x, Number) or isinstance(x, Scalar)

def is_vector(x):
    if hasattr(x, "__len__") and all(is_scalar(i) for i in x):
        return True
    else:
        return False

