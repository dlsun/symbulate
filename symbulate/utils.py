from numbers import Number

def is_scalar(x):
    if isinstance(x, Number):
        return True
    else:
        return False

def is_vector(x):
    if hasattr(x, "__iter__") and all(is_scalar(i) for i in x):
        return True
    else:
        return False

def has_consistent_dimension(x):
    """Checks if every vector in an iterable has the same dimension
    """
    if not all(is_vector(i) for i in x):
        return False
    else:
        lengths = [len(i) for i in x]
        for l in lengths:
            if l != lengths[0]:
                return False
        return True
