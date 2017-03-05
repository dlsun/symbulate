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

def get_dimension(x):
    """Gets the dimension of the vectors in an iterable if it is
    consistent, otherwise returns None
    """
    lengths = [1 if is_scalar(i) else len(i) for i in x]
    for l in lengths:
        if l != lengths[0]:
            return None
    return lengths[0]
