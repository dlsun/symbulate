import numbers


class IndexSet(object):

    def __init__(self):
        return

    def __getitem__(self, t):
        if t in self:
            return t
        else:
            raise KeyError("Time %.2f not in index set." % t)

    def __contains__(self, value):
        return False

    def __eq__(self, other):
        return type(other) == type(self)


class Reals(IndexSet):

    def __init__(self):
        return

    def __contains__(self, value):
        try:
            return -float("inf") < value < float("inf")
        except:
            return False


class Naturals(IndexSet):

    def __init__(self):
        return

    def __contains__(self, value):
        try:
            return (
                value >= 0 and
                (isinstance(value, numbers.Integral) or
                 value.is_integer())
            )
        except:
            return False


class DiscreteTimeSequence(IndexSet):

    def __init__(self, fs):
        self.fs = fs

    def __getitem__(self, n):
        return n / self.fs

    def __contains__(self, value):
        return float(value * self.fs).is_integer()

    def __eq__(self, index):
        return (
            isinstance(index, DiscreteTimeSequence) and
            (self.fs == index.fs)
        )

class Integers(DiscreteTimeSequence):

    def __init__(self):
        self.fs = 1
