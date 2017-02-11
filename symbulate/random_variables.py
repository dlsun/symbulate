import matplotlib.pyplot as plt

from copy import deepcopy as copy

from .probability_space import Event
from .results import RVResults
from .utils import is_scalar, is_vector, get_dimension

class RV:

    def __init__(self, probSpace, fun=lambda x: x):
        self.probSpace = probSpace
        self.fun = fun

    def draw(self):
        return self.fun(self.probSpace.draw())

    def sim(self, n):
        return RVResults(self.draw() for _ in range(n))

    def check_same_probSpace(self, other):
        if self.probSpace != other.probSpace:
            raise Exception("Random variables must be defined on same probability space.")

    def apply(self, function):
        def f_new(outcome):
            return function(self.fun(outcome))
        return RV(self.probSpace, copy(f_new))

    # This allows us to unpack a random vector,
    # e.g., X, Y = RV(BoxModel([0, 1], size=2))
    def __iter__(self):
        test = self.sim(10)
        for i in range(get_dimension(test)):
            yield self[i]

    def __getitem__(self, i):
        # if the indices are a list, return a random vector
        if hasattr(i, "__iter__"):
            return self.apply(lambda x: tuple(x[j] for j in i))
        # otherwise, return the ith value
        else:
            return self.apply(lambda x: x[i])

    # e.g., X + Y or X + 3
    def __add__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x + other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i + j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a + b
            else:
                raise Exception("I don't know how to add those two random variables.")
        return RV(self.probSpace, fun)

    # e.g., 3 + X
    def __radd__(self, other):
        return self.__add__(other)

    # e.g., X - Y or X - 3
    def __sub__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x - other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i - j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a - b
            else:
                raise Exception("I don't know how to subtract those two random variables.")
        return RV(self.probSpace, fun)

    # e.g., 3 - X
    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    # e.g., -X
    def __neg__(self):
        return -1 * self

    # e.g., X * Y or X * 2
    def __mul__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x * other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i * j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a * b
            else:
                raise Exception("I don't know how to multiply those two random variables.")
        return RV(self.probSpace, fun)

    # e.g., 2 * X
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., X / Y or X / 2
    def __truediv__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x / other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i / j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a / b
            else:
                raise Exception("I don't know how to divide those two random variables.")
        return RV(self.probSpace, fun)

    # e.g., 2 / X
    def __rtruediv__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: other / x)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(j / i for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return b / a
            else:
                raise Exception("I don't know how to divide those two random variables.")
        return RV(self.probSpace, fun)

    # e.g., X ** 2
    def __pow__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: x ** other)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(i ** j for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return a ** b
            else:
                raise Exception("I don't know how to raise that random variable to that power.")
        return RV(self.probSpace, fun)

    # e.g., 2 ** X
    def __rpow__(self, other):
        if is_scalar(other):
            return self.apply(lambda x: other ** x)
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            if is_vector(a) and is_vector(b) and len(a) == len(b):
                return tuple(j ** i for i, j in zip(a, b))
            elif is_scalar(a) and is_scalar(b):
                return b ** a
            else:
                raise Exception("I don't know how to raise that random variable to that power.")
        return RV(self.probSpace, fun)

    # Alternative notation for powers: e.g., X ^ 2
    def __xor__(self, other):
        return self.__pow__(other)
    
    # Alternative notation for powers: e.g., 2 ^ X
    def __rxor__(self, other):
        return self.__rpow__(other)

    # Define a joint distribution of two random variables: e.g., X & Y
    def __and__(self, other):
        self.check_same_probSpace(other)
        def fun(outcome):
            a = self.fun(outcome)
            b = other.fun(outcome)
            a = tuple(a) if is_vector(a) else (a, )
            b = tuple(b) if is_vector(b) else (b, )
            return a + b
        return RV(self.probSpace, fun)

    ## The following function all return Events
    ## (Events are used to define conditional distributions)

    # e.g., X < 3
    def __lt__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) < other)
        else:
            raise NotImplementedError

    # e.g., X <= 3
    def __le__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) <= other)
        else:
            raise NotImplementedError

    # e.g., X > 3
    def __gt__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) > other)
        else:
            raise NotImplementedError

    # e.g., X >= 3
    def __ge__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) >= other)
        else:
            raise NotImplementedError

    # e.g., X == 3
    def __eq__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) == other)
        else:
            raise NotImplementedError

    # e.g., X != 3
    def __neq__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) != other)
        else:
            raise NotImplementedError

    # Define conditional distribution of random variable.
    # e.g., X | (X > 3)
    def __or__(self, condition_event):
        # Check that the random variable and event are
        # defined on the same probability space.
        self.check_same_probSpace(condition_event)
        if isinstance(condition_event, Event):
            return RVConditional(self, condition_event)
        else:
            raise NotImplementedError

class RVConditional(RV):

    def __init__(self, random_variable, condition_event):
        self.condition_event = condition_event
        super().__init__(random_variable.probSpace,
                         random_variable.fun)
        
    def draw(self):
        probSpace = self.probSpace
        while True:
            outcome = probSpace.draw()
            if self.condition_event.fun(outcome):
                return self.fun(outcome)

