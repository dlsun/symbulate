import matplotlib.pyplot as plt

from .probability_space import Event
from .results import RVResults
from .utils import is_scalar, is_vector, get_dimension

class RV:

    def __init__(self, probSpace, fun=lambda x: x):
        self.probSpace = probSpace
        self.fun = fun
        """
        Initializes an instance of the random variable class. 
        
        The random variable will be assigned probabilities specific
            to the distribution of the "probSpace" argument.
        """

    def draw(self):
        """
        A function that takes no arguments and returns a single realization
            of the random variable.

        Ex:  X = RV(Normal(0, 1))
             X.draw() might return -0.9, for example.  
        """

        return self.fun(self.probSpace.draw())

    def sim(self, n):
        """Simulate n draws from probability space described by the random variable.

        Args:
          n (int): How many draws to make.

        Returns:
          Results: A list-like object containing the simulation results.
        """

        return RVResults(self.draw() for _ in range(n))

    def check_same_probSpace(self, other):
        if is_scalar(other):
            return
        else:
            self.probSpace.check_same(other.probSpace)

    def apply(self, function):
        """
        Args:
            function: function to apply to the random variable (e.g., log, sqrt, exp)
        
        Input function is applied to the output results.
        """
        
        def f_new(outcome):
            return function(self.fun(outcome))
        return RV(self.probSpace, f_new)

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

    # e.g., abs(X)
    def __abs__(self):
        return self.apply(lambda x: abs(x))

    # The code for most operations (+, -, *, /, ...) is the
    # same, except for the operation itself. The following 
    # factory function takes in the the operation and 
    # generates the code to perform that operation.
    def _operation_factory(self, op):

        def op_fun(self, other):
            self.check_same_probSpace(other)
            if is_scalar(other):
                return self.apply(lambda x: op(x, other))
            elif isinstance(other, RV):
                def fun(outcome):
                    a = self.fun(outcome)
                    b = other.fun(outcome)
                    if is_vector(a) and is_vector(b) and len(a) == len(b):
                        return tuple(op(i, j) for i, j in zip(a, b))
                    elif is_scalar(a) and is_scalar(b):
                        return op(a, b)
                    else:
                        raise Exception("Could not perform operation on the outcomes %s and %s." % (str(a), str(b)))
                return RV(self.probSpace, fun)
            else:
                return NotImplemented

        return op_fun

    # e.g., X + Y or X + 3
    def __add__(self, other):
        op_fun = self._operation_factory(lambda x, y: x + y)
        return op_fun(self, other)

    # e.g., 3 + X
    def __radd__(self, other):
        return self.__add__(other)

    # e.g., X - Y or X - 3
    def __sub__(self, other):
        op_fun = self._operation_factory(lambda x, y: x - y)
        return op_fun(self, other)

    # e.g., 3 - X
    def __rsub__(self, other):
        return -1 * self.__sub__(other)

    # e.g., -X
    def __neg__(self):
        return -1 * self

    # e.g., X * Y or X * 2
    def __mul__(self, other):
        op_fun = self._operation_factory(lambda x, y: x * y)
        return op_fun(self, other)
            
    # e.g., 2 * X
    def __rmul__(self, other):
        return self.__mul__(other)

    # e.g., X / Y or X / 2
    def __truediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: x / y)
        return op_fun(self, other)

    # e.g., 2 / X
    def __rtruediv__(self, other):
        op_fun = self._operation_factory(lambda x, y: y / x)
        return op_fun(self, other)

    # e.g., X ** 2
    def __pow__(self, other):
        op_fun = self._operation_factory(lambda x, y: x ** y)
        return op_fun(self, other)

    # e.g., 2 ** X
    def __rpow__(self, other):
        op_fun = self._operation_factory(lambda x, y: y ** x)
        return op_fun(self, other)

    # Alternative notation for powers: e.g., X ^ 2
    def __xor__(self, other):
        return self.__pow__(other)
    
    # Alternative notation for powers: e.g., 2 ^ X
    def __rxor__(self, other):
        return self.__rpow__(other)

    # Define a joint distribution of two random variables: e.g., X & Y
    def __and__(self, other):
        self.check_same_probSpace(other)
        if isinstance(other, RV):
            def fun(outcome):
                a = self.fun(outcome)
                b = other.fun(outcome)
                a = tuple(a) if is_vector(a) else (a, )
                b = tuple(b) if is_vector(b) else (b, )
                return a + b
            return RV(self.probSpace, fun)
        else:
            raise Exception("Joint distributions are only defined for RVs.")

    ## The following function all return Events
    ## (Events are used to define conditional distributions)

    # e.g., X < 3
    def __lt__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) < other)
        elif isinstance(other, RV):
            return Event(self.probSpace,
                         lambda x: self.fun(x) < other.fun(x))
        else:
            raise NotImplementedError

    # e.g., X <= 3
    def __le__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) <= other)
        elif isinstance(other, RV):
            return Event(self.probSpace,
                         lambda x: self.fun(x) <= other.fun(x))
        else:
            raise NotImplementedError

    # e.g., X > 3
    def __gt__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) > other)
        elif isinstance(other, RV):
            return Event(self.probSpace,
                         lambda x: self.fun(x) > other.fun(x))
        else:
            raise NotImplementedError

    # e.g., X >= 3
    def __ge__(self, other):
        if is_scalar(other):
            return Event(self.probSpace,
                         lambda x: self.fun(x) >= other)
        elif isinstance(other, RV):
            return Event(self.probSpace,
                         lambda x: self.fun(x) >= other.fun(x))
        else:
            raise NotImplementedError

    # e.g., X == 3
    def __eq__(self, other):
        if is_scalar(other) or type(other) == str:
            return Event(self.probSpace,
                         lambda x: self.fun(x) == other)
        elif isinstance(other, RV):
            return Event(self.probSpace,
                         lambda x: self.fun(x) == other.fun(x))
        else:
            raise NotImplementedError

    # e.g., X != 3
    def __ne__(self, other):
        if is_scalar(other) or type(other) == str:
            return Event(self.probSpace,
                         lambda x: self.fun(x) != other)
        elif isinstance(other, RV):
            return Event(self.probSpace,
                         lambda x: self.fun(x) != other.fun(x))
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
        """
        A function that takes no arguments and returns a value from
            the conditional distribution of the random variable.

        e.g. X,Y = RV(Binomial(2, 0.4)**2)
             A = (X | (X + Y == 3)) might return a value of 2, for example.
        
        """
        probSpace = self.probSpace
        while True:
            outcome = probSpace.draw()
            if self.condition_event.fun(outcome):
                return self.fun(outcome)

