import matplotlib.pyplot as plt

from .probability_space import Event
from .results import RVResults
from .utils import is_scalar, is_vector, get_dimension

class RV:
    """Defines a random variable.
    
    A random variable is a function which maps an outcome of
    a probability space to a number.  Simulating a random 
    variable is a two-step process: first, a draw is taken 
    from the underlying probability space; then, the function 
    is applied to that draw to obtain the realized value of
    the random variable.

    Args:
      probSpace (ProbabilitySpace): the underlying probability space
        of the random variable.
      fun (function, optional): a function that maps draws from the 
        probability space to numbers. (By default, the function is the 
        identity function. For named distributions, a draw from the
        underlying probability space is the value of the random
        variable itself, which is why the identity function is the 
        most frequently used.)

    Attributes:
      probSpace (ProbabilitySpace): the underlying probability space
        of the random variable.
      fun (function): a function that maps draws from the probability
        space to numbers.

    Examples:
      # a single draw is a sequence of 0s and 1s, e.g., (0, 0, 1, 0, 1)
      P = BoxModel([0, 1], size=5)
      # X counts the number of 1s in the draw, e.g., 5
      X = RV(P, sum)

      # the function is the identity, so Y has a Normal(0, 1) distribution
      Y = RV(Normal(0, 1)

      # a single draw from BivariateNormal is a tuple of two numbers
      P = BivariateNormal()
      # Z is the smaller of the two numbers
      Z = RV(P, min)
    """

    def __init__(self, probSpace, fun=lambda x: x):
        self.probSpace = probSpace
        self.fun = fun

    def draw(self):
        """A function that takes no arguments and returns a single 
          realization of the random variable.

        Example:
          X = RV(Normal(0, 1))
          X.draw() might return -0.9, for example.  
        """

        return self.fun(self.probSpace.draw())

    def sim(self, n):
        """Simulate n draws from probability space described by the random 
          variable.

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
        """Transform a random variable by a function.

        Args:
          function: function to apply to the random variable
        
        Example:
          X = RV(Exponential(1))
          Y = X.apply(log)

        Note: For most standard functions, you can apply the function to
          the random variable directly. For example, in the example above,
          Y = log(X) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def g(x):
            return log(x ** 2)
          Y = X.apply(g)
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

    ## The following operations all return Events
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
    """Defines a random variable conditional on an event.

    RVConditionals are typically produced when you condition a
    RV on an Event object.

    Args:
      random_variable (RV): the random variable whose conditional
        distribution is desired
      condition_event (Event): the event to condition on

    Attributes:
      random_variable (RV): the random variable whose conditional
        distribution is desired
      condition_event (Event): the event to condition on

    Examples:
      X, Y = RV(Binomial(10, 0.4) ** 2)
      (X | (X + Y == 5)).draw() # returns a value between 0 and 5.
    """

    def __init__(self, random_variable, condition_event):
        self.condition_event = condition_event
        super().__init__(random_variable.probSpace,
                         random_variable.fun)
        
    def draw(self):
        """A function that takes no arguments and returns a value from
          the conditional distribution of the random variable.

        Example:
          X, Y = RV(Binomial(10, 0.4) ** 2)
          (X | (X + Y == 5)).draw() might return a value of 4, for example.
        """
        probSpace = self.probSpace
        while True:
            outcome = probSpace.draw()
            if self.condition_event.fun(outcome):
                return self.fun(outcome)

