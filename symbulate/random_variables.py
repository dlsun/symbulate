from .base import Arithmetic, Transformable, Comparable
from .probability_space import Event
from .result import Vector, join, is_scalar, is_numeric_vector
from .results import RVResults

class RV(Arithmetic, Transformable, Comparable):
    """Defines a random variable.

    A random variable is a function which maps an outcome of
    a probability space to a number.  Simulating a random
    variable is a two-step process: first, a draw is taken
    from the underlying probability space; then, the function
    is applied to that draw to obtain the realized value of
    the random variable.

    Args:
      prob_space (ProbabilitySpace): the underlying probability space
        of the random variable.
      func (function, optional): a function that maps draws from the
        probability space to numbers. (By default, the function is the
        identity function. For named distributions, a draw from the
        underlying probability space is the value of the random
        variable itself, which is why the identity function is the
        most frequently used.)

    Attributes:
      prob_space (ProbabilitySpace): the underlying probability space
        of the random variable.
      func (function): a function that maps draws from the probability
        space to numbers.

    Examples:
      # a single draw is a sequence of 0s and 1s, e.g., (0, 0, 1, 0, 1)
      P = BoxModel([0, 1], size=5)
      # X counts the number of 1s in the draw, e.g., 5
      X = RV(P, sum)

      # the function is the identity, so Y has a Normal(0, 1) distribution
      Y = RV(Normal(0, 1))

      # a single draw from BivariateNormal is a tuple of two numbers
      P = BivariateNormal()
      # Z is the smaller of the two numbers
      Z = RV(P, min)
    """

    def __init__(self, prob_space, func=lambda x: x):
        self.prob_space = prob_space
        self.func = func

    def draw(self):
        """A function that takes no arguments and returns a single
          realization of the random variable.

        Example:
          X = RV(Normal(0, 1))
          X.draw() might return -0.9, for example.
        """
        return self.func(self.prob_space.draw())

    def sim(self, n):
        """Simulate n draws from probability space described by the random
          variable.

        Args:
          n (int): How many draws to make.

        Returns:
          RVResults: A list-like object containing the simulation results.
        """

        return RVResults(self.draw() for _ in range(n))

    def __call__(self, outcome):
        print("Warning: Calling an RV as a function simply applies the "
              "function that defines the RV to the input, regardless of "
              "whether that input is a possible outcome in the underlying "
              "probability space.")
        return self.func(outcome)

    def check_same_prob_space(self, other):
        if hasattr(other, "prob_space"):
            self.prob_space.check_same(other.prob_space)

    def apply(self, func):
        """Transform a random variable by a function.

        Args:
          func: function to apply to the random variable

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
        def _func(outcome):
            return func(self.func(outcome))
        return RV(self.prob_space, _func)

    # This allows us to unpack a random vector,
    # e.g., X, Y = RV(BoxModel([0, 1], size=2))
    def __iter__(self):
        test = self.draw()
        if hasattr(test, "__iter__"):
            for i in range(len(test)):
                yield self[i]
        else:
            raise Exception(
                "To unpack a random vector, the RV needs to "
                "have multiple components."
            )

    def __getitem__(self, n):
        # if n is an RV, return a new random variable
        if isinstance(n, RV):
            return RV(self.prob_space,
                      lambda x: self.func(x)[n.func(x)])
        # if the indices are a list, return a random vector
        elif is_numeric_vector(n):
            return self.apply(
                lambda x: Vector(x[i] for i in n)
            )
        # if the indices are a slice, return a random vector
        elif isinstance(n, slice):
            return self.apply(
                lambda x: Vector(x[i] for i in
                                 range(n.start, n.stop, n.step or 1))
                )
        # otherwise, return the nth value
        return self.apply(lambda x: x[n])

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.)
    def _operation_factory(self, op):

        def _op_func(self, other):
            # operations between this RV and another RV
            if isinstance(other, RV):
                self.check_same_prob_space(other)
                def _func(outcome):
                    return op(self.func(outcome), other.func(outcome))
                return RV(self.prob_space, _func)
            # operations between this RV and a scalar
            return self.apply(lambda x: op(x, other))

        return _op_func

    # The Comparison superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    # Note that a comparison of a random variable returns an Event.
    def _comparison_factory(self, op):

        def _op_func(self, other):
            if is_scalar(other):
                return Event(self.prob_space,
                             lambda x: op(self.func(x), other))
            elif isinstance(other, RV):
                self.check_same_prob_space(other)
                return Event(self.prob_space,
                             lambda x: op(self.func(x), other.func(x)))
            raise NotImplementedError(
                "Comparisons are only defined between two RVs or "
                "between an RV and a scalar."
            )

        return _op_func


    # Define a joint distribution of two random variables: e.g., X & Y
    def __and__(self, other):
        self.check_same_prob_space(other)
        if isinstance(other, RV):
            def _func(outcome):
                return join(self.func(outcome), other.func(outcome))
        elif is_scalar(other):
            def _func(outcome):
                return join(self.func(outcome), other)
        else:
            raise Exception("Joint distributions are only defined for RVs.")
        return RV(self.prob_space, _func)

    def __rand__(self, other):
        self.check_same_prob_space(other)
        if is_scalar(other):
            def _func(outcome):
                return join(other, self.func(outcome))
        return RV(self.prob_space, _func)

    # Define conditional distribution of random variable.
    # e.g., X | (X > 3)
    def __or__(self, condition_event):
        # Check that the random variable and event are
        # defined on the same probability space.
        self.check_same_prob_space(condition_event)
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
        super().__init__(random_variable.prob_space,
                         random_variable.func)

    def draw(self):
        """A function that takes no arguments and returns a value from
          the conditional distribution of the random variable.

        Example:
          X, Y = RV(Binomial(10, 0.4) ** 2)
          (X | (X + Y == 5)).draw() might return a value of 4, for example.
        """
        while True:
            outcome = self.prob_space.draw()
            if self.condition_event.func(outcome):
                return self.func(outcome)
