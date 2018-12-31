import numpy as np

from .base import Logical
from .result import Vector, InfiniteVector, join
from .results import Results


class ProbabilitySpace:
    """Defines a probability space.

    Attributes:
      draw (function): A function explaining how to draw one
        outcome from the probability space.
    """

    def __init__(self, draw):
        self.draw = draw

    def sim(self, n):
        """Simulate n draws from probability space.

        Args:
          n (int): How many draws to make.

        Returns:
          Results: A list-like object containing the simulation results.
        """
        return Results(self.draw() for _ in range(n))

    def check_same(self, other):
        if self != other:
            raise Exception("Events must be defined on same probability space.")

    def apply(self, func):
        """Define a new probability space.

        Args:
          func: function to apply to each realization.

        Returns:
          A new ProbabilitySpace where each realization is func applied to
          a realization from the current probability space.
        """
        def draw():
            return func(self.draw())
        return ProbabilitySpace(draw)

    def __mul__(self, other):
        def draw():
            return join(self.draw(), other.draw())
        return ProbabilitySpace(draw)

    def __pow__(self, exponent):
        if exponent == float("inf"):
            def draw():
                def _func(_):
                    return self.draw()
                return InfiniteVector(_func)
        else:
            def draw():
                return Vector(self.draw() for _ in range(exponent))
        return ProbabilitySpace(draw)


class Event(Logical):

    def __init__(self, prob_space, func):
        self.prob_space = prob_space
        self.func = func

    def check_same_prob_space(self, other):
        self.prob_space.check_same(other.prob_space)

    # The Logical superclass will use this to define the three
    # logical operations: and (&), or (|), not (~).
    def _logical_factory(self, op):

        def _op_func(self, other=None):
            # other will be None when op is the "not" operator
            if other is None:
                return Event(self.prob_space,
                             lambda outcome: op(self.func(outcome)))
            else:
                if isinstance(other, Event):
                    self.check_same_prob_space(other)
                else:
                    raise TypeError(
                        "Logical operations are only defined "
                        "between two Events, not between an Event "
                        "and a %s." % type(other).__name__)
                return Event(self.prob_space,
                             lambda outcome: op(self.func(outcome),
                                                other.func(outcome)))

        return _op_func

    # This prevents users from writing expressions like 2 < X < 5,
    # which evaluate to ((2 < X) and (X < 5)). This unfortunately
    # is not well-defined in Python and cannot be overloaded.
    def __bool__(self):
        raise Exception("Cannot cast an Event to a boolean. "
                        "You may be getting this error if you "
                        "wrote an expression like (2 < X < 5). "
                        "Try ((2 < X) & (X < 5)) instead.")

    def draw(self):
        return self.func(self.prob_space.draw())

    def sim(self, n):
        return Results(self.draw() for _ in range(n))


class BoxModel(ProbabilitySpace):
    """Defines a probability space from a box model.

    Attributes:
      box (list-like or dict-like): The box to sample from.
        The box can be specified either directly as a list
        of objects or indirectly as a dict of objects and
        their counts.
      size (int): How many draws to make.
      replace (bool-like): Sample with replacement or without?
      probs (list): Probabilities of sampling each ticket
        (by default, all tickets are equally likely). Note
        that this is ignored if box is specified as a dict.
      order_matters (bool): Should we count different
        orderings of the same tickets as different outcomes?
        Essentially, this determines whether the draws are
        sorted before returning or not.
    """

    def __init__(self, box, size=None, replace=True, probs=None, order_matters=True):
        if isinstance(box, list):
            self.box = box
            self.probs = probs
        elif isinstance(box, dict):
            self.box = []
            for ticket, count in box.items():
                self.box.extend([ticket] * count)
            self.probs = None
        else:
            raise Exception(
                "Box must be specified either as a list or a dict."
            )
        self.size = None if size == 1 else size
        self.replace = replace
        self.order_matters = order_matters
        self.output_type = Vector
        self.infinite_output_type = InfiniteVector

        # If drawing without replacement, check that the number
        # of draws does not exceed the number of tickets in the box.
        if not self.replace and self.size > len(self.box):
            raise Exception(
                "Cannot draw more tickets (without replacement) "
                "than there are tickets in the box."
            )

    def draw(self):
        """
        A function that takes no arguments and returns a value(s) from the
            "box" argument of the BoxModel.

        Based on BoxModel inputs:
        Number of values returned depends on the input of the "size"
            argument.
        Whether or not a value in the box can appear multiple times
            depends on the "replace" argument.
        If a list of probabilities is specified, values drawn will be drawn
            with the specified probabilities.
        """

        def draw_inds(size):
            return np.random.choice(len(self.box), size, self.replace, self.probs)

        if self.size is None:
            return self.box[draw_inds(None)]
        elif self.size == float("inf"):
            def _func(_):
                return self.box[draw_inds(None)]
            return self.infinite_output_type(_func)
        else:
            draws = [self.box[i] for i in draw_inds(self.size)]
            if not self.order_matters:
                draws.sort()
            return self.output_type(draws)


class DeckOfCards(BoxModel):
    """Defines the probability space for drawing from a deck of cards.

    Attributes:
      size (int): How many draws to make.
      replace (bool): Sample with replacement or without?
      order_matters (bool): Should we count different orderings of
        the same cards as different outcomes or the same outcome?
    """

    def __init__(self, size=None, replace=False, order_matters=True):
        box = []
        for rank in list(range(2, 11)) + ["J", "Q", "K", "A"]:
            for suit in ["Diamonds", "Hearts", "Clubs", "Spades"]:
                box.append((rank, suit))
        super().__init__(box, size, replace,
                         probs=None, order_matters=order_matters)
