import numpy as np

from .result import Vector, InfiniteVector
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

    def __mul__(self, other):
        def draw():
            return self.draw().join(other.draw())
        return ProbabilitySpace(draw)

    def __pow__(self, exponent):
        if exponent == float("inf"):
            def draw():
                result = InfiniteVector()
                def fn(n):
                    return self.draw()
                result.fn = fn
                return result
        else:
            def draw():
                return Vector(self.draw() for _ in range(exponent))
        return ProbabilitySpace(draw)
            

class Event:

    def __init__(self, probSpace, fun):
        self.probSpace = probSpace
        self.fun = fun

    def check_same_probSpace(self, other):
        self.probSpace.check_same(other.probSpace)

    # define the event (A & B)
    def __and__(self, other):
        self.check_same_probSpace(other)
        if isinstance(other, Event):
            return Event(self.probSpace,
                         lambda x: self.fun(x) and other.fun(x))

    # define the event (A | B)
    def __or__(self, other):
        self.check_same_probSpace(other)
        if isinstance(other, Event):
            return Event(self.probSpace,
                         lambda x: self.fun(x) or other.fun(x))

    # define the event (-A)
    def __invert__(self):
        return Event(self.probSpace, lambda x: not self.fun(x))

    # This prevents users from writing expressions like 2 < X < 5,
    # which evaluate to ((2 < X) and (X < 5)). This unfortunately
    # is not well-defined in Python and cannot be overloaded.
    def __bool__(self):
        raise Exception("I do not know how to cast " + 
                        "an event to a boolean. " +
                        "If you wrote an expression " +
                        "like (2 < X < 5), try writing " +
                        "((2 < X) & (X < 5)) instead."
                    )

    def draw(self):
        return self.fun(self.probSpace.draw())

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
            for k, v in box.items():
                self.box.extend([k] * v)
            self.probs = None
        else:
            raise Exception(
                "Box must be specified either as a list or a dict."
            )
        self.size = None if size == 1 else size
        self.replace = replace
        self.order_matters = order_matters

        # If drawing without replacement, check to make sure that
        # the number of draws does not exceed the number in the box.
        if not self.replace and self.size > len(self.box):
            raise Exception(
                "I cannot draw more tickets without replacement "
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
            def f(n, vector):
                return self.box[draw_inds(None)]
            return InfiniteVector(f)
        else:
            draws = [self.box[i] for i in draw_inds(self.size)]
            if not self.order_matters:
                draws.sort()
            return Vector(draws)

class DeckOfCards(BoxModel):
    """Defines the probability space for drawing from a deck of cards.

    Attributes:
      size (int): How many draws to make.
      replace (bool): Sample with replacement or without?
    """
    
    def __init__(self, size=None, replace=False, order_matters=True):
        self.box = []
        for rank in list(range(2, 11)) + ["J", "Q", "K", "A"]:
            for suit in ["Diamonds", "Hearts", "Clubs", "Spades"]:
                self.box.append((rank, suit))
        self.size = None if size == 1 else size
        self.replace = replace
        self.probs = None
        self.order_matters = order_matters

