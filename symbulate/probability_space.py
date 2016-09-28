import numpy as np

from .results import Results

class ProbabilitySpace:
    """Defines a probability space

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

    def __mul__(self, other):
        def draw():
            a = self.draw() if type(self.draw()) == tuple else (self.draw(),)
            b = other.draw() if type(other.draw()) == tuple else (other.draw(),)
            return a + b
        return ProbabilitySpace(draw)

    def __pow__(self, exponent):
        def draw():
            return tuple(self.draw() for _ in range(exponent))
        return ProbabilitySpace(draw)

class BoxModel(ProbabilitySpace):
    """Defines a probability space from a box model.

    Attributes:
      box (list): The box to sample from.
      size (int): How many draws to make.
      replace (bool): Sample with replacement or without?
      probs (list): Probabilities of sampling each ticket
        (by default, all tickets are equally likely)
      order_matters (bool): Should we count different
        orderings of the same tickets as different outcomes?
        Essentially, this determines whether the draws are
        sorted before returning or not.
    """

    def __init__(self, box, size=None, replace=True, probs=None, order_matters=True):
        self.box = box
        self.size = None if size == 1 else size
        self.replace = replace
        self.probs = probs
        self.order_matters = order_matters

    def draw(self):
        inds = np.random.choice(len(self.box), self.size, self.replace, self.probs)
        if self.size is None:
            return self.box[inds]
        else:
            draws = [self.box[i] for i in inds]
            if not self.order_matters:
                draws.sort()
            return tuple(draws)

class DeckOfCards(BoxModel):
    """Defines the probability space for drawing from a deck of cards.

    Attributes:
      size (int): How many draws to make.
      replace (bool): Sample with replacement or without?
    """
    
    def __init__(self, size=None, replace=False):
        self.box = []
        for rank in list(range(2, 11)) + ["J", "Q", "K", "A"]:
            for suit in ["Diamonds", "Hearts", "Clubs", "Spades"]:
                self.box.append((rank, suit))
        self.size = size
        self.replace = replace
        self.probs = None
