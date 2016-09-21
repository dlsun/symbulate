import numpy as np

from .results import Results

class ProbabilitySpace:

    results = Results

    def __init__(self, draw):
        self.draw = draw

    def sim(self, n=None):
        return self.results(self.draw() for _ in range(n))

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

    def __init__(self, box, size=None, replace=True, probs=None):
        self.box = box
        self.size = size
        self.replace = replace
        self.probs = probs

    def draw(self):
        inds = np.random.choice(len(self.box), self.size, self.replace, self.probs)
        if self.size is None:
            return self.box[inds]
        else:
            return tuple(self.box[i] for i in inds)

class DeckOfCards(BoxModel):
    
    def __init__(self, size=None, replace=False):
        self.box = []
        for rank in list(range(2, 11)) + ["J", "Q", "K", "A"]:
            for suit in ["Diamonds", "Hearts", "Clubs", "Spades"]:
                self.box.append((rank, suit))
        self.size = size
        self.replace = replace
        self.probs = None
