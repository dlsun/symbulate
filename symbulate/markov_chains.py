import numpy as np

from .probability_space import ProbabilitySpace
from .random_processes import RandomProcess, TimeIndex
from .seed import get_seed
from .sequences import InfiniteSequence

class MarkovChain(RandomProcess):

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        n = len(initial_dist)
        def draw():
            seed = get_seed()
            def x(t):
                np.random.seed(seed)
                state = np.random.choice(range(n), p=initial_dist)
                for i in range(int(t)):
                    state = np.random.choice(range(n), p=transition_matrix[state])
                if state_labels is None:
                    return state
                else:
                    return state_labels[state]
            return InfiniteSequence(x)

        def fun(x, t):
            return x[t]
                
        super().__init__(ProbabilitySpace(draw), TimeIndex(), fun)
