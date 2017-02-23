import numpy as np

from .probability_space import ProbabilitySpace
from .random_processes import RandomProcess, TimeIndex

class MarkovChain(RandomProcess):

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        n = len(initial_dist)
        def sim():
            def generator():
                i = np.random.choice(range(n), p=initial_dist)
                while True:
                    if state_labels is None:
                        yield i
                    else:
                        yield state_labels[i]
                    i = np.random.choice(range(n), p=transition_matrix[i])
            return generator()

        def fun(x, t):
            for i, state in enumerate(x):
                if i == t:
                    return state
                
        super().__init__(ProbabilitySpace(sim), TimeIndex(), fun)
