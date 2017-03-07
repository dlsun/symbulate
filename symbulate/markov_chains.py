import numpy as np

from .distributions import Exponential
from .probability_space import ProbabilitySpace
from .random_processes import RandomProcess, TimeIndex
from .seed import get_seed
from .sequences import InfiniteSequence


class MarkovChain(RandomProcess):

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        n = len(initial_dist)
        T = TimeIndex(fs=1)
        
        def draw():
            seed = get_seed()
            def x(t):
                np.random.seed(seed)
                state = np.random.choice(range(n), p=initial_dist)
                for _ in range(int(t)):
                    state = np.random.choice(range(n), p=transition_matrix[state])
                if state_labels is None:
                    return state
                else:
                    return state_labels[state]
            return InfiniteSequence(x, T)
        
        def fun(x, t):
            return x[t]
                
        super().__init__(ProbabilitySpace(draw), T, fun)

        
class ContinuousTimeMarkovChain(RandomProcess):

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        n = len(initial_dist)
        T = TimeIndex(fs=float("inf"))

        # check that generator_matrix is valid
        for i, row in enumerate(generator_matrix):
            if sum(row) != 0:
                raise Exception("Rows of a generator matrix must sum to 0.")
            for j, q in enumerate(row):
                if j == i:
                    if row[j] > 0:
                        raise Exception("Diagonal elements of a generator matrix " +
                                        "cannot be positive.")
                else:
                    if row[j] < 0:
                        raise Exception("Off-diagonal elements of a generator matrix " +
                                        "cannot be negative.")

        def draw():
            seed = get_seed()
            def x(t):
                np.random.seed(seed)
                state = np.random.choice(range(n), p=initial_dist)
                total_time = 0
                while True:
                    row = generator_matrix[state]
                    rate = -row[state]
                    total_time += Exponential(rate).draw()
                    if total_time > t:
                        break
                    probs = [p / rate if p >= 0 else 0 for p in row]
                    state = np.random.choice(range(n), p=probs)
                if state_labels is None:
                    return state
                else:
                    return state_labels[state]
            return InfiniteSequence(x, T)

        def fun(x, t):
            return x[t]

        super().__init__(ProbabilitySpace(draw), T, fun)
        
