import numpy as np

from .distributions import Exponential
from .probability_space import ProbabilitySpace
from .random_processes import RandomProcess, TimeIndex
from .random_variables import RV
from .seed import get_seed
from .sequences import InfiniteSequence

EPS = 1e-15

class MarkovChain(RandomProcess):

    def check_transition_matrix(self, transition_matrix):
        for i, row in enumerate(transition_matrix):
            if abs(sum(row) - 1) > EPS:
                raise Exception("Rows of a transition matrix must sum to 1.")
            for j, q in enumerate(row):
                if q < 0:
                    raise Exception("Probabilities cannot be negative.")


    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        m = len(initial_dist)

        self.check_transition_matrix(transition_matrix)
        
        def draw():
            seed = get_seed()
            def x(n):
                np.random.seed(seed)
                state = np.random.choice(range(m), p=initial_dist)
                for _ in range(int(n)):
                    state = np.random.choice(range(m), p=transition_matrix[state])
                if state_labels is None:
                    return state
                else:
                    return state_labels[state]
            return InfiniteSequence(x)
        
        def fun(x, n):
            return x[n]
                
        super().__init__(ProbabilitySpace(draw), TimeIndex(fs=1), fun)

        
class ContinuousTimeMarkovChain(RandomProcess):

    def check_generator(self):
        for i, row in enumerate(self.generator_matrix):
            if abs(sum(row)) > EPS:
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
    
    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        m = len(initial_dist)
        T = TimeIndex(fs=float("inf"))

        self.generator_matrix = generator_matrix
        self.initial_dist = initial_dist
        self.state_labels = state_labels

        # check that generator matrix is valid
        self.check_generator()

        # determine transition matrix
        transition_matrix = []
        for i, row in enumerate(self.generator_matrix):
            rate = -row[i]
            transition_matrix.append(
                [p / rate if j != i else 0 for j, p in enumerate(row)]
            )
        self.transition_matrix = transition_matrix

        # probability space for the states
        P_states = MarkovChain(transition_matrix, initial_dist).probSpace

        # probability space for the jump times
        P_times = Exponential(1) ** float("inf")

        def fun(x, t):
            states, times = x[0], x[1]

            total_time = 0
            n = 0
            while True:
                state = states[n]
                rate = -self.generator_matrix[state][state]
                total_time += times[n] / rate
                if total_time > t:
                    break
                n += 1

            if state_labels is None:
                return state
            else:
                return state_labels[state]

        super().__init__(P_states * P_times, T, fun)

    def States(self):
        def fun(x):
            def f(n):
                state = x[0][n]
                if self.state_labels is None:
                    return state
                else:
                    return self.state_labels[state]
            return InfiniteSequence(f)

        return RV(self.probSpace, fun)
        
    def JumpTimes(self):
        def fun(x):
            def f(n):
                states, times = x[0], x[1]
                total_time = 0
                n = int(n)
                for i in range(n):
                    state = states[i]
                    rate = -self.generator_matrix[state][state]
                    total_time += times[i] / rate                    
                return total_time
            return InfiniteSequence(f)
        
        return RV(self.probSpace, fun)

    def InterjumpTimes(self):

        def fun(x):
            def f(n):
                states, times = x[0], x[1]
                n = int(n)
                state = states[n]
                rate = -self.generator_matrix[state][state]
                return times[n] / rate
            return InfiniteSequence(f)

        return RV(self.probSpace, fun)
