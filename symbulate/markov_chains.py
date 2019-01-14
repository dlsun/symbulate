import numpy as np

from .distributions import Exponential
from .math import inf
from .probability_space import ProbabilitySpace
from .random_variables import RV
from .result import (
    InfiniteVector, ContinuousTimeFunction, DiscreteValued
)

EPS = 1e-15


class MarkovChainResult(InfiniteVector, DiscreteValued):

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        # Check transition matrix
        for row in transition_matrix:
            if abs(sum(row) - 1) > EPS:
                raise Exception("Rows of a transition matrix must sum to 1.")
            for p in row:
                if p < 0:
                    raise Exception("Probabilities cannot be negative.")
        # Check that dimensions agree
        self.transition_matrix = np.asarray(transition_matrix)
        m, n = self.transition_matrix.shape
        if m != n:
            raise Exception("Transition matrix must be square.")
        if len(initial_dist) != n:
            raise Exception("Initial distribution must be a vector whose "
                            "length matches the dimensions of the "
                            "transition matrix.")
        self.initial_dist = initial_dist
        # Process state labels
        if state_labels is not None:
            if len(state_labels) != n:
                raise Exception("There must be as many state labels as "
                                "there are states.")
            self.state_labels = state_labels
        else:
            self.state_labels = range(n)
        self.n_states = n

        # Generate initial state.
        # (self.states stores the indexes of the states, while
        #  self.values stores the labels of the states.)
        state = np.random.choice(range(n), p=self.initial_dist)
        self.states = [state]

        def _func(n):
            m = len(self.states)
            # If nth state not generated yet, generate it.
            if n >= m:
                state = self.states[m - 1]
                for _ in range(m, n + 1):
                    state = np.random.choice(
                        range(self.n_states),
                        p=self.transition_matrix[state, :]
                    )
                    self.states.append(state)
            else:
                state = self.states[n]
            return self.state_labels[state]

        super().__init__(_func)

    def get_states(self):
        return self


class MarkovChainProbabilitySpace(ProbabilitySpace):

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Initialize probability space for a (discrete-time) Markov chain.

        Args:
          transition_matrix: n x n transition matrix
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        def _draw():
            return MarkovChainResult(transition_matrix,
                                     initial_dist,
                                     state_labels)

        super().__init__(_draw)


class MarkovChain(RV):

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Initialize a (discrete-time) Markov chain.

        Args:
          transition_matrix: n x n transition matrix
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        prob_space = MarkovChainProbabilitySpace(
            transition_matrix,
            initial_dist,
            state_labels)
        super().__init__(prob_space)


class ContinuousTimeMarkovChainResult(ContinuousTimeFunction,
                                      DiscreteValued):

    def __init__(self, states, rates,
                 unscaled_interarrival_times,
                 state_labels):
        self.states = states
        self.rates = rates
        self.times = unscaled_interarrival_times
        self.state_labels = state_labels

        # Define an InfiniteVector of the interarrival times.
        def interarrival_times(n):
            for i in range(n + 1):
                state = self.states[i]
                interarrival_time = self.times[i] / self.rates[state]
            return interarrival_time
        self.interarrival_times = InfiniteVector(interarrival_times)

        def _func(t):
            total_time = 0
            n = 0
            while True:
                state = self.states[n]
                total_time += self.times[n] / self.rates[state]
                if total_time > t:
                    return self.state_labels[state]
                n += 1
        super().__init__(_func)


class ContinuousTimeMarkovChainProbabilitySpace(ProbabilitySpace):

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        """Initialize a probability space for a continuous-time Markov chain.

        Args:
          generator_matrix: n x n generator matrix whose rows sum to 0
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        # Check generator matrix
        for i, row in enumerate(generator_matrix):
            if abs(sum(row)) > EPS:
                raise Exception("Rows of a generator matrix must sum to 0.")
            for j, q in enumerate(row):
                if j == i:
                    if q > 0:
                        raise Exception("Diagonal elements of a generator matrix " +
                                        "cannot be positive.")
                else:
                    if q < 0:
                        raise Exception("Off-diagonal elements of a generator matrix " +
                                        "cannot be negative.")
        # Check that dimensions agree
        self.generator_matrix = np.array(generator_matrix)
        m, n = self.generator_matrix.shape
        if m != n:
            raise Exception("Transition matrix must be square.")
        if len(initial_dist) != n:
            raise Exception("Initial distribution must be a vector whose "
                            "length matches the dimensions of the "
                            "transition matrix.")
        self.initial_dist = initial_dist
        # Process state labels
        if state_labels is not None:
            if len(state_labels) != n:
                raise Exception("There must be as many state labels as "
                                "there are states.")
            self.state_labels = state_labels
        else:
            self.state_labels = range(n)
        self.n_states = n

        # determine transition matrix
        transition_matrix = []
        for i, row in enumerate(self.generator_matrix):
            rate = -row[i]
            transition_matrix.append(
                [p / rate if j != i else 0 for j, p in enumerate(row)]
            )
        self.transition_matrix = np.array(transition_matrix)

        # A continuous-time Markov chain is specified by the
        # sequence of states and the unscaled interarrival times.
        def _draw():
            states = MarkovChain(self.transition_matrix,
                                 self.initial_dist).draw()
            rates = -np.diag(self.generator_matrix)
            unscaled_interarrival_times = (Exponential(1) ** inf).draw()
            return ContinuousTimeMarkovChainResult(
                states,
                rates,
                unscaled_interarrival_times,
                self.state_labels)

        super().__init__(_draw)


class ContinuousTimeMarkovChain(RV):

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        """Initialize a continuous-time Markov chain.

        Args:
          generator_matrix: n x n generator matrix whose rows sum to 0
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        prob_space = ContinuousTimeMarkovChainProbabilitySpace(
            generator_matrix,
            initial_dist,
            state_labels)
        super().__init__(prob_space)
