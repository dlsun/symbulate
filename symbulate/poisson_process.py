from .distributions import Exponential
from .math import inf
from .probability_space import ProbabilitySpace
from .result import (
    InfiniteVector,
    ContinuousTimeFunction,
    DiscreteValued
)
from .random_variables import RV
from .random_processes import RandomProcess


class PoissonProcessResult(ContinuousTimeFunction,
                           DiscreteValued):

    def __init__(self, interarrival_times):
        self.interarrival_times = interarrival_times

        def func(t):
            total_time = 0
            for n, time in enumerate(self.interarrival_times):
                total_time += time
                if t < total_time:
                    return n

        super().__init__(func)

    def get_states(self):
        return InfiniteVector(lambda n: n)


class PoissonProcessProbabilitySpace(ProbabilitySpace):

    def __init__(self, rate):
        """Initialize probability space for a Poisson process.

        Args:
          rate: rate of the Poisson process
        """
        self.rate = rate

        def draw():
            interarrival_times = (Exponential(rate=self.rate) ** inf).draw()
            return PoissonProcessResult(interarrival_times)

        super().__init__(draw)


class PoissonProcess(RandomProcess, RV):

    def __init__(self, rate):
        """Initialize a Poisson process.

        Args:
          rate: rate of the Poisson process
        """
        self.rate = rate
        prob_space = PoissonProcessProbabilitySpace(self.rate)
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)
