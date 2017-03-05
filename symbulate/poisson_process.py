from .distributions import Exponential
from .math import inf
from .random_processes import RandomProcess, TimeIndex

class PoissonProcess(RandomProcess):

    def __init__(self, rate):
        self.rate = rate
        self.probSpace = Exponential(rate=rate) ** inf
        self.timeIndex = TimeIndex(fs=inf)
        def fun(x, t):
            n = 0
            total_time = 0
            while True:
                total_time += x[n]
                if total_time > t:
                    break
                else:
                    n += 1
            return n
        self.fun = fun

    def ArrivalTimes(self):
        def fun(x, t):
            total = 0
            for i in range(t):
                total += x[i]
            return total
        return RandomProcess(self.probSpace, TimeIndex(1), fun)

    def InterarrivalTimes(self):
        def fun(x, t):
            return x[t]
        return RandomProcess(self.probSpace, TimeIndex(1), fun)
