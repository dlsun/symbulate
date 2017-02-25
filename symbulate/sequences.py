from .time_index import TimeIndex

class InfiniteSequence:

    def __init__(self, fun, timeIndex=TimeIndex(1)):
        self.fun = fun
        self.timeIndex = timeIndex

    def __getitem__(self, t):
        return self.fun(t)

    def __str__(self):
        if self.timeIndex.fs == float("inf"):
            return "(function)"
        else:
            return str([self.fun(self.timeIndex[i]) 
                        for i in range(50)])
