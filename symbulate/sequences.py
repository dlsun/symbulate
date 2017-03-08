from .time_index import TimeIndex

class InfiniteSequence:

    def __init__(self, fun):
        self.fun = fun

    def __getitem__(self, n):
        if type(n) == int:
            return self.fun(n)
        else:
            raise Exception("Index to a sequence must be an integer.")

    def __call__(self, n):
        return self.__getitem__(n)

    def __str__(self):
        return str(tuple([self.fun(n) for n in range(10)] + ["..."]))

class TimeFunction:

    def __init__(self, fun, timeIndex):
        self.fun = fun
        self.timeIndex = timeIndex

    def __getitem__(self, t):
        return self.fun(t)

    def __call__(self, t):
        return self.fun(t)

    def __str__(self):
        if self.timeIndex.fs == float("inf"):
            return "(continuous-time function)"
        else:
            return str(tuple([self.fun(self.timeIndex[n]) for n in range(10)] + ["..."]))

