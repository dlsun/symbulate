from .probability_space import ProbabilitySpace
from .random_variables import RV
from .random_processes import RandomProcess

def MakeIndependent(*args):
    """Makes RVs or RandomProcesses independent.

    Args:
      *args: Any number of RVs and RandomProcesses

    Returns:
      RVs and RandomProcesses with the same
      marginal distributions as the RVs and
      RandomProcesses, but defined on a common
      probability space so as to be independent.
    """

    def draw():
        outcome = []
        for arg in args:
            outcome.append(arg.probSpace.draw())
        return outcome
    P = ProbabilitySpace(draw)

    outputs = []
    for i, arg in enumerate(args):
        if isinstance(arg, RV):
            # i=i forces Python to bind i now
            def f(x, fun=arg.fun, i=i):
                return fun(x[i])
            outputs.append(RV(P, f))
        elif isinstance(arg, RandomProcess):
            # i=i forces Python to bind i now
            def f(x, t, fun=arg.fun, i=i):
                return fun(x[i], t)
            outputs.append(RandomProcess(
                P, arg.timeIndex, f))
    
    return tuple(outputs)
