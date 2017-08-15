from .probability_space import ProbabilitySpace
from .random_variables import RV
from .random_processes import RandomProcess

def AssumeIndependent(*args):
    """Make RVs or RandomProcesses independent.

    Args:
      *args: Any number of RVs and RandomProcesses

    Returns:
      RVs and RandomProcesses with the same
      marginal distributions as the RVs and
      RandomProcesses, but defined on a common
      probability space so as to be independent.
    """

    # check that none of the RVs (or RandomProcesses)
    # are defined on the same probability space
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            if args[i].probSpace == args[j].probSpace:
                raise Exception(
                    "The RVs or RandomProcesses must be "
                    "currently defined on different "
                    "probability spaces in order to use "
                    "this function.")
    
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
