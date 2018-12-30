from .probability_space import ProbabilitySpace
from .random_variables import RV

def AssumeIndependent(*args):
    """Make RVs independent.

    Args:
      *args: Any number of RVs

    Returns:
      RVs with the same marginal distributions
      as the inputs, but defined on a common
      probability space so as to be independent.
    """

    # Check that none of the RVs are defined on
    # the same probability space.
    for i in range(len(args)):
        if not isinstance(args[i], RV):
            raise Exception(
                "AssumeIndependent(...) can only be "
                "used with RVs, but you passed in a "
                "%s." % type(args[i]).__name__)
        for j in range(i + 1, len(args)):
            if args[i].prob_space == args[j].prob_space:
                raise Exception(
                    "AssumeIndependent(...) can only be "
                    "called on RVs that are initially "
                    "defined on different probability "
                    "spaces."
                    )

    def draw():
        outcome = []
        for arg in args:
            outcome.append(arg.prob_space.draw())
        return outcome
    P = ProbabilitySpace(draw)

    outputs = []
    for i, arg in enumerate(args):
        # i=i forces Python to bind i now
        def _func(x, func=arg.func, i=i):
            return func(x[i])
        outputs.append(RV(P, _func))

    return tuple(outputs)
