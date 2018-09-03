import numpy as np

from .index_sets import (
    DiscreteTimeSequence,
    Reals
)
from .probability_space import ProbabilitySpace
from .result import (
    DiscreteTimeFunction,
    ContinuousTimeFunction
)
from .random_processes import RandomProcess

MACHINE_EPS = 1e-12


def get_gaussian_process_result(mean_fn, cov_fn, index_set=Reals()):

    if isinstance(index_set, DiscreteTimeSequence):
        base_class = DiscreteTimeFunction
    elif isinstance(index_set, Reals):
        base_class = ContinuousTimeFunction
    else:
        raise Exception(
            "Index set for Gaussian process must be Reals or "
            "DiscreteTimeSequence."
        )
    
    class GaussianProcessResult(base_class):
        
        def __init__(self, mean_fn, cov_fn):
            self.mean_fn = mean_fn
            self.cov_fn = cov_fn
            self.mean = np.empty(shape=0)
            self.cov = np.empty(shape=(0, 0))
            self.times = []
            self.values = []

        def fn(self, t0):
            # If a discrete process, t0 is an index.
            # Convert it to a time.
            if isinstance(index_set, DiscreteTimeSequence):
                t0 = int(t0 / index_set.fs)
                
            # Check that t0 is in the index set
            if t0 not in index_set:
                raise KeyError(
                    "Gaussian process is not defined at "
                    "time %.2f." % t0
                )
            
            # if variance is 0, just return the mean
            if self.cov_fn(t0, t0) == 0:
                return self.mean_fn(t0)
        
            # calculate conditional mean and variance
            mean2 = self.mean_fn(t0)
            cov11 = self.cov + MACHINE_EPS * np.identity(len(self.times))
            cov12 = [self.cov_fn(t0, t) for t in self.times]
            cov22 = self.cov_fn(t0, t0)
            cond_mean = (mean2 + (
                cov12 *
                np.linalg.solve(cov11, self.values - self.mean)
            ).sum())
            cond_var = (cov22 - (
                cov12 * np.linalg.solve(cov11, cov12)
            ).sum())
            cond_var = max(cond_var, 0)

            # update mean vector and covariance matrix
            self.mean = np.append(self.mean, mean2)
            self.cov = np.column_stack((self.cov, cov12))
            self.cov = np.vstack((
                self.cov,
                np.append(cov12, cov22)
            ))

            # simulate normal with given mean and variance
            self.times.append(t0)
            value = np.random.normal(cond_mean, np.sqrt(cond_var))
            self.values.append(value)
            return value

    return GaussianProcessResult(mean_fn, cov_fn)

    
class GaussianProcessProbabilitySpace(ProbabilitySpace):

    def __init__(self, mean_fn, cov_fn, index_set=Reals()):
        """Initialize probability space for a Gaussian process.

        Args:
          mean_fn: mean function (function of one argument)
          cov_fn: (auto)covariance function (function of two arguments)
          index_set: index set for the Gaussian process
                     (by default, all real numbers)
        """
        
        def draw():
            return get_gaussian_process_result(
                mean_fn,
                cov_fn,
                index_set)

        super().__init__(draw)


class GaussianProcess(RandomProcess):

    def __init__(self, mean_fn, cov_fn, index_set=Reals()):
        """Initialize Gaussian process.

        Args:
          mean_fn: mean function (function of one argument)
          cov_fn: (auto)covariance function (function of two arguments)
          index_set: index set for the Gaussian process
                     (by default, all real numbers)
        """

        probSpace = GaussianProcessProbabilitySpace(mean_fn,
                                                    cov_fn,
                                                    index_set)
        super().__init__(probSpace)


# Define convenience class for Brownian motion
class BrownianMotionProbabilitySpace(GaussianProcessProbabilitySpace):

    def __init__(self, scale=1, drift=0):
        """Initialize probability space for Brownian motion.

        Args:
          scale: scale parameter of Brownian motion
          drift: drift parameter of Brownian motion
        """
        super().__init__(
            mean_fn=lambda t: drift * t,
            cov_fn=lambda s, t: (scale ** 2) * min(s, t)
        )


class BrownianMotion(RandomProcess):

    def __init__(self, scale=1, drift=0):
        """Initialize Brownian motion.

        Args:
          scale: scale parameter of Brownian motion
          drift: drift parameter of Brownian motion
        """
        probSpace = BrownianMotionProbabilitySpace(
            scale=scale, drift=drift
        )
        super().__init__(probSpace)
