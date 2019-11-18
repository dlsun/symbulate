import numpy as np

from .index_sets import (
    DiscreteTimeSequence,
    Reals
)
from .probability_space import ProbabilitySpace
from .result import (
    DiscreteTimeFunction,
    ContinuousTimeFunction,
    Vector,
    is_number,
    is_numeric_vector
)
from .random_variables import RV
from .random_processes import RandomProcess

MACHINE_EPS = 1e-12


def get_gaussian_process_result(mean_func, cov_func, index_set=Reals()):

    # Determine whether the process is discrete-time or continous-time
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

        def __init__(self, mean_func, cov_func):

            self.mean = np.empty(shape=0)
            self.cov = np.empty(shape=(0, 0))
            self.observed = {}

            def _vfunc(ts):
                # This function assumes that t is an array of times.
                ts = list(ts)

                # Get current times
                times = list(self.observed.keys())

                # If this is a discrete process, t will be an index.
                # Convert it to a time.
                if isinstance(index_set, DiscreteTimeSequence):
                    ts = [t / index_set.fs for t in ts]

                # Check that every t is in the index set
                for t in ts:
                    if t not in index_set:
                        raise KeyError(
                            "Gaussian process is not defined at time %.2f." % t0
                        )

                # Create an object to store the results
                n = len(ts)
                values = np.empty(shape=n)
                values[:] = np.nan

                # Handle times that have already been calculated,
                # as well as times where the variance is 0
                i_delete = []
                for i, t in enumerate(ts):
                    if cov_func(t, t) == 0:
                        values[i] = mean_func(t)
                        i_delete.append(i)
                    elif t in self.observed:
                        values[i] = self.observed[t]
                        i_delete.append(i)
                ts = [t for i, t in enumerate(ts) if i not in i_delete]
                if not ts:
                    return values

                # Simulate values for the remaining times
                mean2 = np.array([mean_func(t) for t in ts])
                cov11 = self.cov + MACHINE_EPS * np.identity(len(times))
                cov12 = np.empty(shape=(len(times), len(ts)))
                for i, s in enumerate(times):
                    for j, t in enumerate(ts):
                        cov12[i, j] = cov_func(s, t)
                cov22 = np.empty(shape=(len(ts), len(ts)))
                for i, s in enumerate(ts):
                    for j, t in enumerate(ts):
                        cov22[i, j] = cov_func(s, t)

                cond_mean = (mean2 + (
                    cov12.T @
                    np.linalg.solve(cov11, list(self.observed.values()) - self.mean)
                ))
                cond_var = (cov22 - (
                    cov12.T @
                    np.linalg.solve(cov11, cov12)
                ))

                # update mean vector and covariance matrix
                self.mean = np.append(self.mean, mean2)
                self.cov = np.block([[cov11, cov12], [cov12.T, cov22]])

                # simulate normal with given mean and variance
                new_values = np.random.multivariate_normal(cond_mean, cond_var)

                # store the new values
                for t, v in zip(ts, new_values):
                    self.observed[t] = v
                values[np.isnan(values)] = new_values
                
                return values

            self.vfunc = _vfunc

            def _func(t):
                return _vfunc([t])[0]
            
            super().__init__(func=_func)
            self.index_set = index_set

    return GaussianProcessResult(mean_func, cov_func)


class GaussianProcessProbabilitySpace(ProbabilitySpace):

    def __init__(self, mean_func, cov_func, index_set=Reals()):
        """Initialize probability space for a Gaussian process.

        Args:
          mean_func: mean function (function of one argument)
          cov_func: (auto)covariance function (function of two arguments)
          index_set: index set for the Gaussian process
                     (by default, all real numbers)
        """

        def draw():
            return get_gaussian_process_result(
                mean_func,
                cov_func,
                index_set)

        super().__init__(draw)


class GaussianProcess(RandomProcess, RV):

    def __init__(self, mean_func, cov_func, index_set=Reals()):
        """Initialize Gaussian process.

        Args:
          mean_func: mean function (function of one argument)
          cov_func: (auto)covariance function (function of two arguments)
          index_set: index set for the Gaussian process
                     (by default, all real numbers)
        """

        prob_space = GaussianProcessProbabilitySpace(mean_func,
                                                     cov_func,
                                                     index_set)
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)


# Define convenience class for Brownian motion
class BrownianMotionProbabilitySpace(GaussianProcessProbabilitySpace):

    def __init__(self, drift=0, scale=1):
        """Initialize probability space for Brownian motion.

        Args:
          drift: drift parameter of Brownian motion
          scale: scale parameter of Brownian motion
        """
        super().__init__(
            mean_func=lambda t: drift * t,
            cov_func=lambda s, t: (scale ** 2) * min(s, t)
        )


class BrownianMotion(RandomProcess, RV):

    def __init__(self, drift=0, scale=1):
        """Initialize Brownian motion.

        Args:
          drift: drift parameter of Brownian motion
          scale: scale parameter of Brownian motion
        """
        prob_space = BrownianMotionProbabilitySpace(
            drift=drift, scale=scale
        )
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)
