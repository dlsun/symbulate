import numpy as np

from .probability_space import ProbabilitySpace
from .sequences import LazyFunction

class GaussianProcess(ProbabilitySpace):

    def __init__(self, mean_fn, cov_fn):
        """Initialize Gaussian process.

        Args:
          mean_fn: mean function (function of one argument)
          cov_fn: covariance function (function of two arguments)
        """
        def fun(arg, cached_args, cached_vals):
            # remove locations where the variance is 0
            cached_args_ = []
            cached_vals_ = []
            for a, v in zip(cached_args, cached_vals):
                if cov_fn(a, a) != 0:
                    cached_args_.append(a)
                    cached_vals_.append(v)

            # if there are no cached values, simulate from
            # (unconditional) normal distribution
            if not cached_args_:
                return np.random.normal(
                    loc=mean_fn(arg),
                    scale=np.sqrt(cov_fn(arg, arg)))
                    
            # calculate mean at arg and at cached_args
            mean1 = np.array([mean_fn(x) for x in cached_args_],
                             dtype="float64")
            mean2 = mean_fn(arg)
        
            # calculate covariance between the different locations
            cov11 = np.array([
                [cov_fn(x1, x2) for x1 in cached_args_]
                for x2 in cached_args_], dtype="float64")
            cov21 = np.array([
                cov_fn(arg, x) for x in cached_args_],
                             dtype="float64")
            cov22 = cov_fn(arg, arg)

            # add 1e-12 to the diagonal for numerical stability
            cov11 += 1e-12 * np.identity(len(cached_args_))

            # calculate conditional mean and variance
            cond_mean = (mean2 +
                         (cov21 * np.linalg.solve(
                             cov11, np.array(cached_vals_) - mean1
                         )).sum())
            cond_var = (cov22 -
                        (cov21 * np.linalg.solve(cov11, cov21)).sum())
            cond_var = max(cond_var, 0)

            # simulate normal with given mean and variance
            return np.random.normal(cond_mean, np.sqrt(cond_var))

        def draw():
            return LazyFunction(fun)

        super().__init__(draw)
