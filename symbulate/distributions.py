import numbers
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from .probability_space import ProbabilitySpace
from .plot import get_next_color
from .result import Scalar, Vector, InfiniteVector

class Distribution(ProbabilitySpace):
    def __init__(self, params, scipy, discrete=True):
        self.params = params

        self.discrete = discrete

        if discrete:
            self.pmf = lambda x: scipy.pmf(x, **self.params)
            self.pdf = self.pmf # add pdf as an alias for pmf
        else:
            self.pdf = lambda x: scipy.pdf(x, **self.params)

        self.cdf = lambda x: scipy.cdf(x, **self.params)
        self.quantile = lambda x: scipy.ppf(x, **self.params)

        self.median = lambda: scipy.median(**self.params)
        self.mean = lambda: scipy.mean(**self.params)
        self.var = lambda: scipy.var(**self.params)
        self.sd = lambda: scipy.std(**self.params)
        self.sim_func = scipy.rvs

        self.xlim = (
            scipy.ppf(0.001, **self.params),
            scipy.ppf(0.999, **self.params)
            )

    def draw(self):
        return Scalar(self.sim_func(**self.params))

    # Override the inherited __pow__ function to take advantage
    # of vectorized simulations.
    def __pow__(self, exponent):
        if exponent == float("inf"):
            def draw():
                def _func(_):
                    return self.sim_func(**self.params)
                return InfiniteVector(_func)
        else:
            def draw():
                return Vector(self.sim_func(**self.params, size=exponent))
        return ProbabilitySpace(draw)

    def plot(self, xlim=None, alpha=None, ax=None, **kwargs):

        # use distribution defaults for xlim if none set
        if xlim is None:
            xlim = self.xlim

        # get the x and y values
        if self.discrete:
            xs = np.arange(int(xlim[0]), int(xlim[1]) + 1)
        else:
            xs = np.linspace(xlim[0], xlim[1], 200)
        ys = self.pdf(xs)

        # determine limits for y-axes based on y values
        ymin, ymax = ys[np.isfinite(ys)].min(), ys[np.isfinite(ys)].max()
        ylim = min(0, ymin - 0.05 * (ymax - ymin)), 1.05 * ymax
        
        # get the current axis if they exist and no axis is specified
        fig = plt.gcf()
        if ax is None and fig.axes:
            ax = plt.gca()

        # if axis already exists, set it to the union of existing and current axis
        if ax is not None:
            xlower, xupper = ax.get_xlim()
            xlim = min(xlim[0], xlower), max(xlim[1], xupper)
            ylower, yupper = ax.get_ylim()
            ylim = min(ylim[0], ylower), max(ylim[1], yupper)
        else:
            ax = plt.gca() # creates new axis

        # set the axis limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        
        # get next color in cycle
        color = get_next_color(ax)

        # plot points for discrete distributions
        if self.discrete:
            ax.scatter(xs, ys, s=40, color=color, alpha=alpha, **kwargs)

        # plot curve
        ax.plot(xs, ys, color=color, alpha=alpha, **kwargs)

        # adjust the axes, base x-axis at 0
        ax.spines["bottom"].set_position("zero")


## Discrete Distributions

class Bernoulli(Distribution):
    """Defines a probability space for a Bernoulli
         distribution.

    Attributes:
      p (float): probability (number between 0 and 1)
        of a "success" (i.e., 1)
    """

    def __init__(self, p):
        if 0 <= p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {
            "p" : p
            }
        super().__init__(params, stats.bernoulli, True)
        self.xlim = (0, 1) # Bernoulli distributions are not defined for x < 0 and x > 1


class Binomial(Distribution):
    """Defines a probability space for a binomial
         distribution.

    Attributes:
      n (int): number of trials
      p (float): probability (number between 0 and 1)
        that each trial results in a "success" (i.e., 1)
    """

    def __init__(self, n, p):

        if n >= 0 and isinstance(n, numbers.Integral):
            self.n = n
        #elif n == 0:
            #raise NotImplementedError
            #TODO
        else:
            raise Exception("n must be a non-negative integer")

        if 0 <= p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {
            "n" : n,
            "p" : p
            }
        super().__init__(params, stats.binom, True)
        self.xlim = (0, n) # Binomial distributions are not defined for x < 0 and x > n


class Hypergeometric(Distribution):
    """Defines a probability space for a hypergeometric
         distribution (which represents the number of
         ones in n draws without replacement from a box
         containing zeros and ones).

    Attributes:
      n (int): number of draws (without replacement)
        from the box
      N0 (int): number of 0s in the box
      N1 (int): number of 1s in the box
    """

    def __init__(self, n, N0, N1):

        if n > 0 and isinstance(n, numbers.Integral):
            self.n = n
        else:
            raise Exception("n must be a positive integer")

        if N0 >= 0 and isinstance(N0, numbers.Integral):
            self.N0 = N0
        else:
            raise Exception("N0 must be a non-negative integer")

        if N1 >= 0 and isinstance(N1, numbers.Integral):
            self.N1 = N1
        else:
            raise Exception("N1 must be a non-negative integer")

        params = {
            "M" : N0 + N1,
            "n" : N1,
            "N" : n
            }

        if N0 + N1 < n:
            raise Exception("N0 + N1 cannot be less than the sample size n")

        super().__init__(params, stats.hypergeom, True)
        self.xlim = (0, n) # Hypergeometric distributions are not defined for x < 0 and x > n


class Geometric(Distribution):
    """Defines a probability space for a geometric
         distribution (which represents the number
         of trials until the first success), including
         the success.

    Attributes:
      p (float): probability (number between 0 and 1)
        that each trial results in a "success" (i.e., 1)
    """

    def __init__(self, p):

        if 0 < p < 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {
            "p" : p
            }
        super().__init__(params, stats.geom, True)
        self.xlim = (1, self.xlim[1]) # Geometric distributions are not defined for x < 1


class NegativeBinomial(Distribution):
    """Defines a probability space for a negative
         binomial distribution (which represents the
         number of trials until r successes), including
         the r successes.

    Attributes:
      r (int): desired number of successes
      p (float): probability (number between 0 and 1)
        that each trial results in a "success" (i.e., 1)
    """

    def __init__(self, r, p):

        if 0 < r and isinstance(r, numbers.Integral):
            self.r = r
        else:
            raise Exception("r must be a positive integer")

        if 0 < p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {
            "n" : r,
            "p" : p,
            "loc" : r
            }
        super().__init__(params, stats.nbinom, True)
        self.xlim = (r, self.xlim[1]) # Negative Binomial distributions are not defined for x < r

    def draw(self):
        """A function that takes no arguments and
            returns a single draw from the Negative Binomial distribution."""

        # Numpy's negative binomial returns numbers in [0, inf),
        # but we want numbers in [r, inf).
        return self.r + np.random.negative_binomial(n=self.r, p=self.p)


class Pascal(Distribution):
    """Defines a probability space for a Pascal
         distribution (which represents the number
         of trials until r successes), not including
         the r successes.

    Attributes:
      r (int): desired number of successes
      p (float): probability (number between 0 and 1)
        that each trial results in a "success" (i.e., 1)
    """

    def __init__(self, r, p):

        if 0 < r and isinstance(r, numbers.Integral):
            self.r = r
        else:
            raise Exception("r must be a positive integer")

        if 0 < p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {
            "n" : r,
            "p" : p
            }
        super().__init__(params, stats.nbinom, True)
        self.xlim = (0, self.xlim[1]) # Pascal distributions are not defined for x < 0


class Poisson(Distribution):
    """Defines a probability space for a Poisson distribution.

    Attributes:
      lam (float): rate parameter for the Poisson distribution
    """

    def __init__(self, lam):

        if 0 < lam:
            self.lam = lam
        else:
            raise Exception("Lambda (lam) must be greater than 0")

        params = {
            "mu" : lam
            }
        super().__init__(params, stats.poisson, True)
        self.xlim = (0, self.xlim[1]) # Poisson distributions are not defined for x < 0


class DiscreteUniform(Distribution):
    """Defines a probability space for a discrete uniform distribution.

    Attributes:
      a (int): lower bound for possible values
      b (int): upper bound for possible values
    """

    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b + 1

        params = {
            "low" : self.a,
            "high" : self.b
            }

        if a >= b:
            raise Exception("b cannot be less than or equal to a")

        super().__init__(params, stats.randint, True)
        self.xlim = (a, b) # Uniform distributions are not defined for x < a and x > b


## Continuous Distributions

class Uniform(Distribution):
    """Defines a probability space for a uniform distribution.

    Attributes:
      a (float): lower bound for possible values
      b (float): upper bound for possible values
    """

    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

        params = {
            "loc" : a,
            "scale" : b - a
            }

        if a > b:
            raise Exception("b cannot be less than a")

        super().__init__(params, stats.uniform, False)
        self.xlim = (a, b) # Uniform distributions are not defined for x < a and x > b


class Normal(Distribution):
    """Defines a probability space for a normal distribution.

    Attributes:
      mean (float): mean parameter of the normal distribution
      sd (float): standard deviation parameter of the normal
        distribution
      var (float): variance parameter of the normal distribution
        (if specified, var parameter will be ignored)
    """
    #TODO edit docstring for Normal Distribution

    def __init__(self, mean=0.0, sd=1.0, var=None):

        #Note: cleaner way to implement this

        if var is None:
            if sd > 0:
                self.scale = sd
            elif sd == 0:
                raise NotImplementedError
                #TODO
            else:
                raise Exception("sd cannot be less than 0")

        else:
            if var > 0:
                self.scale = np.sqrt(var)
            elif var == 0:
                raise NotImplementedError
                #TODO
            else:
                raise Exception("var cannot be less than 0")

        params = {
            "loc" : mean,
            "scale" : self.scale
            }
        super().__init__(params, stats.norm, False)


class Exponential(Distribution):
    """Defines a probability space for an exponential distribution.
       Only one of scale or rate should be set. (The scale is the
       inverse of the rate.)

    Attributes:
      scale (float): scale parameter for gamma distribution
        (often symbolized beta = 1 / lambda)
      rate (float): rate parameter for gamma distribution
        (often symbolized lambda)
    """

    def __init__(self, rate=1.0, scale=None):

        if scale is None:
            if rate > 0:
                self.rate = rate
                self.scale = scale
            else:
                raise Exception("rate must be positive")
        else:
            if scale > 0:
                self.scale = scale
            else:
                raise Exception("scale must be positive")

        params = {
            "scale" : 1. / rate if scale is None else scale
            }
        super().__init__(params, stats.expon, False)
        self.xlim = (0, self.xlim[1]) # Exponential distributions are not defined for x < 0


class Gamma(Distribution):
    """Defines a probability space for a gamma distribution.
       Only one of scale or rate should be set. (The scale is the
       inverse of the rate.)

    Attributes:
      shape (float): shape parameter for gamma distribution
        (often symbolized alpha)
      scale (float): scale parameter for gamma distribution
        (often symbolized beta = 1 / lambda)
      rate (float): rate parameter for gamma distribution
        (often symbolized lambda)
    """

    def __init__(self, shape, rate=1.0, scale=None):

        if 0 < shape:
            self.shape = shape
        else:
            raise Exception("shape parameter must be positive")

        if scale is None:
            if rate > 0:
                self.rate = rate
                self.scale = scale
            else:
                raise Exception("rate must be positive")
        else:
            if scale > 0:
                self.scale = scale
            else:
                raise Exception("scale must be positive")

        params = {
            "a" : shape,
            "scale" : 1. / rate if scale is None else scale
            }
        super().__init__(params, stats.gamma, False)
        self.xlim = (0, self.xlim[1]) # Gamma distributions are not defined for x < 0


class Beta(Distribution):
    """Defines a probability space for a beta distribution.

    Attributes:
      a (float): alpha parameter for beta distribution
      b (float): beta parameter for beta distribution
    """

    def __init__(self, a, b):

        if 0 < a:
            self.a = a
        else:
            raise Exception("a must be positive")

        if 0 < b:
            self.b = b
        else:
            raise Exception("b must be positive")

        params = {
            "a" : a,
            "b" : b
            }
        super().__init__(params, stats.beta, False)
        self.xlim = (0, 1) # Beta distributions are not defined for x < 0 and x > 1


class StudentT(Distribution):
    """Defines a probability space for Student's t distribution.

    Attributes:
      df (int): degrees of freedom
    """

    def __init__(self, df):
        if df > 0:
            self.df = df
        else:
            raise Exception("df must be greater than 0")

        params = {
            "df" : df
            }
        super().__init__(params, stats.t, False)
        if df == 1:
            self.mean = lambda: float('nan')
            self.sd = lambda: float('nan')
            self.var = lambda: float('nan')


class ChiSquare(Distribution):
    """Defines a probability space for a chi-square distribution

    Attributes:
      df (int): degrees of freedom
    """

    def __init__(self, df):
        if df > 0 and isinstance(df, numbers.Integral):
            self.df = df
        else:
            raise Exception("df must be a positive integer")

        params = {
            "df" : df
            }
        super().__init__(params, stats.chi2, False)
        self.xlim = (0, self.xlim[1]) # Chi-Square distributions are not defined for x < 0


class F(Distribution):
    """Defines a probability space for an F distribution

    Attributes:
      dfN (int): degrees of freedom in the numerator
      dfD (int): degrees of freedom in the denominator
    """

    def __init__(self, dfN, dfD):

        if dfN > 0:
            self.dfN = dfN
        else:
            raise Exception("dfN must be greater than 0")

        if dfD > 0:
            self.dfD = dfD
        else:
            raise Exception("dfD must be greater than 0")

        params = {
            "dfn" : dfN,
            "dfd" : dfD
            }
        super().__init__(params, stats.f, False)
        self.xlim = (0, self.xlim[1]) # F distributions are not defined for x < 0


class Cauchy(Distribution):
    """Defines a probability space for a Cauchy distribution

    Attributes:
      The Cauchy distribution has no parameters
    """

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

        params = {
            "loc" : loc,
            "scale" : scale
        }

        super().__init__(params, stats.cauchy, False)

    def draw(self):
        return self.loc + (self.scale * np.random.standard_cauchy())


class LogNormal(Distribution):
    """Defines a probability space for a Log-Normal distribution

       If Y has a LogNormal distribution with parameters mu and sigma, then
       log(Y) has a normal distribution with mean mu and sd sigma.

    Attributes:
      mu (float): mean of the underlying normal distribution
      sigma (float): standard deviation of the underlying normal distribution
    """

    def __init__(self, mu=0.0, sigma=1.0):

        self.norm_mean = mu

        if sigma > 0:
            self.s = sigma
            self.norm_sd = sigma
        else:
            raise Exception("sigma must be greater than 0")

        params = {
            "s" : self.s,
            "scale" : np.exp(mu)
            }
        super().__init__(params, stats.lognorm, False)
        self.xlim = (0, self.xlim[1]) # Log-Normal distributions are not defined for x < 0


class Pareto(Distribution):
    """Defines a probability space for a Pareto distribution.

    Attributes:
      b (float): shape parameter of Pareto distribution
      scale (float): scale parameter of Pareto distribution
          lower bound for possible values
    """

    def __init__(self, b=1.0, scale=1.0):

        if b > 0:
            self.b = b
        else:
            raise Exception("b must be greater than 0")

        if scale > 0:
            self.scale = scale
        else:
            raise Exception("scale must be greater than 0")

        params = {
            "b" : self.b,
            "scale" : self.scale
            }
        super().__init__(params, stats.pareto, False)
        self.xlim = (scale, self.xlim[1]) # Pareto distributions are not defined for x < scale

    def draw(self):
        """A function that takes no arguments and
           returns a single draw from the Pareto distribution."""
        
        # Numpy's Pareto is Lomax distribution, or Type II Pareto
        # but we want the more standard parametrization
        return self.scale * (1 + np.random.pareto(self.b))


# class Weibull(Distribution):
#
#     def __init__(self, eta, beta= ):


class Rayleigh(Distribution):
    """Defines a probability space for a Rayleigh distribution

    Attributes:
      The Rayleigh distribution has no parameters
    """

    def __init__(self):
        params = {}
        super().__init__(params, stats.rayleigh, False)


## Multivariate Distributions

class MultivariateNormal(Distribution):
    """Defines a probability space for a multivariate normal
       distribution.

    Attributes:
      mean (1-D array_like, of length n): mean vector
      cov (2-D array_like, of shape (n, n)): covariance matrix
    """

    def __init__(self, mean, cov):
        if len(mean) != len(cov):
            raise Exception("The dimension of the mean vector" +
                            " is not compatible with the dimensions" +
                            " of the covariance matrix.")

        if len(mean) >= 1:
            self.mean = mean
        else:
            raise Exception("Mean vector and Cov matrix cannot be empty")

        if len(cov) >= 1:
            if all(len(row) == len(mean) for row in cov):
                if np.all(np.linalg.eigvals(cov) >= 0) and np.allclose(cov, np.transpose(cov)):
                    self.cov = cov
                else:
                    raise Exception("Cov matrix is not symmetric and positive semi-definite")
            else:
                raise Exception("Cov matrix is not square")
        else:
            raise Exception("Dimension of cov matrix cannot be less than 1")

        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(x, mean, cov)

    def plot(self):
        raise Exception(
            "Plotting is not currently available for "
            "the multivariate normal distribution."
        )

    def draw(self):
        """A function that takes no arguments and
            returns a single draw from the Multivariate Normal distribution."""

        return Vector(np.random.multivariate_normal(self.mean, self.cov))

    def __pow__(self, exponent):
        if exponent == float("inf"):
            def draw():
                def _func(n):
                    return self.draw()
                return InfiniteVector(_func)
        else:
            def draw():
                return Vector(self.draw() for _ in range(exponent))
        return ProbabilitySpace(draw)



class BivariateNormal(MultivariateNormal):
    """Defines a probability space for a bivariate normal
       distribution.

    Attributes:
      mean1 (float): mean parameter of X
      mean2 (float): mean parameter of Y
      sd1 (float): standard deviation parameter of X
      sd2 (float): standard deviation parameter of Y
      corr (float): correlation between X and Y
      var1 (float): variance parameter of X
        (if specified, sd1 will be ignored)
      var2 (float): variance parameter of Y
        (if specified, sd2 will be ignored)
      cov (float): covariance between X and Y
        (if specified, corr parameter will be ignored)
    """

    def __init__(self,
                 mean1=0.0, mean2=0.0,
                 sd1=1.0, sd2=1.0, corr=0.0,
                 var1=None, var2=None, cov=None):

        if not -1 <= corr <= 1:
            raise Exception("Correlation must be "
                            "between -1 and 1.")

        self.mean = [mean1, mean2]

        if sd1 < 0:
            raise Exception("sd1 cannot be less than 0")
        if sd2 < 0:
            raise Exception("sd2 cannot be less than 0")

        if var1 is None:
            var1 = sd1 ** 2
        if var2 is None:
            var2 = sd2 ** 2
        if var1 < 0 or var2 < 0:
            raise Exception("var1 and var2 cannot be negative")
        if cov is None:
            cov = corr * np.sqrt(var1 * var2)
        self.cov = [[var1, cov], [cov, var2]]
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(x, self.mean, self.cov)



class Multinomial(Distribution):
    """Defines a probability space for a multinomial
       distribution.

    Attributes:
      n (int): number of trials
      p (1-D array_like): probability vector
    """

    def __init__(self, n, p):
        if n >= 0 and isinstance(n, numbers.Integral):
            self.n = n
        #elif n == 0:
            #raise NotImplementedError
            #TODO
        else:
            raise Exception("n must be a non-negative integer")

        if sum(p) == 1 and min(p) >= 0:
            self.p = p
        else:
            raise Exception("Elements of p must be non-negative" +
                            " and sum to 1.")

        self.discrete = False
        self.pdf = lambda x: stats.multinomial(x, n, p)

    def plot(self):
        raise Exception(
            "Plotting is not currently available for "
            "the Multinomial distribution."
        )

    def draw(self):
        """A function that takes no arguments and
            returns a single draw from the multinomial distribution."""

        return Vector(np.random.multinomial(self.n, self.p))

    def __pow__(self, exponent):
        if exponent == float("inf"):
            def draw():
                def _func(_):
                    return self.draw()
                return InfiniteVector(_func)
        else:
            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)
