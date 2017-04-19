import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from .probability_space import ProbabilitySpace
from .plot import configure_axes, get_next_color

class Distribution(ProbabilitySpace):
    def __init__(self, params, scipy, discrete = True):
        self.params = params
        
        if discrete:
            self.pdf = lambda x: scipy.pmf(x, **self.params)
        else:
            self.pdf = lambda x: scipy.pdf(x, **self.params)
        
        self.cdf = lambda x: scipy.cdf(x, **self.params)
        self.quantile = lambda x: scipy.ppf(x, **self.params)
        
        self.median = lambda : scipy.median(**self.params)
        self.mean = lambda : scipy.mean(**self.params)
        self.var = lambda : scipy.var(**self.params)
        self.sd = lambda : scipy.std(**self.params)
        
        self.discrete = discrete
        
        self.xlim = [
            self.mean() - 3 * self.sd(), 
            self.mean() + 3 * self.sd()
            ]
    
    def plot(self, type = None, alpha = None, xlim = None, **kwargs):
        if xlim is None: # if no limits for x-axis are specified, then use the default from plt
            xlower, xupper = self.xlim
        else:
            xlower, xupper = xlim
        
        if (self.discrete):
            xlower = int(xlower)
            xupper = int(xupper)        
            xvals = list(np.arange(xlower, xupper+1, 1))
        else:
            xvals = list(np.linspace(xlower, xupper, 100))
        
        yvals = list(map(self.pdf, xvals))
        
        # get next color in cycle
        axes = plt.gca()
        color = get_next_color(axes)
        
        if (self.discrete):
            plt.scatter(xvals, yvals, s = 40, color = color, alpha = alpha, **kwargs)
        
        plt.plot(xvals, yvals, color = color, alpha = alpha, **kwargs)
        
        configure_axes(axes, xvals, yvals)

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
            # TODO: implement error handling
            pass
        
        params = {
            "p" : p
            }
        super().__init__(params, stats.bernoulli, True)
        self.xlim = [0, 1] # Bernoulli distributions are not defined for x < 0 and x > 1
 
    def draw(self):
        return np.random.binomial(n=1, p=self.p)

class Binomial(Distribution):
    """Defines a probability space for a binomial
         distribution.

    Attributes:
      n (int): number of trials
      p (float): probability (number between 0 and 1)
        that each trial results in a "success" (i.e., 1)
    """

    def __init__(self, n, p):
        self.n = n
        self.p = p
        
        params = {
            "n" : n,
            "p" : p
            }
        super().__init__(params, stats.binom, True)
        self.xlim = [0, n] # Binomial distributions are not defined for x < 0 and x > n

    def draw(self):
        return np.random.binomial(n=self.n, p=self.p)

class Hypergeometric(Distribution):
    """Defines a probability space for a hypergeometric
         distribution (which represents the number of
         ones in n draws without replacement from a box
         containing zeros and ones.

    Attributes:
      n (int): number of draws (without replacement)
        from the box
      N0 (int): number of 0s in the box
      N1 (int): number of 1s in the box
    """

    def __init__(self, n, N0, N1):
        self.n = n
        self.N0 = N0
        self.N1 = N1
        
        params = {
            "M" : N0 + N1,
            "n" : N1,
            "N" : n
            }
        super().__init__(params, stats.hypergeom, True)
        self.xlim = [0, n] # Hypergeometric distributions are not defined for x < 0 and x > n
        
    def draw(self):
        return np.random.hypergeometric(ngood=self.N1, nbad=self.N0, nsample=self.n)

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
        self.p = p
        
        params = {
            "p" : p
            }
        super().__init__(params, stats.geom, True)
        self.xlim[0] = 1 # Geometric distributions are not defined for x < 1
        
    def draw(self):
        return np.random.geometric(p=self.p)

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
        self.r = r
        self.p = p
        
        params = {
            "n" : r,
            "p" : p,
            "loc" : r
            }
        super().__init__(params, stats.nbinom, True)
        self.xlim[0] = r # Negative Binomial distributions are not defined for x < r

    def draw(self):
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
        self.r = r
        self.p = p
        
        params = {
            "n" : r,
            "p" : p
            }
        super().__init__(params, stats.nbinom, True)
        self.xlim[0] = 0 # Pascal distributions are not defined for x < 0
    
    def draw(self):
        # Numpy's negative binomial returns numbers in [0, inf).
        return np.random.negative_binomial(n=self.r, p=self.p)

class Poisson(Distribution):
    """Defines a probability space for a Poisson distribution.

    Attributes:
      lam (float): rate parameter for the Poisson distribution
    """

    def __init__(self, lam):
        self.lam = lam
        
        params = {
            "mu" : lam
            }
        super().__init__(params, stats.poisson, True)
        self.xlim[0] = 0 # Poisson distributions are not defined for x < 0

    def draw(self):
        return np.random.poisson(lam=self.lam)


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
        super().__init__(params, stats.uniform, False)
        self.xlim = [a, b] # Uniform distributions are not defined for x < a and x > b
        
    def draw(self):
        return np.random.uniform(low=self.a, high=self.b)

class Normal(Distribution):
    """Defines a probability space for a normal distribution.

    Attributes:
      mean (float): mean parameter of the normal distribution
      var (float): variance parameter of the normal distribution
      sd (float): standard deviation parameter of the normal 
        distribution (if specified, var parameter will be ignored)
    """

    def __init__(self, mean=0.0, var=1.0, sd=None):
        if sd is None:
            self.scale = np.sqrt(var)
        else:
            self.scale = sd
        
        params = {
            "loc" : mean,
            "scale" : self.scale
            }
        super().__init__(params, stats.norm, False)
    
    def draw(self):
        return np.random.normal(loc=self.mean(), scale=self.scale)

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
        self.scale = scale
        self.rate = rate
        
        params = {
            "scale" : 1. / rate if scale is None else scale
            }
        super().__init__(params, stats.expon, False)
        self.xlim[0] = 0 # Exponential distributions are not defined for x < 0
        
    def draw(self):
        if self.scale is None:
            return np.random.exponential(scale=1. / self.rate)
        else:
            return np.random.exponential(scale=self.scale)

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
        self.shape = shape
        self.scale = scale
        self.rate = rate
        
        params = {
            "a" : shape,
            "scale" : 1. / rate if scale is None else scale
            }
        super().__init__(params, stats.gamma, False)
        self.xlim[0] = 0 # Gamma distributions are not defined for x < 0
            
    def draw(self):
        if self.scale is None:
            return np.random.gamma(self.shape, 1. / self.rate)
        else:
            return np.random.gamma(self.shape, self.scale)

class Beta(Distribution):
    """Defines a probability space for a beta distribution.

    Attributes:
      a (float): alpha parameter for beta distribution
      b (float): beta parameter for beta distribution
    """

    def __init__(self, a, b, scale=None):
        self.a = a
        self.b = b
        
        params = {
            "a" : a,
            "b" : b
            }
        super().__init__(params, stats.beta, False)
        self.xlim = [0, 1] # Beta distributions are not defined for x < 0 and x > 1

    def draw(self):
        return np.random.beta(self.a, self.b)


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
                            "is not compatible with the dimensions" +
                            "of the covariance matrix.")
        self.mean = mean
        self.cov = cov
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(x, mean, cov)
 
    def plot():
        raise Exception("This is not defined for Multivariate Normal distributions.")
    
    def draw(self):
        return tuple(np.random.multivariate_normal(self.mean, self.cov))

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

        if corr is not None and not (-1 <= corr < 1):
            raise Exception("Correlation must be "
                            "between -1 and 1.")

        self.mean = [mean1, mean2]

        if var1 is None:
            var1 = sd1 ** 2
        if var2 is None:
            var2 = sd2 ** 2
        if cov is None:
            cov = corr * np.sqrt(var1 * var2)
        self.cov = [[var1, cov], [cov, var2]]
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(x, self.mean, self.cov)
