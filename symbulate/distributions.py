import numpy as np

from .probability_space import ProbabilitySpace

## Discrete Distributions

class Bernoulli(ProbabilitySpace):
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

    def draw(self):
        return np.random.binomial(n=1, p=self.p)

class Binomial(ProbabilitySpace):
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

    def draw(self):
        return np.random.binomial(n=self.n, p=self.p)

class Hypergeometric(ProbabilitySpace):
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

    def draw(self):
        return np.random.hypergeometric(ngood=self.N1, nbad=self.N0, nsample=self.n)

class Geometric(ProbabilitySpace):
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

    def draw(self):
        return np.random.geometric(p=self.p)

class NegativeBinomial(ProbabilitySpace):
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

    def draw(self):
        return np.random.negative_binomial(n=self.r, p=self.p)

Pascal = NegativeBinomial

class Poisson(ProbabilitySpace):
    """Defines a probability space for a Poisson distribution.

    Attributes:
      lam (float): rate parameter for the Poisson distribution
    """

    def __init__(self, lam):
        self.lam = lam

    def draw(self):
        return np.random.poisson(lam=self.lam)


## Continuous Distributions

class Uniform(ProbabilitySpace):
    """Defines a probability space for a uniform distribution.

    Attributes:
      a (float): lower bound for possible values
      b (float): upper bound for possible values
    """

    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def draw(self):
        return np.random.uniform(low=self.a, high=self.b)

class Normal(ProbabilitySpace):
    """Defines a probability space for a normal distribution.

    Attributes:
      mean (float): mean parameter of the normal distribution
      var (float): variance parameter of the normal distribution
    """

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    
    def draw(self):
        return np.random.normal(loc=self.mean, scale=np.sqrt(self.var))

class Exponential(ProbabilitySpace):
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

    def draw(self):
        if self.scale is None:
            return np.random.exponential(scale=1. / self.rate)
        else:
            return np.random.exponential(scale=self.scale)

class Gamma(ProbabilitySpace):
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
    
    def draw(self):
        if self.scale is None:
            return np.random.gamma(self.shape, 1. / self.rate)
        else:
            return np.random.gamma(self.shape, self.scale)
