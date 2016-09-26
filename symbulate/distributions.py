import numpy as np

from .probability_space import ProbabilitySpace

## Discrete Distributions

class Bernoulli(ProbabilitySpace):

    def __init__(self, p):
        self.p = p

    def draw(self):
        return np.random.binomial(n=1, p=self.p)

class Binomial(ProbabilitySpace):

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def draw(self):
        return np.random.binomial(n=self.n, p=self.p)

class Hypergeometric(ProbabilitySpace):

    def __init__(self, n, N0, N1):
        self.n = n
        self.N0 = N0
        self.N1 = N1

    def draw(self):
        return np.random.hypergeometric(ngood=self.N1, nbad=self.N0, nsample=self.n)

class Geometric(ProbabilitySpace):

    def __init__(self, p):
        self.p = p

    def draw(self):
        return np.random.geometric(p=self.p)

class NegativeBinomial(ProbabilitySpace):

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def draw(self):
        return np.random.negative_binomial(n=self.n, p=self.p)

Pascal = NegativeBinomial

class Poisson(ProbabilitySpace):

    def __init__(self, lam):
        self.lam = lam

    def draw(self):
        return np.random.poisson(lam=self.lam)


## Continuous Distributions

class Uniform(ProbabilitySpace):

    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def draw(self):
        return np.random.uniform(low=self.a, high=self.b)

class Normal(ProbabilitySpace):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    
    def draw(self):
        return np.random.normal(loc=self.mean, scale=np.sqrt(self.var))

class Exponential(ProbabilitySpace):

    def __init__(self, scale=1.0, lam=None):
        self.scale = scale
        self.lam = lam

    def draw(self):
        if self.lam is None:
            return np.random.exponential(scale=self.scale)
        else:
            return np.random.exponential(scale=1. / self.lam)

class Gamma(ProbabilitySpace):

    def __init__(self, shape, scale=1.0, lam=None):
        self.shape = shape
        self.scale = scale
        self.lam = lam
    
    def draw(self):
        if self.lam is None:
            return np.random.gamma(self.shape, self.scale)
        else:
            return np.random.gamma(self.shape, 1. / self.lam)
