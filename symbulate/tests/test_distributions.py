import unittest
import numpy as np
import scipy.stats as stats

from symbulate import *

Nsim = 10000


class TestBernoulli(unittest.TestCase):

    def test_p_one(self):
        X = RV(Bernoulli(1))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 1 for sim in sims))
    
    def test_sum(self):
        X = RV(Bernoulli(.4) ** 5)
        sims = X.apply(sum).sim(Nsim)
        obs = list(sims.tabulate().values())
        exp = Nsim * stats.binom(n=5, p=.4).pmf(range(6))
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > .01)


class TestNormal(unittest.TestCase):

    def test_sum(self):
        X = RV(Normal(mean=-1, sd=2) ** 3)
        sims = X.apply(sum).sim(Nsim)
        cdf = stats.norm(loc=-3,
                         scale=np.sqrt(12)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
