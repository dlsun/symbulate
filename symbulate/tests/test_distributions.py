import unittest
import scipy.stats as stats

from symbulate import *

Nsim = 10000

class TestBernoulli(unittest.TestCase):

    def test_bernolli_sum(self):
        X = RV(Bernoulli(.4) ** 5)
        sims = X.apply(sum).sim(Nsim)
        obs = list(sims.tabulate().values())
        exp = Nsim * stats.binom.pmf(range(6), n=5, p=.4)
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > .001)
