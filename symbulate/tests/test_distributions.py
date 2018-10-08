import unittest
import numpy as np
import scipy.stats as stats

from symbulate import *

Nsim = 10000


class TestBernoulli(unittest.TestCase):

    def test_p_one(self):
        X = RV(Bernoulli(p=1))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 1 for sim in sims))
   
    def test_sum(self):
        exp_list, obs_list = [], []
        X = RV(Bernoulli(p=.4) ** 5)
        sims = X.apply(sum).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(6):
            expected = Nsim * stats.binom(n=5, p=.4).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bernoulli_Binomial_n_1(self):
        exp_list, obs_list = [], []   
        X = RV(Bernoulli(p=.4))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(2): 
            expected = Nsim * stats.binom(n=1, p=.4).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > 0.01)

        
class TestBinomial(unittest.TestCase):
    
    def test_Binomial_p_1(self):
        for nsample in range(1, 1000, 100):
            X = RV(Binomial(n=nsample, p=1.0))
            sims = X.sim(Nsim)
        self.assertTrue(all(sim == nsample for sim in sims))

    def test_Binomial_error_n(self):
        self.assertRaises(Exception, lambda: Binomial(n=-10, p=0.4))
    
    def test_Binomial_additive(self):
        exp_list, obs_list = [], []
        X, Y = RV(Binomial(n=8, p=0.6) * Binomial(n=5, p=0.6))
        sims = (X & Y).sim(Nsim).apply(sum)
        simulated = sims.tabulate()
        for k in range(2, 14):
            expected = Nsim * stats.binom(n=13, p=.6).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue   
        self.assertTrue(pval > 0.01)
    

class TestHypergeometric(unittest.TestCase):

    def test_Hypergeometric_no_failures(self):
        X = RV(Hypergeometric(n=10, N0=0, N1=1000))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))

    def test_Hypergeometric_Binomial_converge(self):
        exp_list, obs_list = [], []
        X = RV(Hypergeometric(n=8, N0=200, N1=800))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(9):
            expected = Nsim * stats.binom(n=8, p=.8).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > 0.01)

    def test_Hypergeometric_error_n_greater(self):
        self.assertRaises(Exception, 
                          lambda: Hypergeometric(n=10, N0=1, N1=8))


class TestGeometric(unittest.TestCase):
    
    def test_Geometric_error(self):
        self.assertRaises(Exception, lambda: Geometric(p=0))

    def test_Geometric_to_NBinom(self):
        exp_list, obs_list = [], []
        X = Geometric(p=0.8)
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(1, 10):
            expected = Nsim  * stats.nbinom(n=1,p=0.8).pmf(k-1)
            if expected > 5:
                exp_list.append(expected) 
                obs_list.append(simulated[k]) 
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > 0.01)


class TestNegativeBinomial(unittest.TestCase):

    def test_NBinom_error_r(self):
        self.assertRaises(Exception, 
                          lambda: NegativeBinomial(r=-10, p=0.6))
    
    def test_NBinom_p_1(self):
        X = NegativeBinomial(r=10, p=1)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))
    
    def test_NBinom_Pascal_additive(self):
        exp_list, obs_list = [], []
        X, Y = RV(Pascal(r=4, p=0.6) * Pascal(r=6, p=0.6))
        sims = (X + Y).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(10, 35):
            expected = Nsim * stats.nbinom(n=10, p=0.6).pmf(k)
            if expected > 5:
                exp_list.append(expected) 
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > .01)

    def test_NBinom_to_Geometric(self):
        exp_list, obs_list = [], []
        X = NegativeBinomial(r=1, p=0.8)
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(10):
            expected = Nsim * stats.geom(p=0.8).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval  = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > 0.01)


class TestPascal(unittest.TestCase):

    def test_Pascal_error_r(self):
        self.assertRaises(Exception, lambda: Pascal(r=0, p=0.3))
    
    def test_Pascal_p_1(self):
        X = Pascal(r=10, p=1.0)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 0 for sim in sims))


class TestPoisson(unittest.TestCase):
    
    def test_Poisson_error(self):
        self.assertRaises(Exception, lambda: Poisson(lam=0))

    def test_Poisson_additive(self):
        exp_list, obs_list = [], []
        X, Y = RV(Poisson(lam=4) * Poisson(lam=7))
        sims = (X + Y).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(25):
            expected = Nsim * stats.poisson(mu=11).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > .01)

    def test_conditional_Poisson_add(self):
        obs_list, exp_list = [], []
        X,Y = RV(Poisson(lam=6) * Poisson(lam=7))
        sims = (X|(X+Y == 12)).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(12):
            expected = Nsim * stats.binom(n=12, p=6 / 13).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > .01)


class TestUniform(unittest.TestCase):
    
    def test_Uniform_error(self):
        self.assertRaises(Exception, lambda: Uniform(a=6, b=-1))

    def test_conditional_exp_uniform(self):
        X, Y = RV(Exponential(rate=3) ** 2)
        sims = (X | (X < 3) & (X + Y > 3)).sim(1000)
        cdf = stats.uniform(loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_ChiSquare(self):
        X = RV(Uniform(a=0, b=1))
        sims = (-2 * log(X)).sim(Nsim)
        cdf = stats.chi2(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Uniform_to_Exponential(self):
        X = RV(Uniform(a=0, b=1))
        Y = -1 / 5 * log(X)
        sims = Y.sim(Nsim)
        cdf = stats.expon(scale=1/5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Uniform_to_Beta(self):
        X = RV(Uniform(a=0, b=1))
        sims = (X ** 15).sim(Nsim)
        cdf = stats.beta(a=1/15, b=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Uniform_to_Cauchy(self):
        X = RV(Uniform(a=0, b=1))
        sims = (pi * (X - 1/2)).apply(tan).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Pareto(self):
        X = RV(Uniform(a=0, b=1))
        sims = (2 * X ** (-1 / 0.1)).sim(10000)
        cdf = stats.pareto(b=0.1, loc=0, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
        
class TestNormal(unittest.TestCase):

    def test_Normal_error(self):
        self.assertRaises(Exception, lambda: Normal(mean=0, var=-10)) 
    
    def test_sum(self):
        X = RV(Normal(mean=-1, sd=2) ** 3)
        sims = X.apply(sum).sim(Nsim)
        cdf = stats.norm(loc=-3,
                         scale=np.sqrt(12)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_sum_Standard_Normal(self):
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X + Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_subtract_Standard_Normal(self):
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X - Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_standardize(self):
        X = RV(Normal(mean=8, var=4))
        X_stand = (X - 8) / 2
        sims = X_stand.sim(Nsim)
        cdf = stats.norm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_standardize_to_Normal(self):
        Z = RV(Normal(mean=0, sd=1))
        X = 10 + 5 * Z
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=10, scale=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_to_Gamma(self):
        X = RV(Normal(mean=0, var=1))
        X = X ** 2
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=1/2, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_to_ChiSquare(self):
        X = RV(Normal(mean=0, var=1))
        X = X ** 2
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_to_Cauchy(self):
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X / Y).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_to_F(self):
        A, B, C, V, W, X, Y, Z = RV(Normal(mean=0, var=1) ** 8)
        sims = ((((A ** 2) + (B ** 2) + (C ** 2)) / 3) /
               (((V ** 2) + (W ** 2) + (X ** 2) + (Y ** 2) + (Z ** 2)) 
                                                            / 5)).sim(Nsim)
        cdf = stats.f(dfn=3, dfd=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_sum_Normal_to_ChiSquare(self):
        X, Y, Z, A, B = RV(Normal(mean=0, var=1) ** 5)
        sims = ((X ** 2) + (Y ** 2) + (Z ** 2) + (A ** 2) + (B ** 2)).sim(Nsim)
        cdf = stats.chi2(df=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

 
class TestExponential(unittest.TestCase):
    
    def test_Exponential_error(self):
        self.assertRaises(Exception, lambda: Exponential(rate=-5))

    def test_Exponential_to_Gamma(self):
        X = RV(Exponential(rate=0.9))
        sims = X.sim(Nsim)
        cdf = stats.gamma(scale=1/0.9, a=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_sum_Gamma(self):
        X, Y, Z, A = RV(Exponential(rate=0.9) ** 4)
        sims = (X + Y + Z + A).sim(Nsim)
        cdf = stats.gamma(scale=1/0.9, a=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_ChiSquare(self):
        X = RV(Exponential(rate=1/2))
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Exponential_to_Pareto(self):
        X = RV(Exponential(rate=2))
        sims = (3 * exp(X)).sim(Nsim)
        cdf = stats.pareto(b=2, loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Exponential_to_Weibull(self):
        X = RV(Exponential(rate=5))
        sims = X.sim(Nsim)
        cdf= stats.frechet_r(scale=1/5, c=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Exponential_to_Rayleigh(self):
        X = RV(Exponential(rate=5))
        sims = sqrt(X).sim(Nsim)
        cdf = stats.rayleigh(scale=1 / sqrt(2 * 5)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Poisson_Exponential_to_Geometric(self):
        def poisson_exp():
            x = Exponential(rate=1 / lam).draw() 
            z = RV(Poisson(x)).draw()
            return z

        exp_list, obs_list, lam = [], [], 1/5
        P = ProbabilitySpace(poisson_exp)
        A = RV(P)
        sims = (A + 1).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(40):
            expected = Nsim * stats.geom(p=1 / (1 + lam)).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > .01)


class TestGamma(unittest.TestCase):

    def test_Gamma_shape_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape=-5, rate=40))

    def test_Gamma_rate_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape=4, rate=-10))

    def test_Gamma_to_Exponential(self):
        X = Gamma(shape=1, rate=1/ 0.9)
        sims = X.sim(Nsim)
        cdf = stats.expon(scale=0.9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
        
    def test_Gamma_reshape(self):
        X = RV(Gamma(shape=9, scale=4))
        sims = (X * 8).sim(Nsim)
        cdf = stats.gamma(scale=4 * 8, a=9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
 
    def test_Gamma_additive(self):
        X, Y = RV(Gamma(shape=10, scale=0.5) * 
                  Gamma(shape=8, scale=0.5))
        sims = (X + Y).sim(Nsim)
        cdf = stats.gamma(scale=0.5, a=18).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Gamma_to_Beta(self):
        X, Y = RV(Gamma(shape=5, scale=8) * Gamma(shape=4, scale=8))
        sims = (X / (X + Y)).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Gamma_to_F(self):
        X, Y = RV(Gamma(shape=2, rate=5) * Gamma(shape=4, rate=7))
        sims = ((4 * 5 * X) / (2 * 7 * Y)).sim(Nsim)
        cdf = stats.f(dfn=2 * 2, dfd=2 * 4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Gamma_to_ChiSquare(self):
        X = RV(Gamma(shape=10/2, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=10).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)


class TestBeta(unittest.TestCase):
    
    def test_Beta_error_a(self):
        self.assertRaises(Exception, lambda: Beta(a=-10, b=3))

    def test_Beta_error_b(self):
        self.assertRaises(Exception, lambda: Beta(a=3, b=-10))

    def test_Beta_to_Uniform(self):
        X = Beta(a=1, b=1)
        sims = X.sim(Nsim)
        cdf = stats.uniform(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Beta_symmetry(self):
        X = RV(Beta(a=4, b=5))
        sims = (1 - X).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Beta_to_Exponential(self):
        X = RV(Beta(a=0.7, b=1))
        sims = (-log(X)).sim(Nsim)
        cdf = stats.expon(scale=1 / 0.7).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Beta_to_F(self):
        X = RV(Beta(a=10/2, b=12/2))
        sims = (12 * X / (10 * (1 - X))).sim(Nsim)
        cdf = stats.f(dfn=10, dfd=12).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)


class TestStudentT(unittest.TestCase):
    
    def test_StudentT_df_error(self):
        self.assertRaises(Exception, lambda: StudentT(df=0))

    def test_StudentT_to_Normal(self):
        X = StudentT(df=Nsim)
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=0,scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_ChiSquare_to_StudentT(self):
        X, Y = RV(Normal(mean=0, var=1) * ChiSquare(df=5))
        sims = (X / sqrt(Y / 5)).sim(Nsim)
        cdf = stats.t(df=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)    


class TestChiSquare(unittest.TestCase):
    
    def test_ChiSquare_error(self):
        self.assertRaises(Exception, lambda: ChiSquare(df=0.5))

    def test_ChiSquare_to_Gamma(self):
        X = RV(ChiSquare(df=10))
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=5, scale=1/0.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_to_F(self):
        X, Y = RV(ChiSquare(df=3) * ChiSquare(df=5))
        sims = ((X / 3) / (Y / 5)).sim(Nsim)
        cdf = stats.f(dfn=3, dfd=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_ChiSquare_to_Beta(self):
        X, Y = RV(ChiSquare(df=4) * ChiSquare(df=5))
        sims = (X / (X + Y)).sim(Nsim)
        cdf = stats.beta(a=4/2, b=5/2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
    

class TestF(unittest.TestCase):

    def test_F_error(self):
        self.assertRaises(Exception, lambda: F(dfN=0, dfD=5))

    def test_inverse_T(self):
        X = RV(F(dfN=4, dfD=8))
        sims = (1 / X).sim(Nsim)
        cdf = stats.f(dfn=8, dfd=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_StudentT_to_F(self):
        X = RV(StudentT(df=15))
        sims = (X ** 2).sim(Nsim)
        cdf = stats.f(dfn=1, dfd=15).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_F_to_Beta(self):
        X = RV(F(dfN=5, dfD=8))
        sims = ((5 * X / 8) / (1 + (5 * X / 8))).sim(Nsim)
        cdf = stats.beta(a=5/2, b=8/2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)


class TestCauchy(unittest.TestCase):

    def test_Cauchy_mean(self):
        X = Cauchy()
        math.isnan(X.mean())

    def test_Cauchy_to_T(self):
        X = RV(Cauchy())
        sims = X.sim(Nsim)
        cdf = stats.t(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_inverse(self):
        X = RV(Cauchy())
        sims = (1 / X).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_additive(self):
        X, Y = RV(Cauchy()**2)
        sims = (X + Y).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestLognormal(unittest.TestCase):

    def test_LogNormal_error(self):
        self.assertRaises(Exception, lambda: LogNormal(mu=0, sigma=-5)) 

    def test_LogNormal_to_Normal(self):
        X = LogNormal(mu=10, sigma=5)
        sims = X.sim(Nsim).apply(log)
        cdf = stats.norm(loc=10, scale=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_to_LogNormal(self):
        X = RV(Normal(mean=10, sd=5))
        sims = X.apply(exp).sim(Nsim)
        cdf = stats.lognorm(s=5, scale=exp(10)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_LogNormal_Product(self):
        X, Y = RV(LogNormal(mu=10, sigma=5) * LogNormal(mu=11, sigma=6))
        sims = (X * Y).sim(Nsim)
        cdf = stats.lognorm(s=sqrt(25 + 36), scale=exp(21)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

class TestPareto(unittest.TestCase):

    def test_Pareto_check_mean(self):
        x = stats.pareto(b=-3) 
        math.isnan(x.mean())

    def test_Pareto_to_Exponential(self):
        X = RV(Pareto(b=1.5, scale=0.1))
        sims = (log(X / 0.1)).sim(Nsim)
        cdf = stats.expon(scale=1/1.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)


class TestWeibull(unittest.TestCase):
    def test_Weibull_to_Exponential(self):
        '''X = RV(Weibull(scale=1, c=10))
           sims = X.sim(Nsim)
           cdf = stats.expon(scale=1/10).cdf
           pval = stats.kstest(sims, cdf).pvalue
           self.assertTrue(pval > .01)
        '''
        pass

class TestRayleigh(unittest.TestCase):

    def test_Rayleigh_Normal(self):
        A, B = RV(Normal(mean=0, var=1) * Normal(mean=0, var=1))
        sims = (A ** 2 + B ** 2).apply(sqrt).sim(Nsim)
        cdf = stats.rayleigh.cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Rayleigh_to_Chi(self):
        X = RV(Rayleigh())
        sims = X.sim(Nsim)
        cdf = stats.chi(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)


class TestMultivariateNormal(unittest.TestCase):
     
    def test_MultivariateNormal_mean_cov_error(self):
        self.assertRaises(Exception, 
             lambda: MultivariateNormal(mean=[2], cov=[[2,2], [3,4]]))

    def test_MultivariateNormal_cov_square_error(self):
        self.assertRaises(Exception, 
             lambda: MultivariateNormal(mean=[2,4], cov=[[2,4,5],[2,1]]))


class TestBivariateNormal(unittest.TestCase):
    
    def test_BivariateNormal_error1(self):
        self.assertRaises(Exception, lambda: 
             BivariateNormal(mean1=3, mean2=4, sd1=-3, sd2=3, corr=0.9))

    def test_BivariateNormal_error2(self):
        self.assertRaises(Exception, lambda: 
             BivariateNormal(mean1=3, mean2=4, sd1=3, sd2=3, corr=1.1))
 
    def test_LinCom_BivNormal(self):
        X, Y = RV(BivariateNormal(mean1=30, mean2=50, sd1=8, sd2=6, corr=-0.4))
        Z = 4 * X - 2 * Y
        Z_mean = 4 * 30 + (-2) * 50
        Z_var = 16 * 64 + 4 * 36 + 2 * (4) * (-2) * (-0.4 * 8 * 6)
        sims = Z.sim(Nsim)
        cdf = stats.norm(loc=Z_mean, scale=sqrt(Z_var)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
    
    def test_BivNormal_condDistr_r(self):
        for c in [-0.9, 0.9, 0.1]:
            X,Y = RV(BivariateNormal(mean1=20, mean2=10, sd1=3, sd2=5, corr=c))
            sims = (Y | (abs(X - 21) < 0.1)).sim(500)
            cdf = stats.norm(loc=10 + c * 5 / 3 * (21 - 20), 
                             scale=5 * sqrt(1 - c ** 2)).cdf
            pval = stats.kstest(sims, cdf).pvalue
            self.assertTrue(pval > 0.01)


