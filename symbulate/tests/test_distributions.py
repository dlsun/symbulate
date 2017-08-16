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
        key_list = [keys for keys in sims.tabulate()]
        obs = list(sims.tabulate().values())
        exp = [Nsim * stats.binom(n = 5, p = .4).pmf(k) for k in key_list]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bernoulli_Binomial_n_1(self):
        exp, obs, new_key = [], [], []    
        X = RV(Bernoulli(.4))
        sims = X.sim(Nsim)
        key_list = list(sims.tabulate().keys())
        for k in key_list:
            outcome = Nsim * stats.binom(n = 1, p = .4).pmf(k)
            if outcome > 5:
                exp.append(outcome)
                new_key.append(k)
        obs = [sims.tabulate()[k] for k in new_key]        
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)
        
class TestBinomial(unittest.TestCase):
    
    def test_Binomial_p_1(self):
        for nsample in range(1, 1000, 100):
            X = RV(Binomial(nsample, 1.0))
            sims = X.sim(Nsim)
        self.assertTrue(all(sim == nsample  for sim in sims))

    # warning message will appear for n = 0 Binomial 
    def test_Binomial_n_0(self):
        X = RV(Binomial(0, 0.5))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 0 for sim in sims))

    def test_Binomial_error_n(self):
        self.assertRaises(Exception, lambda: Binomial(-10, 0.4))
    
    def test_Binomial_additive(self):
        exp, obs, new_key = [], [], []
        X,Y = RV(Binomial(8, 0.6) * Binomial(5, 0.6))
        sims = (X&Y).sim(Nsim).apply(sum)
        key_list = list(sims.tabulate().keys())
        for k in key_list:
            outcome = Nsim * stats.binom(n = 13, p = .6).pmf(k)
            if outcome > 5:
                exp.append(outcome)
                new_key.append(k)
        obs = [sims.tabulate()[k] for k in new_key]
        pval = stats.chisquare(obs, exp).pvalue   
        self.assertTrue(pval > 0.01)
    

class TestHypergeometric(unittest.TestCase):

    def test_Hypergeometric_no_failures(self):
        X = RV(Hypergeometric(10, 0, 1000))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))

    def test_Hypergeometric_Binomial_converge(self):
        exp, obs, new_key = [], [], []
        X = RV(Hypergeometric(8, 200, 800))
        sims = X.sim(Nsim)
        key_list = list(sims.tabulate().keys())
        for k in key_list:
            outcome = Nsim * stats.binom(n = 8, p = .8).pmf(k)
            if outcome > 5:
                exp.append(outcome)
                new_key.append(k)
        obs = [sims.tabulate()[k] for k in new_key]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_Hypergeometric_error(self):
        self.assertRaises(Exception, lambda: Hypergeometric(10, 0, 8))


class TestGeometric(unittest.TestCase):
    
    def test_Geometric_error(self):
        self.assertRaises(Exception, lambda: Geometric(0))

       #TODO
    def test_Geometric_to_NBinom(self):
        #X = Geometric(0.8)
        #sims = X.sim(Nsim)
        #key_list = list(sims.tabulate().keys())
        #obs = list(sims.tabulate().values())
        #exp = [Nsim * stats.nbinom.pmf(k, 1, 0.8) for k in key_list] 
        #pval = stats.chisquare(obs, exp).pvalue
        #self.assertTrue(pval > 0.01)
        pass


class TestNegativeBinomial(unittest.TestCase):

    def test_NBinom_error_r(self):
        self.assertRaises(Exception, lambda: NegativeBinomial(-10, 0.6))
    
    def test_NBinom_p_1(self):
        X = NegativeBinomial(10, 1)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))
    
    #TODO
    def test_NBinom_additive(self):
        #X,Y = RV(NegativeBinomial(45, 0.6) * NegativeBinomial(10, 0.6))
        #sims = (X+Y).sim(Nsim)
        #key_list = list(sims.tabulate().keys())        
        #obs = list(sims.tabulate().values())
        #exp = [Nsim * stats.nbinom(55, 0.6).pmf(k) for k in key_list]
        #pval = stats.chisquare(obs, exp).pvalue
        #self.assertTrue(pval > .01)
        pass

    def test_NBinom_to_Geometric(self):
        exp, obs, new_key = [], [], []
        X = NegativeBinomial(r = 1, p = 0.8)
        sims = X.sim(Nsim)
        key_list = list(sims.tabulate().keys())
        for k in key_list:
            outcome = Nsim * stats.geom(0.8).pmf(k)
            if outcome > 5:
                exp.append(outcome)
                new_key.append(k)
        obs = [sims.tabulate()[k] for k in new_key]
        pval  = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)


class TestPascal(unittest.TestCase):

    def test_Pascal_error_r(self):
        self.assertRaises(Exception, lambda: Pascal(r = 0, p = 0.3))
    
    def test_Pascal_p_1(self):
        X = Pascal(r = 10, p = 1.0)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 0 for sim in sims))


class TestPoisson(unittest.TestCase):
    
    def test_Poisson_error(self):
        self.assertRaises(Exception, lambda: Poisson(lam = 0))

    def test_Poisson_additive(self):
        exp, obs, new_key = [], [], []
        X,Y = RV(Poisson(4) * Poisson(7)) 
        sims = (X+Y).sim(Nsim)
        key_list = list(sims.tabulate().keys())
        for k in key_list:
            outcome = Nsim * stats.poisson(11).pmf(k)
            if outcome > 5:
                exp.append(outcome)
                new_key.append(k)
        obs = [sims.tabulate()[k] for k in new_key]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > .01)

    def test_conditional_Poisson_add(self):
        exp, obs, new_key = [], [], []
        X,Y = RV(Poisson(4) * Poisson(7))
        sims = (X+Y).sim(Nsim)
        key_list = list(sims.tabulate().keys())
        for k in key_list:
            outcome = Nsim * stats.poisson(11).pmf(k)
            if outcome > 5:
                exp.append(outcome)
                new_key.append(k)
        obs = [sims.tabulate()[k] for k in new_key]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > .01)


#class DiscreteUniform(unittest.TestCase):


class TestUniform(unittest.TestCase):
    
    def test_Uniform_error(self):
        self.assertRaises(Exception, lambda: Uniform(6, -1))

    def test_conditional_exp_uniform(self):
        X,Y = RV(Exponential(3) ** 2)
        sims = (X|(X < 3) & (X+Y > 3)).sim(1000)
        exp = stats.uniform(0,3).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)


class TestNormal(unittest.TestCase):

    def test_Normal_error(self):
        self.assertRaises(Exception, lambda: Normal(0, -10)) 
    
    def test_sum(self):
        X = RV(Normal(mean=-1, sd=2) ** 3)
        sims = X.apply(sum).sim(Nsim)
        cdf = stats.norm(loc=-3,
                         scale=np.sqrt(12)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_Normal_standardize(self):
        X = RV(Normal(mean = 8, var = 4))
        X_stand = (X - 8) / 2
        sims = X_stand.sim(Nsim)
        cdf = stats.norm(loc = 0, scale = 1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_standardize_to_Normal(self):
        Z = RV(Normal(0, 1))
        X = 10 + 5 * Z
        sims = X.sim(Nsim)
        cdf = stats.norm(loc = 10, scale = 5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_normal_to_Gamma(self):
        X = RV(Normal(0, 1))
        X = X ** 2
        sims = X.sim(Nsim)
        cdf = stats.gamma(a = 1/2, scale = 2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

    def test_normal_to_ChiSquare(self):
        X = RV(Normal(0, 1))
        X = X ** 2
        sims = X.sim(Nsim)
        cdf = stats.chi2(df = 1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)


class TestExponential(unittest.TestCase):
    
    def test_Exponential_error(self):
        self.assertRaises(Exception, lambda: Exponential(-5))

    def test_Exponential_to_Gamma(self):
        X = RV(Exponential(rate = 0.9))
        sims = X.sim(Nsim)
        exp = stats.gamma(scale = 1/0.9, a = 1).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_sum_Gamma(self):
        X,Y,Z,A = RV(Exponential(rate = 0.9) ** 4)
        sims = (X+Y+Z+A).sim(Nsim)
        exp = stats.gamma(scale = 1/0.9, a = 4).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)


class TestGamma(unittest.TestCase):

    def test_Gamma_shape_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape = -5, rate = 40))

    def test_Gamma_rate_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape = 4, rate = -10))

    def test_Gamma_to_Exponential(self):
        X = Gamma(shape = 1, rate = 1/ 0.9)
        sims = X.sim(Nsim)
        exp = stats.expon(scale = 0.9).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)

        
    def test_Gamma_reshape(self):
        def g(x):
            return x*8

        X = RV(Gamma(shape = 9, scale = 4))
        sims = X.sim(Nsim).apply(g)
        exp = stats.gamma(scale = 4 * 8, a = 9).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)
 
    def test_Gamma_additive(self):
        X,Y = RV(Gamma(shape = 10, scale = 0.5) * Gamma(shape = 8, scale = 0.5))
        sims = (X+Y).sim(Nsim)
        exp = stats.gamma(scale = 0.5, a = 18).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)


class TestBeta(unittest.TestCase):
    
    def test_Beta_error_a(self):
        self.assertRaises(Exception, lambda: Beta(a = -10, b = 3))

    def test_Beta_error_b(self):
        self.assertRaises(Exception, lambda: beta(a = 3, b = -10))

    def test_Beta_to_Uniform(self):
        X = Beta(a = 1, b = 1)
        sims = X.sim(Nsim)
        exp = stats.uniform(0, 1).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)

    def test_Beta_to_ChiSquare(self):
        X,Y = RV(ChiSquare(3) * ChiSquare(5))
        sims = ((X/3) / (Y/5)).sim(Nsim)
        exp = stats.f(dfn = 3, dfd = 5).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)


class TestStudentT(unittest.TestCase):
    
    def test_StudentT_df_error(self):
        self.assertRaises(Exception, lambda: StudentT(0))

    def test_StudentT_to_Normal(self):
        X = StudentT(1000000)
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=0,scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)

class TestChiSquare(unittest.TestCase):
    
    def test_ChiSquare_error(self):
        self.assertRaises(Exception, lambda: ChiSquare(0.5))

    def test_ChiSquare_to_Gamma(self):
        X = RV(ChiSquare(df = 10))
        sims = X.sim(Nsim)
        exp = stats.gamma(a = 5, scale = 1/ 0.5).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_to_F(self):
        X,Y = RV(ChiSquare(3) * ChiSquare(5))
        sims = ((X/3) / (Y/5)).sim(Nsim)
        exp = stats.f(dfn = 3, dfd = 5).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)

    
class TestF(unittest.TestCase):

    def test_F_error(self):
        self.assertRaises(Exception, lambda: F(0, 5))

    def test_inverse_T(self):
        X = RV(F(4, 8))
        sims = (1/X).sim(Nsim)
        exp = stats.f(dfn = 8, dfd = 4).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)

    def test_StudentT_to_T(self):
        X = StudentT(15)
        sims = X.sim(Nsim).apply(square)
        exp = stats.f(dfn = 1, dfd = 15).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)

    def test_StudentT_to_T2(self):
        X = RV(StudentT(15))
        X = X**2
        sims = X.sim(Nsim)
        exp = stats.f(dfn = 1, dfd = 15).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)


class TestCauchy(unittest.TestCase):

    def test_Cauchy_mean(self):
        X = Cauchy()
        math.isnan(X.mean())


class TestLognormal(unittest.TestCase):

    def test_LogNormal_error(self):
        self.assertRaises(Exception, lambda: LogNormal(mean = 0, var = -5)) 

    def test_LogNormal_to_Normal(self):
        X = LogNormal(mean = 10, sd = 5)
        sims = X.sim(Nsim).apply(math.log)
        exp = stats.norm(loc = 10, scale = 5).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)


class TestPareto(unittest.TestCase):

    def test_Pareto_check_mean(self):
        x = stats.pareto(-3) 
        math.isnan(x.mean())


class TestRayleigh(unittest.TestCase):

    def test_Rayleigh_Normal(self):
        A,B = RV(Normal(0, 1) * Normal(0, 1))
        sims = (A**2 + B**2).sim(Nsim).apply(math.sqrt)
        exp = stats.rayleigh.cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > .01)


class TestMultivariateNormal(unittest.TestCase):
     
    def test_MultivariateNormal_mean_cov_error(self):
        self.assertRaises(Exception, lambda: MultivariateNormal([2], [[2,2], [3,4]]))

    def test_MultivariateNormal_cov_square_error(self):
        self.assertRaises(Exception, lambda: MultivariateNormal([2,4], [[2,4,5],[2,1]]))


class TestBivariateNormal(unittest.TestCase):
    
    def test_BivariateNormal_error1(self):
        self.assertRaises(Exception, lambda: BivariateNormal(3.0, 4.0, -3.0, 3.0, 0.9))

    def test_BivariateNormal_error2(self):
        self.assertRaises(Exception, lambda: BivariateNormal(3.0, 4.0, 3.0, 3.0, 1.1))
 
    def test_BivNormal_condDistr_r(self):
        X,Y = RV(BivariateNormal(20, 10, 3, 5, 0.5))
        sims = (Y | (abs(X - 21) < 0.1)).sim(1000)
        exp = stats.norm(10 + 0.5 * 5 / 3 * (21 - 20), 5 * math.sqrt(1-(0.5)**2)).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)

    #if approx 1 (0.9999), it fails?
    def test_BivNormal_condDistr_r(self):
        X,Y = RV(BivariateNormal(20, 10, 3, 5, 0.9))
        sims = (Y | (abs(X - 21) < 0.1)).sim(1000)
        exp = stats.norm(10 + 0.9 * 5 / 3 * (21 - 20), 5 * math.sqrt(1-(0.9)**2)).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)

    #if approx -1 (-0.99999), it fails
    def test_BivNormal_condDistr_r(self):
        X,Y = RV(BivariateNormal(20, 10, 3, 5, -0.9))
        sims = (Y | (abs(X - 21) < 0.1)).sim(1000)
        exp = stats.norm(10 + (-0.9) * 5 / 3 * (21 - 20), 5 * math.sqrt(1-(-0.9)**2)).cdf
        pval = stats.kstest(sims, exp).pvalue
        self.assertTrue(pval > 0.01)



def square(x):
    return x**2
def sqrt(x):
    return x**1/2
