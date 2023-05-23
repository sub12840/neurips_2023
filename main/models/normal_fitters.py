import numpy as np
from scipy.optimize import minimize
import scipy.stats as scs

class UnivariateGaussianMD(object):
    """
    Fit W1 Univariate minimum distance estimator
    """
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def _optimize_dist(self, theta):
        estimates_tmp = self.noise_terms*theta[1] +theta[0]
        wasserstein_d = np.abs(estimates_tmp - self.barycenter)
        return np.mean(wasserstein_d)
    
    def fit(self, X, sampling_obs=40000, mc_sampled=20, maxit=10000):
        bary_enlarged = np.repeat(X, sampling_obs/X.shape[0])
        bary_enlarged.sort()

        noise_terms = scs.norm(0,1).rvs(size=(mc_sampled, bary_enlarged.shape[0]))
        noise_terms.sort()

        self.barycenter = np.broadcast_to(bary_enlarged, noise_terms.shape)
        self.noise_terms = noise_terms

        theta_init = np.random.uniform(0,1,(2,))
        response = minimize(self._optimize_dist,
                            theta_init,
                            method='nelder-mead',
                            options={'xatol': 1e-8, 'maxiter': maxit, 'disp': True})
        
        self.mu, self.sigma = response.x

    def sample(self, n, mc_samples=20):
        samples_ = scs.norm(loc=self.mu,
                            scale=self.sigma).rvs(size=(mc_samples,n))
        samples_.sort(axis=1)
        samples_avg = samples_.mean(axis=0)
        return samples_avg
    

class MVGaussianMLE(object):
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def fit(self, X):
        X = X[...,np.newaxis] if X.ndim == 2 else X
        self.mu = X.mean(0)
        self.sigma = np.einsum('ijk,ikj->jk', X-self.mu, X-self.mu) / (X.shape[0]-1)

    def prob(self, X):
        factor1 = (2*np.pi)**(-self.mu.shape[0]/2)*np.linalg.det(self.sigma)**(-1/2)
        factor2 = np.exp((-1/2)*np.einsum('ijk,jl,ilk->ik', X-self.mu, np.linalg.inv(self.sigma), X-self.mu))
        return factor1*factor2
    
    def sample(self, n):
        sample = scs.multivariate_normal(mean=self.mu.squeeze(),
                                         cov=self.sigma).rvs(n)
        return sample
