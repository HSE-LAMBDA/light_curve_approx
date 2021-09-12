import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel as C
from scipy.stats import wilcoxon

def add_log_lam(passband, passband2lam):
    log_lam = np.array([passband2lam[i] for i in passband])
    return log_lam

def create_aug_data(t_min, t_max, n_passbands, n_obs=1000):
    t = []
    passband = []
    for i_pb in range(n_passbands):
        t += list(np.linspace(t_min, t_max, n_obs))
        passband += [i_pb]*n_obs
    return np.array(t), np.array(passband)

def bootstrap_estimate_mean_stddev(arr, n_samples=10000):
    arr = np.array(arr)
    np.random.seed(0)
    bs_samples = np.random.randint(0, len(arr), size=(n_samples, len(arr)))
    bs_samples = arr[bs_samples].mean(axis=1)
    sigma = np.sqrt(np.sum((bs_samples - bs_samples.mean())**2) / (n_samples - 1))
    return np.mean(bs_samples), sigma

def wilcoxon_pvalue(arr1, arr2):
    return wilcoxon(arr1, arr2).pvalue


class GaussianProcessesAugmentation(object):
    
    def __init__(self, passband2lam, kernel = C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(), flag_err = False):
        """
        Light Curve Augmentation based on Gaussian Processes Regression
        
        Parameters:
        -----------
        passband2lam : dict
            A dictionary, where key is a passband ID and value is Log10 of its wave length.
            Example: 
                passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
        """
        
        self.passband2lam = passband2lam
        
        self.ss = None
        self.reg = None
        self.kernel = kernel
        self.flag_err = flag_err
    
    def fit(self, t, flux, flux_err, passband):
        """
        Fit an augmentation model.
        
        Parameters:
        -----------
        t : array-like
            Timestamps of light curve observations.
        flux : array-like
            Flux of the light curve observations.
        flux_err : array-like
            Flux errors of the light curve observations.
        passband : array-like
            Passband IDs for each observation.
        """
        
        t        = np.array(t)
        flux     = np.array(flux)
        flux_err = np.array(flux_err)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        
        X_error = (flux_err / np.std(flux)).reshape(-1)

        self.ss = StandardScaler()
        #self.sserr = StandardScaler(with_mean=False)
        
        #self.ss = RobustScaler((45.0, 55.0))
        X_ss = self.ss.fit_transform(X)
        #X_error = self.sserr.fit_transform(X_error).reshape(-1)

        
        #kernel = C(1.0) * RBF([1.0, 1.0]) + WhiteKernel()
#         kernel = C(1.0) * Matern() * RBF([1.0, 1.0]) + Matern() #+ WhiteKernel()
        if self.flag_err:
            self.reg = GaussianProcessRegressor(kernel=self.kernel, alpha=X_error**2, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5, normalize_y=True, random_state=42)

        else:
            self.reg = GaussianProcessRegressor(kernel=self.kernel, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5, normalize_y=True, random_state=42)
            
        self.reg.fit(X_ss, flux)
    
    
    def predict(self, t, passband, copy=True):
        """
        Apply the augmentation model to the given observation mjds.
        
        Parameters:
        -----------
        t : array-like
            Timestamps of light curve observations.
        passband : array-like
            Passband IDs for each observation.
            
        Returns:
        --------
        flux_pred : array-like
            Flux of the light curve observations, approximated by the augmentation model.
        flux_err_pred : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        """
        
        t        = np.array(t)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        X_ss = self.ss.transform(X)
        #X_ss = X
        
        flux_pred, flux_err_pred = self.reg.predict(X_ss, return_std=True)
        
        return flux_pred, flux_err_pred
        
    
    def augmentation(self, t_min, t_max, n_obs=100):
        """
        The light curve augmentation.
        
        Parameters:
        -----------
        t_min, t_max : float
            Min and max timestamps of light curve observations.
        n_obs : int
            Number of observations in each passband required.
            
        Returns:
        --------
        t_aug : array-like
            Timestamps of light curve observations.
        flux_aug : array-like
            Flux of the light curve observations, approximated by the augmentation model.
        flux_err_aug : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        passband_aug : array-like
            Passband IDs for each observation.
        """
        
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs = n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)
        
        return t_aug, flux_aug, flux_err_aug, passband_aug