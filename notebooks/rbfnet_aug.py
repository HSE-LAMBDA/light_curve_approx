import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso


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

def rbf(X, c, s):
    diffs = (X ** 2).sum(axis=1).reshape(-1, 1) + (c ** 2).sum(axis=1).reshape(1, -1) \
          - 2 * X @ c.T
    diffs = -1 * diffs / (2 * s ** 2).reshape(1, -1)
    return np.exp(diffs)

def calculate_cluster_std(centers):
    diffs = (centers ** 2).sum(axis=1).reshape(-1, 1) + (centers ** 2).sum(axis=1).reshape(1, -1) \
          - 2 * centers @ centers.T
    return np.sqrt(np.maximum(diffs, 0)).max() / np.sqrt(2 * centers.shape[0])


class RBFNetAugmentation(object):
    
    def __init__(self, passband2lam, n_hidden=18, regularization=None, reg_alpha=1e-2):
        """
        Light Curve Augmentation based on Radial Basis Function Network
        https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/
        
        Parameters:
        -----------
        passband2lam : dict
            A dictionary, where key is a passband ID and value is Log10 of its wave length.
            Example: 
                passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
                                 
        n_hidden : int
            The number of hidden units. Must be divisible by the number of passbands.
            
        regularization : one of {None, 'l1', 'l2'}
            The type of regularization to use when training the final layer.
            
        reg_alpha : float
            Regularization coefficient, bigger means stronger regularization and weaker model.
        """
        assert n_hidden % len(passband2lam) == 0
        
        self.passband2lam = passband2lam
        self.n_hidden = n_hidden
        self.regularization = regularization
        self.reg_alpha = reg_alpha
        
        self.ss = None
        
        self.centers = None
        self.stds = None
        
        self.reg = None
        
    
    
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
        
        self.ss = StandardScaler()
        X_ss = self.ss.fit_transform(X)
        
        n_time = self.n_hidden // len(self.passband2lam)
        n_lam = len(self.passband2lam)
        time_centers = np.linspace(X_ss[:, 0].min(), X_ss[:, 1].max(), n_time)
        lam_centers = (np.array(list(self.passband2lam.values())) - self.ss.mean_[1]) / self.ss.var_[1] ** 0.5
        self.centers = np.concatenate([
            np.concatenate([time_centers] * n_lam).reshape(-1, 1),
            np.concatenate([lam_centers] * n_time).reshape(n_time, n_lam).T.reshape(-1, 1)
        ], axis=1)
        self.stds = np.full(self.n_hidden, calculate_cluster_std(self.centers))
        
        X_ss_rbf = rbf(X_ss, self.centers, self.stds)
        
        if self.regularization == 'l2':
            self.reg = Ridge(self.reg_alpha)
        elif self.regularization == 'l1':
            self.reg = Lasso(self.reg_alpha)
        else:
            self.reg = LinearRegression()
           
        self.reg.fit(X_ss_rbf, flux)
    
    
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
        X_ss_rbf = rbf(X_ss, self.centers, self.stds)
        
        flux_pred = self.reg.predict(X_ss_rbf)
        flux_err_pred = np.empty(flux_pred.shape)
        
        return np.maximum(0, flux_pred), flux_err_pred
        
    
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
        
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)
        
        return t_aug, flux_aug, flux_err_aug, passband_aug