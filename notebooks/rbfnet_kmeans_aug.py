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

def calculate_cluster_std(points, center):
    n_points = points.shape[0]
    if n_points < 2:
        return 1
    return np.sqrt(
        ((points - center) ** 2).sum() / (n_points - 1)
    )


class RBFNetAugmentation(object):
    
    def __init__(self, passband2lam, n_hidden=10, regularization=None):
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
            The number of hidden units, equivalent to the number of clusters generated by KMeans.
            
        regularization : one of {None, 'l1', 'l2'}
            The type of regularization to use when training the final layer.
        """
        
        self.passband2lam = passband2lam
        self.n_hidden = n_hidden
        self.regularization = regularization
        
        self.ss = None
        
        self.clustering = None
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
        
        self.clustering = KMeans(self.n_hidden)
        self.clustering.fit(X_ss)
        self.centers = self.clustering.cluster_centers_
        self.stds = np.empty(self.n_hidden)
        labels = self.clustering.predict(X_ss)
        for label in range(self.n_hidden):
            self.stds[label] = calculate_cluster_std(X_ss[labels == label], self.centers[label])
        X_ss_rbf = rbf(X_ss, self.centers, self.stds)
        print(X)
        print(X_ss)
        print(X_ss_rbf)
        
        if self.regularization == 'l2':
            self.reg = Ridge()
        elif self.regularization == 'l1':
            self.reg = Lasso()
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
        
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)
        
        return t_aug, flux_aug, flux_err_aug, passband_aug