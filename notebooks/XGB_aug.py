import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler



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


class XGBRegressionAugmentation(object):
    def __init__(self, passband2lam):
        """
        Light Curve Augmentation based on XGBRegressor
        
        Parameters:
        -----------
        passband2lam : dict
            A dictionary, where key is a passband ID and value is Log10 of its wave length.
            Example: 
                passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
        """
        self.passband2lam = passband2lam
        
        self.reg      = None
        self.X_scaler = None
        self.y_scaler = None
        self.ss_t     = None
        
    
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
        self.ss_t = StandardScaler().fit(np.array(t).reshape((-1, 1)))
        flux     = np.array(flux)
        flux_err = np.array(flux_err)
        X = np.concatenate(self.__array_joining(t, passband), axis=1)
        
        self.X_scaler = StandardScaler().fit(X)
        X_ss = self.X_scaler.transform(X)
        
        self.y_scaler = StandardScaler().fit(flux.reshape((-1, 1)))
        y_ss = self.y_scaler.transform(flux.reshape((-1, 1)))
        
        self.reg = XGBRegressor(n_estimators=100, max_depth=6, objective='reg:squarederror')
        self.reg.fit(X_ss, y_ss)
        
    
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
            Flux of the light curve observations, approximated by the augmentation model.d
        flux_err_pred : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        """
        X = np.concatenate(self.__array_joining(t, passband), axis=1)
        
        X_ss = self.X_scaler.transform(X)
        
        flux_pred = self.y_scaler.inverse_transform(self.reg.predict(X_ss))
        return np.maximum(flux_pred, 0), np.empty(flux_pred.shape)
    
    
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
        flux_err_pred : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        passband_aug : array-like
            Passband IDs for each observation.
        """
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)
        return t_aug, flux_aug, flux_err_aug, passband_aug
    
        
    def __array_joining(self, t, passband):
        t        = self.ss_t.transform(np.array(t).reshape((-1, 1)))
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        array_for_concatenate = [
            t.reshape((-1, 1)),
            np.power(t, 2).reshape((-1, 1)),
            np.power(t, 3).reshape((-1, 1)),
            1 / (t + 10).reshape((-1, 1)),
            np.exp(t).reshape((-1, 1)),
            np.exp(-t).reshape((-1, 1)),
            np.sin(t).reshape((-1, 1)),
            np.cos(t).reshape((-1, 1)),
            np.sinh(t).reshape((-1, 1)),
            np.cosh(t).reshape((-1, 1)),
            log_lam.reshape((-1, 1)),
            np.power(log_lam, 2).reshape((-1, 1)),
            np.power(log_lam, 3).reshape((-1, 1))
        ]
        return array_for_concatenate
