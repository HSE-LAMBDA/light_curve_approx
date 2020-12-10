import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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


class LinearRegressionAugmentation(object):
    def __init__(self, passband2lam):
        self.passband2lam = passband2lam
        self.ss = StandardScaler()
        self.reg = None
        
    
    def fit(self, t, flux, flux_err, passband):
        t        = np.array(t)
        flux     = np.array(flux)
        flux_err = np.array(flux_err)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        X_ss = self.ss.fit_transform(X)
        
        self.reg = LinearRegression()
        self.reg.fit(X_ss, flux)
        
    
    def predict(self, t, passband, copy=True):
        t        = np.array(t)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        X_ss = self.ss.transform(X)
        flux_pred = self.reg.predict(X_ss)
        return flux_pred
    
    def augmentation(self, t_min, t_max, n_obs=100):
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs)
        flux_aug = self.predict(t_aug, passband_aug, copy=True)
        return t_aug, flux_aug, passband_aug
    
        