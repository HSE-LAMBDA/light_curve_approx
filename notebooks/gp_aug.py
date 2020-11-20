import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C


def add_log_lam(data, passband2lam):
    passbands = data.passband.values
    log_lam = [passband2lam[i] for i in passbands]
    data['log_lam'] = log_lam
    return data

def create_aug_data(mjd_min, mjd_max, n_passbands, n_obs=1000):
    dfs = []
    for passband in range(n_passbands):
        df = pd.DataFrame()
        df['mjd'] = np.linspace(mjd_min, mjd_max, n_obs)
        df['passband'] = passband
        df['flux'] = 0
        df['flux_err'] = 0
        dfs.append(df)
    data = pd.concat(dfs, axis=0)
    return data


class GaussianProcessesAugmentation(object):
    
    def __init__(self, passband2lam):
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
    
    
    def fit(self, data):
        """
        Fit an augmentation model.
        
        Parameters:
        -----------
        data : panda.DataFrame
            A Data Frame with light curve observations. 
            Mandatory columns: ['mjd', 'passband', 'flux', 'flux_err'].
        """
        
        data = add_log_lam(data, self.passband2lam)
        
        X = data[['mjd', 'log_lam']].values
        y = data['flux'].values
        
        self.ss = StandardScaler()
        X_ss = self.ss.fit_transform(X)
        
        kernel = C(1.0) * RBF([1.0, 1.0]) + WhiteKernel()
        self.reg = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, 
                                            optimizer="fmin_l_bfgs_b", random_state=42)

        self.reg.fit(X_ss, y)
    
    
    def predict(self, data, copy=True):
        """
        Apply the augmentation model to the given observation mjds.
        
        Parameters:
        -----------
        data : panda.DataFrame
            A Data Frame with light curve observations. 
            Mandatory columns: ['mjd', 'passband', 'flux', 'flux_err'].
            
        Returns:
        --------
        out_data : panda.DataFrame
            A Data Frame with light curve observations, where 'flux' and 'flux_err' are estimated by the augmentation model. 
            Columns: ['mjd', 'passband', 'flux', 'flux_err'].
        """
        
        data = add_log_lam(data, self.passband2lam)
        
        X     = data[['mjd', 'log_lam']].values
        y     = data['flux'].values
        y_err = data['flux_err'].values
        
        X_ss = self.ss.transform(X)
        
        y_pred, y_std_pred = self.reg.predict(X_ss, return_std=True)
        
        if copy:
            out_data = data.copy()
        else:
            out_data = data
        
        out_data['flux']     = y_pred
        out_data['flux_err'] = y_std_pred
        
        return out_data
        
    
    def augmentation(self, data, n_obs=100):
        """
        The light curve augmentation.
        
        Parameters:
        -----------
        data : panda.DataFrame
            A Data Frame with light curve observations. 
            Mandatory columns: ['mjd', 'passband', 'flux', 'flux_err'].
            
        Returns:
        --------
        aug_data : panda.DataFrame
            A Data Frame with light curve observations, where 'flux' and 'flux_err' are estimated by the augmentation model. 
            Columns: ['mjd', 'passband', 'flux', 'flux_err'].
        """
        
        mjd = data['mjd'].values
        
        aug_data = create_aug_data(mjd.min(), mjd.max(), len(self.passband2lam), n_obs)
        aug_data = self.predict(aug_data, copy=True)
        
        return aug_data