import numpy as np
#import gp_aug
import pandas as pd
from sklearn.model_selection import train_test_split


passband2lam  = {0: 1, 1: 2} # green, red 
color = {1: 'red', 0: 'green'}


def get_object(df, name_in_BTSdf):
    """df - csv with all obj"""
    assert isinstance(name_in_BTSdf, str), 'Попробуйте ввести название объекта из ZTF'
    if name_in_BTSdf[:2] == 'ZT':
        df_num = df[df.object_id == name_in_BTSdf]
        return df_num
    else:
        return None

def get_passband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve

def compile_obj(t, flux, flux_err, passband):
    obj = pd.DataFrame()
    obj['mjd']      = t
    obj['flux']     = flux
    obj['flux_err'] = flux_err
    obj['passband'] = passband
    return obj

def bootstrap_estimate_mean_stddev(arr, n_samples=10000):
    arr = np.array(arr)
    np.random.seed(0)
    bs_samples = np.random.randint(0, len(arr), size=(n_samples, len(arr)))
    bs_samples = arr[bs_samples].mean(axis=1)
    sigma = np.sqrt(np.sum((bs_samples - bs_samples.mean())**2) / (n_samples - 1))
    return np.mean(bs_samples), sigma