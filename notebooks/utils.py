import numpy as np
import pandas as pd


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

def regression_quality_metrics_report(y_true, y_pred):
    """
    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,)
        Estimated targets as returned by a regressor.
        
    Returns
    -------
    List of metric values: [rmse, mae, rse, rae, mape, rmsle]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse  = np.sqrt( mean_squared_error(y_true, y_pred) )
    mae   = mean_absolute_error(y_true, y_pred)
    
    rse  = np.sqrt( ( (y_true - y_pred)**2 ).sum() / ( (y_true - y_true.mean())**2 ).sum() )
    rae  = np.abs( y_true - y_pred ).sum() / np.abs( y_true - y_true.mean() ).sum()
    mape = 100. / len(y_true) * np.abs( 1. - y_pred/y_true ).sum()
    
    return [rmse, mae, rse, rae, mape]
