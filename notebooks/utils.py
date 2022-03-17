import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from scipy import stats


def nlpd_metric(flux, flux_pred, flux_err_pred):
    """
    The Negative Log Predictive Density (NLPD) metric calculation.
    Source: http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf
    
    Parameters:
    -----------
    flux : array-like
        Flux of the light curve observations.
    flux_pred : array-like
        Flux of the light curve observations, approximated by the augmentation model.
    flux_err_pred : array-like
        Flux errors of the light curve observations, estimated by the augmentation model.
        
    Returns:
    --------
    metric : float
        NLPD metrc value.
    """
    
    metric = (flux - flux_pred)**2 / (2 * flux_err_pred**2) + np.log(flux_err_pred) + 0.5 * np.log(2 * np.pi)
    
    return metric.mean()


def nrmse_metric(flux, flux_err, flux_pred):
    """
    The normalized Root Mean Squared Error (nRMSE) metric. 
    Source: http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf
    
    Parameters:
    -----------
    flux : array-like
        Flux of the light curve observations.
    flux_err : array-like
            Flux errors of the light curve observations.
    flux_pred : array-like
        Flux of the light curve observations, approximated by the augmentation model.
        
    Returns:
    --------
    metric : float
        nRMSE metrc value.
    """
    
    metric = (flux - flux_pred)**2 / flux_err**2
    
    return np.sqrt(metric.mean())


def picp_metric(flux, flux_pred, flux_err_pred, alpha=0.90):
    """
    The Prediction Interval Coverage Probability (PICP) metric. 
    Source: https://www.sciencedirect.com/science/article/pii/S0893608006000153?via%3Dihub
    
    Parameters:
    -----------
    flux : array-like
        Flux of the light curve observations.
    flux_pred : array-like
        Flux of the light curve observations, approximated by the augmentation model.
    flux_err_pred : array-like
        Flux errors of the light curve observations, estimated by the augmentation model.
    alpha : float [0, 1]
        Fraction of the distribution inside confident intervals.
        
    Returns:
    --------
    metric : float
        PICP metrc value.
    """
    
    p_left, p_right = stats.norm.interval(alpha=alpha, loc=flux_pred, scale=flux_err_pred)
    metric = (flux > p_left) * (flux <= p_right)
    
    return metric.mean()



def regression_quality_metrics_report(flux, flux_pred, flux_err=None, flux_err_pred=None, alpha=0.90):
    """
    Parameters:
    -----------
    flux : array-like
        Flux of the light curve observations.
    flux_pred : array-like
        Flux of the light curve observations, approximated by the augmentation model.
    flux_err : array-like
            Flux errors of the light curve observations.
    flux_err_pred : array-like
        Flux errors of the light curve observations, estimated by the augmentation model.
    alpha : float [0, 1]
        Fraction of the distribution inside confident intervals.
        
    Returns
    -------
    List of metric values: [rmse, mae, rse, rae, mape, rmsle, nlpd, nrmse, picp]
    """
    
    flux = np.array(flux)
    flux_pred = np.array(flux_pred)
    
    rmse  = np.sqrt( mean_squared_error(flux, flux_pred) )
    mae   = mean_absolute_error(flux, flux_pred)
    
    rse  = np.sqrt( ( (flux - flux_pred)**2 ).sum() / ( (flux - flux.mean())**2 ).sum() )
    rae  = np.abs( flux - flux_pred ).sum() / np.abs( flux - flux.mean() ).sum()
    mape = 100. / len(flux) * np.abs( 1. - flux_pred/flux ).sum()
    
    if (flux_err is not None) and (flux_err_pred is not None):
        
        nlpd  = nlpd_metric(flux, flux_pred, flux_err_pred)
        nrmse = nrmse_metric(flux, flux_err, flux_pred)
        picp  = picp_metric(flux, flux_pred, flux_err_pred, alpha)
    
        return [rmse, mae, rse, rae, mape, nlpd, nrmse, picp]
    
    else:
        return [rmse, mae, rse, rae, mape]