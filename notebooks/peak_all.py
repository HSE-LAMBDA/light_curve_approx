import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm
import utils
from utils import regression_quality_metrics_report
from sklearn.model_selection import train_test_split
import gp_aug as gp_aug_old
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import aug_standart_ztf
from gp_aug import add_log_lam
from gp_aug import create_aug_data
from gp_aug import bootstrap_estimate_mean_stddev
from gp_aug import wilcoxon_pvalue
from gp_aug import GaussianProcessesAugmentation
from sklearn.gaussian_process.kernels import RBF, Matern,\
RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel as C
import fulu
from fulu import single_layer_aug
from fulu import bnn_aug
from fulu import nf_aug
from fulu import mlp_reg_aug
from fulu import gp_aug
import warnings
warnings.filterwarnings('ignore')

N_OBS = 2000
path = '/home/masha/AstroAugumetation/'
path_to = "/home/masha/AstroAugumetation/ZTF_BTS_02_02__02_04_2021.csv"


bts = pd.read_csv(path_to, sep =',')
bts = bts.drop('Unnamed: 0', 1) 

df_all = pd.read_csv(path + 'ANTARES.csv')
df_all = df_all.drop('Unnamed: 0', 1)

color = {1: 'red', 0: 'green'}
passband2lam  = {0: 1, 1: 2} # green, red 
models_dict = {'BNN': bnn_aug.BayesianNetAugmentation(passband2lam), 'NF': nf_aug.NormalizingFlowAugmentation(passband2lam), 'NN (pytorch)': single_layer_aug.SingleLayerNetAugmentation(passband2lam), 'NN (sklearn)': mlp_reg_aug.MLPRegressionAugmentation(passband2lam), 'GP': gp_aug.GaussianProcessesAugmentation(passband2lam), 'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()': gp_aug_old.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(),  False), 'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug_old.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel())}
cool_objects = ['ZTF20aahbamv', 'ZTF19abirmkt']

def peak_all_d(model_name = 'NN (sklearn)', n_obs = N_OBS, TEST_SIZE = 0.5):
    model = models_dict[model_name]
    def estimate_curve_peak_aug(name, n_obs=n_obs):
        #anobj_test, anobj_test_pred, anobj_aug, flux_predd, anobj_train = augum_gp(anobject, kernel, n_obs=n_obs)

        #всё что есть в одну дату - складывается
        anobject = aug_standart_ztf.get_object(df_all, name)
        anobject_train, anobject_test = train_test_split(anobject, test_size = TEST_SIZE, random_state=11)

        model.fit(anobject_train['mjd'].values, anobject_train['flux'].values, 
                  anobject_train['flux_err'].values, anobject_train['passband'].values)

        # predict flux for unseen observations
        flux_pred, flux_err_pred = model.predict(anobject_test['mjd'].values, anobject_test['passband'].values)

        # augmentation
        t_aug, flux_aug, flux_err_aug, passbands_aug = model.augmentation(anobject['mjd'].min(), 
                                                                          anobject['mjd'].max(), 
                                                                          n_obs=n_obs)
        anobject_test_pred = anobject_test.copy()
        anobject_test_pred['flux'], anobject_test_pred['flux_err'] = flux_pred, flux_err_pred

        anobject_aug = aug_standart_ztf.compile_obj(t_aug, flux_aug, flux_err_aug, passbands_aug)

        curve = anobject_aug[['mjd', 'flux']].groupby('mjd', as_index=False).sum()
        return curve['mjd'][curve['flux'].argmax()]

    def estimate_curve_peak_obs(name, n_obs=n_obs):

        anobject = aug_standart_ztf.get_object(df_all, name)
        anobject_train, anobject_test = train_test_split(anobject, test_size = TEST_SIZE, random_state=11)

        model.fit(anobject_train['mjd'].values, anobject_train['flux'].values, 
                  anobject_train['flux_err'].values, anobject_train['passband'].values)

        # predict flux for unseen observations
        flux_pred, flux_err_pred = model.predict(anobject_test['mjd'].values, anobject_test['passband'].values)

        # augmentation
        t_aug, flux_aug, flux_err_aug, passbands_aug = model.augmentation(anobject['mjd'].min(), 
                                                                          anobject['mjd'].max(), 
                                                                          n_obs=n_obs)
        anobject_test_pred = anobject_test.copy()
        anobject_test_pred['flux'], anobject_test_pred['flux_err'] = flux_pred, flux_err_pred

        anobject_aug = aug_standart_ztf.compile_obj(t_aug, flux_aug, flux_err_aug, passbands_aug)

        curve = anobject[['mjd', 'flux']].groupby('mjd', as_index=False).sum()

        return curve['mjd'][curve['flux'].argmax()]

    def residuals_histogram(all_objects):
        plt.figure(figsize=(10, 7))
        plt.hist(all_objects['peak_time'].values - all_objects['pred_peakmjd'].values, bins=50)
        plt.xlabel('mjd residuals', fontsize=15)
        plt.show()


    def plot_light_curves_with_peak(name, true_peak_mjd=None, title="", n_obs=N_OBS, save=None):
        #anobj_test, anobj_test_pred, anobj_aug, flux_predd, anobj_train = augum_gp(anobject, kernel, n_obs=n_obs)
        #pred_peak_mjd = GP_estimate_curve_peak(anobject, kernel, n_obs=n_obs)

        anobject = aug_standart_ztf.get_object(df_all, name)

        anobject_train, anobject_test = train_test_split(anobject, test_size = TEST_SIZE, random_state=11)

        model.fit(anobject_train['mjd'].values, anobject_train['flux'].values, 
                  anobject_train['flux_err'].values, anobject_train['passband'].values)

        # predict flux for unseen observations
        flux_pred, flux_err_pred = model.predict(anobject_test['mjd'].values, anobject_test['passband'].values)

        # augmentation
        t_aug, flux_aug, flux_err_aug, passbands_aug = model.augmentation(anobject['mjd'].min(), 
                                                                          anobject['mjd'].max(), 
                                                                          n_obs=N_OBS)
        anobject_test_pred = anobject_test.copy()
        anobject_test_pred['flux'], anobject_test_pred['flux_err'] = flux_pred, flux_err_pred

        anobject_aug = aug_standart_ztf.compile_obj(t_aug, flux_aug, flux_err_aug, passbands_aug)


        fig = plt.figure(figsize=(35,35), dpi = 150)
        plt.rcParams.update({'font.size': 30})
        fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.15)
        fig.suptitle(model_name)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)

        ax1.spines['bottom'].set_linewidth(3)
        ax1.spines['top'].set_linewidth(3)
        ax1.spines['left'].set_linewidth(3)
        ax1.spines['right'].set_linewidth(3)

        ax2.spines['bottom'].set_linewidth(3)
        ax2.spines['top'].set_linewidth(3)
        ax2.spines['left'].set_linewidth(3)
        ax2.spines['right'].set_linewidth(3)

        aug_standart_ztf.plot_light_curves_ax_band(anobject_test, anobject_train, anobject_aug, ax1, ax2, name)

        curve = anobject_aug[['mjd', 'flux']].groupby('mjd', as_index=False).sum()
        pred_peak_mjd = curve['mjd'][curve['flux'].argmax()]
        ax1.plot(curve['mjd'].values, curve['flux'].values, label='sum', linewidth=5.5, color='pink')
        ax2.plot(curve['mjd'].values, curve['flux'].values, label='sum', linewidth=5.5, color='pink')
        #plt.xlabel('Modified Julian Date', size=14)
        #plt.xticks(size=14)
        #plt.ylabel('Flux', size=14)
        #plt.yticks(size=14)

        ax1.axvline(pred_peak_mjd, label='pred peak', color='red', linestyle = '--', linewidth=5.5)
        ax2.axvline(pred_peak_mjd, label='pred peak', color='red', linestyle = '--', linewidth=5.5)
        
        if true_peak_mjd is not None:
            ax1.axvline(true_peak_mjd, label='true peak', color='black', linewidth=5.5)
            ax2.axvline(true_peak_mjd, label='true peak', color='black', linewidth=5.5)
            
        ax1.legend(loc='best', ncol=3, fontsize=30)
        ax2.legend(loc='best', ncol=3, fontsize=30)
        #plt.title(title, size=14)

        if save is not None:
            plt.savefig(save, format='pdf')

        plt.show()

#     metadata = pd.DataFrame()
#     metadata.insert(0, 'object_id', df_all['object_id'])
#     metadata = metadata.drop_duplicates()

#     metadata['pred_peakmjd'] = metadata['object_id'].apply(lambda name: estimate_curve_peak_aug(name))
    #metadata['peak_time'] = metadata['object_id'].apply(lambda name: estimate_curve_peak_obs(name))

#     [rmse, mae, rse, rae, mape] = utils.regression_quality_metrics_report(metadata['peak_time'].values, 
#                                                                           metadata['pred_peakmjd'].values)
#     print("RMSE: ", rmse, "+-", bootstrap_estimate_mean_stddev([rmse]))
#     print("MAE: ", mae, "+-", bootstrap_estimate_mean_stddev([mae]))
#     print("RSE: ", rse, "+-", bootstrap_estimate_mean_stddev([rse]))
#     print("RAE: ", rae, "+-", bootstrap_estimate_mean_stddev([rae]))
#     print("MAPE: ", mape, "+-", bootstrap_estimate_mean_stddev([mape]))

#     residuals_histogram(metadata)

    for object_id in tqdm(cool_objects):
        plot_light_curves_with_peak(object_id, save=model_name + "_" + object_id) #,
#                                     metadata[metadata['object_id']==object_id]['peak_time'].iloc[0])