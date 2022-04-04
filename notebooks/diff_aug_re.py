import fulu
import pandas as pd
from fulu import single_layer_aug
from fulu import bnn_aug
from fulu import nf_aug
from fulu import mlp_reg_aug
from fulu import gp_aug
import numpy as np
from sklearn.model_selection import train_test_split
import aug_standart_ztf_re as aug_standart_ztf
import utils_re as utils
from importlib import reload
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
from aug_standart_ztf import bootstrap_estimate_mean_stddev
from sklearn.gaussian_process.kernels import RBF, Matern, \
RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel as C
import os
import time

np.random.seed(0)

diff_between_format_bts_and_mjd = 2458000 - 2400000.5
path = os.getcwd()
path_to = "{}/ZTF_BTS_23_29__22_09_2021.csv".format(os.getcwd())
color = {1: 'red', 0: 'green'}
passband2lam  = {0: 1, 1: 2} # green, red 
models_dict = {'BNN': bnn_aug.BayesianNetAugmentation(passband2lam),
               'NF': nf_aug.NormalizingFlowAugmentation(passband2lam),
               'NN (pytorch)': single_layer_aug.SingleLayerNetAugmentation(passband2lam),
               'NN (sklearn)': mlp_reg_aug.MLPRegressionAugmentation(passband2lam),
               'GP': gp_aug.GaussianProcessesAugmentation(passband2lam),
               'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()),
               'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()), 'GP with err': gp_aug.GaussianProcessesAugmentation(passband2lam, use_err = True), 'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(), use_err = True),
               'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel(), use_err = True)}

calculate_mjd = lambda x: x + diff_between_format_bts_and_mjd


bts = pd.read_csv(path_to, sep =',')
bts = bts.drop('Unnamed: 0', 1) 

df_all = pd.read_csv(path + '/ANTARES_NEW.csv')
#df_all = pd.read_csv(path + '/ANTARES_NEW_10_in_g_r_bands.csv')
df_all = df_all.drop('Unnamed: 0', 1)

#print("Названия колонок в таблице ANTARES.csv со всеми кривыми блеска: \n\n", df_all.columns, "\n\n")
print("Количество объектов: ", len(df_all['object_id'].unique()))

obj_names = df_all['object_id'].unique()
df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0

def aug(step = 1000, model_name = 'NN (pytorch)', N_OBS = 2000, TEST_SIZE = 0.5):
    model = models_dict[model_name]
    print(model_name)
#     for name, i in tqdm(zip(obj_names[::step], range(len(obj_names[::step])))):
#         # fit augmentation model
#         anobject = aug_standart_ztf.get_object(df_all, name)
#         anobject_train, anobject_test = train_test_split(anobject, test_size = TEST_SIZE, random_state=11)

#         model.fit(anobject_train['mjd'].values, anobject_train['flux'].values, 
#               anobject_train['flux_err'].values, anobject_train['passband'].values)

#         # predict flux for unseen observations
#         flux_pred, flux_err_pred = model.predict(anobject_test['mjd'].values, anobject_test['passband'].values)

#         # augmentation
#         t_aug, flux_aug, flux_err_aug, passbands_aug = model.augmentation(anobject['mjd'].min(), 
#                                                                       anobject['mjd'].max(), 
#                                                                       n_obs=N_OBS)
#         anobject_test_pred = anobject_test.copy()
#         anobject_test_pred['flux'], anobject_test_pred['flux_err'] = flux_pred, flux_err_pred

#         anobject_aug = aug_standart_ztf.compile_obj(t_aug, flux_aug, flux_err_aug, passbands_aug)

#     plt.show()    
    report = pd.DataFrame(columns=["ID", 'RMSE', 'MAE', 'RSE', 'RAE', 'MAPE', 'NLPD', 'NRMSEO', 'NRMSEP', 'PICP_68', 'PICP_95'])
    timest = []
    for name, i in tqdm(zip(df_all['object_id'].unique(), range(len(df_all['object_id'].unique())))):
        # fit augmentation model
        
        anobject = aug_standart_ztf.get_object(df_all, name)
        anobject_train, anobject_test = train_test_split(anobject, test_size = TEST_SIZE, random_state=11)
        
        start_time_moment =  time.time()
        
        model.fit(anobject_train['mjd'].values, anobject_train['flux'].values, 
              anobject_train['flux_err'].values, anobject_train['passband'].values)

        # predict flux for unseen observations
        flux_pred, flux_err_pred = model.predict(anobject_test['mjd'].values, anobject_test['passband'].values)

        # augmentation
        t_aug, flux_aug, flux_err_aug, passbands_aug = model.augmentation(anobject['mjd'].min(), 
                                                                      anobject['mjd'].max(), 
                                                                      n_obs=N_OBS)
        end_time_moment =  time.time()
        time_of_work = end_time_moment - start_time_moment
        timest.append(time_of_work)
        if name == 'ZTF20aahbamv':
            model.plot(title='ZTF20aahbamv {}'.format(model_name))
            plt.show()
        anobject_test_pred = anobject_test.copy()
        anobject_test_pred['flux'], anobject_test_pred['flux_err'] = flux_pred, flux_err_pred

        anobject_aug = aug_standart_ztf.compile_obj(t_aug, flux_aug, flux_err_aug, passbands_aug)

        metrics = utils.regression_quality_metrics_report(anobject_test['flux'].values,
                                                          flux_pred,
                                                          anobject_test['flux_err'].values,
                                                          flux_err_pred)
        report.loc[len(report), :] = [i] + list(metrics)
        
    print("Среднее время работы для 1-го объекта (fit, predict,augmentation): {}".format(np.nanmean(timest)))
    
    for i in tqdm(report.columns[1:].values):
        mean, std = bootstrap_estimate_mean_stddev(report[i].values)
        print('Среднее значение метрики ' + i + ' {}'.format(mean), 
              "+-", std)
