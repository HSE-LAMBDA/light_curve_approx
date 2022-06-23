import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')
import utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm.notebook import tqdm
#from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.gaussian_process.kernels import RBF, Matern, \
RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel as C
from importlib import reload
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import fulu
from fulu import single_layer_aug
from fulu import bnn_aug
from fulu import nf_aug
from fulu import mlp_reg_aug
from fulu import gp_aug
#import gp_aug as gp_aug_old
#from gp_aug import bootstrap_estimate_mean_stddev
from copy import deepcopy
import os
from joblib import Parallel, delayed
from  sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_curve, auc, recall_score, precision_score
from sklearn.utils import resample


passband2lam  = {0: 1, 1: 2} # green, red 
color = {1: 'red', 0: 'green'}
params_mlp = {'passband2lam': passband2lam, 
                              'hidden_layer_sizes': (20, 10),
                              'solver': 'lbfgs',
                              'activation': 'tanh',
                              'learning_rate_init': 0.03171745,
                              'max_iter': 90, 
                              'batch_size': 1, 
                              'weight_decay': 0.03109669}
params_nf = {'passband2lam': passband2lam, 
                          'batch_size': 500, 
                          'n_epochs': 3000, 
                          'lr': 0.00500001, 
                          'device': 'cpu', 
                          'weight_decay': 0.}

params_bnn = {'passband2lam': passband2lam, 
                          'n_hidden': 20, 
                          'prior_sigma': 9.99997363e-02, 
                          'n_epochs': 3000, 
                          'lr': 1.00030572e-02, 
                          'kl_weight': 1.01967102e-04, 
                          'optimizer': 'Adam', 
                          'device': 'cpu', 
                          'weight_decay': 3.32160019e-06}

params_mlp_p = {'passband2lam': passband2lam, 
                      'n_hidden': 20, 
                      'activation': 'tanh', 
                      'n_epochs': 1000, 
                      'batch_size': 500, 
                      'lr': 0.00996565, 
                      'optimizer': 'Adam', 
                      'device': 'auto', 
                      'weight_decay': 0.00034349}

# models_dict = {'BNN': bnn_aug.BayesianNetAugmentation(passband2lam), \
#                'NF': nf_aug.NormalizingFlowAugmentation(passband2lam),\
#                'NN (pytorch)': single_layer_aug.SingleLayerNetAugmentation(passband2lam),\
#                'NN (sklearn)': mlp_reg_aug.MLPRegressionAugmentation(passband2lam),\
#                'GP': gp_aug.GaussianProcessesAugmentation(passband2lam),\
# 'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()':gp_aug_old.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(),  False),\
#                'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug_old.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel())}
models_dict = {'BNN': bnn_aug.BayesianNetAugmentation(**params_bnn),
               'NF': nf_aug.NormalizingFlowAugmentation(**params_nf),
               'NN (pytorch)': single_layer_aug.SingleLayerNetAugmentation(**params_mlp_p),
               'NN (sklearn)': mlp_reg_aug.MLPRegressionAugmentation(**params_mlp),
               'GP': gp_aug.GaussianProcessesAugmentation(passband2lam),
               'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()),
               'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()),
               'GP C(1.0) * Matern([1, 1]) + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0) * Matern([1, 1]) + WhiteKernel()),
               'GP with err': gp_aug.GaussianProcessesAugmentation(passband2lam, use_err = True),
               'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(), use_err = True),
               'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel(), use_err = True),
              'GP C(1.0) * Matern([1, 1]) + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0) * Matern([1, 1]) + WhiteKernel(), use_err = True)}
#                'GP': gp_aug.GaussianProcessesAugmentation(passband2lam),
#                'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()),
#                'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()), 'GP with err': gp_aug.GaussianProcessesAugmentation(passband2lam, use_err = True), 'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(), use_err = True),
#                'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel() with err': gp_aug.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel(), use_err = True)}

diff_between_format_bts_and_mjd = 2458000 - 2400000.5
N_OBS = 256#128
N_EPOCHES = 300


calculate_mjd = lambda x: x + diff_between_format_bts_and_mjd


path = os.getcwd()
path_to = "{}/ZTF_BTS_23_29__22_09_2021.csv".format(os.getcwd())

bts = pd.read_csv(path_to, sep =',')
bts = bts.drop('Unnamed: 0', 1) 

df_all = pd.read_csv(path + '/ANTARES_NEW.csv')
#df_all = pd.read_csv(path + 'ANTARES_10_in_g_r_bands.csv')
df_all = df_all.drop('Unnamed: 0', 1)

print("Названия колонок в таблице ANTARES_NEW.csv со всеми кривыми блеска: \n\n", df_all.columns, "\n\n")
print("Количество объектов: ", len(df_all['object_id'].unique()))

obj_names = df_all['object_id'].unique()


# df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
# df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0
df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-91T', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-pec', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Iax', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-91bg', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-CSM', 'obj_type'] = 1
df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0


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

def get_images(outputs, silent=False):
    images = []
    labels = []
    for img in outputs:
        if img is not None:
            labels.append(img[0])
            images.append(img[1]) # завернуть в лист если хочу другую размерность 4 dim
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def aug_one(model, name):
    good_id = name
    anobject = get_object(df_all, good_id)

    model.fit(anobject['mjd'].values, anobject['flux'].values, anobject['flux_err'].values, anobject['passband'].values)

    t_aug, flux_aug, flux_err_aug, passband_aug = model.augmentation(anobject['mjd'].min(), 
                                                                 anobject['mjd'].max(), n_obs=N_OBS)

    anobject_aug = compile_obj(t_aug, flux_aug, flux_err_aug, passband_aug)


    flux_aug = anobject_aug['flux'].values

    data_array = flux_aug.reshape((2, N_OBS))#.T
    #all_data.append([data_array])

    # add target value for this curve
#     if np.isin(anobject['obj_type'].values, 0).all():
#         all_target_classes.append(0)
#     elif np.isin(anobject['obj_type'].values, 1).all():
#         all_target_classes.append(1)
    true_class = int(anobject['obj_type'].to_numpy()[0])
    return (true_class, data_array) #all_data, all_target_classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
                                    nn.Conv1d(2, 8, 3, padding=1),
                                    nn.LayerNorm((8, 256)),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2),
                                    nn.Conv1d(8, 16, 3, padding=1),
                                    nn.LayerNorm((16, 128)),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2),
                                    nn.Conv1d(16, 32, 3, padding=1),
                                    nn.LayerNorm((32, 64)),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2),
                                    nn.Flatten(),
                                    nn.Dropout(0.33),
                                    nn.Linear(16 * 64, 1),
                                    nn.Sigmoid()
                                )

    def forward(self, x):
        x = self.cnn(x)
        return x

def gen_report(y_test, y_test_pred, n_iters=1000, decimals=3):
    
    metrics = []
    inds = np.arange(len(y_test))
    for i in range(n_iters):
        inds_boot = resample(inds)
        roc_auc = roc_auc_score(y_test[inds_boot], y_test_pred[inds_boot])
        logloss = log_loss(y_test[inds_boot], y_test_pred[inds_boot], eps=10**-6)
        accuracy = accuracy_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        precision, recall, _ = precision_recall_curve(y_test[inds_boot], y_test_pred[inds_boot])
        pr_auc = auc(recall, precision)
        recall = recall_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        precision = precision_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        metrics.append([roc_auc, pr_auc, logloss, accuracy, recall, precision])
    metrics = np.array(metrics)
    report = pd.DataFrame(columns=["ROC_AUC", 'PR-AUC', 'LogLoss', 'Accuracy', 'Recall', 'Precision'], 
                          data=[metrics.mean(axis=0), metrics.std(axis=0)], 
                          index=['mean', 'std'])
    return report

def classification(model_name = 'GP', n_obs = N_OBS, n_epoches = N_EPOCHES):
    all_data = []
    all_target_classes = []

    model = models_dict[model_name]
    print(model_name)

    outputs = Parallel(n_jobs=-1)(delayed(aug_one)(model, name) for name in df_all['object_id'].unique())
    all_data, all_target_classes = get_images(outputs)
    
    all_data = np.array(all_data)
    all_target_classes = np.array(all_target_classes)
    print(all_data.shape, all_target_classes.shape)


    # train / test split data
    X_train, X_test_val, y_train, y_test_val = train_test_split(all_data, 
                                                            all_target_classes,
                                                            test_size=0.4,
                                                            random_state=11)

    X_val, X_test, y_val, y_test = train_test_split(X_test_val, 
                                                            y_test_val,
                                                            test_size=0.5,
                                                            random_state=11)
    # normalize input data
    X_train_norm = np.array((X_train - X_train.mean()) / X_train.std(), dtype=np.float32)
    X_test_norm = np.array((X_test - X_train.mean()) / X_train.std(), dtype=np.float32)
    X_val_norm = np.array((X_val - X_train.mean()) / X_train.std(), dtype=np.float32)

    # convert train data to tensors
    X_train_tensor = torch.from_numpy(X_train_norm)
    y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32))

    # create train data loader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                                  shuffle=True)#, num_workers=2)

    # convert test data to tensors
    X_test_tensor = torch.from_numpy(X_test_norm)
    y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32))

    # create test data loader
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                 shuffle=False)#, num_workers=2)

    # convert val data to tensors
    X_val_tensor = torch.from_numpy(X_val_norm)
    y_val_tensor = torch.from_numpy(np.array(y_val, dtype=np.float32))

    # create val data loader
    val_data = TensorDataset(X_val_tensor, y_val_tensor)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                                 shuffle=False)#, num_workers=2)
    
    #print(X_train_tensor.size())
    net = Net()
    criterion = nn.BCELoss()#reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.0002, weight_decay=0.001)#optim.SGD(net.parameters(), lr=0.001)#, momentum=0.8)
    epochs = np.arange(n_epoches)

    best_loss_val = float('inf')
    best_state_on_val = None

    for epoch in tqdm(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        net.train()
        for info in trainloader:
            # get the inputs; info is a list of [inputs, labels]
            inputs, labels = info

            # zero the parameter gradients
            for param in net.parameters():
                param.grad = None

            # forward + backward + optimize
            #print(inputs.size())
            #print(net(inputs).size())
            outputs = net(inputs).reshape(1)#(61)
            #print(outputs.size())
            #print(labels.size())
            loss = criterion(outputs, labels)
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        # print mean loss for the epoch
        cur_loss = epoch_loss / X_train_norm.shape[0]
        plt.plot(epoch, cur_loss, '.', color='red')
        if (epoch + 1) % 10 == 0:
            print('[%5d] error: %.3f' % (epoch + 1, cur_loss))

        net.eval()
        epoch_loss_val = 0.0
        for info in valloader:
            # get the inputs; info is a list of [inputs, labels]
            inputs, labels = info

            # forward
            outputs = net(inputs).reshape(1)
            loss = criterion(outputs, labels)

            epoch_loss_val += loss.item()

        cur_loss_val = epoch_loss_val / X_val_norm.shape[0]
        plt.plot(epoch, cur_loss_val, '.', color='blue')

        if epoch_loss_val <= best_loss_val:
            best_loss_val = epoch_loss_val
            best_state_on_val = deepcopy(net.state_dict())

    plt.legend(['train_loss', 'val_loss'])        
    plt.show()

    net.load_state_dict(best_state_on_val)
    print('Finished Training')
    
    y_test_pred = net(X_test_tensor).detach().numpy()[:, 0]
    
    report = gen_report(y_test, y_test_pred)
    print(report)
#     # check models' accuracy
#     test_accuracy = get_model_accuracy(net, testloader)
#     train_accuracy = get_model_accuracy(net, trainloader)

#     print('Test accuracy of the network on the %d test images: %.4f' % 
#           (X_test_norm.shape[0], test_accuracy))
#     print('Train accuracy of the network on the %d train images: %.4f' % 
#           (X_train_norm.shape[0], train_accuracy))
#     y_test = []
#     y_probs = []
#     y_probs_0 = []
#     y_probs_1 = []

#     with torch.no_grad():
#         for test_info in testloader:
#             images, test_labels = test_info
#             test_outputs = net(images)

#             # get output value
#             prob = test_outputs.item()

#             # check true target valur    
#             true_class = int(test_labels.item())

#             # compare output to threshold
#             if true_class == 0:
#                 y_probs_0.append(prob)
#             else:
#                 y_probs_1.append(prob)

#             # get predicted target value
#             y_test.append(true_class)
#             y_probs.append(prob)

#     y_test = np.array(y_test)
#     y_probs = np.array(y_probs)

#     assert np.array(y_probs).min() >= 0
#     assert np.array(y_probs).max() <= 1

#     N = len(y_probs)

#     # sample predicted values
#     sample_coeffs = np.random.randint(0, N, (10000, 1000))
#     sample_prob = y_probs[sample_coeffs]
#     sample_test = y_test[sample_coeffs]
#     sample_pred = sample_prob > 0.5

#     assert len(sample_test) == len(sample_prob)
#     assert len(sample_prob) == len(sample_pred)
#     T = len(sample_test)

#     # calculated mean accuracy
#     accuracy = [(sample_pred[i] == sample_test[i]).mean() for i in range(T)]
#     y_pred = np.array(y_probs) > 0.5
#     print("LogLoss = %.4f" % log_loss(y_test, y_pred))

#     # calculate mean log loss
#     logloss = [log_loss(sample_test[i], sample_pred[i]) for i in range(T)]
#     # compare distibution of output values

#     rc('xtick', labelsize=16)
#     rc('ytick', labelsize=16)

#     fig = plt.figure(dpi=80, figsize=(13, 5.5))

#     bins_number = 22
#     step = 1 / bins_number
#     hist_0, _ = np.histogram(y_probs_0, bins=bins_number, range=(0.0, 1.0))
#     hist_1, _ = np.histogram(y_probs_1, bins=bins_number, range=(0.0, 1.0))
#     x = np.arange(0 + step/2, 1, step)

#     ax1 = fig.add_subplot(121)
#     ax1.yaxis.tick_right()
#     plt.title("Распределение P(X in 1) для X in 0", fontsize=21, pad=7)

#     ax1.bar(x, hist_0, color="xkcd:magenta", width=0.038, log=True)
#     ax1.minorticks_on()
#     ax1.tick_params('y', length=10, width=1, which='major')
#     ax1.tick_params('y', length=5, width=1, which='minor')
#     ax1.tick_params('x', length=7, width=1, which='major')
#     ax1.tick_params('x', length=0, width=1, which='minor')
#     plt.xlim((0.0, 1.0))
#     plt.ylim((ax1.get_ylim()[0], int(ax1.get_ylim()[1]) // 100 * 100))
#     plt.grid(True)

#     ax2 = fig.add_subplot(122)
#     ax2.yaxis.tick_right()
#     plt.title("Распределение P(X in 1) для X in 1", fontsize=21, pad=7)

#     ax2.bar(x, hist_1, color="xkcd:cornflower", width=0.038, log=True)
#     ax2.minorticks_on()
#     ax2.tick_params('y', length=10, width=1, which='major')
#     ax2.tick_params('y', length=5, width=1, which='minor')
#     ax2.tick_params('x', length=7, width=1, which='major')
#     ax2.tick_params('x', length=0, width=1, which='minor')
#     plt.xlim((0.0, 1.0))
#     plt.ylim(ax1.get_ylim())
#     plt.grid(True)

#     plt.show()

    # compare distibution of output values

#     rc('xtick', labelsize=14)
#     rc('ytick', labelsize=14)

#     fig = plt.figure(dpi=80, figsize=(10, 6))
#     ax = fig.add_subplot(111)
#     ax.yaxis.tick_right()
#     plt.title("Распределение выходных значений классификатора", fontsize=18, pad=7)
#     plt.xlabel("Вероятность принадлежности объекта первому классу", fontsize=15)
#     plt.ylabel("Количество объектов", fontsize=15)

#     bins_number = 40
#     step = 1 / bins_number
#     hist_0, _ = np.histogram(y_probs_0, bins=bins_number, range=(0.0, 1.0))
#     hist_1, _ = np.histogram(y_probs_1, bins=bins_number, range=(0.0, 1.0))
#     x = np.arange(0 + step/2, 1, step)

#     plt.bar(x, hist_0, color='r', width=0.02, log=True, label="нулевой класс", alpha=0.3)
#     plt.bar(x, hist_1, color='b', width=0.02, log=True, label="первый класс", alpha=0.3)

#     ax.minorticks_on()
#     ax.tick_params('y', length=8, width=1, which='major')
#     ax.tick_params('y', length=4, width=1, which='minor')
#     ax.tick_params('x', length=7, width=1, which='major')
#     ax.tick_params('x', length=4, width=1, which='minor')

#     plt.xlim((0.0, 1.0))
#     plt.grid(True)
#     plt.legend(frameon=True, loc=(0.67, 0.81), fontsize=16, shadow=0.1)

#     plt.show()

#     rc('xtick', labelsize=16)
#     rc('ytick', labelsize=16)

#     print("Test ROC-AUC: %.4f, test PR-AUC: %.4f" % (roc_auc_score(y_test, y_probs), 
#                                                          average_precision_score(y_test, y_probs)))

#     # calculate mean AUC-ROC & AUC-PR
#     auc_roc = [roc_auc_score(sample_test[i], sample_prob[i]) for i in range(T)]
#     auc_pr = [average_precision_score(sample_test[i], sample_prob[i]) for i in range(T)]

#     precision, recall, _ = precision_recall_curve(y_test, y_probs)

#     fig = plt.figure(dpi=80, figsize=(13, 5.5))
#     ax = fig.add_subplot(121)
#     plt.grid(True)
#     ax.set_title("PR-кривая", fontsize=21, pad=7)
#     ax.fill_between(recall, precision, alpha=0.6, color="xkcd:apple green", lw=2)
#     ax.set_xlim(0, 1.)
#     ax.set_ylim(0, 1.02)

#     fpr, tpr, thresholds = roc_curve(y_test, y_probs)

#     ax = fig.add_subplot(122)
#     plt.grid(True)
#     ax.set_title("ROC-кривая", fontsize=21, pad=7)
#     ax.fill_between(fpr, tpr, alpha=0.6, color="xkcd:goldenrod", lw=2)
#     ax.set_xlim(0, 1.)
#     ax.set_ylim(0, 1.02)

#     plt.show()
    
#     mean_logloss, std_logloss = bootstrap_estimate_mean_stddev(logloss)
#     mean_accuracy, std_accuracy = bootstrap_estimate_mean_stddev(accuracy)
#     mean_auc_roc, std_auc_roc = bootstrap_estimate_mean_stddev(auc_roc)
#     mean_auc_pr, std_auc_pr = bootstrap_estimate_mean_stddev(auc_pr)
    
    
#     print("LogLoss:  mean = %.4f, std = %.4f" % (mean_logloss, std_logloss))
#     print("Accuracy: mean = %.4f, std = %.4f" % (mean_accuracy, std_accuracy))
#     print("AUC-ROC:  mean = %.4f, std = %.4f" % (mean_auc_roc, std_auc_roc))
#     print("AUC-PR:   mean = %.4f, std = %.4f" % (mean_auc_pr, std_auc_pr))   