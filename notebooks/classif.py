import numpy as np
import pandas as pd
import sklearn
import warnings
warnings.filterwarnings('ignore')
import aug_standart_ztf
from aug_standart_ztf import compile_obj
import utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm.notebook import tqdm
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
import gp_aug as gp_aug_old
from gp_aug import bootstrap_estimate_mean_stddev


passband2lam  = {0: 1, 1: 2} # green, red 
color = {1: 'red', 0: 'green'}
models_dict = {'BNN': bnn_aug.BayesianNetAugmentation(passband2lam), \
               'NF': nf_aug.NormalizingFlowAugmentation(passband2lam),\
               'NN (pytorch)': single_layer_aug.SingleLayerNetAugmentation(passband2lam),\
               'NN (sklearn)': mlp_reg_aug.MLPRegressionAugmentation(passband2lam),\
               'GP': gp_aug.GaussianProcessesAugmentation(passband2lam),\
'GP C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel()':gp_aug_old.GaussianProcessesAugmentation(passband2lam, C(1.0)*RBF([1.0, 1.0]) + Matern() + WhiteKernel(),  False),\
               'GP C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel()': gp_aug_old.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel())}

diff_between_format_bts_and_mjd = 2458000 - 2400000.5
N_OBS = 50
N_EPOCHES = 600#400


calculate_mjd = lambda x: x + diff_between_format_bts_and_mjd


path = '/home/masha/AstroAugumetation/'
path_to = "/home/masha/AstroAugumetation/ZTF_BTS_02_02__02_04_2021.csv"

bts = pd.read_csv(path_to, sep =',')
bts = bts.drop('Unnamed: 0', 1) 

#df_all = pd.read_csv(path + 'ANTARES.csv')
df_all = pd.read_csv(path + 'ANTARES_10_in_g_r_bands.csv')
df_all = df_all.drop('Unnamed: 0', 1)

print("Названия колонок в таблице ANTARES.csv со всеми кривыми блеска: \n\n", df_all.columns, "\n\n")
print("Количество объектов: ", len(df_all['object_id'].unique()))

obj_names = df_all['object_id'].unique()


df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0


def classification(model_name = 'GP', n_obs = N_OBS, n_epoches = N_EPOCHES):
    all_data = []
    all_target_classes = []

    model = models_dict[model_name]
    print(model)
    for good_id in tqdm(obj_names):
        anobject = aug_standart_ztf.get_object(df_all, good_id)

        anobject_train, anobject_test = train_test_split(anobject, test_size=0.5, random_state=11)
        

        model.fit(anobject_train['mjd'].values, anobject_train['flux'].values, anobject_train['flux_err'].values, anobject_train['passband'].values)

    # predict flux for unseen observations
        flux_pred, flux_err_pred = model.predict(anobject_test['mjd'].values, anobject_test.passband)#, copy=True)

    # augmentation
        t_aug, flux_aug, flux_err_aug, passband_aug = model.augmentation(anobject['mjd'].min(), 
                                                                     anobject['mjd'].max(), n_obs=n_obs)

        anobject_test_pred = compile_obj(anobject_test['mjd'].values, flux_pred, 
                                      flux_err_pred, anobject_test['passband'].values)
        anobject_aug = compile_obj(t_aug, flux_aug, flux_err_aug, passband_aug)


        flux_aug = anobject_aug['flux'].values

        data_array = flux_aug.reshape((2, n_obs)).T
        all_data.append([data_array])

        # add target value for this curve
        if np.isin(anobject['obj_type'].values, 0).all():
            all_target_classes.append(0)
        elif np.isin(anobject['obj_type'].values, 1).all():
            all_target_classes.append(1)

    all_data = np.array(all_data)
    all_target_classes = np.array(all_target_classes)
    print(all_data.shape, all_target_classes.shape)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(1, 4, kernel_size=(15, 2)) 
            self.conv2 = nn.Conv2d(4, 8, kernel_size=(10, 1))
            self.pool = nn.MaxPool2d(1,4)
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(56,64)
            self.fc2 = nn.Linear(64,1)

        def forward(self, x):

            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))

            x = self.pool(x)
            x = self.dropout(x)

            x = x.view(-1,56)

            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x

    def get_model_accuracy(net, data_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for test_info in data_loader:
                images, test_labels = test_info
                test_outputs = net(images)
                prob = test_outputs.item()
                if prob > 0.5:
                    predicted = 1
                else:
                    predicted = 0
                total += test_labels.size(0)
                correct += (predicted == test_labels.item())

        return correct / total


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
                                                  shuffle=True, num_workers=2)

    # convert test data to tensors
    X_test_tensor = torch.from_numpy(X_test_norm)
    y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32))

    # create test data loader
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                 shuffle=False, num_workers=2)

    # convert val data to tensors
    X_val_tensor = torch.from_numpy(X_val_norm)
    y_val_tensor = torch.from_numpy(np.array(y_val, dtype=np.float32))

    # create val data loader
    val_data = TensorDataset(X_val_tensor, y_val_tensor)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                                 shuffle=False, num_workers=2)

    net = Net()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)#, momentum=0.8)
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
            outputs = net(inputs).reshape(1)
            loss = criterion(outputs, labels)
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
            best_state_val = net.state_dict()

    plt.legend(['train_loss', 'val_loss'])        
    plt.show()

    net.load_state_dict(best_state_val)
    print('Finished Training')

    # check models' accuracy
    test_accuracy = get_model_accuracy(net, testloader)
    train_accuracy = get_model_accuracy(net, trainloader)

    print('Test accuracy of the network on the %d test images: %.4f' % 
          (X_test_norm.shape[0], test_accuracy))
    print('Train accuracy of the network on the %d train images: %.4f' % 
          (X_train_norm.shape[0], train_accuracy))
    y_test = []
    y_probs = []
    y_probs_0 = []
    y_probs_1 = []

    with torch.no_grad():
        for test_info in testloader:
            images, test_labels = test_info
            test_outputs = net(images)

            # get output value
            prob = test_outputs.item()

            # check true target valur    
            true_class = int(test_labels.item())

            # compare output to threshold
            if true_class == 0:
                y_probs_0.append(prob)
            else:
                y_probs_1.append(prob)

            # get predicted target value
            y_test.append(true_class)
            y_probs.append(prob)

    y_test = np.array(y_test)
    y_probs = np.array(y_probs)

    assert np.array(y_probs).min() >= 0
    assert np.array(y_probs).max() <= 1

    N = len(y_probs)

    # sample predicted values
    sample_coeffs = np.random.randint(0, N, (10000, 1000))
    sample_prob = y_probs[sample_coeffs]
    sample_test = y_test[sample_coeffs]
    sample_pred = sample_prob > 0.5

    assert len(sample_test) == len(sample_prob)
    assert len(sample_prob) == len(sample_pred)
    T = len(sample_test)

    # calculated mean accuracy
    accuracy = [(sample_pred[i] == sample_test[i]).mean() for i in range(T)]
    y_pred = np.array(y_probs) > 0.5
    print("LogLoss = %.4f" % log_loss(y_test, y_pred))

    # calculate mean log loss
    logloss = [log_loss(sample_test[i], sample_pred[i]) for i in range(T)]
    # compare distibution of output values

    rc('xtick', labelsize=16)
    rc('ytick', labelsize=16)

    fig = plt.figure(dpi=80, figsize=(13, 5.5))

    bins_number = 22
    step = 1 / bins_number
    hist_0, _ = np.histogram(y_probs_0, bins=bins_number, range=(0.0, 1.0))
    hist_1, _ = np.histogram(y_probs_1, bins=bins_number, range=(0.0, 1.0))
    x = np.arange(0 + step/2, 1, step)

    ax1 = fig.add_subplot(121)
    ax1.yaxis.tick_right()
    plt.title("Распределение P(X in 1) для X in 0", fontsize=21, pad=7)

    ax1.bar(x, hist_0, color="xkcd:magenta", width=0.038, log=True)
    ax1.minorticks_on()
    ax1.tick_params('y', length=10, width=1, which='major')
    ax1.tick_params('y', length=5, width=1, which='minor')
    ax1.tick_params('x', length=7, width=1, which='major')
    ax1.tick_params('x', length=0, width=1, which='minor')
    plt.xlim((0.0, 1.0))
    plt.ylim((ax1.get_ylim()[0], int(ax1.get_ylim()[1]) // 100 * 100))
    plt.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.yaxis.tick_right()
    plt.title("Распределение P(X in 1) для X in 1", fontsize=21, pad=7)

    ax2.bar(x, hist_1, color="xkcd:cornflower", width=0.038, log=True)
    ax2.minorticks_on()
    ax2.tick_params('y', length=10, width=1, which='major')
    ax2.tick_params('y', length=5, width=1, which='minor')
    ax2.tick_params('x', length=7, width=1, which='major')
    ax2.tick_params('x', length=0, width=1, which='minor')
    plt.xlim((0.0, 1.0))
    plt.ylim(ax1.get_ylim())
    plt.grid(True)

    plt.show()

    # compare distibution of output values

    rc('xtick', labelsize=14)
    rc('ytick', labelsize=14)

    fig = plt.figure(dpi=80, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    plt.title("Распределение выходных значений классификатора", fontsize=18, pad=7)
    plt.xlabel("Вероятность принадлежности объекта первому классу", fontsize=15)
    plt.ylabel("Количество объектов", fontsize=15)

    bins_number = 40
    step = 1 / bins_number
    hist_0, _ = np.histogram(y_probs_0, bins=bins_number, range=(0.0, 1.0))
    hist_1, _ = np.histogram(y_probs_1, bins=bins_number, range=(0.0, 1.0))
    x = np.arange(0 + step/2, 1, step)

    plt.bar(x, hist_0, color='r', width=0.02, log=True, label="нулевой класс", alpha=0.3)
    plt.bar(x, hist_1, color='b', width=0.02, log=True, label="первый класс", alpha=0.3)

    ax.minorticks_on()
    ax.tick_params('y', length=8, width=1, which='major')
    ax.tick_params('y', length=4, width=1, which='minor')
    ax.tick_params('x', length=7, width=1, which='major')
    ax.tick_params('x', length=4, width=1, which='minor')

    plt.xlim((0.0, 1.0))
    plt.grid(True)
    plt.legend(frameon=True, loc=(0.67, 0.81), fontsize=16, shadow=0.1)

    plt.show()

    rc('xtick', labelsize=16)
    rc('ytick', labelsize=16)

    print("Test ROC-AUC: %.4f, test PR-AUC: %.4f" % (roc_auc_score(y_test, y_probs), 
                                                         average_precision_score(y_test, y_probs)))

    # calculate mean AUC-ROC & AUC-PR
    auc_roc = [roc_auc_score(sample_test[i], sample_prob[i]) for i in range(T)]
    auc_pr = [average_precision_score(sample_test[i], sample_prob[i]) for i in range(T)]

    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    fig = plt.figure(dpi=80, figsize=(13, 5.5))
    ax = fig.add_subplot(121)
    plt.grid(True)
    ax.set_title("PR-кривая", fontsize=21, pad=7)
    ax.fill_between(recall, precision, alpha=0.6, color="xkcd:apple green", lw=2)
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.02)

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    ax = fig.add_subplot(122)
    plt.grid(True)
    ax.set_title("ROC-кривая", fontsize=21, pad=7)
    ax.fill_between(fpr, tpr, alpha=0.6, color="xkcd:goldenrod", lw=2)
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.02)

    plt.show()
    
    mean_logloss, std_logloss = bootstrap_estimate_mean_stddev(logloss)
    mean_accuracy, std_accuracy = bootstrap_estimate_mean_stddev(accuracy)
    mean_auc_roc, std_auc_roc = bootstrap_estimate_mean_stddev(auc_roc)
    mean_auc_pr, std_auc_pr = bootstrap_estimate_mean_stddev(auc_pr)
    
    
    print("LogLoss:  mean = %.4f, std = %.4f" % (mean_logloss, std_logloss))
    print("Accuracy: mean = %.4f, std = %.4f" % (mean_accuracy, std_accuracy))
    print("AUC-ROC:  mean = %.4f, std = %.4f" % (mean_auc_roc, std_auc_roc))
    print("AUC-PR:   mean = %.4f, std = %.4f" % (mean_auc_pr, std_auc_pr))   