import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
class RBF(nn.Module):
    def __init__(self, in_features, out_features, basis_func=gaussian):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 2)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        return self.basis_func(distances)
    

class RBFNetRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10):
        super(RBFNetRegressor, self).__init__()
        self.net = nn.Sequential(
            RBF(n_inputs, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, x):
        return self.net(x)


class FitRBFNetRegressor(object):
    
    def __init__(self, n_hidden=10, n_epochs=10, batch_size=64, lr=0.01, lam=0., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    
    def fit(self, X, y):
        # Scaling
        X_ss = self.scaler.fit_transform(X)
        y_ss = self.y_scaler.fit_transform(y[:, None])
        # Estimate model
        self.model = RBFNetRegressor(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden).to(device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y_ss, dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        # Enable droout
        self.model.train(True)
        
        best_loss = float('inf')
        best_state = None
        
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                loss = loss_func(y_batch, y_pred_batch)
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
            if np.mean(loss_history) < best_loss:
                best_loss = np.mean(loss_history)
                best_state = self.model.state_dict()
        self.model.load_state_dict(best_state)
    
    def predict(self, X):
        # Scaling
        X_ss = self.scaler.transform(X)
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = self.y_scaler.inverse_transform(y_pred)
        return y_pred


class RBFNetAugmentation(object):
    
    def __init__(self, passband2lam, **reg_kwargs):
        """
        Light Curve Augmentation based on Radial Basis Function Network
        
        Parameters:
        -----------
        passband2lam : dict
            A dictionary, where key is a passband ID and value is Log10 of its wave length.
            Example: 
                passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
                                 
        **reg_kwargs : keyword arguments for FitRBFNetRegressor
        """        
        self.passband2lam = passband2lam
        self.reg_kwargs = reg_kwargs
        
        self.ss = None
        self.reg = None
        
    
    
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
        
        t        = np.array(t)
        flux     = np.array(flux)
        flux_err = np.array(flux_err)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        
        self.reg = FitRBFNetRegressor(**self.reg_kwargs)
        self.reg.fit(X, flux)
    
    
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
            Flux of the light curve observations, approximated by the augmentation model.
        flux_err_pred : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        """
        
        t        = np.array(t)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        
        flux_pred = self.reg.predict(X)
        flux_err_pred = np.empty(flux_pred.shape)
        
        return np.maximum(0, flux_pred), flux_err_pred
        
    
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
        flux_err_aug : array-like
            Flux errors of the light curve observations, estimated by the augmentation model.
        passband_aug : array-like
            Passband IDs for each observation.
        """
        
        t_aug, passband_aug = create_aug_data(t_min, t_max, len(self.passband2lam), n_obs)
        flux_aug, flux_err_aug = self.predict(t_aug, passband_aug, copy=True)
        
        return t_aug, flux_aug, flux_err_aug, passband_aug