import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

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


class NNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10):
        super(NNRegressor, self).__init__()        
        self.fc1 = nn.Linear(n_inputs, n_hidden//2)
        self.fc2 = nn.Linear(n_hidden//2, n_hidden//4)
        self.fc3 = nn.Linear(n_hidden//4, n_hidden//4)
        self.fc4 = nn.Linear(n_hidden//4, 1)
        
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        x = self.act1(x)
        x = self.fc4(x)
        return x
    
    
class FitNNRegressor(object):
    
    def __init__(self, n_hidden=40, n_epochs=10, batch_size=64, lr=0.01, lam=1., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
        self.scaler = StandardScaler()
        

    def fit(self, X, y):
        # Scaling
        X_ss = self.scaler.fit_transform(X)
        # Estimate model
        self.model = NNRegressor(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden).cuda(device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        
        best_loss = float('inf')
        best_state = None
        
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                loss = loss_func(y_batch, y_pred_batch)
                
                lam = torch.tensor(self.lam)
                l2_reg = torch.tensor(0.).to(device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lam * l2_reg
                
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
            
        
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
        return y_pred

    
class NNProcessesAugmentation(object):
    
    def __init__(self, passband2lam):
        self.passband2lam = passband2lam
        self.ss = None
        self.reg = None

    
    def fit(self, t, flux, flux_err, passband):
        t        = np.array(t)
        flux     = np.array(flux)
        flux_err = np.array(flux_err)
        passband = np.array(passband)
        log_lam  = add_log_lam(passband, self.passband2lam)
        
        X = np.concatenate((t.reshape(-1, 1), log_lam.reshape(-1, 1)), axis=1)
        self.ss = StandardScaler()
        X_ss = self.ss.fit_transform(X)
        
        self.reg = FitNNRegressor(n_hidden=400, n_epochs=100, batch_size=1, lr=0.01, lam=1., optimizer='RMSprop')
        
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


        