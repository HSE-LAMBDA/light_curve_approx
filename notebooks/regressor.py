import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        nn.init.normal_(self.centres, 0, 1)
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


class NNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10):
        super(NNRegressor, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        return x
    
    
class FitNNRegressor(object):
    
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
        
    
    def fit(self, X, y, model_type="NN"):
        # Scaling
        X_ss = self.scaler.fit_transform(X)
        # Estimate model
        if model_type == "NN":
            self.model = NNRegressor(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden).to(device)
        elif model_type == "RBF":
            self.model = RBFNetRegressor(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden).to(device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
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
        return y_pred