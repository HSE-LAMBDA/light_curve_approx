import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class NNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10):
        super(NNRegressor, self).__init__()
        self.seq = nn.Sequential(nn.Linear(n_inputs, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, 1))

    def forward(self, x):
        return self.seq(x)
    
    
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
    
    def fit(self, X, y):
        # Estimate model
        self.model = NNRegressor(n_inputs=X.shape[1], n_hidden=self.n_hidden).to(device)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lam)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred