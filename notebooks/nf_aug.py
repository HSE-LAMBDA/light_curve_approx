import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable


DEVICE = torch.device('cpu')


class InvertibleLayer(nn.Module):
    def __init__(self, var_size):
        super(InvertibleLayer, self).__init__()

        self.var_size = var_size

    def f(self, x, y):
        '''
        Implementation of forward pass.

        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition

        Return:
          torch.Tensor of shape [batch_size, var_size], torch.Tensor of shape [batch_size]
        '''
        pass

    def g(self, x, y):
        '''
        Implementation of backward (inverse) pass.

        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition

        Return:
        А  torch.Tensor of shape [batch_size, var_size]
        '''
        pass
    
    
class NormalizingFlow(nn.Module):
    
    def __init__(self, layers, prior):
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior

    def log_prob(self, x, y):
        '''
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        log_likelihood = None

        for layer in self.layers:
            x, change = layer.f(x, y)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(x)

        return log_likelihood.mean()

    def sample(self, y):
        '''
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        
        x = self.prior.sample((len(y), ))
        for layer in self.layers[::-1]:
            x = layer.g(x, y)

        return x
    
    
class RealNVP(InvertibleLayer):
    
    def __init__(self, var_size, cond_size, mask, hidden=10):
        super(RealNVP, self).__init__(var_size=var_size)

        self.mask = mask

        self.nn_t = nn.Sequential(
            nn.Linear(var_size+cond_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size)
            )
        self.nn_s = nn.Sequential(
            nn.Linear(var_size+cond_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, var_size),
            )

    def f(self, x, y):
        '''
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        xy = torch.cat((x * self.mask[None, :], y), dim=1)
        t = self.nn_t(xy)
        s = self.nn_s(xy)

        new_x = (x * torch.exp(s) + t) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        log_det = (s * (1 - self.mask[None, :])).sum(dim=-1)
        return new_x, log_det

    def g(self, x, y):
        '''
        x: torch.Tensor of shape [batch_size, var_size]
            Data
        y: torch.Tensor of shape [batch_size, cond_size]
            Condition
        '''
        xy = torch.cat((x * self.mask[None, :], y), dim=1)
        t = self.nn_t(xy)
        s = self.nn_s(xy)

        new_x = ((x - t) * torch.exp(-s)) * (1 - self.mask[None, :]) + x * self.mask[None, :]
        return new_x
    
    
class NFFitter(object):
    
    def __init__(self, var_size=2, cond_size=2, normalize_y=True, batch_size=500, n_epochs=3000, lr=0.005, randomize_x=True):
        
        self.normalize_y = normalize_y
        self.randomize_x = randomize_x
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        
        prior = torch.distributions.MultivariateNormal(torch.zeros(var_size), torch.eye(var_size))

        layers = []
        for i in range(8):
            layers.append(RealNVP(var_size=var_size, cond_size=cond_size+randomize_x, mask=((torch.arange(var_size) + i) % 2)))

        self.nf = NormalizingFlow(layers=layers, prior=prior)
        self.opt = torch.optim.Adam(self.nf.parameters(), lr=self.lr)
        
        
    def reshape(self, y):
        try:
            y.shape[1]
            return y
        except:
            return y.reshape(-1, 1)
    
    
    def fit(self, X, y, y_std=None):
        
        # reshape
        y = self.reshape(y)
        
        # normalize
        if self.normalize_y:
            self.ss_y = StandardScaler()
            y = self.ss_y.fit_transform(y)
            
        if y_std is not None:
            y_std = self.reshape(y_std)
            if self.normalize_y:
                y_std /= self.ss_y.scale_
        else:
            y_std = np.zeros_like(y)
            
        #noise = np.random.normal(0, 1, (y.shape[0], 1))
        #y = np.concatenate((y, noise), axis=1)
        
        # numpy to tensor
        y_real = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        y_real_std = torch.tensor(y_std, dtype=torch.float32, device=DEVICE)
        X_cond = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        
        # tensor to dataset
        dataset_real = TensorDataset(y_real, y_real_std, X_cond)
        
        criterion = nn.MSELoss()
        self.loss_history = []

        # Fit GAN
        for epoch in range(self.n_epochs):
            for i, (y_batch, std_batch, x_batch) in enumerate(
                DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)
            ):   
                noise = np.random.normal(0, 1, (len(y_batch), 1))
                noise = torch.tensor(noise, dtype=torch.float32, device=DEVICE)
                
                y_batch = torch.normal(y_batch, std_batch)
                y_batch = torch.cat((y_batch, noise), dim=1)
                
                if self.randomize_x:
                    noise = np.random.normal(0, 1, (len(x_batch), 1))
                    noise = torch.tensor(noise, dtype=torch.float32, device=DEVICE)
                    x_batch = torch.cat((x_batch, noise), dim=1)
                
                #y_pred = self.nf.sample(x_batch)
                
                # caiculate loss
                loss = -self.nf.log_prob(y_batch, x_batch)
                #loss = criterion(y_batch, y_pred)
                
                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                    
                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())            
        
    def predict(self, X):
        #noise = np.random.normal(0, 1, (X.shape[0], 1))
        #X = np.concatenate((X, noise), axis=1)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if self.randomize_x:
            noise = np.random.normal(0, 1, (len(X), 1))
            noise = torch.tensor(noise, dtype=torch.float32, device=DEVICE)
            X = torch.cat((X, noise), dim=1)
        y_pred = self.nf.sample(X).cpu().detach().numpy()[:, 0]
        # normalize
        if self.normalize_y:
            y_pred = self.ss_y.inverse_transform(y_pred)
        return y_pred
    
    def predict_n_times(self, X, n_times=100):
        predictions = []
        for i in range(n_times):
            y_pred = self.predict(X)
            predictions.append(y_pred)
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std


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
    

class NFAugmentation(object):
    
    def __init__(self, passband2lam):
        """
        Light Curve Augmentation based on Normalizing Flows Regression
        
        Parameters:
        -----------
        passband2lam : dict
            A dictionary, where key is a passband ID and value is Log10 of its wave length.
            Example: 
                passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}
        """
        
        self.passband2lam = passband2lam
        
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
        
        self.ss = StandardScaler()
        X_ss = self.ss.fit_transform(X)
        
        self.reg = NFFitter(var_size=2, cond_size=2, normalize_y=True, batch_size=500,
                            n_epochs=3000, lr=0.005, randomize_x=True)
        self.reg.fit(X_ss, flux, flux_err)
    
    
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
        X_ss = self.ss.transform(X)
        
        flux_pred, flux_err_pred = self.reg.predict_n_times(X_ss, n_times=100)
        
        return flux_pred, flux_err_pred
        
    
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