from sklearn.model_selection import train_test_split
from itertools import chain
import numpy as np

def binned_train_test_split(*arrays, t, bin_size=5, 
                            test_size=None, train_size=None, 
                            random_state=None, shuffle=True, 
                            bounds_in_train=True):
    
    """
    Split arrays or matrices into random binned train and test subsets
    
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    t : array-like
        Timestamps of light curve observations.
    bin_size : float
        Size of each time bin.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test bins. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train bins. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    bounds_in_train : bool, default=True
        Whether or not to include bins on bounds into train sample.
        

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """

    bins = np.arange(t.min(), t.max(), bin_size)
    bin_inds = np.digitize(t, bins)
    
    i_bins = np.unique(bin_inds)
    if len(i_bins) < 2:
        raise ValueError("At least 2 bins are required. Reduce bin_size please.")
    
    if bounds_in_train:
        if len(i_bins) < 4:
            raise ValueError("At least 4 bins are required. Reduce bin_size please.")
        bounds = i_bins[[0, -1]]
        i_bins = i_bins[1:-1]
    
    i_bins_train, i_bins_test = train_test_split(i_bins, 
                                                 test_size=test_size, 
                                                 train_size=train_size, 
                                                 random_state=random_state, 
                                                 shuffle=shuffle)
    
    if bounds_in_train:
        i_bins_train = np.concatenate((i_bins_train, bounds), axis=0)
    
    mask_train = np.in1d(bin_inds, i_bins_train)
    mask_test = np.in1d(bin_inds, i_bins_test)

    return list(chain.from_iterable((a[mask_train], a[mask_test]) for a in arrays))