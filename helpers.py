"""
A set of helper functions for TIMBRE.

@author: Gautam Agarwal
"""
import numpy as np
from random import sample
from keras import models
from scipy import signal


def test_train(lapID, which_phase, n_folds=5, which_fold=0):
    """
    Returns test and train samples
    
    Parameters:
    - lapID: contains info about trial number and maze arm of each sample
    - which_phase: which phase of the session to use (see get_data\get_behav for info)
    - n_folds: how many folds to assign
    - which_fold: which fold to return values for
    
    Returns:
    - train_inds: which samples to use for training model
    - test_inds: which samples to use for testing model
    """
    ctr = np.zeros(3)
    use_sample = lapID[:, 3] == which_phase
    if which_phase == 2:  # period where rat is staying at port
        use_sample = use_sample & (lapID[:, 2] == 1)  # only use correct trials
    fold_assign = -np.ones(np.size(use_sample))
    for i in range(int(np.max(lapID[:, 0]))):
        inds = (lapID[:, 0] == i) & use_sample
        if np.sum(inds):
            which_arm = int(lapID[inds, 1][0])
            fold_assign[inds] = ctr[which_arm] % n_folds
            ctr[which_arm] += 1
    test_inds = fold_assign == which_fold
    train_inds = np.isin(fold_assign, np.arange(n_folds)) & ~test_inds
    train_inds = balanced_indices(lapID[:, 1], train_inds)
    return test_inds, train_inds


def balanced_indices(vector, bool_indices):
    """
    Returns indices that balance the number of samples for each label in vector

    Parameters:
    vector: The input vector from which to select indices.
    bool_indices: A boolean array indicating which indices in the vector to consider.

    Returns:
    list: A list of indices representing a balanced selection of the unique values in the subset of the vector.
    
    Generated using ChatGPT
    """
    # Convert boolean indices to actual indices
    actual_indices = np.where(bool_indices)[0]

    # Extract the elements and their corresponding indices
    selected_elements = [(vector[i], i) for i in actual_indices]

    # Find unique elements
    unique_elements = np.unique(vector[bool_indices])

    # Group elements by value and collect their indices
    elements_indices = {element: [] for element in unique_elements}
    for value, idx in selected_elements:
        if value in elements_indices:
            elements_indices[value].append(idx)

    # Find the minimum count among the unique elements
    min_count = min(len(elements_indices[element]) for element in unique_elements)

    # Create a balanced set of indices
    balanced_indices_set = []
    for element in unique_elements:
        if len(elements_indices[element]) >= min_count:
            balanced_indices_set.extend(sample(elements_indices[element], min_count))

    return np.array(balanced_indices_set)


def group_by_pos(pos, num_bins, train_inds):
    """
    Subdivides track into bins for training linear classifier on demodulated LFP
    
    Parameters
    ----------
    pos : a vector that contains the position of the animal along the track
    num_bins : a scalar int that indicates how many bins to divide the track into

    Returns
    -------
    pos : a vector of binned positions
    """
    pos = pos - np.min(pos[train_inds])
    pos = pos / (np.max(pos[train_inds]) + 10 ** -8)
    pos = np.int32(np.floor(pos * num_bins))
    return pos


def layer_output(X, m, layer_num):
    """
    Returns response of one of TIMBRE's layers

    Parameters:
    - X: Input data
    - m: Trained model
    - layer_num: Which layer's output to return

    Returns:
    - Layer's response to input
    """
    # stack the real and imaginary components of the data
    X = np.concatenate((np.real(X), np.imag(X)), axis=1)
    m1 = models.Model(inputs=m.inputs, outputs=m.layers[layer_num].output)
    return m1.predict(X)  # return output of layer layer_num


def accumarray(subs, vals, size=None, fill_value=0):
    """
    Averages all values that are associated with the same index. Does this separately for each column of vals.
    Useful for visualizing dependency of layer outputs on behavioral features. 

    Parameters:
    - subs: An MxN array of subscripts, where M is the number of entries in vals and N is the number of dimensions of the output.
    - vals: An MxK matrix of values.
    - size: Tuple specifying the size of the output array (default is based on the maximum index in each column of subs)
    - fill_value: The value to fill in cells of the output that have no entries (default is 0).

    Returns:
    - result: An array of accumulated values.
    
    Generated using ChatGPT
    """
    subs = subs.astype(int)
    if subs.ndim == 1:
        subs = subs[:, np.newaxis]
    if size is None:
        size = tuple(np.max(subs, axis=0) + 1)
    else:
        assert len(size) == subs.shape[1], "Size mismatch between size and subs."

    # Handle single column vals
    if len(vals.shape) == 1:
        vals = vals[:, np.newaxis]

    # Convert subscripts to linear indices.
    indices = np.ravel_multi_index(tuple(subs.T), size)

    K = vals.shape[1]
    result = np.full((*size, K), fill_value, dtype=float)

    for k in range(K):
        total = np.bincount(indices, weights=vals[:, k], minlength=np.prod(size))
        count = np.bincount(indices, minlength=np.prod(size))
        with np.errstate(divide='ignore', invalid='ignore'):  # Ignore divide by zero and invalid operations
            averaged = np.where(count != 0, total / count, fill_value)
        result[..., k] = averaged.reshape(size)

    return result if K > 1 else result.squeeze(-1)


def filter_data(data, cutoff, fs, filt_type='high', order=5, use_hilbert=False):
    """
    Applies a column-wise zero-phase filter to data
    
    Parameters:
    data : a T x N array with filtered data
    cutoff : cutoff frequency (should be 2 numbers for 'band')
    fs : sampling rate
    filt_type : specify as 'high', 'low', or 'band'.
    order : filter order. The default is 5.
    use_hilbert: whether to apply a Hilbert transform (default = False)

    Returns
    -------
    data : a T x N array with filtered data

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=filt_type, analog=False)
    data = signal.filtfilt(b, a, data, axis=0)
    if use_hilbert:
        data = signal.hilbert(data, axis=0)

    return data


def whiten(X, inds_train, fudge_factor=10 ** -5):
    """
    Decorrelates the input data

    Parameters:
    - X: A TxN array of data, can be complex-valued
    - inds_train: which samples to use to estimate correlations
    - fudge_factor: adds a small constant to lessen the influence of small, noisy directions in the data

    Returns:
    - X: decorrelated data
    - u: directions of highest variance in original data
    - Xv: scaling factor used to normalize decorrelated data
    """
    _, _, u = np.linalg.svd(X[inds_train, :], full_matrices=False, compute_uv=True)
    X = X @ np.conj(u.T)
    Xv = np.var(X[inds_train, :], axis=0)
    Xv = np.sqrt(Xv + sum(Xv) * fudge_factor)
    X = X / Xv
    return X, u, Xv
