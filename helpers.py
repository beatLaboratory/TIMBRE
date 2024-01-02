"""
A set of helper functions for running TIMBRE, whitening data, and visualizing outputs.

@author: Gautam Agarwal
"""

from keras.callbacks import EarlyStopping
from keras import models, layers, optimizers, backend, constraints, activations
import complexnn
import numpy as np
from keras import utils as np_utils
from random import sample
from scipy import signal

def TIMBRE(X,Y,inds_test,inds_train,hidden_nodes=0,learn_rate=.001,is_categorical=True):
  """
  Learns oscillatory patterns that are predictive of class labels
  
  Parameters:
  - X = Multi-channel data (T samples x N channels, complex-valued)
  - Y = Category labels (T samples, integer-valued)
  - inds_test = test indices (Either T x 1 boolean, or U x 1 integers)
  - inds_train = train indices (Either T x 1 boolean, or U x 1 integers)
  - hidden_nodes = how many nodes to use (no hidden layer if set to 0)
  - learn_rate = how quickly the network learns
  - is_categorical = whether the output consists of discrete classes 
 
  Returns:
  - model: trained network
  - fittedModel: history of loss and accuracy for test and train data
  - test_acc: accuracy on test data after training
  """
  
  #stack the real and imaginary components of the data
  X = np.concatenate((np.real(X), np.imag(X)), axis = 1) 
  #use one-hot encoding for the class labels
  if is_categorical:
      Y = np_utils.to_categorical(Y)
      my_loss = 'categorical_crossentropy'
  else:
      my_loss = 'kde'
  backend.clear_session()
  # Early Stopping: stop training model when test loss stops decreasing
  es = EarlyStopping(monitor = 'val_loss', patience = 1)
  # Specify the algorithm and step size used by gradient descent
  adam = optimizers.Adam(learning_rate=learn_rate)
  if hidden_nodes > 0:
    num_chans = hidden_nodes
  else:
    num_chans = Y.shape[1]
  model = models.Sequential()
  # Layer 1: Takes a complex-valued projection of the input
  model.add(complexnn.dense.ComplexDense(num_chans, input_shape=(X.shape[1],), use_bias=False, kernel_constraint = constraints.unit_norm()))
  # Layer 2: Converts complex-valued output of layer 0 to a real-valued magnitude
  model.add(layers.Lambda(lambda x: (x[:,:x.shape[1]//2]**2 + x[:,x.shape[1]//2:]**2)**.5))
  # Layer 3: Softmax of layer 2
  model.add(layers.Activation(activations.softmax))
  if hidden_nodes > 0: #Need another layer for output
    model.add(layers.Dense(Y.shape[1], activation='softmax'))
  model.compile(loss=my_loss, optimizer=adam,metrics = ['accuracy'])
  # Train the model
  fittedModel = model.fit(X[inds_train,:], Y[inds_train,:], epochs = 100,
    verbose = 2, validation_data=(X[inds_test,:], Y[inds_test,:]),
    shuffle=True, callbacks=[es])
  test_acc = fittedModel.history['val_accuracy'][-1]
  return model, fittedModel, test_acc

def carrier_based(X,Y,inds_test,inds_train,learn_rate=.001,is_categorical=True,subgroups = []):
  """
  Predicts output using demodulated LFP and linear regression
  
  Parameters:
  - X = Multi-channel data (T samples x N channels, complex-valued)
  - Y = Category labels (T samples, integer-valued)
  - inds_test = test indices (Either T x 1 boolean, or U x 1 integers)
  - inds_train = train indices (Either T x 1 boolean, or U x 1 integers)
  - learn_rate = how quickly the network learns
  - is_categorical = whether the output consists of discrete classes 
 
  Returns:
  - model: trained network
  - fittedModel: history of loss and accuracy for test and train data
  - test_acc: accuracy on test data after training
  """
  #demodulate the LFP using the carrier (defined as the first PC)
  X = X*np.exp(1j*-np.angle(X[:,0][:,np.newaxis]))
  #stack the real and imaginary components of the data
  X = np.concatenate((np.real(X), np.imag(X)), axis = 1)
  #use one-hot encoding for the class labels
  if is_categorical:
      if len(subgroups):
        Yc = np_utils.to_categorical(Y*np.max(subgroups+1)+subgroups)
      else:
        Yc = np_utils.to_categorical(Y)
      my_loss = 'categorical_crossentropy'
  else:
      my_loss = 'kde'

  backend.clear_session()
  # Early Stopping: stop training model when test loss stops decreasing
  es = EarlyStopping(monitor = 'val_loss', patience = 1)
  # Specify the algorithm and step size used by gradient descent
  adam = optimizers.Adam(learning_rate=learn_rate)
  num_chans = Yc.shape[1]
  model = models.Sequential()
  # Layer 1: Takes a complex-valued projection of the input
  model.add(layers.Dense(num_chans, input_shape=(X.shape[1],), use_bias=True, kernel_constraint = constraints.unit_norm()))
  # Layer 2: Softmax of layer 1
  model.add(layers.Activation(activations.softmax))
  #model.add(layers.Dense(Y.shape[1],activation='softmax'))
  model.compile(loss=my_loss, optimizer=adam,metrics = ['accuracy'])
  # Train the model
  fittedModel = model.fit(X[inds_train], Yc[inds_train], epochs = 100,
    verbose = 2, validation_data=(X[inds_test], Yc[inds_test]),
    shuffle=True, callbacks=[es])
  if len(subgroups):
    print(model.predict(X[inds_test]))
    test_acc = np.mean(np.floor(np.argmax(model.predict(X[inds_test]),axis=1)/np.max(subgroups+1))==Y[inds_test])
  else:
    test_acc = fittedModel.history['val_accuracy'][-1]
  return model, fittedModel, test_acc

def layer_output(X,m,layer_num):
  """
  Returns response of one of TIMBRE's layers
  
  Parameters:
  - X: Input data
  - m: Trained model
  - layer_num: Which layer's output to return
  
  Returns:
  - Layer's response to input
  """
  #stack the real and imaginary components of the data
  X = np.concatenate((np.real(X), np.imag(X)), axis = 1) 
  m1 = models.Model(inputs=m.input, outputs=m.layers[layer_num].output)
  return m1.predict(X) #return output of layer layer_num

def test_train(lapID,which_phase,n_folds = 5,which_fold = 0):
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
    use_sample = lapID[:,3] == which_phase
    if which_phase == 2: # period where rat is staying at port
        use_sample = use_sample & (lapID[:,2] == 1) #only use correct trials
    fold_assign = -np.ones(np.size(use_sample))
    for i in range(int(np.max(lapID[:,0]))):
        inds = (lapID[:,0] == i) & use_sample
        if np.sum(inds):
            which_arm = int(lapID[inds,1][0])
            fold_assign[inds] = ctr[which_arm]%n_folds
            ctr[which_arm] += 1
    test_inds = fold_assign == which_fold
    train_inds = np.isin(fold_assign, np.arange(n_folds)) & ~test_inds
    train_inds = balanced_indices(lapID[:,1],train_inds)
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

def whiten(X,inds_train,fudge_factor=10**-5):
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
    _,_,u = np.linalg.svd(X[inds_train,:],full_matrices=False,compute_uv=True)
    X = X@np.conj(u.T)
    Xv = np.var(X[inds_train,:],axis=0)
    Xv = np.sqrt(Xv+sum(Xv)*fudge_factor)
    X = X/Xv
    return X, u, Xv

def filter_data(data, cutoff, fs, filt_type='high', order=5, use_hilbert = False):
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
    data = signal.filtfilt(b, a, data,axis=0)
    if use_hilbert:
        data = signal.hilbert(data,axis=0)
        
    return data

def group_by_pos(pos,num_bins):
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
    pos = pos - np.min(pos)
    pos = pos / (np.max(pos)+10**-8)
    pos = np.int32(np.floor(pos*num_bins))
    return pos


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