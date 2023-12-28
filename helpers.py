"""
A set of helper functions for running TIMBRE, whitening data, and visualizing outputs.

@author: Gautam Agarwal
"""

from keras.callbacks import EarlyStopping
from keras import models, layers, optimizers, backend, constraints, activations
import complexnn
import numpy as np
from keras import utils as np_utils

def TIMBRE(X,Y,inds_test,inds_train,hidden_nodes=0,learn_rate=.001):
  """
  Learns oscillatory patterns that are predictive of class labels
  
  Parameters:
  - X = Multi-channel data (T samples x N channels, complex-valued)
  - Y = Category labels (T samples, integer-valued)
  - inds_test = test indices (Either T x 1 boolean, or U x 1 integers)
  - inds_train = train indices (Either T x 1 boolean, or U x 1 integers)
  - hidden_nodes = how many nodes to use (no hidden layer if set to 0)
  - learn_rate = how quickly the network learns
 
  Returns:
  - model: trained network
  - fittedModel: history of loss and accuracy for test and train data
  """
  
  #stack the real and imaginary components of the data
  X = np.concatenate((np.real(X), np.imag(X)), axis = 1) 
  #use one-hot encoding for the class labels
  Y = np_utils.to_categorical(Y)                          
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
  model.compile(loss='categorical_crossentropy', optimizer=adam,metrics = ['accuracy'])
  # Train the model
  fittedModel = model.fit(X[inds_train,:], Y[inds_train,:], epochs = 100,
    verbose = 2, validation_data=(X[inds_test,:], Y[inds_test,:]),
    shuffle=True, callbacks=[es])
  return model, fittedModel

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