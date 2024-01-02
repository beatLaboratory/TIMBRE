"""
Created on Tue Jan  2 14:54:42 2024

@author: Gautam Agarwal
"""
from keras.callbacks import EarlyStopping
from keras import models, layers, optimizers, backend, constraints, activations
import complexnn
import numpy as np
from keras import utils as np_utils

def TIMBRE(X,Y,inds_test,inds_train,hidden_nodes=0,learn_rate=.001,is_categorical=True,verbosity=0):
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
  - verbosity = amount of model training info to output (default = 0)
 
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
    verbose = 0, validation_data=(X[inds_test,:], Y[inds_test,:]),
    shuffle=True, callbacks=[es])
  test_acc = fittedModel.history['val_accuracy'][-1]
  return model, fittedModel, test_acc