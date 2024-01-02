"""
Created on Tue Jan  2 14:57:25 2024

@author: Gautam Agarwal
"""
from keras.callbacks import EarlyStopping
from keras import models, layers, optimizers, backend, constraints, activations
import numpy as np
from keras import utils as np_utils

def carrier_based(X,Y,inds_test,inds_train,learn_rate=.001,is_categorical=True,subgroups=[],verbosity=0):
  """
  Predicts output from the demodulated LFP using a linear model
  
  Parameters:
  - X = Multi-channel whitened data (T samples x N channels, complex-valued - channel 0 is 1st PC)
  - Y = Category labels (T samples, integer-valued)
  - inds_test = test indices (Either T x 1 boolean, or U x 1 integers)
  - inds_train = train indices (Either T x 1 boolean, or U x 1 integers)
  - learn_rate = how quickly the network learns
  - is_categorical = whether the output consists of discrete classes 
  - verbosity = amount of model training info to output (default = 0)
 
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
    verbose = 0, validation_data=(X[inds_test], Yc[inds_test]),
    shuffle=True, callbacks=[es])
  if len(subgroups):
    test_acc = np.mean(np.floor(np.argmax(model.predict(X[inds_test]),axis=1)/np.max(subgroups+1))==Y[inds_test])
  else:
    test_acc = fittedModel.history['val_accuracy'][-1]
  return model, fittedModel, test_acc