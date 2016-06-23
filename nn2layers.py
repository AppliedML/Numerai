from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

#read data
train_tour1 = pd.read_csv('numerai_training_data.csv')

#separate target and features
data_feature = np.asarray(train_tour1.ix[:,0:1])
data_target = np.asarray( train_tour1.target)

# convert list of labels to binary class matrix
target = np_utils.to_categorical(data_target) 

# pre-processing: divide by max and substract mean
scale = np.max(data_feature)
data_feature /= scale

mean = np.std(data_feature)
data_feature -= mean

input_dim = data_feature.shape[1]
nb_classes = target.shape[1]

print("input dimensions:", input_dim)
print("target dimensions:", nb_classes)

# Deep Learning 2 Layers
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))

## layer 2
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))

# Output layer
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print("Training...")
model.fit(data_feature, target, nb_epoch=10, batch_size=20, validation_split=0.5, show_accuracy=True)


