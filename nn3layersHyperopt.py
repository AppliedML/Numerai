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

from sklearn.preprocessing import FunctionTransformer

#read data
train_tour1 = pd.read_csv('numerai_training_data.csv')
feature = pd.DataFrame(train_tour1.ix[:,0:21])
target = pd.DataFrame(train_tour1.target)

#log feature
transformer = FunctionTransformer(np.log1p)
feature_log = transformer.transform(feature)

#add all feature
feature_log = pd.DataFrame(feature_log)
feature_all = pd.concat([feature, feature_log], axis =1 )

#separate target and features
feature_all = np.asarray(feature_all)
target = np.asarray(target)

# convert list of labels to binary class matrix
target = np_utils.to_categorical(target) 

# pre-processing: divide by max and substract mean
scale = np.max(feature_all)
feature_all /= scale

mean = np.std(feature_all)
feature_all-= mean

input_dim = feature_all.shape[1]
nb_classes = target.shape[1]

print("input dimensions:", input_dim)
print("target dimensions:", nb_classes)

from sklearn.cross_validation import train_test_split

X_train , X_test, y_train, y_test = train_test_split(feature_all, target, test_size = 0.2)



from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss
import sys

X = X_train
y = y_train
X_val = X_test
y_val = y_test

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.choice('units3', [64,128, 256, 512, 1024]), 
                    'dropout3': hp.choice('dropout3', [.1, .3, .5, .9]),
                    'activation3': hp.choice('activation3',['relu','tanh'])}
                    ]),

            'units1': hp.choice('units1', [64,128, 256, 512, 1024]),
            'units2': hp.choice('units2', [64,128, 256, 512, 1024]),

            'dropout1': hp.choice('dropout1', [.1, .3, .5, .9]),
            'dropout2': hp.choice('dropout2', [.1, .3, .5, .9]),

            'batch_size' : hp.choice('batch_size', [32,64, 128]),

            'nb_epochs' : hp.choice('nb_epochs', [15,20,25,30]),
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation1': hp.choice('activation1',['relu','tanh']),
            'activation2': hp.choice('activation2',['relu','tanh'])
        }

def f_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    model.add(Activation(params['activation1']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation2']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
        model.add(Activation(params['choice']['activation3']))
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 1)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 1)
    acc = log_loss(y_val, pred_auc)

    print("\n")
    print('logloss:', acc)
    sys.stdout.flush() 
    return {'loss': acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=30, trials=trials)
print(best)