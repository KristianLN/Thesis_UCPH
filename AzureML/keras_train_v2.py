########## KERAS APPROACH #############

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import argparse
import os
import re
import time
import glob

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout, Activation,LeakyReLU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization


import tensorflow as tf

from azureml.core import Run

from utils.generate_features import generateFeatures_multi_final
from utils.preprocessing_features_and_labels import align_features_and_labels_multi_final, pre_processing


print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-source', type=str, dest='data_source', default='data', help='data folder mounting point')
parser.add_argument('--nn-type', type=str, dest='nn_type', default='ffnn', help='Choose the neural network type')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=128,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=128,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.0, help='X_1')
parser.add_argument('--n-layers', type=int, dest='n_layers', default=1, help='X_2')
parser.add_argument('--l2-penalty', type=float, dest='l2_penalty', default=0.0, help='X_3')
parser.add_argument('--batch-norm', type=bool, dest='batch_norm', default=False, help='X_4')
parser.add_argument('--batch-shuffle', type=bool, dest='batch_shuffle', default=False, help='X_5')
parser.add_argument('--activation-inner', type=str, dest='activation_inner', default='relu', help='X_6')
parser.add_argument('--activation-output', type=str, dest='activation_output', default='softmax', help='X_7')

## Newly added
parser.add_argument('--featureset', type=int, dest='feature_set',default=1, help='specifying the feature set')
parser.add_argument('--pre-processing', type=str, dest = 'pre_processing',default=None, help='pre-processing of training and validation data')
parser.add_argument('--feature-lags', type=int, dest='feature_lags', default=0, help='number of lagged features')
parser.add_argument('--n-epochs', type=int, dest = 'n_epochs', default=20, help='number of epochs to run')
parser.add_argument('--label-type', type=int, dest = 'label_type', default=1, help='label type')
parser.add_argument('--pastobs-in-percentage', type=bool, dest = 'pastobs_in_percentage', default=False, help='Option to specify if the pastobs feature should be in percentage')
parser.add_argument('--resume-from', type=str, default=None,
                    help='location of the model or checkpoint files from where to resume the training')
args = parser.parse_args()

previous_model_location = args.resume_from
n_feature_lags = args.feature_lags
features_to_use = args.feature_set
pre_procesing_applied = args.pre_processing
n_epochs = args.n_epochs
label_type = args.label_type
pastobs_in_percentage = args.pastobs_in_percentage
nn_type = args.nn_type
dropout_ratio = args.dropout_ratio
n_layers = args.n_layers
l2_penalty = args.l2_penalty
batch_norm = args.batch_norm
batch_shuffle = args.batch_shuffle
activation_inner = args.activation_inner
activation_output = args.activation_output
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
batch_size = args.batch_size
learning_rate = args.learning_rate

# start an Azure ML run
run = Run.get_context()

datasetName = 'aggregatedTAQ_60sec_wsectors'

# get the input dataset by name
dataset = run.input_datasets[datasetName]

### load train and test set into numpy arrays
# load the TabularDataset to pandas DataFrame
data = dataset.to_pandas_dataframe().set_index(['Column1','Column2'])
data.columns = ['open','high','low','close','spread_open',
                              'spread_high','spread_low','spread_close',
                              'bidsize_open','bidsize_high','bidsize_low','bidsize_close',
                              'ofrsize_open','ofrsize_high','ofrsize_low','ofrsize_close',
                              'Ticker','sector']

# Removing the XNTK ticker
data = data[~data.Ticker.isin(['XNTK'])]

etfs = ['IYH','IYM','IYK','IYJ','IYG','IYW','IYC','IYR','IDU','IYZ','IYE','IYF','SPY','DIA','QQQ']

# Extracting the sector ETFs to a separate variable
sectorETFS = data[data.Ticker.isin(etfs)]

# Removing the ETFs
data = data[~data.Ticker.isin(etfs)]



########### Generate Features ################

# feature_sets = {0:'base',
#                 1:'base with sectors',
#                 # 2:'technical',
#                 # 3:'technical with sectors'
#                 2:'base with technical',
#                 # 3:'base with techinical and sectors',
#                 3:'full'}

if features_to_use == 0:
    listOfFeatures = [
                        'pastobs',
                        'spread',
                        'bidsize',
                        'ofrsize',
                        'pastreturns',
                        'intradaytime'
                    ]

elif features_to_use == 1:

    listOfFeatures = [
                    'pastobs',
                    'spread',
                    'bidsize',
                    'ofrsize',
                    # 'pastreturns',
                    'intradaytime',
                    'sector'
                    ]

elif features_to_use == 2:
    listOfFeatures = [
                        'pastobs',
                        'spread',
                        'bidsize',
                        'ofrsize',
                        'pastreturns',
                        'intradaytime',
                        'stok',
                        'stod',
                        'sstod',
                        'wilr',
                        'roc',
                        'rsi',
                        'atr',
                        'cci',
                        'dpo',
                        'sma',
                        'ema',
                        'macd',
                        'macd_diff',
                        'macd_signal',
                        'dis5',
                        'dis10'
                        ]

elif features_to_use == 3:

    listOfFeatures = [
                    'pastobs',
                    'spread',
                    'bidsize',
                    'ofrsize',
                    'pastreturns',
                    'intradaytime',
                    'stok',
                    'stod',
                    'sstod',
                    'wilr',
                    'roc',
                    'rsi',
                    'atr',
                    'cci',
                    'dpo',
                    'sma',
                    'ema',
                    'macd',
                    'macd_diff',
                    'macd_signal',
                    'dis5',
                    'dis10',
                    'sector'
                    ]

elif features_to_use == 4:
    raise NotImplementedError
    ##

elif features_to_use == 5:
    raise NotImplementedError
    ##

elif features_to_use == 6:
    raise NotImplementedError

## Extracting the features

features = generateFeatures_multi_final(data = data,
                                        listOfFeatures = listOfFeatures,
                                        feature_lags = n_feature_lags,
                                        sectorETFS=sectorETFS,
                                        pastobs_in_percentage=pastobs_in_percentage)


########### Generate Labels ################

price_candles = data[['open','high','low','close','Ticker']]

########### Align Data ################

# Label options:

'''
Label construction:
- Binary median split
	- positive/negative returns in our case, at 1 minute horizon.

- Multi 33-33-33 chunks

- Multi 20-60-20 chunks

- Multi 20-20-20-20-20 chunks

- Multi 10-20-40-20-10 chunks
'''

if label_type == 0:
    n_classes = 2
    label_split = [] # empty means equal split

elif label_type == 1:
    n_classes = 3
    label_split = [] # empty means equal split

elif label_type == 2:
    n_classes = 3
    label_split = [0, 0.2, 0.8, 1] # for non-equal, include 0 and 1

elif label_type == 3:
    n_classes = 5
    label_split = [] # empty means equal split

elif label_type == 4:
    n_classes = 5
    label_split = [0, 0.1, 0.3, 0.7, 0.9, 1] # for non-equal, include 0 and 1

# from imported function (see testing_preprocessing_features_and_labels.ipynb for thorough experimenting with all the cut-offs):
X, y, indices = align_features_and_labels_multi_final(price_candles = price_candles,
                                                        all_features = features,
                                                        prediction_horizon = 1,
                                                        n_feature_lags = n_feature_lags,
                                                        n_classes = n_classes, # 5,
                                                        label_split = label_split,
                                                        safe_burn_in = False,
                                                        data_sample = 'full',
                                                        splitType='global',
                                                        noise=False,
                                                        ticker_dummies=False)


# Adding ticker dummies
tickers = X.pop('ticker')
X = pd.concat([X, pd.get_dummies(tickers, prefix='d_ticker', drop_first=False)], axis=1)

########### Splitting data ################

# Let's have a proper split (along tickers & dates)
train_size = 0.8

# Sort the indices
tempIndices = indices.sort_values(['days','timestamps','ticker'])

# Sorting the data
X = X.loc[tempIndices.index,:]#.head(66)
y = y.loc[tempIndices.index,:]

# extracting the first date for the validation data.
first_val_day = int(np.floor(indices.days.unique().shape[0]*0.8))

# Splitting the data
X_train = X[tempIndices.days<tempIndices.days.unique()[first_val_day]]
y_train = y[tempIndices.days<tempIndices.days.unique()[first_val_day]]

X_test = X[tempIndices.days>=tempIndices.days.unique()[first_val_day]]
y_test = y[tempIndices.days>=tempIndices.days.unique()[first_val_day]]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]


########### Pre-processing: none right now ################

if  pre_procesing_applied == None:
    # do nothing here
    pass

elif  pre_procesing_applied == 'std':

    # Standardize some features
    ppdict1 = {i:'std' for i in X_train.columns if 'd_' != i[0:2]}
    # Keep some in actual levels (Dummies in this case).
    ppdict2 = {i:'act' for i in X_train.columns if 'd_' == i[0:2]}

    # Merging the two
    ppdict = {**ppdict1,**ppdict2}

    x_train,x_test = pre_processing(x_train,x_test,pp_dict)

elif pre_procesing_applied == 'minmax':

    # Standardize some features
    ppdict1 = {i:'minmax' for i in X_train.columns if 'd_' != i[0:2]}
    # Keep some in actual levels (Dummies in this case).
    ppdict2 = {i:'act' for i in X_train.columns if 'd_' == i[0:2]}

    # Merging the two
    ppdict = {**ppdict1,**ppdict2}

    x_train,x_test = pre_processing(x_train,x_test,pp_dict)

elif pre_procesing_applied == 'pow':

    # Standardize some features
    ppdict1 = {i:'pow' for i in X_train.columns if 'd_' != i[0:2]}
    # Keep some in actual levels (Dummies in this case).
    ppdict2 = {i:'act' for i in X_train.columns if 'd_' == i[0:2]}

    # Merging the two
    ppdict = {**ppdict1,**ppdict2}

    x_train,x_test = pre_processing(x_train,x_test,pp_dict)

elif pre_procesing_applied == 'quantgau':

    # Standardize some features
    ppdict1 = {i:'quantgau' for i in X_train.columns if 'd_' != i[0:2]}
    # Keep some in actual levels (Dummies in this case).
    ppdict2 = {i:'act' for i in X_train.columns if 'd_' == i[0:2]}

    # Merging the two
    ppdict = {**ppdict1,**ppdict2}

    x_train,x_test = pre_processing(x_train,x_test,pp_dict)

elif pre_procesing_applied == 'individual':

    # Keep some in actual levels (Dummies in this case).
    ppdict2 = {i:'act' for i in X_train.columns if 'd_' == i[0:2]}

    # Merging the two
    ppdict = {**ppdict1,**ppdict2}

    x_train,x_test = pre_processing(x_train,x_test,pp_dict)

elif pre_procesing_applied == 'stacked':

    # Standardize some features

    for j in ['pow','std','minmax']:

        ppdict1 = {i:j for i in X_train.columns if 'd_' != i[0:2]}

        # Keep some in actual levels (Dummies in this case).
        ppdict2 = {i:'act' for i in X_train.columns if 'd_' == i[0:2]}

        # Merging the two
        ppdict = {**ppdict1,**ppdict2}

        x_train,x_test = pre_processing(x_train,x_test,pp_dict)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]

n_inputs = X_train.shape[1] #28 * 28
n_outputs = n_classes # 2
n_epochs = n_epochs# 20

# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]

y_train = one_hot_encode(y_train.values.astype(np.int), n_outputs)
y_test = one_hot_encode(y_test.values.astype(np.int), n_outputs)

print('Shapes after one-hot-encoding labels')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# if nn_type =='ffnn':
#     # Build a simple MLP model
#     model = Sequential()
#
#     for layer in np.arange(n_layers):
#         if layer == 0:
#             # first hidden layer
#             model.add(Dense(n_h1, input_shape=(n_inputs,)))
#
#             if batch_norm:
#                 model.add(BatchNormalization())
#
#             if activation_inner == 'leakyrelu':
#                 model.add(LeakyReLU())
#             else:
#                 model.add(Activation(activation_inner))
#             # dropout
#             if dropout_ratio > 0:
#                 model.add(Dropout(dropout_ratio))
#         else:
#
#             # second hidden layer
#             model.add(Dense(n_h2, activation=activation_inner))
#             if batch_norm:
#                 model.add(BatchNormalization())
#
#             if activation_inner == 'leakyrelu':
#                 model.add(LeakyReLU())
#             else:
#                 model.add(Activation(activation_inner))
#
#             # dropout
#             if dropout_ratio > 0:
#                 model.add(Dropout(dropout_ratio))
#
#     # output layer
#     model.add(Dense(n_outputs))
#
#     if batch_norm:
#         model.add(BatchNormalization())
#
#     model.add(Activation(activation_output))
#
# elif nn_type == 'lstm':
#
#     # LSTM Data Prep
#     # reshape input to be 3D [samples, timesteps, features]
#     X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
#     X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
#
#     # LSTM
#     model = Sequential()
#
#     for layer in np.arange(n_layers):
#
#         if layer==0:
#
#             #Adding the first LSTM layer and some Dropout regularisation
#             model.add(LSTM(units = n_h1, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2]))) # Be sure on the shape!!(n_inputs, 1)
#
#             if batch_norm:
#                 model.add(BatchNormalization())
#
#             # model.add(Activation(activation_inner))
#
#             # dropout
#             if dropout_ratio > 0:
#                 model.add(Dropout(dropout_ratio))
#
#         elif layer < (n_layers-1):
#             # Adding a second LSTM layer and some Dropout regularisation
#             model.add(LSTM(units = n_h2, return_sequences = True))
#             if batch_norm:
#                 model.add(BatchNormalization())
#
#             # model.add(Activation(activation_inner))
#
#             # dropout
#             if dropout_ratio > 0:
#                 model.add(Dropout(dropout_ratio))
#
#         else:
#             # Adding a second LSTM layer and some Dropout regularisation
#             model.add(LSTM(units = n_h2, return_sequences = False))
#             if batch_norm:
#                 model.add(BatchNormalization())
#
#             # model.add(Activation(activation_inner))
#
#             # dropout
#             if dropout_ratio > 0:
#                 model.add(Dropout(dropout_ratio))
#
#     # Adding the output layer
#     model.add(Dense(units = n_outputs))
#
#     if batch_norm:
#         model.add(BatchNormalization())
#
#     model.add(Activation(activation_output))
#
#
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=learning_rate),
#               metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])

 model = keras.Sequential()
  #model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  #hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
  #model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))

  # hp_l2 = hp.Choice('l2_rate', values = [1e10, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4])
  model.add(keras.layers.Dense(1,
            input_shape=(n_inputs,),
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(hp_l2)))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  # hp_learning_rate = hp.Choice('learning_rate', values = [1e-1, 1e-2, 1e-3, 1e-4])

  # model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
  #               #loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
  #               loss = keras.losses.BinaryCrossentropy(from_logits = False),
  #               metrics = ['accuracy', keras.metrics.AUC(name='auc')])
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])

  return model

# start an Azure ML run
run = Run.get_context()


class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])
        run.log('AUC', log['val_auc'])

        run.log('Train Loss', log['loss'])
        run.log('Train Accuracy', log['accuracy'])
        run.log('Train AUC', log['auc'])


history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[LogRunMetrics()],
                    shuffle=batch_shuffle
                    )

score = model.evaluate(X_test, y_test, verbose=0)

print("Scores", score)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test AUC', score[1])
print('Test AUC:', score[1])

run.log('Final test accuracy', score[2])
print('Test accuracy:', score[2])

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
