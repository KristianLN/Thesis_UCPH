
########## KERAS APPROACH #############

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # turn off GPU usage
import glob
import pandas as pd
import time

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import Callback

import tensorflow as tf

#from azureml.core import Run
#from utils import load_data, one_hot_encode

from utils.generate_features import generateFeatures_multi_v2
from utils.preprocessing_features_and_labels import align_features_and_labels_multi_final

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=128,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=128,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')

args = parser.parse_args()

# data_folder = args.data_folder

# print('training dataset is stored here:', data_folder)

# X_train_path = glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'), recursive=True)[0]
# X_test_path = glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'), recursive=True)[0]
# y_train_path = glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'), recursive=True)[0]
# y_test_path = glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'), recursive=True)[0]

# X_train = load_data(X_train_path, False) / 255.0
# X_test = load_data(X_test_path, False) / 255.0
# y_train = load_data(y_train_path, True).reshape(-1)
# y_test = load_data(y_test_path, True).reshape(-1)



# # start an Azure ML run
# run = Run.get_context()

# # get the input dataset by name
# dataset = run.input_datasets['aggregateTAQ_60sec'] # [data_source]

# # data_folder = args.data_folder
# # print('Data folder:', data_folder)

# # load train and test set into numpy arrays
# # note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
# # X_train = load_data(glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'),
# #                               recursive=True)[0], False) / np.float32(255.0)
# # X_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'),
# #                              recursive=True)[0], False) / np.float32(255.0)
# # y_train = load_data(glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'),
# #                               recursive=True)[0], True).reshape(-1)
# # y_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'),
# #                              recursive=True)[0], True).reshape(-1)

# ### load train and test set into numpy arrays
# # load the TabularDataset to pandas DataFrame
# data = dataset.to_pandas_dataframe().set_index(['Column1','Column2'])
# data.columns = ['open','high','low','close','spread_open',
#                               'spread_high','spread_low','spread_close',
#                               'bidsize_open','bidsize_high','bidsize_low','bidsize_close',
#                               'ofrsize_open','ofrsize_high','ofrsize_low','ofrsize_close',
#                               'Ticker']

# # Reading in sector information
# stockInfo = run.input_datasets['stockInfo_v1'].to_pandas_dataframe() #pd.read_csv('stockInfo_v1.csv',header=[0,1])
# stockInfo.columns = ['ticker','sector','exchange','marketCap'] # probably already named

# # Creating a table with stock information based on the tickers available in the data.
# uniqueTickers = data.Ticker.unique()
# stockTable = stockInfo[stockInfo.ticker.isin(uniqueTickers)]

# Do we extract new data or read in?
readIn = True
# run load_data()
if readIn:
    
    # Listing the data files 
    #path = '../../../Google Drev/Thesis/Data/TAQ/AggregatedTAQ'
    path = 'F:/AggregatedTAQ/round3'
    datafiles = os.listdir(path)
    content = np.concatenate([['\n\n'],[str(j)+': '+i+'\n' for j,i in enumerate(datafiles) if 'csv' in i],['\n\n']])
    
    # Asking for user input
    file = 2 #input('Which one do you want to load? %s'%''.join(content))
    data = pd.read_csv(path + '/' + datafiles[int(file)],
                       header = None,
                       names=['open','high','low','close',
                              'spread_open','spread_high','spread_low','spread_close',
                              'bidsize_open','bidsize_high','bidsize_low','bidsize_close',
                              'ofrsize_open','ofrsize_high','ofrsize_low','ofrsize_close',
                              'Ticker'])

# Reading in sector information
stockInfo = pd.read_csv('../utils/stockInfo_v1.csv',header=[0,1])
stockInfo.columns = ['ticker','sector','exchange','marketCap']

# Creating a table with stock information based on the tickers available in the data.
uniqueTickers = data.Ticker.unique()
stockTable = stockInfo[stockInfo.ticker.isin(uniqueTickers)]









### drop ETFs
etfs = ['IYH','IYM','IYK','IYJ','IYG','IYW','IYC','IYR','IDU','IYZ','IYE','IYF']

# Extracting the sector ETFs to a separate variable
sectorETFS = data[data.Ticker.isin(etfs)]

# Removing the ETFs
data = data[~data.Ticker.isin(etfs)]


########### Generate Features ################

n_feature_lags = 5

features = generateFeatures_multi_v2(data = data, 
                                  listOfFeatures = [
                                                    'pastobs',
                                                    'spread',
                                                    'bidsize',
                                                    'ofrsize',
#                                                     'stok',
#                                                     'stod',
#                                                     'sstod',
#                                                     'wilr',
#                                                     'roc',
#                                                     'rsi',
#                                                     'atr',
#                                                     'cci',
#                                                     'dpo',
#                                                     'sma',
#                                                     'ema',
#                                                     'macd',
#                                                       'macd_diff',
#                                                       'macd_signal',
#                                                     'dis5',
#                                                     'dis10',
                                                      'sector'
                                                   ], 
                                   feature_lags = n_feature_lags
                                     ,stockTable=stockTable)


########### Generate Labels ################

n_classes = 2
# extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
price_candles = data[['open','high','low','close','Ticker']]

########### Align Data ################

# from imported function (see testing_preprocessing_features_and_labels.ipynb for thorough experimenting with all the cut-offs):    
X, y = align_features_and_labels_multi_final(price_candles = price_candles, 
                                             all_features = features,
                                             prediction_horizon = 1, 
                                             n_feature_lags = n_feature_lags, 
                                             n_classes = n_classes, # 5,
                                             safe_burn_in = False, 
                                             data_sample = 'full',
                                             splitType='global',
                                             noise=False,
                                             ticker_dummies=False)

########### Splitting data ################

# Let's have a proper split (along tickers & dates)
train_size = 0.8
data_splits = pd.DataFrame()
data_splits = X.index.to_series().groupby(X['ticker']).agg(['first','last']).reset_index()

data_splits['val_size'] = ((1-train_size) * (data_splits['last'] - data_splits['first'])).astype(int)
data_splits['val_start_idx'] = data_splits['last'] - data_splits['val_size']
data_splits['val_end_idx'] = data_splits['last'] + 1 # to get the last observation included

data_splits['train_start_idx'] =  data_splits['first']
data_splits['train_end_idx'] = data_splits['val_start_idx']

# Store ranges
train_ranges = [list(x) for x in zip(data_splits['train_start_idx'], data_splits['train_end_idx'])]
val_ranges = [list(x) for x in zip(data_splits['val_start_idx'], data_splits['val_end_idx'])]

# Adding ticker dummies
tickers = X.pop('ticker')
X = pd.concat([X, pd.get_dummies(tickers, prefix='ticker', drop_first=False)], axis=1)


X_train = pd.concat([X.iloc[start:end, :] for (start, end) in train_ranges]).reset_index(drop=True).values
y_train = pd.concat([y.iloc[start:end] for (start, end) in train_ranges]).reset_index(drop=True).values.reshape(-1)

X_test = pd.concat([X.iloc[start:end, :] for (start, end) in val_ranges]).reset_index(drop=True).values
y_test = pd.concat([y.iloc[start:end] for (start, end) in val_ranges]).reset_index(drop=True).values.reshape(-1)




########### Pre-processing: none right now ################




print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]

n_inputs = X_train.shape[1] #28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 2 
n_epochs = 20
batch_size = args.batch_size
learning_rate = args.learning_rate

# y_train = one_hot_encode(y_train, n_outputs)
# y_test = one_hot_encode(y_test, n_outputs)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Build a simple MLP model
model = Sequential()
# first hidden layer
model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
# second hidden layer
model.add(Dense(n_h2, activation='relu'))
# output layer
model.add(Dense(n_outputs, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=['accuracy'])

# start an Azure ML run
# run = Run.get_context()


class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
#         run.log('Loss', log['val_loss'])
#         run.log('Accuracy', log['val_accuracy'])
        pass
#         print(log)
#         print(epoch, '-- Training accuracy:', log['accuracy'], '\b Validation accuracy:', log['val_accuracy'])

start_time = time.perf_counter()

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[LogRunMetrics()])

score = model.evaluate(X_test, y_test, verbose=0)

# log a single value
#run.log("Final test loss", score[0])
print('Test loss:', score[0])

#run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

# plt.figure(figsize=(6, 3))
# plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
# plt.plot(history.history['val_accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
# plt.plot(history.history['val_loss'], 'r--', label='Loss', lw=4, alpha=0.5)
# plt.legend(fontsize=12)
# plt.grid(True)

# log an image
#run.log_image('Accuracy vs Loss', plot=plt)

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

stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))


