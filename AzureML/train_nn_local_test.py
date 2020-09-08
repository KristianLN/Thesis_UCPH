
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import argparse
import os
import re
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # turn off GPU usage
import tensorflow as tf
import time
import glob

# from azureml.core import Run
# from utils import load_data
from tensorflow.keras import Model, layers

from utils.generate_features import generateFeatures_multi_v2
from utils.preprocessing_features_and_labels import align_features_and_labels_multi_final

# config = tf.ConfigProto()
# sess = tf.Session(config=config)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First hidden layer.
        self.h1 = layers.Dense(n_h1, activation=tf.nn.relu)
        # Second hidden layer.
        self.h2 = layers.Dense(n_h2, activation=tf.nn.relu)
        self.out = layers.Dense(n_outputs)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        if not is_training:
            # Apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


def cross_entropy_loss(y, logits):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        logits = neural_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(y, logits)

    # Variables to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


print("TensorFlow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-source', type=str, dest='data_source', default='data', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=128,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=128,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
parser.add_argument('--resume-from', type=str, default=None,
                    help='location of the model or checkpoint files from where to resume the training')
args = parser.parse_args()

previous_model_location = args.resume_from
# You can also use environment variable to get the model/checkpoint files location
# previous_model_location = os.path.expandvars(os.getenv("AZUREML_DATAREFERENCE_MODEL_LOCATION", None))


# start an Azure ML run
#run = Run.get_context()

# get the input dataset by name
#dataset = run.input_datasets['aggregateTAQ_60sec'] # [data_source]

# data_folder = args.data_folder
# print('Data folder:', data_folder)

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
# X_train = load_data(glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'),
#                               recursive=True)[0], False) / np.float32(255.0)
# X_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'),
#                              recursive=True)[0], False) / np.float32(255.0)
# y_train = load_data(glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'),
#                               recursive=True)[0], True).reshape(-1)
# y_test = load_data(glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'),
#                              recursive=True)[0], True).reshape(-1)

### load train and test set into numpy arrays
# load the TabularDataset to pandas DataFrame
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

n_feature_lags = 0

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

#train_ds.shape, train_y.shape, validate_ds.shape, val_y.shape, train_y.shape[0] + val_y.shape[0]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

training_set_size = X_train.shape[0]


########### Pre-processing: none right now ################







#n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 2
learning_rate = args.learning_rate
n_epochs = 20
batch_size = args.batch_size

# Build neural network model.
neural_net = NeuralNet()

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate) #SGD(learning_rate)

# # start an Azure ML run
# run = Run.get_context()

if previous_model_location:
    # Restore variables from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)
    checkpoint_file_path = tf.train.latest_checkpoint(previous_model_location)
    checkpoint.restore(checkpoint_file_path)
    checkpoint_filename = os.path.basename(checkpoint_file_path)
    num_found = re.search(r'\d+', checkpoint_filename)
    if num_found:
        start_epoch = int(num_found.group(0))
        print("Resuming from epoch {}".format(str(start_epoch)))

start_time = time.perf_counter()
for epoch in range(0, n_epochs):

    # randomly shuffle training set
    #indices = np.random.permutation(training_set_size)
    #X_train = X_train[indices]
    #y_train = y_train[indices]

    # batch index
    b_start = 0
    b_end = b_start + batch_size
    for _ in range(training_set_size // batch_size):
        # get a batch
        X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]

        # update batch index for the next batch
        b_start = b_start + batch_size
        b_end = min(b_start + batch_size, training_set_size)

        # train
        run_optimization(X_batch, y_batch)

    # evaluate training set
    pred = neural_net(X_batch, is_training=False)
    acc_train = accuracy(pred, y_batch)

    # evaluate validation set
    pred = neural_net(X_test, is_training=False)
    acc_val = accuracy(pred, y_test)

    # log accuracies
    #run.log('training_acc', np.float(acc_train))
    #run.log('validation_acc', np.float(acc_val))
    print(epoch, '-- Training accuracy:', acc_train, '\b Validation accuracy:', acc_val)

    # Save checkpoints in the "./outputs" folder so that they are automatically uploaded into run history.
    checkpoint_dir = './outputs/'
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)

    if epoch % 2 == 0:
        checkpoint.save(checkpoint_dir)

#run.log('final_acc', np.float(acc_val))
os.makedirs('./outputs/model', exist_ok=True)

# files saved in the "./outputs" folder are automatically uploaded into run history
# this is workaround for https://github.com/tensorflow/tensorflow/issues/33913 and will be fixed once we move to >tf2.1
neural_net._set_inputs(X_train)
tf.saved_model.save(neural_net, './outputs/model/')

stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))
