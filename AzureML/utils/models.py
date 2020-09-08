import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import time
import h5py
import copy
import datetime
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
from IPython.display import clear_output


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

    def input_function():#data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds
    return input_function

def performanceTesting(X,Y, cv_folds,seed,ppDict,verbose):
    # ensure expected data type
    #X = X.values.astype(np.float32)
    #Y = Y.astype(np.int)#.values
    scoreHold = []
    # nested cross-validation, this is the outer fold initialization
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                            random_state=seed).split(X,Y)

    # store training and test accuracies, best parameters,
    # selected features, ensemble accuracies
    # total_train_acc = np.zeros(shape=(cv_folds, len(models)))
    # total_test_acc = np.zeros(shape=(cv_folds, len(models)))
    # total_best_params = np.zeros(shape=(cv_folds, len(models)),dtype=np.object)
    # total_features = np.zeros(shape=(cv_folds, len(models)),dtype=np.object)
    # total_ens_test_acc = np.zeros(shape=(cv_folds, 4)) # 4 different ensembles

    for counter, (train_index, test_index) in enumerate(kf):
        if verbose:
            print("Outer CV fold {}".format(counter))

    #     # create directory for output, if write_output is active
    #     if write_output:
    #         odir = os.path.join(outputdir, str(counter))
    #         if not os.path.exists(odir):
    #             os.makedirs(odir)
        #print(train_index)
        # get training/test splits
        X_train, X_test = X.loc[train_index,:].reset_index(drop=True),\
                            X.loc[test_index,:].reset_index(drop=True)

        Y_train, Y_test = Y[train_index], Y[test_index]

        # store predictions and errors within each outer fold
    #     total_pred_proba = np.zeros(shape=(len(test_index), len(models)))
    #     total_Y_pred = np.zeros(shape=(len(test_index), len(models)))
    #     total_Y_error_proba = np.zeros(shape=(len(test_index), len(models)))
    #     total_Y_error_class = np.zeros(shape=(len(test_index), len(models)))

    #     # store predictions on inner folds for training stacked ensembles
    #     total_inner_pred = np.zeros(shape=(len(train_index), len(models)))
    #     total_inner_pred_proba = np.zeros(shape=(len(train_index), len(models)))

        # initialize inner kf
    #     inner_kf = cross_validation.StratifiedKFold(Y_train, n_folds=cv_folds,
    #                                       shuffle=True, random_state = seed)
        #print(X_train.shape,X_test.shape)
        # perform pre-processing on current outer training and test set
        X_train, X_test = pre_processing(X_train, X_test, ppDict,100)
        #print(X_train.dtypes,'\n')
        #print(X_test.dtypes,'\n')
#         print(np.sum(np.sum(X_train.isnull(),axis=None)),
#               np.sum(np.sum(X_train==np.inf,axis=None)))
#         print(np.sum(np.sum(X_test.isnull(),axis=None)),
#               np.sum(np.sum(X_test==np.inf,axis=None)))
#         print(np.sum(X_train.isnull(),axis=None),
#               np.sum(X_train==np.inf,axis=None))
#         print(np.sum(X_test.isnull(),axis=None),
#               np.sum(X_test==np.inf,axis=None))

        clf = LogisticRegression(random_state=0,
                                 max_iter=1000,
                                 solver='lbfgs',
                                 C = 0.1).fit(X_train, Y_train)
#         clf.predict(X[:2, :])

        scoreHold.append(clf.score(X_test, Y_test))
    print('Scores: ',scoreHold)
    print('Average Score: %.3f (%.3f)'% (np.mean(scoreHold),np.std(scoreHold)))

def scoreFunction(testData,predData,scoreToReturn, scoresToPrint = []):

    individualPredictions = pd.DataFrame([pred['probabilities'] for pred in predData])
    individualClasses = pd.DataFrame([pred['classes'] for pred in predData]).astype(int)

    scoreDict = \
                {'roc_auc_ovr': roc_auc_score(testData.astype(int),
                                              individualPredictions,
                                              multi_class='ovr' # 'ovo'/'ovr'
                                             ),
                 'roc_auc_ovo': roc_auc_score(testData.astype(int),
                                              individualPredictions,
                                              multi_class='ovo' # 'ovo'/'ovr'
                                             ),
                 'f1_score': f1_score(testData.astype(int),
                                      individualClasses,
                                      average='macro',
                                     ),
                 'log_loss': log_loss(testData.astype(int),
                                      individualPredictions,
                                     ),
                }
    if len(scoresToPrint) > 0 :
        # Print the desired scores
        for score in scoresToPrint:
            print(score + ': %.5f' % scoreDict[score])

    return round(scoreDict[scoreToReturn],5)
