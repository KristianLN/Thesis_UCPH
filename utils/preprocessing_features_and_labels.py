import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import time
import h5py
import copy
import datetime
import ta

################################################## Table of Content ##################################################
# extract_labels() : extract labels given classes and group_style (we have only equal: 5 x 20% bins right now)
# align_features_and_labels(): burn-in features, extract labels (calls extract_labels()) and align indices to features

######################################################################################################################

def extract_labels(data = '', classes = 5, group_style = 'equal'):
    
   # returns = ((data.T[-1][1:]/data.T[-1][0:-1])-1)*100
    returns = ((data[1:, -1] / data[:-1, -1])-1)*100
    
    labels = np.zeros(returns.shape[0])

    if group_style == 'equal':
        thresholdsMin = [np.array_split(np.sort(returns),classes)[i].min() for i in np.arange(classes)]
        thresholdsMax = [np.array_split(np.sort(returns),classes)[i].max() for i in np.arange(classes)]    
    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')
    
    for i in np.arange(classes):
        if i == 0:
            labels[(returns <= thresholdsMax[i])] = i

        elif i == (classes-1):
            labels[(returns >= thresholdsMin[i])] = i

        else:
            labels[(returns >= thresholdsMin[i]) & (returns<=thresholdsMax[i])] = i  
            
    return labels, returns, [thresholdsMin, thresholdsMax]


def align_features_and_labels(candles, prediction_horizon, features, n_feature_lags, n_classes,
                              safe_burn_in = False, data_sample = 'full'):
    
    if not safe_burn_in:
        assert data_sample == 'full'
        # we assume data_sample is full and that we can continue features from yesterday's values.
        # that we have a single burn-in at the beginning and that's it

        # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
        burned_in_idx = np.where((np.sum(np.isnan(features.values), axis=1) == 0) == True)[0][0]

        # calculate end-point cut-off to match with labels
        end_point_cut = max(prediction_horizon, n_feature_lags + 1)

        # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
        burned_in_features = features.iloc[burned_in_idx : -end_point_cut, :].reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

        # slice away the burned-in indices from labels
        labels, _, _ = extract_labels(data = candles[burned_in_idx+n_feature_lags:, :], 
                                      classes = n_classes, group_style = 'equal')
        # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :], 
        #                                             classes = n_classes, group_style = 'equal')

        # check if there are remaining NaNs are burn-in (means error)
        remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
        if remaining_nans > 0:
            raise ValueError('Had NaN in burned_in_features after burn-in')   
            
    return burned_in_features, labels # call the function as X, y = align_features_and_labels(.) if you like