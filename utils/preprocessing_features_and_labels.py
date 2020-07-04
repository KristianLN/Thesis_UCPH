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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
# Do you wanna see?

# Setting up the Scalers!
mm_scaler = MinMaxScaler()
scaler = StandardScaler()
norm_scaler = Normalizer()
pt = PowerTransformer()
ptNst = PowerTransformer(standardize=False)
qtUni = QuantileTransformer(n_quantiles=100)
qtGau = QuantileTransformer(n_quantiles=100,output_distribution='normal')



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

def pre_processing(rawData,ppDict,subBy,verbose=False):

    # Creating empty lists to hold the content of our pre-processing dictonary
    key = []
    item = []

    # Extracting the items of the pre-processing dictonary
    for k,i in ppDict.items():
        key.append(k)
        item.append(i)

    # Numping
    key = np.array(key)
    item = np.array(item)

    # Creating an empty dataframe to store the pre-processed data.
    preproX = pd.DataFrame()

    # Pre-processing the data according to the desired ways.
    for ele in np.unique(item):
        if verbose:
            print('Pre-Processing Procedure: ',ele)

        # Return the actual values
        if ele.lower() == 'act':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the raw feature to the new frame
            preproX[key[item==ele]] = rawData[key[item==ele]]

        # Return the actual values demeaned
        elif ele.lower() == 'actde':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the demeaned features to the new frame
    #         print(X[key[item==ele]].head())
    #         print(X[key[item==ele]].mean())
    #         print((X[key[item==ele]]-X[key[item==ele]].mean()).head())
            preproX[key[item==ele]] = rawData[key[item==ele]]-rawData[key[item==ele]].mean()

        # Return the features quantiale transformed (gaussian)
        elif ele.lower() == 'quantgau':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            preproX[key[item==ele]] = pd.DataFrame(qtGau.fit_transform(rawData[key[item==ele]].values))

        # Return the features standardized
        elif ele.lower() == 'std':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            preproX[key[item==ele]] = pd.DataFrame(scaler.fit_transform(rawData[key[item==ele]].values))

        # Return the features substracted a certain amount
        elif ele.lower() == 'sub':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            preproX[key[item==ele]] = rawData[key[item==ele]]-subBy

        # Return the features power transformed (standardized)
        elif ele.lower() == 'pow':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            preproX[key[item==ele]] = pd.DataFrame(pt.fit_transform(rawData[key[item==ele]].values))

        # Return the features min-max-normalised
        elif ele.lower() == 'minmax':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            preproX[key[item==ele]] = pd.DataFrame(mm_scaler.fit_transform(rawData[key[item==ele]].values))

        # Return the features norm scale
        elif ele.lower() == 'norm':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            preproX[key[item==ele]] = pd.DataFrame(norm_scaler.fit_transform(rawData[key[item==ele]].values))

    return preproX

def pre_processing_final(rawData_train,
                        rawData_test,
                        ppDict,
                        subBy,
                        verbose=False):

    # Creating empty lists to hold the content of our pre-processing dictonary
    key = []
    item = []

    # Extracting the items of the pre-processing dictonary
    for k,i in ppDict.items():
        key.append(k)
        item.append(i)

    # Numping
    key = np.array(key)
    item = np.array(item)

    # Creating an empty dataframe to store the pre-processed data.
    pp_train = pd.DataFrame()
    pp_test = pd.DataFrame()

    # Pre-processing the data according to the desired ways.
    for ele in np.unique(item):
        if verbose:
            print('Pre-Processing Procedure: ',ele)

        # Return the actual values
        if ele.lower() == 'act':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the raw feature to the new frame
            # preproX[key[item==ele]] = rawData[key[item==ele]]
            pp_train[key[item==ele]] = rawData_train[key[item==ele]]
            pp_test[key[item==ele]] = rawData_test[key[item==ele]]

        # Return the actual values demeaned
        elif ele.lower() == 'actde':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the demeaned features to the new frame
    #         print(X[key[item==ele]].head())
    #         print(X[key[item==ele]].mean())
    #         print((X[key[item==ele]]-X[key[item==ele]].mean()).head())
            # preproX[key[item==ele]] = rawData[key[item==ele]]-rawData[key[item==ele]].mean()
            pp_train[key[item==ele]] = rawData_train[key[item==ele]]-rawData_train[key[item==ele]].mean()
            pp_test[key[item==ele]] = rawData_test[key[item==ele]]-rawData_train[key[item==ele]].mean()

        # Return the features quantiale transformed (gaussian)
        elif ele.lower() == 'quantgau':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')


            # preproX[key[item==ele]] = pd.DataFrame(qtGau.fit_transform(rawData[key[item==ele]].values))
            # Adding the transformed features to the new frame
            qtGau.fit(rawData_train[key[item==ele]].values)
            pp_train[key[item==ele]] = pd.DataFrame(qtGau.transform(rawData_train[key[item==ele]].values))
            pp_test[key[item==ele]] = pd.DataFrame(qtGau.transform(rawData_test[key[item==ele]].values))

        elif ele.lower() == 'quantuni':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')


            # preproX[key[item==ele]] = pd.DataFrame(qtGau.fit_transform(rawData[key[item==ele]].values))
            # Adding the transformed features to the new frame
            qtUni.fit(rawData_train[key[item==ele]].values)
            pp_train[key[item==ele]] = pd.DataFrame(qtUni.transform(rawData_train[key[item==ele]].values))
            pp_test[key[item==ele]] = pd.DataFrame(qtUni.transform(rawData_test[key[item==ele]].values))

        # Return the features standardized
        elif ele.lower() == 'std':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            # preproX[key[item==ele]] = pd.DataFrame(scaler.fit_transform(rawData[key[item==ele]].values))
            scaler.fit(rawData_train[key[item==ele]].values)
            pp_train[key[item==ele]] = pd.DataFrame(scaler.transform(rawData_train[key[item==ele]].values))
            pp_test[key[item==ele]] = pd.DataFrame(scaler.transform(rawData_test[key[item==ele]].values))

        # Return the features substracted a certain amount
        elif ele.lower() == 'sub':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            # preproX[key[item==ele]] = rawData[key[item==ele]]-subBy
            pp_train[key[item==ele]] = rawData_train[key[item==ele]]-subBy
            pp_test[key[item==ele]] = rawData_test[key[item==ele]]-subBy

        # Return the features power transformed (standardized)
        elif ele.lower() == 'pow':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            # preproX[key[item==ele]] = pd.DataFrame(pt.fit_transform(rawData[key[item==ele]].values))
            pt.fit(rawData_train[key[item==ele]].values)
            pp_train[key[item==ele]] = pd.DataFrame(pt.transform(rawData_train[key[item==ele]].values))
            pp_test[key[item==ele]] = pd.DataFrame(pt.transform(rawData_test[key[item==ele]].values))

        # Return the features min-max-normalised
        elif ele.lower() == 'minmax':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            # preproX[key[item==ele]] = pd.DataFrame(mm_scaler.fit_transform(rawData[key[item==ele]].values))
            mm_scaler.fit(rawData_train[key[item==ele]].values)
            pp_train[key[item==ele]] = pd.DataFrame(mm_scaler.transform(rawData_train[key[item==ele]].values))
            pp_test[key[item==ele]] = pd.DataFrame(mm_scaler.transform(rawData_test[key[item==ele]].values))

        # Return the features norm scale
        elif ele.lower() == 'norm':
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            # preproX[key[item==ele]] = pd.DataFrame(norm_scaler.fit_transform(rawData[key[item==ele]].values))
            norm_scaler.fit(rawData_train[key[item==ele]].values)
            pp_train[key[item==ele]] = pd.DataFrame(norm_scaler.transform(rawData_train[key[item==ele]].values))
            pp_test[key[item==ele]] = pd.DataFrame(norm_scaler.transform(rawData_test[key[item==ele]].values))

    # Rearanging columns before we return it
    pp_train,pp_test = pp_train[rawData_train.columns],pp_test[rawData_test.columns]

    # Return preprocessed data
    return pp_train.reset_index(drop=True),pp_test.reset_index(drop=True)
