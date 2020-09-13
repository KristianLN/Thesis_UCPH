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
# extract_labels_multi() : extract labels over all tickers given classes and group_style (we have only equal: 5 x 20% bins right now)
# align_features_and_labels(): burn-in features, extract labels (calls extract_labels()) and align indices to features
# align_features_and_labels_multi(): burn-in features, extract labels (calls extract_labels_multi()) and align indices to features over all tickers


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


# _multi works with align_features_and_labels_multi() and assumes data is a vector of close prices (not a matrix)
def extract_labels_multi_v1(data = None, classes = 5, group_style = 'equal'):

   # returns = ((data.T[-1][1:]/data.T[-1][0:-1])-1)*100
    returns = ((data[1:] / data[:-1]) -1) * 100

    labels = np.zeros(returns.shape[0])

    if group_style == 'equal':
        thresholdsMin = [np.array_split(np.sort(returns), classes)[i].min() for i in np.arange(classes)]
        thresholdsMax = [np.array_split(np.sort(returns), classes)[i].max() for i in np.arange(classes)]
    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')

    for i in np.arange(classes):
        if i == 0:
            labels[(returns <= thresholdsMax[i])] = i

        elif i == (classes-1):
            labels[(returns >= thresholdsMin[i])] = i

        else:
            labels[(returns >= thresholdsMin[i]) & (returns<=thresholdsMax[i])] = i

    return labels #, returns, [thresholdsMin, thresholdsMax]

def extract_labels_multi_v2(data = None,
                        classes = 5,
                        group_style = 'equal',
                        splits=None):

# attempts on 13-08-2020 to fix label issue (this version is not fully tested / might not work correctly)
   # returns = ((data.T[-1][1:]/data.T[-1][0:-1])-1)*100
    returns = ((data[1:] / data[:-1]) -1) * 100
    # If returns are exact zero, perhaps because there hasn't been any price updates over a candle, we add a little bit of noise, to ensure that the labels are evenly distributed.
    returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

    labels = np.zeros(returns.shape[0])

    if group_style == 'equal':
        if splits is None:
            splits = np.array_split(np.sort(returns),classes)

        for i in np.arange(classes):

            labels[np.isin(returns,splits[i])] = i

    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')

    return labels #, returns, [thresholdsMin, thresholdsMax]

# attempts on 13-08-2020 to fix label issue (this version is not fully tested / might not work correctly)
def extract_labels_multi_v3(data = None,
                        classes = 5,
                        group_style = 'equal',
                        splits=None,noise=True):

   # returns = ((data.T[-1][1:]/data.T[-1][0:-1])-1)
    returns = ((data[1:] / data[:-1]) -1)
    # If returns are exact zero, perhaps because there hasn't been any price updates over a candle, we add a little bit of noise, to ensure that the labels are evenly distributed.
    if noise:
        returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

    labels = np.zeros(returns.shape[0])

    if group_style == 'equal':
        if splits is None:
            splits = np.array_split(np.sort(returns),classes)

        for i in np.arange(classes):

            labels[np.isin(returns,splits[i])] = i

    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')

    return labels #, returns, [thresholdsMin, thresholdsMax]


## Works only for two classes
# attempts on 13-08-2020 to fix label issue (this version is not fully tested / might not work correctly)
def extract_labels_multi_v4(data = None,
                        classes = 5,
                        group_style = 'equal',
                        global_median=None):

   # returns = ((data.T[-1][1:]/data.T[-1][0:-1])-1)
    returns = ((data[1:] / data[:-1]) -1)
    # If returns are exact zero, perhaps because there hasn't been any price updates over a candle, we add a little bit of noise, to ensure that the labels are evenly distributed.
    # if noise:
        # returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

    labels = np.zeros(returns.shape[0])

    if group_style == 'equal':
        # if splits is None:
            # splits = np.array_split(np.sort(returns),classes)

        # for i in np.arange(classes):

        labels[returns>global_median] = 1
        labels[returns<=global_median] = 0

    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')

    return labels #, returns, [thresholdsMin, thresholdsMax]


## Works for all number of classes
# this version takes data in a direct returns for a specific ticker
def extract_labels_multi_v5(data = None,
                        classes = 5,
                        group_style = 'equal',
                        splits=None):

    if group_style == 'equal':

        labels = pd.cut(data, bins=splits, labels=False, right=False, include_lowest=True)

        # we need right=False (open right-handside in split interval) to get median into the positive class
        # this makes the last point nan, we fix it here
        if sum(np.isnan(labels)) > 0:
            print(f'Number of NaNs in label: {sum(np.isnan(labels))}. 1 is expected')
            print(f'Returns that lead to NaNs in label: {data[np.where(np.isnan(labels))]}')
            assert sum(np.isnan(labels)) <= 1, "There should be max 1 NaN"

            if data[np.where(np.isnan(labels))] >= splits[-1]:
                labels[np.where(np.isnan(labels))] = classes - 1 # assign last label id
            else:
                print(data[np.where(np.isnan(labels))], splits[-1])
                raise ValueError('There is a label NaN where its underlying return is not max of dataset, which it should be')

    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')

    return labels


## v5
def extract_labels_multi_final(data = None,
                                classes = 5,
                                group_style = 'equal',
                                splits=None):

    # this version takes data in a direct returns for a specific ticker

    if group_style == 'equal':
        # if splits is None:
            # splits = np.array_split(np.sort(returns),classes)

        # for i in np.arange(classes):

        #labels[returns > global_median] = 1
        #labels[returns <= global_median] = 0

        labels = pd.cut(data, bins=splits, labels=False, right=False, include_lowest=True)

        # we need right=False (open right-handside in split interval) to get median into the positive class
        # this makes the last point nan, we fix it here
        if sum(np.isnan(labels)) > 0:
            print(f'Number of NaNs in label: {sum(np.isnan(labels))}. 1 is expected')
            print(f'Returns that lead to NaNs in label: {data[np.where(np.isnan(labels))]}')
            assert sum(np.isnan(labels)) <= 1, "There should be max 1 NaN"

            if data[np.where(np.isnan(labels))] >= splits[-1]:
                labels[np.where(np.isnan(labels))] = classes - 1 # assign last label id
            else:
                print(data[np.where(np.isnan(labels))], splits[-1])
                raise ValueError('There is a label NaN where its underlying return is not max of dataset, which it should be')

    elif group_style != 'equal':
        raise ValueError(f'group_style {group_style} not implemented')

    return labels


def align_features_and_labels(candles, prediction_horizon, features, n_feature_lags, n_classes,
                              safe_burn_in = False, data_sample = 'full'):

    # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
    price_candles = candles.iloc[:, :4].values

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
        labels, _, _ = extract_labels(data = price_candles[burned_in_idx+n_feature_lags:, :],
                                      classes = n_classes, group_style = 'equal')
        # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
        #                                             classes = n_classes, group_style = 'equal')

        # check if there are remaining NaNs are burn-in (means error)
        remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
        if remaining_nans > 0:
            raise ValueError('Had NaN in burned_in_features after burn-in')

    return burned_in_features, labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like


# _multi has multi-ticker support
def align_features_and_labels_multi_v1(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full'):

    all_burned_in_features = pd.DataFrame()
    all_labels = pd.DataFrame()

    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):

        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]

            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

            # slice away the burned-in indices from labels
            labels = extract_labels_multi(data = ticker_prices[(burned_in_idx+n_feature_lags):],
                                          classes = n_classes,
                                          group_style = 'equal')
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        burned_in_features['ticker'] = ticker_name
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    return all_burned_in_features, all_labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like

# attempts on 13-08-2020 to fix label issue (this version is not fully tested / might not work correctly)
def align_features_and_labels_multi_v2(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full',
                                        splitType='global'):

    all_burned_in_features = pd.DataFrame()
    all_labels = pd.DataFrame()
    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
        splits = np.array_split(np.sort(returns),n_classes)

    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):

        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]

            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

            # slice away the burned-in indices from labels
            labels = extract_labels_multi(data = ticker_prices[(burned_in_idx+n_feature_lags):],
                                          classes = n_classes,
                                          group_style = 'equal',
                                          splits = splits)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        burned_in_features['ticker'] = ticker_name
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    return all_burned_in_features, all_labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like

# attempts on 13-08-2020 to fix label issue (this version is not fully tested / might not work correctly)
def align_features_and_labels_multi_v3(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full',
                                        splitType='global',
                                        noise = True):

    all_burned_in_features = pd.DataFrame()
    all_labels = pd.DataFrame()
    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        # returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
        returns = np.concatenate([((price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
                         price_candles[price_candles.Ticker==ticker]['close'].values[:-1])-1) for ticker\
                          in price_candles.Ticker.unique()])
        if noise:
            returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

        splits = np.array_split(np.sort(returns),n_classes)

    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):

        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]

            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

            # slice away the burned-in indices from labels
            labels = extract_labels_multi(data = ticker_prices[(burned_in_idx+n_feature_lags):],
                                          classes = n_classes,
                                          group_style = 'equal',
                                          splits = splits,noise=noise)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        burned_in_features['ticker'] = ticker_name
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    return all_burned_in_features, all_labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like

# attempts on 13-08-2020 to fix label issue (this version is not fully tested / might not work correctly)
def align_features_and_labels_multi_v4(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full',
                                        splitType='global',
                                        noise = True):

    all_burned_in_features = pd.DataFrame()
    all_labels = pd.DataFrame()
    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        # returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
        returns = np.concatenate([((price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
                         price_candles[price_candles.Ticker==ticker]['close'].values[:-1])-1) for ticker\
                          in price_candles.Ticker.unique()])
        if noise:
            returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

        # splits = np.array_split(np.sort(returns),n_classes)
        global_median = np.median(returns)
    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):

        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]

            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

            # slice away the burned-in indices from labels
            labels = extract_labels_multi(data = ticker_prices[(burned_in_idx+n_feature_lags):],
                                          classes = n_classes,
                                          group_style = 'equal',
                                          global_median = global_median)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        burned_in_features['ticker'] = ticker_name
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    return all_burned_in_features, all_labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like


# this version calculates global returns, add ticker to the output, and inputs direct ticker-wise returns for extract label
def align_features_and_labels_multi_v5(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full',
                                        splitType='global',
                                        noise = False):

    all_burned_in_features = pd.DataFrame()
    all_labels = pd.DataFrame()

    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        # returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
#         returns = np.concatenate([((price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
#                          price_candles[price_candles.Ticker==ticker]['close'].values[:-1])-1) for ticker\
#                           in price_candles.Ticker.unique()])

        returns = []
        tickers = []
        for ticker in price_candles.Ticker.unique():

            ticker_returns = (price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
                                 price_candles[price_candles.Ticker==ticker]['close'].values[:-1]) - 1
            ticker_names = [ticker for i in range(len(ticker_returns))]

            returns.append(ticker_returns)
            tickers.append(ticker_names)

        # concatenate returns and add noise
        returns = np.concatenate(returns)
        if noise:
            returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

        tickers = np.concatenate(tickers)

        _, splits = pd.qcut(returns, q=n_classes, labels=False, retbins=True)
        #print(splits)

        returns = pd.DataFrame({'returns': returns, 'Ticker': tickers})



    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):
        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        #ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values
        ticker_returns = returns[returns.Ticker==ticker_name]['returns'].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]

            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

            # slice away the burned-in indices from labels
            labels = extract_labels_multi_final(data = ticker_returns[(burned_in_idx+n_feature_lags):],
                                                classes = n_classes,
                                                group_style = 'equal',
                                                splits = splits)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        burned_in_features['ticker'] = ticker_name
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    return all_burned_in_features, all_labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like


# Included the option to return ticker label as dummies.
def align_features_and_labels_multi_v6(price_candles,
                                            all_features,
                                            prediction_horizon,
                                            n_feature_lags,
                                            n_classes,
                                            safe_burn_in = False,
                                            data_sample = 'full',
                                            splitType='global',
                                            noise = False,
                                            ticker_dummies = None):

    all_burned_in_features = pd.DataFrame()
    all_labels = pd.DataFrame()

    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        # returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
#         returns = np.concatenate([((price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
#                          price_candles[price_candles.Ticker==ticker]['close'].values[:-1])-1) for ticker\
#                           in price_candles.Ticker.unique()])

        returns = []
        tickers = []
        for ticker in price_candles.Ticker.unique():

            ticker_returns = (price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
                                 price_candles[price_candles.Ticker==ticker]['close'].values[:-1]) - 1
            ticker_names = [ticker for i in range(len(ticker_returns))]

            returns.append(ticker_returns)
            tickers.append(ticker_names)

        # concatenate returns and add noise
        returns = np.concatenate(returns)
        if noise:
            returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

        tickers = np.concatenate(tickers)

        _, splits = pd.qcut(returns, q=n_classes, labels=False, retbins=True)
        #print(splits)

        returns = pd.DataFrame({'returns': returns, 'Ticker': tickers})



    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):
        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        #ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values
        ticker_returns = returns[returns.Ticker==ticker_name]['returns'].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]

            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise

            # slice away the burned-in indices from labels
            labels = extract_labels_multi_final(data = ticker_returns[(burned_in_idx+n_feature_lags):],
                                                classes = n_classes,
                                                group_style = 'equal',
                                                splits = splits)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        # Adding the ticker
        # burned_in_features['ticker'] = ticker_name
        burned_in_features.loc[:,'ticker'] = ticker_name

        # Returning the ticker as dummies
        if ticker_dummies is not None:

            tickers = burned_in_features.pop('ticker')
            burned_in_features = pd.concat([burned_in_features, pd.get_dummies(tickers, prefix='ticker', drop_first=False)], axis=1)
        # Adding the burned in data
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features.reset_index(drop=True)])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

        return all_burned_in_features.reset_index(drop=True), all_labels.reset_index(drop=True) # call the function as X, y = align_features_and_labels(.) if you like

# Now extracting the time indices to be used to sort afterwards.
def align_features_and_labels_multi_v7(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full',
                                        splitType='global',
                                        noise = False,
                                        ticker_dummies = False):

    all_burned_in_features = pd.DataFrame()
    all_burned_in_indices = pd.DataFrame()
    all_labels = pd.DataFrame()

    dailyIndices = pd.DataFrame({'days':price_candles.index.get_level_values(0),
                                      'timestemps':price_candles.index.get_level_values(1),
                                      'ticker':price_candles.Ticker})

    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        # returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
#         returns = np.concatenate([((price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
#                          price_candles[price_candles.Ticker==ticker]['close'].values[:-1])-1) for ticker\
#                           in price_candles.Ticker.unique()])

        returns = []
        tickers = []

        for ticker in price_candles.Ticker.unique():

            ticker_returns = (price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
                                 price_candles[price_candles.Ticker==ticker]['close'].values[:-1]) - 1
            ticker_names = [ticker for i in range(len(ticker_returns))]

            returns.append(ticker_returns)
            tickers.append(ticker_names)

        # concatenate returns and add noise
        returns = np.concatenate(returns)
        if noise:
            returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

        tickers = np.concatenate(tickers)

        _, splits = pd.qcut(returns, q=n_classes, labels=False, retbins=True)
        #print(splits)

        returns = pd.DataFrame({'returns': returns, 'Ticker': tickers})

    keepCheck = []

    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):
        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        ticker_indices = dailyIndices[dailyIndices.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        #ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values
        ticker_returns = returns[returns.Ticker==ticker_name]['returns'].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]
            keepCheck.append(burned_in_idx)
            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise
            burned_in_indices = ticker_indices.iloc[burned_in_idx : -end_point_cut, :]
            # slice away the burned-in indices from labels
            labels = extract_labels_multi_final(data = ticker_returns[(burned_in_idx+n_feature_lags):],
                                                classes = n_classes,
                                                group_style = 'equal',
                                                splits = splits)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        # Adding the ticker
        burned_in_features.loc[:,'ticker'] = ticker_name

        # Adding the burned in data
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features.reset_index(drop=True)])
        all_burned_in_indices = pd.concat([all_burned_in_indices, burned_in_indices.reset_index(drop=True)])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    # Returning the ticker as dummies
    if ticker_dummies:

        tickers = all_burned_in_features.pop('ticker')
        all_burned_in_features = pd.concat([all_burned_in_features, pd.get_dummies(tickers, prefix='d_ticker', drop_first=False)], axis=1)
#     print('Are all burned_in_idx the same?', all(keepCheck==keepCheck[0]))
#     print(dailyIndicies.head(50))
    return all_burned_in_features.reset_index(drop=True),\
            all_labels.reset_index(drop=True),\
            all_burned_in_indices.reset_index(drop=True)

# v7
def align_features_and_labels_multi_final(price_candles,
                                        all_features,
                                        prediction_horizon,
                                        n_feature_lags,
                                        n_classes,
                                        safe_burn_in = False,
                                        data_sample = 'full',
                                        splitType='global',
                                        noise = False,
                                        ticker_dummies = False):

    all_burned_in_features = pd.DataFrame()
    all_burned_in_indices = pd.DataFrame()
    all_labels = pd.DataFrame()

    dailyIndices = pd.DataFrame({'days':price_candles.index.get_level_values(0),
                                      'timestemps':price_candles.index.get_level_values(1),
                                      'ticker':price_candles.Ticker})

    if splitType.lower() == 'global':
        # Making the splits for the labels based on all tickers
        # returns = ((price_candles['close'].values[1:] / price_candles['close'].values[:-1]) -1) * 100
#         returns = np.concatenate([((price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
#                          price_candles[price_candles.Ticker==ticker]['close'].values[:-1])-1) for ticker\
#                           in price_candles.Ticker.unique()])

        returns = []
        tickers = []

        for ticker in price_candles.Ticker.unique():

            ticker_returns = (price_candles[price_candles.Ticker==ticker]['close'].values[1:]/\
                                 price_candles[price_candles.Ticker==ticker]['close'].values[:-1]) - 1
            ticker_names = [ticker for i in range(len(ticker_returns))]

            returns.append(ticker_returns)
            tickers.append(ticker_names)

        # concatenate returns and add noise
        returns = np.concatenate(returns)
        if noise:
            returns[returns==0] = np.random.normal(0,1,sum(returns==0))/1000000

        tickers = np.concatenate(tickers)

        _, splits = pd.qcut(returns, q=n_classes, labels=False, retbins=True)
        #print(splits)

        returns = pd.DataFrame({'returns': returns, 'Ticker': tickers})

    keepCheck = []

    for ticker_iter, ticker_name in enumerate(all_features.ticker.unique()):
        ticker_features = all_features[all_features.ticker==ticker_name].copy(deep=True)
        ticker_indices = dailyIndices[dailyIndices.ticker==ticker_name].copy(deep=True)
        # removing the "ticker" variable from ticker_features as np.isnan() does not like non-numericals
        #ticker_features = ticker_features.iloc[:, ticker_features.columns != 'ticker']
        ticker_features.drop('ticker', axis=1, inplace=True)
        # extract first 4 columns as the lag0 or raw OHLC prices (used for labelling)
        #ticker_prices = price_candles[price_candles.Ticker==ticker_name]['close'].values # candles.iloc[:, :4].values
        ticker_returns = returns[returns.Ticker==ticker_name]['returns'].values

        if not safe_burn_in:
            assert data_sample == 'full'
            # we assume data_sample is full and that we can continue features from yesterday's values.
            # that we have a single burn-in at the beginning and that's it

            # get first index that has no NaNs (the sum checks for True across columns, we look for sum == 0 and where that is first True)
            burned_in_idx = np.where((np.sum(np.isnan(ticker_features.values), axis=1) == 0) == True)[0][0]
            keepCheck.append(burned_in_idx)
            # calculate end-point cut-off to match with labels
            end_point_cut = max(prediction_horizon, n_feature_lags + 1)

            # slice away the observations used for burn-in (taking off 1 at the end to match with labels [slice off "prediction_horizon"])
            burned_in_features = ticker_features.iloc[burned_in_idx : -end_point_cut, :] #.reset_index(drop=True) # features[burned_in_idx:] latter is sligthly faster but maybe not as precise
            burned_in_indices = ticker_indices.iloc[burned_in_idx : -end_point_cut, :]
            # slice away the burned-in indices from labels
            labels = extract_labels_multi_final(data = ticker_returns[(burned_in_idx+n_feature_lags):],
                                                classes = n_classes,
                                                group_style = 'equal',
                                                splits = splits)
            # labels, returns, thresholds = extract_labels(data = candles[burned_in_idx + n_feature_lags : , :],
            #                                             classes = n_classes, group_style = 'equal')

            # check if there are remaining NaNs are burn-in (means error)
            remaining_nans = np.where(np.isnan(burned_in_features.values))[0].size
            if remaining_nans > 0:
                raise ValueError('Had NaN in burned_in_features after burn-in')

        # Adding the ticker
        burned_in_features.loc[:,'ticker'] = ticker_name

        # Adding the burned in data
        all_burned_in_features = pd.concat([all_burned_in_features, burned_in_features.reset_index(drop=True)])
        all_burned_in_indices = pd.concat([all_burned_in_indices, burned_in_indices.reset_index(drop=True)])
        all_labels = pd.concat([all_labels, pd.Series(labels)])
        print(ticker_name + " done")

    # Returning the ticker as dummies
    if ticker_dummies:

        tickers = all_burned_in_features.pop('ticker')
        all_burned_in_features = pd.concat([all_burned_in_features, pd.get_dummies(tickers, prefix='d_ticker', drop_first=False)], axis=1)
#     print('Are all burned_in_idx the same?', all(keepCheck==keepCheck[0]))
#     print(dailyIndicies.head(50))
    return all_burned_in_features.reset_index(drop=True),\
            all_labels.reset_index(drop=True),\
            all_burned_in_indices.reset_index(drop=True)


def pre_processing_initial(rawData,ppDict,subBy,verbose=False):

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

def pre_processing_extended(rawData_train,
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

def pre_processing_v1(rawData_train,
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

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the raw feature to the new frame
            pp_train[cols] = rawData_train[cols]
            pp_test[cols] = rawData_test[cols]

        # Return the actual values demeaned
        elif ele.lower() == 'actde':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the demeaned features to the new frame
            pp_train[cols] = rawData_train[cols]-rawData_train[cols].mean()
            pp_test[cols] = rawData_test[cols]-rawData_train[cols].mean()

        # Return the features quantiale transformed (gaussian)
        elif ele.lower() == 'quantgau':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            qtGau.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(qtGau.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(qtGau.transform(rawData_test[cols].values))

        elif ele.lower() == 'quantuni':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            qtUni.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(qtUni.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(qtUni.transform(rawData_test[cols].values))

        # Return the features standardized
        elif ele.lower() == 'std':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            scaler.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(scaler.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(scaler.transform(rawData_test[cols].values))

        # Return the features substracted a certain amount
        elif ele.lower() == 'sub':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pp_train[cols] = rawData_train[cols]-subBy
            pp_test[cols] = rawData_test[cols]-subBy

        elif ele.lower() == 'log':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pp_train[cols] = np.log(rawData_train[cols])
            pp_test[cols] = np.log(rawData_test[cols])

        # Return the features power transformed (standardized)
        elif ele.lower() == 'pow':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pt.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(pt.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(pt.transform(rawData_test[cols].values))

        # Return the features min-max-normalised
        elif ele.lower() == 'minmax':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            mm_scaler.fit(rawData_train[cols].values) if len(cols) > 1 else mm_scaler.fit(rawData_train[cols].values.reshape(-1,1))
            pp_train[cols] = pd.DataFrame(mm_scaler.transform(rawData_train[cols].values)) if len(cols) > 1 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
            pp_test[cols] = pd.DataFrame(mm_scaler.transform(rawData_test[cols].values)) if len(cols) > 1 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))

        # Return the features norm scale
        elif ele.lower() == 'norm':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            norm_scaler.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(norm_scaler.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(norm_scaler.transform(rawData_test[cols].values))

    # Rearanging columns before we return it
    pp_train,pp_test = pp_train[rawData_train.columns],pp_test[rawData_test.columns]

    # Return preprocessed data
    return pp_train.reset_index(drop=True),pp_test.reset_index(drop=True)

def pre_processing_v2(rawData_train,
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

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the raw feature to the new frame
            pp_train[cols] = rawData_train[cols]
            pp_test[cols] = rawData_test[cols]

        # Return the actual values demeaned
        elif ele.lower() == 'actde':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the demeaned features to the new frame
            pp_train[cols] = rawData_train[cols]-rawData_train[cols].mean()
            pp_test[cols] = rawData_test[cols]-rawData_train[cols].mean()

        # Return the features quantiale transformed (gaussian)
        elif ele.lower() == 'quantgau':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            qtGau.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(qtGau.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(qtGau.transform(rawData_test[cols].values))

        elif ele.lower() == 'quantuni':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            qtUni.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(qtUni.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(qtUni.transform(rawData_test[cols].values))

        # Return the features standardized
        elif ele.lower() == 'std':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            scaler.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(scaler.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(scaler.transform(rawData_test[cols].values))

        # Return the features substracted a certain amount
        elif ele.lower() == 'sub':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pp_train[cols] = rawData_train[cols]-subBy
            pp_test[cols] = rawData_test[cols]-subBy

        elif ele.lower() == 'log':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pp_train[cols] = np.log(rawData_train[cols])
            pp_test[cols] = np.log(rawData_test[cols])

        # Return the features power transformed (standardized)
        elif ele.lower() == 'pow':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if (t in c)] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pt.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(pt.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(pt.transform(rawData_test[cols].values))

        # Return the features min-max-normalised
        elif ele.lower() == 'minmax':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)
#             print(cols)
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')
#             print(rawData_train[cols].values)
#             print(rawData_train[cols].values.shape)
            #print(rawData_train[cols].values.reshape(-1,2))
            #print(rawData_train[cols].values.reshape(-1,2).shape)
            # Adding the transformed features to the new frame
            mm_scaler.fit(rawData_train[cols].values)# if len(cols) > 10 else mm_scaler.fit(rawData_train[cols].values.reshape(-1,1))
#             print(pd.DataFrame(mm_scaler.transform(rawData_train[cols].values)))
            pp_train[cols] = pd.DataFrame(mm_scaler.transform(rawData_train[cols].values))#,columns=cols# if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
#             print(pp_train)
#             print(pd.DataFrame(mm_scaler.transform(rawData_test[cols].values)))
            pp_test[cols] = pd.DataFrame(mm_scaler.transform(rawData_test[cols].values))#,columns = cols#if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))
#             print(pp_test)
#             pp_train[cols] = mm_scaler.transform(rawData_train[cols].values)# if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
#             pp_test[cols] = mm_scaler.transform(rawData_test[cols].values)# if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))

#             mm_scaler.fit(rawData_train[cols].values) if len(cols) > 1 else mm_scaler.fit(rawData_train[cols].values.reshape(-1,1))
#             pp_train[cols] = pd.DataFrame(mm_scaler.transform(rawData_train[cols].values)) if len(cols) > 1 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
#             pp_test[cols] = pd.DataFrame(mm_scaler.transform(rawData_test[cols].values)) if len(cols) > 1 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))

        # Return the features norm scale
        elif ele.lower() == 'norm':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            norm_scaler.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(norm_scaler.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(norm_scaler.transform(rawData_test[cols].values))

    # Rearanging columns before we return it
    pp_train,pp_test = pp_train[rawData_train.columns],pp_test[rawData_test.columns]

    # Return preprocessed data

    return pp_train.reset_index(drop=True),pp_test.reset_index(drop=True)

def pre_processing(rawData_train,
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

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the raw feature to the new frame
            pp_train[cols] = rawData_train[cols]
            pp_test[cols] = rawData_test[cols]

        # Return the actual values demeaned
        elif ele.lower() == 'actde':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the demeaned features to the new frame
            pp_train[cols] = rawData_train[cols]-rawData_train[cols].mean()
            pp_test[cols] = rawData_test[cols]-rawData_train[cols].mean()

        # Return the features quantiale transformed (gaussian)
        elif ele.lower() == 'quantgau':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            qtGau.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(qtGau.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(qtGau.transform(rawData_test[cols].values))

        elif ele.lower() == 'quantuni':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            qtUni.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(qtUni.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(qtUni.transform(rawData_test[cols].values))

        # Return the features standardized
        elif ele.lower() == 'std':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            scaler.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(scaler.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(scaler.transform(rawData_test[cols].values))

        # Return the features substracted a certain amount
        elif ele.lower() == 'sub':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pp_train[cols] = rawData_train[cols]-subBy
            pp_test[cols] = rawData_test[cols]-subBy

        elif ele.lower() == 'log':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pp_train[cols] = np.log(rawData_train[cols])
            pp_test[cols] = np.log(rawData_test[cols])

        # Return the features power transformed (standardized)
        elif ele.lower() == 'pow':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if (t in c)] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            pt.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(pt.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(pt.transform(rawData_test[cols].values))

        # Return the features min-max-normalised
        elif ele.lower() == 'minmax':

            # Account for lags and preprocess all lags the same way
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)
#             print(cols)
            if verbose:
                print('Columns Processed:',key[item==ele],'\n')
#             print(rawData_train[cols].values)
#             print(rawData_train[cols].values.shape)
            #print(rawData_train[cols].values.reshape(-1,2))
            #print(rawData_train[cols].values.reshape(-1,2).shape)
            # Adding the transformed features to the new frame
            mm_scaler.fit(rawData_train[cols].values)# if len(cols) > 10 else mm_scaler.fit(rawData_train[cols].values.reshape(-1,1))
#             print(pd.DataFrame(mm_scaler.transform(rawData_train[cols].values)))
            pp_train[cols] = pd.DataFrame(mm_scaler.transform(rawData_train[cols].values))#,columns=cols# if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
#             print(pp_train)
#             print(pd.DataFrame(mm_scaler.transform(rawData_test[cols].values)))
            pp_test[cols] = pd.DataFrame(mm_scaler.transform(rawData_test[cols].values))#,columns = cols#if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))
#             print(pp_test)
#             pp_train[cols] = mm_scaler.transform(rawData_train[cols].values)# if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
#             pp_test[cols] = mm_scaler.transform(rawData_test[cols].values)# if len(cols) > 10 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))

#             mm_scaler.fit(rawData_train[cols].values) if len(cols) > 1 else mm_scaler.fit(rawData_train[cols].values.reshape(-1,1))
#             pp_train[cols] = pd.DataFrame(mm_scaler.transform(rawData_train[cols].values)) if len(cols) > 1 else pd.DataFrame(mm_scaler.transform(rawData_train[cols].values.reshape(-1,1)))
#             pp_test[cols] = pd.DataFrame(mm_scaler.transform(rawData_test[cols].values)) if len(cols) > 1 else pd.DataFrame(mm_scaler.transform(rawData_test[cols].values.reshape(-1,1)))

        # Return the features norm scale
        elif ele.lower() == 'norm':

            # Account for lags and preprocess all lags the same way
#             cols = [[c for c in rawData_train.columns if t in c] for t in key[item==ele]]
            cols = [[c for c in rawData_train.columns if ((t==c) | (t in c) & ('lag' in c))] for t in key[item==ele]]
            cols = np.concatenate(cols)

            if verbose:
                print('Columns Processed:',key[item==ele],'\n')

            # Adding the transformed features to the new frame
            norm_scaler.fit(rawData_train[cols].values)
            pp_train[cols] = pd.DataFrame(norm_scaler.transform(rawData_train[cols].values))
            pp_test[cols] = pd.DataFrame(norm_scaler.transform(rawData_test[cols].values))

    # Rearanging columns before we return it
    pp_train,pp_test = pp_train[rawData_train.columns],pp_test[rawData_test.columns]

    # Return preprocessed data

    return pp_train.reset_index(drop=True),pp_test.reset_index(drop=True)
