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
# candleCreateNP() : Original candle Function
# candleCreateNP_vect(): Basic vectorized function
# candleCreateNP_vect_v2(): Final vectorized function, with possibility to create artificial holes in the data.
# candleCreateNP_vect_v3(): Final vectorized function
# generateFeatures(); Generate a wide range of technical features - not tested in this format - only copied from CrunchTAQ

######################################################################################################################

# Original function

def candleCreateNP(data
                    ,step):

    aggregateMinute = np.arange(0,60,step)
    aggregateHour = np.arange(9,16,1)
    aggregateDate = np.arange(len(data.Date.unique()))

    remove = 30//step

    candleNP = np.zeros((((len(aggregateDate)*len(aggregateMinute)*len(aggregateHour))-int(remove*len(aggregateDate))),4))

    numpiedData = data[['Date','Hour','Minute']].to_numpy()
    numpiedData = numpiedData.T
    numpiedPrice = data['price'].to_numpy()

    ii = 0
    for l in data.Date.unique():
        for i in aggregateHour:
            for j in aggregateMinute:
                if (i == 9) & (j <30):
                    continue

                p1 = numpiedPrice[((numpiedData[0]==l)&\
                                     (numpiedData[1]==i)&\
                                     (numpiedData[2]>=j))&((numpiedData[0]==l)&\
                                                           (numpiedData[1]==i)&\
                                                           (numpiedData[2]<j+step))]
                if len(p1) > 0:
                    candleNP[ii] = np.array([p1[0],p1.max(),p1.min(),p1[-1]])
                else:
                    # if no new prices in the interval considered, use the previous pne
                    candleNP[ii] = candleNP[ii-1]
                ii += 1

    return candleNP

# Basic vectorized function

# set up as function
def candleCreateNP_vect(data
                        ,step
                        ,verbose=True):

    data['hour_min_col'] = data['Hour'] + data['Minute']/60
    if verbose:
        print(f"min and max of new hour_min_col: \
              {data['hour_min_col'].min()}, {data['hour_min_col'].max()}")

    # setup time_bins to group each timestamp
    delta = step/60

    time_bins = np.arange(9.5-delta, 16, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})
    # return as numpy if preferred
    return OHLC

# Final vectorized function with possibility to create artificial holes
def candleCreateNP_vect_v2(data
                            ,step
                            ,verbose=False
                            ,createHoles=False
                            ,holes=10
                            ,fillHoles=True):

    data['hour_min_col'] = data['Hour'] + data['Minute']/60
    if verbose:
        print(f"min and max of new hour_min_col: \
              {data['hour_min_col'].min()}, {data['hour_min_col'].max()}")

    # setup time_bins to group each timestamp
    delta = step/60
    time_bins = np.arange(9.5-delta, 16, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})
    # Create some holes in the candle-table, so we can verify that the method works.
    if createHoles:
        np.random.seed(2021)
        for dt in OHLC.index.get_level_values(0).unique():
            dropedElements = []

            for inx in np.arange(holes):

                rm = np.random.randint(0,len(time_bins)-2,1,)
                dropedElements.append(rm[0])
                OHLC = OHLC.drop((dt,rm[0]))

            if verbose:
                print('Elements dropped: ',dropedElements)

    #Let check if we are missing any values
    dayz = len(OHLC.index.get_level_values(0).unique())
    if len(OHLC.index.get_level_values(1))!=((len(time_bins)-2)*dayz):

        ##### Creating our temporary table, with all the indices that is surposed to be in the actual candle-table.
        ## Creating the multiIndex-index
        mtInd = pd.MultiIndex.from_product([OHLC.index.get_level_values(0).unique(),np.arange(len(time_bins)-2)],
                                   names=['Date','time_group'])

        ## Creating the multiIndex-columns
        mtCol = pd.MultiIndex.from_product([['price'],['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            # Storing the indices to be filled
            toBeFilled = tempDf[tempDf.price['open'].isna()].index

            # Fill out the empty ones!
            tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled],
                                                  ('price','high'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled],
                                                  ('price','low'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled],
                                                  ('price','close'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]})

        # Return the complete data
        return tempDf

    else:

        # return as numpy if preferred
        return OHLC#.values



# Final vectorized function
def createCandles(data
                    ,step
                    ,verbose=False
                    ,fillHoles=True):

    data['hour_min_col'] = data['Hour'] + data['Minute']/60
    if verbose:
        print(f"min and max of new hour_min_col: \
              {data['hour_min_col'].min()}, {data['hour_min_col'].max()}")

    # setup time_bins to group each timestamp
    delta = step/60
    time_bins = np.arange(9.5-delta, 16, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, close (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})
    #Let check if we are missing any values
    dayz = len(OHLC.index.get_level_values(0).unique())
    if len(OHLC.index.get_level_values(1))!=((len(time_bins)-2)*dayz):

        ##### Creating our temporary table, with all the indices that is surposed to be in the actual candle-table.
        ## Creating the multiIndex-index
        mtInd = pd.MultiIndex.from_product([OHLC.index.get_level_values(0).unique(),np.arange(len(time_bins)-2)],
                                   names=['Date','time_group'])

        ## Creating the multiIndex-columns
        mtCol = pd.MultiIndex.from_product([['price'],['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            # Storing the indices to be filled
            toBeFilled = tempDf[tempDf.price['open'].isna()].index

            # Fill out the empty ones!
            tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled],
                                                  ('price','high'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled],
                                                  ('price','low'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled],
                                                  ('price','close'):tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]})

        # Return the complete data
        return tempDf

    else:

        # return as numpy if preferred
        return OHLC

# Final vectorized function
def createCandles_test(data
                        ,step
                        ,verbose=False
                        ,fillHoles=True):

    data['hour_min_col'] = data['Hour'] + data['Minute']/60
    if verbose:
        print(f"min and max of new hour_min_col: \
              {data['hour_min_col'].min()}, {data['hour_min_col'].max()}")

    # setup time_bins to group each timestamp
    delta = step/60
    time_bins = np.arange(9.5-delta, 16, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})
    #Let check if we are missing any values
    dayz = len(OHLC.index.get_level_values(0).unique())
    if len(OHLC.index.get_level_values(1))!=((len(time_bins)-2)*dayz):

        ##### Creating our temporary table, with all the indices that is surposed to be in the actual candle-table.
        ## Creating the multiIndex-index
        mtInd = pd.MultiIndex.from_product([OHLC.index.get_level_values(0).unique(),np.arange(len(time_bins)-2)],
                                   names=['Date','time_group'])

        ## Creating the multiIndex-columns
        mtCol = pd.MultiIndex.from_product([['price'],['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            # Storing the indices to be filled
            toBeFilled = tempDf[tempDf.price['open'].isna()].index

            # Fill out the empty ones!
            dataToFillIn = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]
            tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'):dataToFillIn,
                                                  ('price','high'):dataToFillIn,
                                                  ('price','low'):dataToFillIn,
                                                  ('price','close'):dataToFillIn})

        # Return the complete data
        return tempDf

    else:

        # return as numpy if preferred
        return OHLC

########### Not tested yet - only copied from CrunchTAQ!!
def generateFeatures(data
                    ,listOfFeatures=[]
                    ,featureWindow=1):
    # The input data is build up as follows:
    # Open, high, low and close.

    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':

            # Creating column names
            if isinstance(data.loc[(data.index.get_level_values(0).unique()[0],0)],pd.Series):
                cn = [['open_'+str(i),
                       'high_'+str(i),
                       'low_'+str(i),
                       'close_'+str(i)] for i in np.arange(featureWindow)]
                colnames = []

                for ele in cn:
                    colnames += ele
            else:
                # Made ready if we at some point moved to the data being a scalar series.
                raise ValueError('Im not ready to take on a scalar series.')

            # Create a variable to temporary store the new features
            tempFeatures = np.zeros((data.shape[0]-featureWindow+1,featureWindow*data.shape[1]))

            stepper = np.arange(featureWindow,len(tempFeatures)+featureWindow)

            i = 0
            # Creating the features
            for s in stepper:

                tempFeatures[i] = data.iloc[i:s].values.flatten()

                i += 1

            # Adding the features
            for colnm,feat in zip(colnames,tempFeatures.T):
                featuresPD[colnm] = feat

        # Stochastic K
        elif feature.lower() == 'stok':

            tempFeatures= ta.momentum.stoch(data.price['high'],
                                            data.price['low'],
                                            data.price['close'])
            # Adding the feature
            featuresPD['stok'] = tempFeatures

        # Stochastic D
        elif feature.lower() == 'stod':

            tempFeatures= ta.momentum.stoch_signal(data.price['high'],
                                                   data.price['low'],
                                                   data.price['close'])
            # Adding the feature
            featuresPD['stod'] = tempFeatures

        # Slow Stochastic D
        elif feature.lower() == 'sstod':

            tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(data.price['high'],
                                                                          data.price['low'],
                                                                          data.price['close']))
            # Adding the feature
            featuresPD['sstod'] = tempFeatures

        # Williams %R
        elif feature.lower() == 'wilr':

            tempFeatures= ta.momentum.wr(data.price['high'],
                                         data.price['low'],
                                         data.price['close'])
            # Adding the feature
            featuresPD['wilr'] = tempFeatures

        # Rate Of Change
        elif feature.lower() == 'roc':

            tempFeatures= ta.momentum.roc(data.price['close'])

            # Adding the feature
            featuresPD['roc'] = tempFeatures

        # Relative Strength Index
        elif feature.lower() == 'rsi':

            tempFeatures= ta.momentum.rsi(data.price['close'])

            # Adding the feature
            featuresPD['rsi'] = tempFeatures

        # Average True Range
        elif feature.lower() == 'atr':

            tempFeatures= ta.volatility.average_true_range(data.price['high'],
                                                           data.price['low'],
                                                           data.price['close'])
            # Adding the feature
            featuresPD['atr'] = tempFeatures

        # Commodity Channel Index
        elif feature.lower() == 'cci':

            tempFeatures= ta.trend.cci(data.price['high'],
                                       data.price['low'],
                                       data.price['close'])
            # Adding the feature
            featuresPD['cci'] = tempFeatures

         # Detrended Price Ocillator
        elif feature.lower() == 'dpo':

            tempFeatures= ta.trend.dpo(data.price['close'])

            # Adding the feature
            featuresPD['dpo'] = tempFeatures

        # Simple Moving Average
        elif feature.lower() == 'sma':

            tempFeatures= ta.trend.sma_indicator(data.price['close'])

            # Adding the feature
            featuresPD['sma'] = tempFeatures

        # Exponential Moving Average
        elif feature.lower() == 'ema':

            tempFeatures= ta.trend.ema_indicator(data.price['close'])

            # Adding the feature
            featuresPD['ema'] = tempFeatures

        # Moving Average Convergence Divergence
        elif feature.lower() == 'macd':

            tempFeatures= ta.trend.macd(data.price['close'])

            # Adding the feature
            featuresPD['macd'] = tempFeatures

         # Disparity 5
        elif feature.lower() == 'dis5':

            tempFeatures= (data.price['close']/ta.trend.sma_indicator(data.price['close'],5))*100

            # Adding the feature
            featuresPD['dis5'] = tempFeatures

        # Disparity 10
        elif feature.lower() == 'dis10':

            tempFeatures= (data.price['close']/ta.trend.sma_indicator(data.price['close'],10))*100

            # Adding the feature
            featuresPD['dis10'] = tempFeatures

    return featuresPD
