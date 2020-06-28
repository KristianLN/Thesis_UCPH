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
def candleCreateNP_vect_v3(data
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
def candleCreateNP_vect_v4(data
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
              
    ###Let check if we are missing any values       
    # number of days
    dayz = len(OHLC.index.get_level_values(0).unique())
              
    # if 
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
        return OHLC.values
              
              
# v5 vectorized: uses new time_bins to go with fully floated version (hour_min_col) of the Timestamp vector
# in v5 we subtract only 1 from len(time_bins) "(len(time_bins)-1)"
# as time_bins now go from 9.5 to 16 (including both 9.5 and 16), only the end-point 16 is excessive and must be removed
# v5 also has "sample" as input variable for setting time_bins to [9.5, 16] ('stable' and default) or [9, 16.5] ('full')
# v5 has also "numpied" as input variable for returning either full pandas df with multiindex or the numpied values (prices)
def candleCreateNP_vect_v5(data
                        ,step
                        ,verbose=False
                        ,fillHoles=True
                        ,sample='stable'
                        ,numpied=True):

    # v1-v4:
    #data['hour_min_col'] = data['Hour'] + data['Minute']/60
    
    # v5:
    # generate hour_min_col to hold floated Timestamp for time binning into candles
    Timestamp_dt = data['Timestamp'].dt
    Timestamp_float = Timestamp_dt.hour \
                      + Timestamp_dt.minute/60 \
                      + Timestamp_dt.second/(60*60) \
                      + Timestamp_dt.microsecond/(60*60*10**6)
    data['hour_min_col'] = Timestamp_float    
              
    if verbose:
        print(f"min and max of new hour_min_col: \
              {data['hour_min_col'].min()}, {data['hour_min_col'].max()}")

    # setup time_bins to group each timestamp
    delta = step/60
              
    if sample == 'full':
        time_bins = np.arange(9, 16.5+delta, delta)
    else:
        time_bins = np.arange(9.5, 16+delta, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})
              
    ###Let check if we are missing any values       
    # number of days
    dayz = len(OHLC.index.get_level_values(0).unique())
              
    # if 
    if len(OHLC.index.get_level_values(1))!=((len(time_bins)-1)*dayz):

        ##### Creating our temporary table, with all the indices that is surposed to be in the actual candle-table.
        ## Creating the multiIndex-index
        mtInd = pd.MultiIndex.from_product([OHLC.index.get_level_values(0).unique(), np.arange(len(time_bins)-1)],
                                   names=['Date','time_group'])

        ## Creating the multiIndex-columns
        mtCol = pd.MultiIndex.from_product([['price'], ['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            # Storing the indices to be filled
            toBeFilled = tempDf[tempDf.price['close'].isna()].index

            # Fill out the empty ones!
            dataToFillIn = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]
            tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'): dataToFillIn,
                                                  ('price','high'): dataToFillIn,
                                                  ('price','low'): dataToFillIn,
                                                  ('price','close'): dataToFillIn})

        # Return the complete data
        if numpied:
            return tempDf.values
        else:
            return tempDf  

    else:
              
        # return as numpy if preferred      
        if numpied:
            return OHLC.values
        else:
            return OHLC            
              
              
# Final vectorized function (currently v5)
def candleCreateNP_vect_final(data
                        ,step
                        ,verbose=False
                        ,fillHoles=True
                        ,sample='stable'
                        ,numpied=True):

    # v1-v4:
    #data['hour_min_col'] = data['Hour'] + data['Minute']/60
    
    # v5:
    # generate hour_min_col to hold floated Timestamp for time binning into candles
    Timestamp_dt = data['Timestamp'].dt
    Timestamp_float = Timestamp_dt.hour \
                      + Timestamp_dt.minute/60 \
                      + Timestamp_dt.second/(60*60) \
                      + Timestamp_dt.microsecond/(60*60*10**6)
    data['hour_min_col'] = Timestamp_float    
              
    if verbose:
        print(f"min and max of new hour_min_col: \
              {data['hour_min_col'].min()}, {data['hour_min_col'].max()}")

    # setup time_bins to group each timestamp
    delta = step/60
              
    if sample == 'full':
        time_bins = np.arange(9, 16.5+delta, delta)
    else:
        time_bins = np.arange(9.5, 16+delta, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})
              
    ###Let check if we are missing any values       
    # number of days
    dayz = len(OHLC.index.get_level_values(0).unique())
              
    # if 
    if len(OHLC.index.get_level_values(1))!=((len(time_bins)-1)*dayz):

        ##### Creating our temporary table, with all the indices that is surposed to be in the actual candle-table.
        ## Creating the multiIndex-index
        mtInd = pd.MultiIndex.from_product([OHLC.index.get_level_values(0).unique(), np.arange(len(time_bins)-1)],
                                   names=['Date','time_group'])

        ## Creating the multiIndex-columns
        mtCol = pd.MultiIndex.from_product([['price'], ['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            # Storing the indices to be filled
            toBeFilled = tempDf[tempDf.price['close'].isna()].index

            # Fill out the empty ones!
            dataToFillIn = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]
            tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'): dataToFillIn,
                                                  ('price','high'): dataToFillIn,
                                                  ('price','low'): dataToFillIn,
                                                  ('price','close'): dataToFillIn})

        # Return the complete data
        if numpied:
            return tempDf.values
        else:
            return tempDf

    else:
              
        # return as numpy if preferred      
        if numpied:
            return OHLC.values
        else:
            return OHLC
              

########### Not tested yet - only copied from CrunchTAQ!!
def generateFeatures(data
                    ,listOfFeatures=[]
                    ,featureWindow=1):
    # The input data is build up as follows:
    # Open, high, low and close.
#     dataPD = pd.DataFrame({'open':data.T[0],
#                             'high':data.T[1],
#                             'low':data.T[2],
#                             'close':data.T[3]})
    dataPD = pd.DataFrame({'open':data[:,0],
                             'high':data[:,1],
                             'low':data[:,2],
                             'close':data[:,3]})              
    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':

            # Creating column names
            if isinstance(data[0],np.ndarray):
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
            tempFeatures = np.zeros((len(data)-featureWindow+1,featureWindow*len(data[0])))

            stepper = np.arange(featureWindow,len(tempFeatures)+featureWindow)

            i = 0
            # Creating the features
            for s in stepper:

                tempFeatures[i] = data[i:s].flatten()

                i += 1

            # Adding the features
            for colnm,feat in zip(colnames,tempFeatures.T):
                featuresPD[colnm] = feat

        # Stochastic K
        elif feature.lower() == 'stok':

            tempFeatures= ta.momentum.stoch(dataPD.high,
                                            dataPD.low,
                                            dataPD.close)
            # The below is implemented as Stochastic D at the moment.
            # tempFeatures= ta.momentum.stoch_signal(dataPD.high,
            #                                 dataPD.low,
            #                                 dataPD.close)
            # Adding the feature
            featuresPD['stok'] = tempFeatures

        # Stochastic D
        elif feature.lower() == 'stod':

            tempFeatures= ta.momentum.stoch_signal(dataPD.high,
                                                   dataPD.low,
                                                   dataPD.close)
            # Adding the feature
            featuresPD['stod'] = tempFeatures

        # Slow Stochastic D
        elif feature.lower() == 'sstod':

            tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
                                                                          dataPD.low,
                                                                          dataPD.close))
            # Adding the feature
            featuresPD['sstod'] = tempFeatures

        # Williams %R
        elif feature.lower() == 'wilr':

            tempFeatures= ta.momentum.wr(dataPD.high,
                                         dataPD.low,
                                         dataPD.close)
            # Adding the feature
            featuresPD['wilr'] = tempFeatures

        # Rate Of Change
        elif feature.lower() == 'roc':

            tempFeatures= ta.momentum.roc(dataPD.close)

            # Adding the feature
            featuresPD['roc'] = tempFeatures

        # Relative Strength Index
        elif feature.lower() == 'rsi':

            tempFeatures= ta.momentum.rsi(dataPD.close)

            # Adding the feature
            featuresPD['rsi'] = tempFeatures

        # Average True Range
        elif feature.lower() == 'atr':

            tempFeatures= ta.volatility.average_true_range(dataPD.high,
                                                           dataPD.low,
                                                           dataPD.close)
            # Adding the feature
            featuresPD['atr'] = tempFeatures

        # Commodity Channel Index
        elif feature.lower() == 'cci':

            tempFeatures= ta.trend.cci(dataPD.high,
                                       dataPD.low,
                                       dataPD.close)
            # Adding the feature
            featuresPD['cci'] = tempFeatures

         # Detrended Price Ocillator
        elif feature.lower() == 'dpo':

            tempFeatures= ta.trend.dpo(dataPD.close)

            # Adding the feature
            featuresPD['dpo'] = tempFeatures

        # Simple Moving Average
        elif feature.lower() == 'sma':

            tempFeatures= ta.trend.sma_indicator(dataPD.close)

            # Adding the feature
            featuresPD['sma'] = tempFeatures

        # Exponential Moving Average
        elif feature.lower() == 'ema':

            tempFeatures= ta.trend.ema_indicator(dataPD.close)

            # Adding the feature
            featuresPD['ema'] = tempFeatures

        # Moving Average Convergence Divergence
        elif feature.lower() == 'macd':

            tempFeatures= ta.trend.macd(dataPD.close)
            # tempFeatures= ta.trend.macd_diff(dataPD.close)
            # tempFeatures= ta.trend.macd_signal(dataPD.close)
            # Adding the feature
            featuresPD['macd'] = tempFeatures

         # Disparity 5
        elif feature.lower() == 'dis5':

            tempFeatures= (dataPD.close/ta.trend.sma_indicator(dataPD.close,5))*100

            # Adding the feature
            featuresPD['dis5'] = tempFeatures

        # Disparity 10
        elif feature.lower() == 'dis10':

            tempFeatures= (dataPD.close/ta.trend.sma_indicator(dataPD.close,10))*100

            # Adding the feature
            featuresPD['dis10'] = tempFeatures

    return featuresPD
              
              
def generateFeatures_v2(data
                        ,listOfFeatures=[]
                        ,feature_lags=1):
    # The input data is build up as follows:
    # Open, high, low and close.
    dataPD = pd.DataFrame({'open':data[:,0],
                             'high':data[:,1],
                             'low':data[:,2],
                             'close':data[:,3]})              
    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':
            featuresPD['open'] = dataPD.open
            featuresPD['high'] = dataPD.high
            featuresPD['low'] = dataPD.low
            featuresPD['close'] = dataPD.close

        # Stochastic K
        elif feature.lower() == 'stok':

            tempFeatures= ta.momentum.stoch(dataPD.high,
                                            dataPD.low,
                                            dataPD.close)
            # The below is implemented as Stochastic D at the moment.
            # tempFeatures= ta.momentum.stoch_signal(dataPD.high,
            #                                 dataPD.low,
            #                                 dataPD.close)
            # Adding the feature
            featuresPD['stok'] = tempFeatures

        # Stochastic D
        elif feature.lower() == 'stod':

            tempFeatures= ta.momentum.stoch_signal(dataPD.high,
                                                   dataPD.low,
                                                   dataPD.close)
            # Adding the feature
            featuresPD['stod'] = tempFeatures

        # Slow Stochastic D
        elif feature.lower() == 'sstod':

            tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
                                                                          dataPD.low,
                                                                          dataPD.close))
            # Adding the feature
            featuresPD['sstod'] = tempFeatures

        # Williams %R
        elif feature.lower() == 'wilr':

            tempFeatures= ta.momentum.wr(dataPD.high,
                                         dataPD.low,
                                         dataPD.close)
            # Adding the feature
            featuresPD['wilr'] = tempFeatures

        # Rate Of Change
        elif feature.lower() == 'roc':

            tempFeatures= ta.momentum.roc(dataPD.close)

            # Adding the feature
            featuresPD['roc'] = tempFeatures

        # Relative Strength Index
        elif feature.lower() == 'rsi':

            tempFeatures= ta.momentum.rsi(dataPD.close)

            # Adding the feature
            featuresPD['rsi'] = tempFeatures

        # Average True Range
        elif feature.lower() == 'atr':

            tempFeatures= ta.volatility.average_true_range(dataPD.high,
                                                           dataPD.low,
                                                           dataPD.close)
            # Adding the feature
            featuresPD['atr'] = tempFeatures

        # Commodity Channel Index
        elif feature.lower() == 'cci':

            tempFeatures= ta.trend.cci(dataPD.high,
                                       dataPD.low,
                                       dataPD.close)
            # Adding the feature
            featuresPD['cci'] = tempFeatures

         # Detrended Price Ocillator
        elif feature.lower() == 'dpo':

            tempFeatures= ta.trend.dpo(dataPD.close)

            # Adding the feature
            featuresPD['dpo'] = tempFeatures

        # Simple Moving Average
        elif feature.lower() == 'sma':

            tempFeatures= ta.trend.sma_indicator(dataPD.close)

            # Adding the feature
            featuresPD['sma'] = tempFeatures

        # Exponential Moving Average
        elif feature.lower() == 'ema':

            tempFeatures= ta.trend.ema_indicator(dataPD.close)

            # Adding the feature
            featuresPD['ema'] = tempFeatures

        # Moving Average Convergence Divergence
        elif feature.lower() == 'macd':
            
            # note: having all 3 causes multicollinearity. Maybe not a problem in ML, let's see :-)
            # macd is the difference between two EMAs
            # macd_signal is an EMA of the above macd line
            # macd_diff is the so-called histogram (just bars really) of the time-wise difference between macd and macd_signal

            # Adding the features
            featuresPD['macd'] = ta.trend.macd(dataPD.close)            
            featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close) 
            featuresPD['macd_signal'] = ta.trend.macd_signal(dataPD.close)

         # Disparity 5
        elif feature.lower() == 'dis5':

            tempFeatures= (dataPD.close/ta.trend.sma_indicator(dataPD.close,5))*100

            # Adding the feature
            featuresPD['dis5'] = tempFeatures

        # Disparity 10
        elif feature.lower() == 'dis10':

            tempFeatures= (dataPD.close/ta.trend.sma_indicator(dataPD.close,10))*100

            # Adding the feature
            featuresPD['dis10'] = tempFeatures     
            
        # Bollinger Bands
        elif feature.lower() == 'bb':
            
            # Define Bollinger Bands function to extract from
            bb_function = ta.volatility.BollingerBands(close=dataPD.close, n=20, ndev=2)
            
            # Adding the features
            featuresPD['bb_mavg'] = bb_function.bollinger_mavg()
            featuresPD['bb_hband'] = bb_function.bollinger_hband()
            featuresPD['bb_lband'] = bb_function.bollinger_lband()
            featuresPD['bb_pband'] = bb_function.bollinger_pband()
            featuresPD['bb_wband'] = bb_function.bollinger_wband()
         
    # if we want any lags:
    if feature_lags > 0:

        # collect names of all raw features (before any lagging) 
        all_raw_features = featuresPD.columns
        
        # loop through each lag and shift all features at once
        for roll_i in np.arange(feature_lags + 1): # + 1 as we treat feature_lags = 1 as having both lag0 and lag1
            
            # define new column name (feature_name_ + lagX) where X = roll_i is the shifting parameter
            new_col_names = [feature_name + '_lag' + str(roll_i) for feature_name in all_raw_features]
            
            # Shift/roll all raw features with the shifting parameter roll_i and save as new columns. 
            # The shift parameter must be negative (we want lag0 to be the 'newest'/'latest')
            featuresPD[new_col_names] = featuresPD[all_raw_features].shift( - (feature_lags - roll_i))
            
        # remove all raw features
        featuresPD = featuresPD.loc[:, ~featuresPD.columns.isin(all_raw_features)]
                
    return featuresPD
              
              
# final is currently _v2
def generateFeatures_final(data
                        ,listOfFeatures=[]
                        ,feature_lags=1):
    # The input data is build up as follows:
    # Open, high, low and close.
    dataPD = pd.DataFrame({'open':data[:,0],
                             'high':data[:,1],
                             'low':data[:,2],
                             'close':data[:,3]})              
    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':
            featuresPD['open'] = dataPD.open
            featuresPD['high'] = dataPD.high
            featuresPD['low'] = dataPD.low
            featuresPD['close'] = dataPD.close

        # Stochastic K
        elif feature.lower() == 'stok':

            tempFeatures= ta.momentum.stoch(dataPD.high,
                                            dataPD.low,
                                            dataPD.close)
            # The below is implemented as Stochastic D at the moment.
            # tempFeatures= ta.momentum.stoch_signal(dataPD.high,
            #                                 dataPD.low,
            #                                 dataPD.close)
            # Adding the feature
            featuresPD['stok'] = tempFeatures

        # Stochastic D
        elif feature.lower() == 'stod':

            tempFeatures= ta.momentum.stoch_signal(dataPD.high,
                                                   dataPD.low,
                                                   dataPD.close)
            # Adding the feature
            featuresPD['stod'] = tempFeatures

        # Slow Stochastic D
        elif feature.lower() == 'sstod':

            tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
                                                                          dataPD.low,
                                                                          dataPD.close))
            # Adding the feature
            featuresPD['sstod'] = tempFeatures

        # Williams %R
        elif feature.lower() == 'wilr':

            tempFeatures= ta.momentum.wr(dataPD.high,
                                         dataPD.low,
                                         dataPD.close)
            # Adding the feature
            featuresPD['wilr'] = tempFeatures

        # Rate Of Change
        elif feature.lower() == 'roc':

            tempFeatures= ta.momentum.roc(dataPD.close)

            # Adding the feature
            featuresPD['roc'] = tempFeatures

        # Relative Strength Index
        elif feature.lower() == 'rsi':

            tempFeatures= ta.momentum.rsi(dataPD.close)

            # Adding the feature
            featuresPD['rsi'] = tempFeatures

        # Average True Range
        elif feature.lower() == 'atr':

            tempFeatures= ta.volatility.average_true_range(dataPD.high,
                                                           dataPD.low,
                                                           dataPD.close)
            # Adding the feature
            featuresPD['atr'] = tempFeatures

        # Commodity Channel Index
        elif feature.lower() == 'cci':

            tempFeatures= ta.trend.cci(dataPD.high,
                                       dataPD.low,
                                       dataPD.close)
            # Adding the feature
            featuresPD['cci'] = tempFeatures

         # Detrended Price Ocillator
        elif feature.lower() == 'dpo':

            tempFeatures= ta.trend.dpo(dataPD.close)

            # Adding the feature
            featuresPD['dpo'] = tempFeatures

        # Simple Moving Average
        elif feature.lower() == 'sma':

            tempFeatures= ta.trend.sma_indicator(dataPD.close)

            # Adding the feature
            featuresPD['sma'] = tempFeatures

        # Exponential Moving Average
        elif feature.lower() == 'ema':

            tempFeatures= ta.trend.ema_indicator(dataPD.close)

            # Adding the feature
            featuresPD['ema'] = tempFeatures

        # Moving Average Convergence Divergence
        elif feature.lower() == 'macd':
            
            # note: having all 3 causes multicollinearity. Maybe not a problem in ML, let's see :-)
            # macd is the difference between two EMAs
            # macd_signal is an EMA of the above macd line
            # macd_diff is the so-called histogram (just bars really) of the time-wise difference between macd and macd_signal

            # Adding the features
            featuresPD['macd'] = ta.trend.macd(dataPD.close)            
            featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close) 
            featuresPD['macd_signal'] = ta.trend.macd_signal(dataPD.close)

         # Disparity 5
        elif feature.lower() == 'dis5':

            tempFeatures= (dataPD.close/ta.trend.sma_indicator(dataPD.close,5))*100

            # Adding the feature
            featuresPD['dis5'] = tempFeatures

        # Disparity 10
        elif feature.lower() == 'dis10':

            tempFeatures= (dataPD.close/ta.trend.sma_indicator(dataPD.close,10))*100

            # Adding the feature
            featuresPD['dis10'] = tempFeatures     
            
        # Bollinger Bands
        elif feature.lower() == 'bb':
            
            # Define Bollinger Bands function to extract from
            bb_function = ta.volatility.BollingerBands(close=dataPD.close, n=20, ndev=2)
            
            # Adding the features
            featuresPD['bb_mavg'] = bb_function.bollinger_mavg()
            featuresPD['bb_hband'] = bb_function.bollinger_hband()
            featuresPD['bb_lband'] = bb_function.bollinger_lband()
            featuresPD['bb_pband'] = bb_function.bollinger_pband()
            featuresPD['bb_wband'] = bb_function.bollinger_wband()
        
    # if we want any lags:
    if feature_lags > 0:

        # collect names of all raw features (before any lagging) 
        all_raw_features = featuresPD.columns
        
        # loop through each lag and shift all features at once
        for roll_i in np.arange(feature_lags + 1): # + 1 as we treat feature_lags = 1 as having both lag0 and lag1
            
            # define new column name (feature_name_ + lagX) where X = roll_i is the shifting parameter
            new_col_names = [feature_name + '_lag' + str(roll_i) for feature_name in all_raw_features]
            
            # Shift/roll all raw features with the shifting parameter roll_i and save as new columns. 
            # The shift parameter must be negative (we want lag0 to be the 'newest'/'latest')
            featuresPD[new_col_names] = featuresPD[all_raw_features].shift( - (feature_lags - roll_i))
            
        # remove all raw features
        featuresPD = featuresPD.loc[:, ~featuresPD.columns.isin(all_raw_features)]
                
    return featuresPD 
