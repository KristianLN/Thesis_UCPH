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



# v6 has return_spreads that returns spread based on quotes input data
def candleCreateNP_vect_v6(data
                        ,step
                        ,verbose=False
                        ,fillHoles=True
                        ,sample='stable'
                        ,numpied=True
                        ,return_spreads=False):

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

    if return_spreads:
        assert 'spread' in data.columns, 'The input data is not quotes data which it must be for return_spread == True'
        spreads = data.groupby(['Date','time_group'])[['spread']].agg(['first', 'last'])
        spreads = spreads.rename(columns={'first':'open',
                                          'last':'close'})
        OHLC = pd.concat([OHLC, spreads], axis=1)

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
        if not return_spreads:
            mtCol = pd.MultiIndex.from_product([['price'], ['open','high','low','close']])
        else:
            mtCol = pd.MultiIndex.from_tuples([('price', 'open'),
                                               ('price', 'high'),
                                               ('price', 'low'),
                                               ('price', 'close'),
                                               ('spread', 'open'),
                                               ('spread', 'close')])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            if not return_spreads:
                # Storing the indices to be filled
                toBeFilled = tempDf[tempDf.price['close'].isna()].index

                # Fill out the empty ones!
                dataToFillIn = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]
                tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'): dataToFillIn,
                                                      ('price','high'): dataToFillIn,
                                                      ('price','low'): dataToFillIn,
                                                      ('price','close'): dataToFillIn})

            else:
                # Storing the indices to be filled
                toBeFilled_price = tempDf[tempDf.price['close'].isna()].index
                toBeFilled_spread = tempDf[tempDf.spread['close'].isna()].index

                # Fill out the empty ones!
                dataToFillIn_price = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled_price]
                dataToFillIn_spread = tempDf.spread['close'].fillna(method='ffill').loc[toBeFilled_spread]

                tempDf.loc[toBeFilled_price, ('price')] = pd.DataFrame({('price','open'): dataToFillIn_price,
                                                                          ('price','high'): dataToFillIn_price,
                                                                          ('price','low'): dataToFillIn_price,
                                                                          ('price','close'): dataToFillIn_price,
                                                                          })

                tempDf.loc[toBeFilled_spread, ('spread')] = pd.DataFrame({('spread','open'): dataToFillIn_spread,
                                                                          ('spread','close'): dataToFillIn_spread,
                                                                           })

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


# v7 has return_extended that returns candles for spread, offer, ask based on quotes input data
# Time bins are fixed to match correct time intervals.
def candleCreateNP_vect_v7(data
                        ,step
                        ,verbose=False
                        ,fillHoles=True
                        ,sample='stable'
                        ,numpied=True
                        ,return_extended=None):

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
        time_bins = np.arange(9.5, 16+delta, delta)
    else:
        time_bins = np.arange(10, 15.5+delta, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})

    if return_extended: #return_spreads:
        if 'spread' in return_extended:
            assert 'spread' in data.columns, 'The input data is not quotes data which it must be for return_extended[spread] == True'
            spreads = data.groupby(['Date','time_group'])[['spread']].agg(['first', 'max', 'min', 'last'])
            spreads = spreads.rename(columns={'first':'open'
                                            ,'max':'high'
                                            ,'min':'low'
                                            ,'last':'close'})
            OHLC = pd.concat([OHLC, spreads], axis=1)

        if 'bidsize' in return_extended:
            assert 'bidsize' in data.columns, 'The input data is not quotes data which it must be for return_extended[bidsize] == True'
            bidsize = data.groupby(['Date','time_group'])[['bidsize']].agg(['first', 'max', 'min', 'last'])
            bidsize = bidsize.rename(columns={'first':'open'
                                            ,'max':'high'
                                            ,'min':'low'
                                            ,'last':'close'})
            OHLC = pd.concat([OHLC, bidsize], axis=1)

        if 'ofrsize' in return_extended:
            assert 'ofrsize' in data.columns, 'The input data is not quotes data which it must be for return_extended[ofrsize] == True'
            ofrsize = data.groupby(['Date','time_group'])[['ofrsize']].agg(['first', 'max', 'min', 'last'])
            ofrsize = ofrsize.rename(columns={'first':'open'
                                            ,'max':'high'
                                            ,'min':'low'
                                            ,'last':'close'})
            OHLC = pd.concat([OHLC, ofrsize], axis=1)

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
        #if not return_spreads:
        if return_extended is None:
            mtCol = pd.MultiIndex.from_product([['price'], ['open','high','low','close']])
        else:
            cols_to_include = ['price']
            # append each specified variable from return_extended (spread/bidsize/ofrsize)
            [cols_to_include.append(extended_col) for extended_col in return_extended]
            # generate MultiIndex
            mtCol = pd.MultiIndex.from_product([cols_to_include, ['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            if return_extended is None: #not return_spreads:
                # Storing the indices to be filled
                toBeFilled = tempDf[tempDf.price['close'].isna()].index

                # Fill out the empty ones!
                dataToFillIn = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]
                tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'): dataToFillIn,
                                                      ('price','high'): dataToFillIn,
                                                      ('price','low'): dataToFillIn,
                                                      ('price','close'): dataToFillIn})

            else:
                # Storing the indices to be filled
                toBeFilled_price = tempDf[tempDf.price['close'].isna()].index

                # Fill out the empty ones!
                dataToFillIn_price = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled_price]

                tempDf.loc[toBeFilled_price, ('price')] = pd.DataFrame({('price','open'): dataToFillIn_price,
                                                                        ('price','high'): dataToFillIn_price,
                                                                        ('price','low'): dataToFillIn_price,
                                                                        ('price','close'): dataToFillIn_price,
                                                                       })
                if 'spread' in return_extended:
                    # Storing the indices to be filled
                    toBeFilled_spread = tempDf[tempDf.spread['close'].isna()].index

                    # Fill out the empty ones
                    dataToFillIn_spread = tempDf.spread['close'].fillna(method='ffill').loc[toBeFilled_spread]

                    tempDf.loc[toBeFilled_spread, ('spread')] = pd.DataFrame({('spread','open'): dataToFillIn_spread,
                                                                              ('spread','high'): dataToFillIn_spread,
                                                                              ('spread','low'): dataToFillIn_spread,
                                                                              ('spread','close'): dataToFillIn_spread,
                                                                             })

                if 'bidsize' in return_extended:
                    # Storing the indices to be filled
                    toBeFilled_bidsize = tempDf[tempDf.bidsize['close'].isna()].index

                    # Fill out the empty ones
                    dataToFillIn_bidsize = tempDf.bidsize['close'].fillna(method='ffill').loc[toBeFilled_bidsize]

                    tempDf.loc[toBeFilled_bidsize, ('bidsize')] = pd.DataFrame({('bidsize','open'): dataToFillIn_bidsize,
                                                                                  ('bidsize','high'): dataToFillIn_bidsize,
                                                                                  ('bidsize','low'): dataToFillIn_bidsize,
                                                                                  ('bidsize','close'): dataToFillIn_bidsize,
                                                                                 })

                if 'ofrsize' in return_extended:
                    # Storing the indices to be filled
                    toBeFilled_ofrsize = tempDf[tempDf.ofrsize['close'].isna()].index

                    # Fill out the empty ones
                    dataToFillIn_ofrsize = tempDf.ofrsize['close'].fillna(method='ffill').loc[toBeFilled_ofrsize]

                    tempDf.loc[toBeFilled_ofrsize, ('ofrsize')] = pd.DataFrame({('ofrsize','open'): dataToFillIn_ofrsize,
                                                                                  ('ofrsize','high'): dataToFillIn_ofrsize,
                                                                                  ('ofrsize','low'): dataToFillIn_ofrsize,
                                                                                  ('ofrsize','close'): dataToFillIn_ofrsize,
                                                                                 })

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


# Final vectorized function (currently v7)
def candleCreateNP_vect_final(data
                             ,step
                             ,verbose=False
                             ,fillHoles=True
                             ,sample='stable'
                             ,numpied=True
                             ,return_extended=None):

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
        time_bins = np.arange(9.5, 16+delta, delta)
    else:
        time_bins = np.arange(10, 15.5+delta, delta)

    # put each timestamp into a bucket according to time_bins defined by the step variable
    data['time_group'] = pd.cut(data['hour_min_col'], bins=time_bins, right=True, labels=False)

    # group by date and time_group, extract price, take it open, max, min, last (open, high, low, close)
    OHLC = data.groupby(['Date','time_group'])[['price']].agg(['first', 'max', 'min', 'last'])
    OHLC = OHLC.rename(columns={'first':'open'
                                ,'max':'high'
                                ,'min':'low'
                                ,'last':'close'})

    if return_extended: #return_spreads:
        if 'spread' in return_extended:
            assert 'spread' in data.columns, 'The input data is not quotes data which it must be for return_extended[spread] == True'
            spreads = data.groupby(['Date','time_group'])[['spread']].agg(['first', 'max', 'min', 'last'])
            spreads = spreads.rename(columns={'first':'open'
                                            ,'max':'high'
                                            ,'min':'low'
                                            ,'last':'close'})
            OHLC = pd.concat([OHLC, spreads], axis=1)

        if 'bidsize' in return_extended:
            assert 'bidsize' in data.columns, 'The input data is not quotes data which it must be for return_extended[bidsize] == True'
            bidsize = data.groupby(['Date','time_group'])[['bidsize']].agg(['first', 'max', 'min', 'last'])
            bidsize = bidsize.rename(columns={'first':'open'
                                            ,'max':'high'
                                            ,'min':'low'
                                            ,'last':'close'})
            OHLC = pd.concat([OHLC, bidsize], axis=1)

        if 'ofrsize' in return_extended:
            assert 'ofrsize' in data.columns, 'The input data is not quotes data which it must be for return_extended[ofrsize] == True'
            ofrsize = data.groupby(['Date','time_group'])[['ofrsize']].agg(['first', 'max', 'min', 'last'])
            ofrsize = ofrsize.rename(columns={'first':'open'
                                            ,'max':'high'
                                            ,'min':'low'
                                            ,'last':'close'})
            OHLC = pd.concat([OHLC, ofrsize], axis=1)

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
        #if not return_spreads:
        if return_extended is None:
            mtCol = pd.MultiIndex.from_product([['price'], ['open','high','low','close']])
        else:
            cols_to_include = ['price']
            # append each specified variable from return_extended (spread/bidsize/ofrsize)
            [cols_to_include.append(extended_col) for extended_col in return_extended]
            # generate MultiIndex
            mtCol = pd.MultiIndex.from_product([cols_to_include, ['open','high','low','close']])

        ## Creating the table itself
        tempDf = pd.DataFrame(np.nan
                              ,columns=mtCol
                              ,index=mtInd)

        # Filling the non-empty elements of OHLC into the temp-table
        tempDf.loc[OHLC.index]=OHLC.copy(deep=True)

        # To see that the filling mechanism works:
        if fillHoles:

            if return_extended is None: #not return_spreads:
                # Storing the indices to be filled
                toBeFilled = tempDf[tempDf.price['close'].isna()].index

                # Fill out the empty ones!
                dataToFillIn = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled]
                tempDf.loc[toBeFilled] = pd.DataFrame({('price','open'): dataToFillIn,
                                                      ('price','high'): dataToFillIn,
                                                      ('price','low'): dataToFillIn,
                                                      ('price','close'): dataToFillIn})

            else:
                # Storing the indices to be filled
                toBeFilled_price = tempDf[tempDf.price['close'].isna()].index

                # Fill out the empty ones!
                dataToFillIn_price = tempDf.price['close'].fillna(method='ffill').loc[toBeFilled_price]

                tempDf.loc[toBeFilled_price, ('price')] = pd.DataFrame({('price','open'): dataToFillIn_price,
                                                                        ('price','high'): dataToFillIn_price,
                                                                        ('price','low'): dataToFillIn_price,
                                                                        ('price','close'): dataToFillIn_price,
                                                                       })
                if 'spread' in return_extended:
                    # Storing the indices to be filled
                    toBeFilled_spread = tempDf[tempDf.spread['close'].isna()].index

                    # Fill out the empty ones
                    dataToFillIn_spread = tempDf.spread['close'].fillna(method='ffill').loc[toBeFilled_spread]

                    tempDf.loc[toBeFilled_spread, ('spread')] = pd.DataFrame({('spread','open'): dataToFillIn_spread,
                                                                              ('spread','high'): dataToFillIn_spread,
                                                                              ('spread','low'): dataToFillIn_spread,
                                                                              ('spread','close'): dataToFillIn_spread,
                                                                             })

                if 'bidsize' in return_extended:
                    # Storing the indices to be filled
                    toBeFilled_bidsize = tempDf[tempDf.bidsize['close'].isna()].index

                    # Fill out the empty ones
                    dataToFillIn_bidsize = tempDf.bidsize['close'].fillna(method='ffill').loc[toBeFilled_bidsize]

                    tempDf.loc[toBeFilled_bidsize, ('bidsize')] = pd.DataFrame({('bidsize','open'): dataToFillIn_bidsize,
                                                                                  ('bidsize','high'): dataToFillIn_bidsize,
                                                                                  ('bidsize','low'): dataToFillIn_bidsize,
                                                                                  ('bidsize','close'): dataToFillIn_bidsize,
                                                                                 })

                if 'ofrsize' in return_extended:
                    # Storing the indices to be filled
                    toBeFilled_ofrsize = tempDf[tempDf.ofrsize['close'].isna()].index

                    # Fill out the empty ones
                    dataToFillIn_ofrsize = tempDf.ofrsize['close'].fillna(method='ffill').loc[toBeFilled_ofrsize]

                    tempDf.loc[toBeFilled_ofrsize, ('ofrsize')] = pd.DataFrame({('ofrsize','open'): dataToFillIn_ofrsize,
                                                                                  ('ofrsize','high'): dataToFillIn_ofrsize,
                                                                                  ('ofrsize','low'): dataToFillIn_ofrsize,
                                                                                  ('ofrsize','close'): dataToFillIn_ofrsize,
                                                                                 })

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


# Includes spread and sizes, and takes a PD dataframe instead of a numpy array.
def generateFeatures_v3(data,listOfFeatures=[],feature_lags=1):
    # The input data is build up as follows:
    # Open, high, low and close.
#     dataPD = pd.DataFrame({'open':data[:,0],
#                              'high':data[:,1],
#                              'low':data[:,2],
#                              'close':data[:,3]})
    dataPD = data.copy(deep=True)
#     print(dataPD.columns)
#     print([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

    dataPD.columns = np.concatenate([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

#     dataPD = dataPD.loc[:,['price_open','price_high','price_low','price_close']].rename(columns=['open','high','low','close'])
    dataPD = dataPD.rename(columns={'price_open':'open',
                                    'price_high':'high',
                                    'price_low':'low',
                                    'price_close':'close'})
    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':
            featuresPD['open'] = dataPD.open
            featuresPD['high'] = dataPD.high
            featuresPD['low'] = dataPD.low
            featuresPD['close'] = dataPD.close

        elif feature.lower() == 'spread':
            featuresPD['spread_open'] = dataPD.spread_open
            featuresPD['spread_high'] = dataPD.spread_high
            featuresPD['spread_low'] = dataPD.spread_low
            featuresPD['spread_close'] = dataPD.spread_close

        elif feature.lower() == 'bidsize':
            featuresPD['bidsize_open'] = dataPD.bidsize_open
            featuresPD['bidsize_high'] = dataPD.bidsize_high
            featuresPD['bidsize_low'] = dataPD.bidsize_low
            featuresPD['bidsize_close'] = dataPD.bidsize_close

        elif feature.lower() == 'ofrsize':
            featuresPD['ofrsize_open'] = dataPD.ofrsize_open
            featuresPD['ofrsize_high'] = dataPD.ofrsize_high
            featuresPD['ofrsize_low'] = dataPD.ofrsize_low
            featuresPD['ofrsize_close'] = dataPD.ofrsize_close

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

        # Adjust price feature
    if 'pastobs' in listOfFeatures:
        if feature_lags > 0:
            priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
            # print(priceCols)
            tempClose = featuresPD.close_lag0.copy(deep=True)
#             print('\n')

#             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
            featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
#             print('\n')
#             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
#             print(tempClose)
            featuresPD.loc[:,'close_lag0'] = tempClose
        else:
#             tempClose = copy.deepcopy(featuresPD.close.values)
            tempClose = featuresPD.close.copy(deep=True)

#             print(tempClose)
#             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
            featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
#             print('\n')
#             print(featuresPD.loc[:,['open','high','low','close']])
#             print(featuresPD.close)
#             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
            featuresPD.loc[:,'close'] = tempClose

    return featuresPD


# v4 does not change column names: in latest data extraction (constructing candles on-the-fly), the column are already fine
def generateFeatures_v4(data,listOfFeatures=[],feature_lags=1):
    # The input data is build up as follows:
    # Open, high, low and close.
#     dataPD = pd.DataFrame({'open':data[:,0],
#                              'high':data[:,1],
#                              'low':data[:,2],
#                              'close':data[:,3]})
    dataPD = data.copy(deep=True)
#     print(dataPD.columns)
#     print([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

    ### in latest data extraction (constructing candles on-the-fly), the column are already fine
#     dataPD.columns = np.concatenate([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

# #     dataPD = dataPD.loc[:,['price_open','price_high','price_low','price_close']].rename(columns=['open','high','low','close'])
#     dataPD = dataPD.rename(columns={'price_open':'open',
#                                     'price_high':'high',
#                                     'price_low':'low',
#                                     'price_close':'close'})
    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':
            featuresPD['open'] = dataPD.open
            featuresPD['high'] = dataPD.high
            featuresPD['low'] = dataPD.low
            featuresPD['close'] = dataPD.close

        elif feature.lower() == 'spread':
            featuresPD['spread_open'] = dataPD.spread_open
            featuresPD['spread_high'] = dataPD.spread_high
            featuresPD['spread_low'] = dataPD.spread_low
            featuresPD['spread_close'] = dataPD.spread_close

        elif feature.lower() == 'bidsize':
            featuresPD['bidsize_open'] = dataPD.bidsize_open
            featuresPD['bidsize_high'] = dataPD.bidsize_high
            featuresPD['bidsize_low'] = dataPD.bidsize_low
            featuresPD['bidsize_close'] = dataPD.bidsize_close

        elif feature.lower() == 'ofrsize':
            featuresPD['ofrsize_open'] = dataPD.ofrsize_open
            featuresPD['ofrsize_high'] = dataPD.ofrsize_high
            featuresPD['ofrsize_low'] = dataPD.ofrsize_low
            featuresPD['ofrsize_close'] = dataPD.ofrsize_close

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

        # Adjust price feature
    if 'pastobs' in listOfFeatures:
        if feature_lags > 0:
            priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
            # print(priceCols)
            tempClose = featuresPD.close_lag0.copy(deep=True)
#             print('\n')

#             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
            featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
#             print('\n')
#             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
#             print(tempClose)
            featuresPD.loc[:,'close_lag0'] = tempClose
        else:
#             tempClose = copy.deepcopy(featuresPD.close.values)
            tempClose = featuresPD.close.copy(deep=True)

#             print(tempClose)
#             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
            featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
#             print('\n')
#             print(featuresPD.loc[:,['open','high','low','close']])
#             print(featuresPD.close)
#             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
            featuresPD.loc[:,'close'] = tempClose

    return featuresPD



# final is currently _v4
def generateFeatures_final(data,listOfFeatures=[],feature_lags=1):
    # The input data is build up as follows:
    # Open, high, low and close.
#     dataPD = pd.DataFrame({'open':data[:,0],
#                              'high':data[:,1],
#                              'low':data[:,2],
#                              'close':data[:,3]})
    dataPD = data.copy(deep=True)
#     print(dataPD.columns)
#     print([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

    ### in latest data extraction (constructing candles on-the-fly), the column are already fine
#     dataPD.columns = np.concatenate([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

# #     dataPD = dataPD.loc[:,['price_open','price_high','price_low','price_close']].rename(columns=['open','high','low','close'])
#     dataPD = dataPD.rename(columns={'price_open':'open',
#                                     'price_high':'high',
#                                     'price_low':'low',
#                                     'price_close':'close'})
    featuresPD = pd.DataFrame()

    for feature in listOfFeatures:

        # Past observations
        if feature.lower() == 'pastobs':
            featuresPD['open'] = dataPD.open
            featuresPD['high'] = dataPD.high
            featuresPD['low'] = dataPD.low
            featuresPD['close'] = dataPD.close

        elif feature.lower() == 'spread':
            featuresPD['spread_open'] = dataPD.spread_open
            featuresPD['spread_high'] = dataPD.spread_high
            featuresPD['spread_low'] = dataPD.spread_low
            featuresPD['spread_close'] = dataPD.spread_close

        elif feature.lower() == 'bidsize':
            featuresPD['bidsize_open'] = dataPD.bidsize_open
            featuresPD['bidsize_high'] = dataPD.bidsize_high
            featuresPD['bidsize_low'] = dataPD.bidsize_low
            featuresPD['bidsize_close'] = dataPD.bidsize_close

        elif feature.lower() == 'ofrsize':
            featuresPD['ofrsize_open'] = dataPD.ofrsize_open
            featuresPD['ofrsize_high'] = dataPD.ofrsize_high
            featuresPD['ofrsize_low'] = dataPD.ofrsize_low
            featuresPD['ofrsize_close'] = dataPD.ofrsize_close

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

        # Adjust price feature
    if 'pastobs' in listOfFeatures:
        if feature_lags > 0:
            priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
            # print(priceCols)
            tempClose = featuresPD.close_lag0.copy(deep=True)
#             print('\n')

#             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
            featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
#             print('\n')
#             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
#             print(tempClose)
            featuresPD.loc[:,'close_lag0'] = tempClose
        else:
#             tempClose = copy.deepcopy(featuresPD.close.values)
            tempClose = featuresPD.close.copy(deep=True)

#             print(tempClose)
#             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
            featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
#             print('\n')
#             print(featuresPD.loc[:,['open','high','low','close']])
#             print(featuresPD.close)
#             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
            featuresPD.loc[:,'close'] = tempClose

    return featuresPD


# multi_v1 is multi-Ticker (having support of multi-Ticker in previous versions become too messy)
def generateFeatures_multi_v1(data, listOfFeatures=[], feature_lags=1):
    # The input data is build up as follows:
    # Open, high, low and close.
#     dataPD = pd.DataFrame({'open':data[:,0],
#                              'high':data[:,1],
#                              'low':data[:,2],
#                              'close':data[:,3]})
    #dataPD = data.copy(deep=True)
#     print(dataPD.columns)
#     print([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

    ### in latest data extraction (constructing candles on-the-fly), the column are already fine
#     dataPD.columns = np.concatenate([[j+'_'+i for i in dataPD.columns.get_level_values(1).unique()] for j in dataPD.columns.get_level_values(0).unique()])

# #     dataPD = dataPD.loc[:,['price_open','price_high','price_low','price_close']].rename(columns=['open','high','low','close'])
#     dataPD = dataPD.rename(columns={'price_open':'open',
#                                     'price_high':'high',
#                                     'price_low':'low',
#                                     'price_close':'close'})

    multi_features = pd.DataFrame()

    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)

        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = dataPD.spread_open
                featuresPD['spread_high'] = dataPD.spread_high
                featuresPD['spread_low'] = dataPD.spread_low
                featuresPD['spread_close'] = dataPD.spread_close

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

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

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        print(ticker_name + " done")

    return multi_features

# Includes sector dummies and split of macd-related features into separate calls.
def generateFeatures_multi_v2(data, listOfFeatures=[], feature_lags=1,stockTable = None):

    # try:
    if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None

    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)

        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = dataPD.spread_open
                featuresPD['spread_high'] = dataPD.spread_high
                featuresPD['spread_low'] = dataPD.spread_low
                featuresPD['spread_close'] = dataPD.spread_close

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close)

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

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
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

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")

    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        sectors = data.pop('sector')
        multi_features = pd.concat([multi_features, pd.get_dummies(sectors
                                                                , prefix='sector'
                                                                , drop_first=False)]
                                , axis=1)

    return multi_features

def generateFeatures_multi_v3(data,
                                listOfFeatures=[],
                                feature_lags=1,
                                stockTable = None):

    # try:
    if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None

    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)

        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = dataPD.spread_open
                featuresPD['spread_high'] = dataPD.spread_high
                featuresPD['spread_low'] = dataPD.spread_low
                featuresPD['spread_close'] = dataPD.spread_close

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close)

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

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
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

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")

    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        sectors = data.pop('sector')
        multi_features = pd.concat([multi_features,
                                    pd.get_dummies(sectors
                                                    ,prefix='d_sector'
                                                    ,drop_first=False)]
                                                    ,axis=1)

    return multi_features

# KLN: Included relative return calculations
# FMNS: making relative sector return code compatible with Azure's reset_index (making level_0 == Column1, etc.)
# Note: There is no additional shift in sector returns (tempSector)
def generateFeatures_multi_v4(data,
                                 listOfFeatures=[],
                                 feature_lags=1,
                                 #stockTable = None,
                                 sectorETFS=None):

    # try:
    #if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        #data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        # table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
        #                        index=['level_0','level_1'],columns='Ticker')
        try:
            table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
                               index=['level_0','level_1'],columns='Ticker')
        except:
            table = pd.pivot_table(sectorETFS.reset_index()[['Column1', 'Column2','close','Ticker']],
                               index=['Column1', 'Column2'],columns='Ticker')
        table.columns = table.columns.get_level_values(1)

        tempSector = pd.DataFrame(np.concatenate([np.array([0 for i in np.arange(table.shape[1])])\
                                          .reshape((1,table.shape[1])),
                                          ((table.values[1:]/table.values[0:-1])-1)]),
                          index=table.index,
                          columns=table.columns).fillna(0)


    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)
#         print(ticker_name)
        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = dataPD.spread_open
                featuresPD['spread_high'] = dataPD.spread_high
                featuresPD['spread_low'] = dataPD.spread_low
                featuresPD['spread_close'] = dataPD.spread_close

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close)

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

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
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

            # Sector return
            elif feature.lower() == 'sector':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                             dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)
#                 print(returnPD.shape)
                relativeReturns = pd.DataFrame(returnPD.values - tempSector.values,
                                               columns=tempSector.columns,
                                               index=tempSector.index)

                featuresPD[['relReturns_'+i for i in relativeReturns.columns]] = relativeReturns
#                 featuresPD[relativeReturns.columns.str()] = relativeReturns
#                 colnames =
#                 featuresPD['relative_return'] = pd.concat([APPLE,relativeReturns],axis=1)



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

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")
#     print(multi_features.index)
    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        #sectors = data.pop('sector')
        sectors = data.loc[:,'sector'].copy(deep=True)
        tempIndices = multi_features.index
        multi_features = multi_features.reset_index(drop=True)
#         print(multi_features.index)
#         print(pd.get_dummies(sectors
#                             , prefix='d_sector'
#                             , drop_first=False))
        multi_features = pd.concat([multi_features, pd.get_dummies(sectors
                                                                , prefix='d_sector'
                                                                , drop_first=False).reset_index(drop=True)]
                                                                , axis=1)
        multi_features.index = tempIndices

    return multi_features

# Included intraday time and past returns as features
# added option change between past obs in percentage or not.
def generateFeatures_multi_v5(data,
                                 listOfFeatures=[],
                                 feature_lags=1,
                                 #stockTable = None,
                                 sectorETFS=None,
                                 pastobs_in_percentage = False):

    # try:
    #if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        #data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        # table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
        #                        index=['level_0','level_1'],columns='Ticker')
        try:
            table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
                               index=['level_0','level_1'],columns='Ticker')
        except:
            table = pd.pivot_table(sectorETFS.reset_index()[['Column1', 'Column2','close','Ticker']],
                               index=['Column1', 'Column2'],columns='Ticker')
        table.columns = table.columns.get_level_values(1)

        tempSector = pd.DataFrame(np.concatenate([np.array([0 for i in np.arange(table.shape[1])])\
                                          .reshape((1,table.shape[1])),
                                          ((table.values[1:]/table.values[0:-1])-1)]),
                          index=table.index,
                          columns=table.columns).fillna(0)

    # Calculate number of days in data
    nDays = data.index.get_level_values(0).unique().shape[0]

    # Getting the number of candles per hour
    candles_per_hour = data.index.get_level_values(1).unique().shape[0]/6.5

    # Setting up the dataframe
    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)
#         print(ticker_name)
        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = dataPD.spread_open
                featuresPD['spread_high'] = dataPD.spread_high
                featuresPD['spread_low'] = dataPD.spread_low
                featuresPD['spread_close'] = dataPD.spread_close

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close)

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

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
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

            # Sector return
            elif feature.lower() == 'sector':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                                        dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)
#                 print(returnPD.shape)
                relativeReturns = pd.DataFrame(returnPD.values - tempSector.values,
                                               columns=tempSector.columns,
                                               index=tempSector.index)

                featuresPD[['relReturns_'+i for i in relativeReturns.columns]] = relativeReturns
#                 featuresPD[relativeReturns.columns.str()] = relativeReturns
#                 colnames =
#                 featuresPD['relative_return'] = pd.concat([APPLE,relativeReturns],axis=1)

            # past return
            elif feature.lower() == 'pastreturns':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                                        dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)

                featuresPD.loc[:,'pastreturns'] = returnPD.values

            # time bandit
            elif feature.lower() == 'intradaytime':
#                 print(dataPD.shape)
                intradaytime = pd.DataFrame({'return':9.5+(1/candles_per_hour)+data.index.get_level_values(1).unique()/candles_per_hour})
                # print(days)
                fullPeriodIntradayTime = np.tile(intradaytime.values,(nDays,1)).flatten()

                featuresPD.loc[:,'intradaytime'] = fullPeriodIntradayTime

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
                ## New
                featuresPD.index = featuresPD.index.shift( - (feature_lags - roll_i))

            # remove all raw features
            featuresPD = featuresPD.loc[:, ~featuresPD.columns.isin(all_raw_features)]

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                # featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].divide(featuresPD.close_lag0, axis=0)-1
                else:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                # featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].divide(featuresPD.close, axis=0)-1
                else:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")
#     print(multi_features.index)
    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        #sectors = data.pop('sector')
        sectors = data.loc[:,'sector'].copy(deep=True)
        tempIndices = multi_features.index
        multi_features = multi_features.reset_index(drop=True)
#         print(multi_features.index)
#         print(pd.get_dummies(sectors
#                             , prefix='d_sector'
#                             , drop_first=False))
        multi_features = pd.concat([multi_features, pd.get_dummies(sectors
                                                                , prefix='d_sector'
                                                                , drop_first=False).reset_index(drop=True)]
                                                                , axis=1)
        multi_features.index = tempIndices

    return multi_features

# Included intraday time and past returns as features
# added option change between past obs in percentage or not.
##### included ,
##      fillna=True
def generateFeatures_multi_v6(data,
                                 listOfFeatures=[],
                                 feature_lags=1,
                                 #stockTable = None,
                                 sectorETFS=None,
                                 pastobs_in_percentage = False):

    # try:
    #if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        #data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        # table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
        #                        index=['level_0','level_1'],columns='Ticker')
        try:
            table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
                               index=['level_0','level_1'],columns='Ticker')
        except:
            table = pd.pivot_table(sectorETFS.reset_index()[['Column1', 'Column2','close','Ticker']],
                               index=['Column1', 'Column2'],columns='Ticker')
        table.columns = table.columns.get_level_values(1)

        tempSector = pd.DataFrame(np.concatenate([np.array([0 for i in np.arange(table.shape[1])])\
                                          .reshape((1,table.shape[1])),
                                          ((table.values[1:]/table.values[0:-1])-1)]),
                          index=table.index,
                          columns=table.columns).fillna(0)

    # Calculate number of days in data
    nDays = data.index.get_level_values(0).unique().shape[0]

    # Getting the number of candles per hour
    candles_per_hour = data.index.get_level_values(1).unique().shape[0]/6.5

    # Setting up the dataframe
    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)
#         print(ticker_name)
        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = dataPD.spread_open
                featuresPD['spread_high'] = dataPD.spread_high
                featuresPD['spread_low'] = dataPD.spread_low
                featuresPD['spread_close'] = dataPD.spread_close

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close,
                                                fillna=True)

                # Adding the feature
                featuresPD['stok'] = tempFeatures

            # Stochastic D
            elif feature.lower() == 'stod':

                tempFeatures= ta.momentum.stoch_signal(dataPD.high,
                                                       dataPD.low,
                                                       dataPD.close,
                                                       fillna=True)
                # Adding the feature
                featuresPD['stod'] = tempFeatures

            # Slow Stochastic D
            elif feature.lower() == 'sstod':

                tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
                                                                              dataPD.low,
                                                                              dataPD.close,
                                                                              fillna=True))
                # Adding the feature
                featuresPD['sstod'] = tempFeatures

            # Williams %R
            elif feature.lower() == 'wilr':

                tempFeatures= ta.momentum.wr(dataPD.high,
                                             dataPD.low,
                                             dataPD.close,
                                             fillna=True)
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
                                           dataPD.close,
                                           fillna=True)
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

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
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

            # Sector return
            elif feature.lower() == 'sector':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                                        dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)
#                 print(returnPD.shape)
                relativeReturns = pd.DataFrame(returnPD.values - tempSector.values,
                                               columns=tempSector.columns,
                                               index=tempSector.index)

                featuresPD[['relReturns_'+i for i in relativeReturns.columns]] = relativeReturns
#                 featuresPD[relativeReturns.columns.str()] = relativeReturns
#                 colnames =
#                 featuresPD['relative_return'] = pd.concat([APPLE,relativeReturns],axis=1)

            # past return
            elif feature.lower() == 'pastreturns':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                                        dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)

                featuresPD.loc[:,'pastreturns'] = returnPD.values

            # time bandit
            elif feature.lower() == 'intradaytime':
#                 print(dataPD.shape)
                intradaytime = pd.DataFrame({'return':9.5+(1/candles_per_hour)+data.index.get_level_values(1).unique()/candles_per_hour})
                # print(days)
                fullPeriodIntradayTime = np.tile(intradaytime.values,(nDays,1)).flatten()

                featuresPD.loc[:,'intradaytime'] = fullPeriodIntradayTime

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
                ## New
                featuresPD.index = featuresPD.index.shift( - (feature_lags - roll_i))

            # remove all raw features
            featuresPD = featuresPD.loc[:, ~featuresPD.columns.isin(all_raw_features)]

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                # featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].divide(featuresPD.close_lag0, axis=0)-1
                else:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                # featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].divide(featuresPD.close, axis=0)-1
                else:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")
#     print(multi_features.index)
    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        #sectors = data.pop('sector')
        sectors = data.loc[:,'sector'].copy(deep=True)
        tempIndices = multi_features.index
        multi_features = multi_features.reset_index(drop=True)
#         print(multi_features.index)
#         print(pd.get_dummies(sectors
#                             , prefix='d_sector'
#                             , drop_first=False))
        multi_features = pd.concat([multi_features, pd.get_dummies(sectors
                                                                , prefix='d_sector'
                                                                , drop_first=False).reset_index(drop=True)]
                                                                , axis=1)
        multi_features.index = tempIndices

    return multi_features

## Updated the features with the following chanages;
#       Removed close prise in levels (across all lags)
#       SMA/EMA: Changed to difference to close price
#       Disparity are now in percentage, insterad of a share previous (only done to match definition)
#       Removed SStod as it where the same as Stod
#       Spread: Now normalised with the corresponding prise instead of being in levels.
def generateFeatures_multi_v7(data,
                                 listOfFeatures=[],
                                 feature_lags=1,
                                 #stockTable = None,
                                 sectorETFS=None,
                                 pastobs_in_percentage = False):

    # try:
    #if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        #data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        # table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
        #                        index=['level_0','level_1'],columns='Ticker')
        try:
            table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
                               index=['level_0','level_1'],columns='Ticker')
        except:
            table = pd.pivot_table(sectorETFS.reset_index()[['Column1', 'Column2','close','Ticker']],
                               index=['Column1', 'Column2'],columns='Ticker')
        table.columns = table.columns.get_level_values(1)

        tempSector = pd.DataFrame(np.concatenate([np.array([0 for i in np.arange(table.shape[1])])\
                                          .reshape((1,table.shape[1])),
                                          ((table.values[1:]/table.values[0:-1])-1)]),
                          index=table.index,
                          columns=table.columns).fillna(0)

    # Calculate number of days in data
    nDays = data.index.get_level_values(0).unique().shape[0]

    # Getting the number of candles per hour
    candles_per_hour = data.index.get_level_values(1).unique().shape[0]/6.5

    # Setting up the dataframe
    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)
#         print(ticker_name)
        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD['open'] = dataPD.open
                featuresPD['high'] = dataPD.high
                featuresPD['low'] = dataPD.low
                featuresPD['close'] = dataPD.close

            ## Made it normalised spreads.
            elif feature.lower() == 'spread':
                featuresPD['spread_open'] = (dataPD.spread_open / dataPD.open)*100
                featuresPD['spread_high'] = (dataPD.spread_high / dataPD.high)*100
                featuresPD['spread_low'] = (dataPD.spread_low / dataPD.low)*100
                featuresPD['spread_close'] = (dataPD.spread_close / dataPD.close)*100

            elif feature.lower() == 'bidsize':
                featuresPD['bidsize_open'] = dataPD.bidsize_open
                featuresPD['bidsize_high'] = dataPD.bidsize_high
                featuresPD['bidsize_low'] = dataPD.bidsize_low
                featuresPD['bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD['ofrsize_open'] = dataPD.ofrsize_open
                featuresPD['ofrsize_high'] = dataPD.ofrsize_high
                featuresPD['ofrsize_low'] = dataPD.ofrsize_low
                featuresPD['ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close,
                                                fillna=True)

                # Adding the feature
                featuresPD['stok'] = tempFeatures

            # Stochastic D
            elif feature.lower() == 'stod':

                tempFeatures= ta.momentum.stoch_signal(dataPD.high,
                                                       dataPD.low,
                                                       dataPD.close,
                                                       fillna=True)
                # Adding the feature
                featuresPD['stod'] = tempFeatures

            ## Removed as it is the same as the above
            # Slow Stochastic D
            # elif feature.lower() == 'sstod':
            #
            #     tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
            #                                                                   dataPD.low,
            #                                                                   dataPD.close,
            #                                                                   fillna=True))
            #     # Adding the feature
            #     featuresPD['sstod'] = tempFeatures

            ## Remved as it provides some issues
            # Williams %R
            # elif feature.lower() == 'wilr':
            #
            #     tempFeatures= ta.momentum.wr(dataPD.high,
            #                                  dataPD.low,
            #                                  dataPD.close,
            #                                  fillna=True)
            #     # Adding the feature
            #     featuresPD['wilr'] = tempFeatures

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
                                           dataPD.close
                                           ,fillna=True)
                # Adding the feature
                featuresPD['cci'] = tempFeatures

             # Detrended Price Ocillator
            elif feature.lower() == 'dpo':

                tempFeatures= ta.trend.dpo(dataPD.close)

                # Adding the feature
                featuresPD['dpo'] = tempFeatures

            ## Changed to return the difference to the currenct close price, instead of simply the SMA
            # Simple Moving Average
            elif feature.lower() == 'sma':

                tempFeatures = dataPD.close - ta.trend.sma_indicator(dataPD.close)

                # Adding the feature
                featuresPD['sma'] = tempFeatures

            ## Changed to return the difference to the currenct close price, instead of simply the EMA
            # Exponential Moving Average
            elif feature.lower() == 'ema':

                tempFeatures= dataPD.close - ta.trend.ema_indicator(dataPD.close)

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

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD['macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
                featuresPD['macd_signal'] = ta.trend.macd_signal(dataPD.close)

             # Disparity 5
            elif feature.lower() == 'dis5':

                tempFeatures= ((dataPD.close/ta.trend.sma_indicator(dataPD.close,5))-1)*100

                # Adding the feature
                featuresPD['dis5'] = tempFeatures

            # Disparity 10
            elif feature.lower() == 'dis10':

                tempFeatures= ((dataPD.close/ta.trend.sma_indicator(dataPD.close,10))-1)*100

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

            # Sector return
            elif feature.lower() == 'sector':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                             dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)
#                 print(returnPD.shape)
                relativeReturns = pd.DataFrame(returnPD.values - tempSector.values,
                                               columns=tempSector.columns,
                                               index=tempSector.index)

                featuresPD[['relReturns_'+i for i in relativeReturns.columns]] = relativeReturns
#                 featuresPD[relativeReturns.columns.str()] = relativeReturns
#                 colnames =
#                 featuresPD['relative_return'] = pd.concat([APPLE,relativeReturns],axis=1)

            # past return
            elif feature.lower() == 'pastreturns':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                             dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)

                featuresPD.loc[:,'pastreturns'] = returnPD.values

            # time bandit
            elif feature.lower() == 'intradaytime':
#                 print(dataPD.shape)
                intradaytime = pd.DataFrame({'return':9.5+(1/candles_per_hour)+data.index.get_level_values(1).unique()/candles_per_hour})
                # print(days)
                fullPeriodIntradayTime = np.tile(intradaytime.values,(nDays,1)).flatten()

                featuresPD.loc[:,'intradaytime'] = fullPeriodIntradayTime

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

                if roll_i > 0:
                    featuresPD.index = pd.MultiIndex.from_arrays([pd.Series(featuresPD.index.get_level_values(0)).shift(-1),pd.Series(featuresPD.index.get_level_values(1)).shift(-1)])
            # remove all raw features
            featuresPD = featuresPD.loc[:, ~featuresPD.columns.isin(all_raw_features)]

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                # featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].divide(featuresPD.close_lag0, axis=0)-1
                else:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                # featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].divide(featuresPD.close, axis=0)-1
                else:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD['ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")
#     print(multi_features.index)
    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        #sectors = data.pop('sector')
        sectors = data.loc[:,'sector'].copy(deep=True)
        tempIndices = multi_features.index
        multi_features = multi_features.reset_index(drop=True)
#         print(multi_features.index)
#         print(pd.get_dummies(sectors
#                             , prefix='d_sector'
#                             , drop_first=False))
        multi_features = pd.concat([multi_features, pd.get_dummies(sectors
                                                                , prefix='d_sector'
                                                                , drop_first=False).reset_index(drop=True)]
                                                                , axis=1)
        multi_features.index = tempIndices

        closeCols = np.concatenate([[c for c in multi_features.columns if c.startswith(t,0,len(t))] for t in ['close']])

        multi_features_wo_close = multi_features.drop(closeCols,axis = 1)

    return multi_features_wo_close

# THis is v7
def generateFeatures_multi_final(data,
                                 listOfFeatures=[],
                                 feature_lags=1,
                                 #stockTable = None,
                                 sectorETFS=None,
                                 pastobs_in_percentage = False):

    # try:
    #if (stockTable is not None) & ('sector' not in data.columns.str.lower()):
# if (stockTable != None) & ('sector' not in data.columns.str.lower()):
    # Appending the stock information to the data.
        #data = data.merge(stockTable[['ticker','sector']],left_on='Ticker',right_on='ticker',how='left')
    # except:
        # None
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        # table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
        #                        index=['level_0','level_1'],columns='Ticker')
        try:
            table = pd.pivot_table(sectorETFS.reset_index()[['level_0','level_1','close','Ticker']],
                               index=['level_0','level_1'],columns='Ticker')
        except:
            table = pd.pivot_table(sectorETFS.reset_index()[['Column1', 'Column2','close','Ticker']],
                               index=['Column1', 'Column2'],columns='Ticker')
        table.columns = table.columns.get_level_values(1)

        tempSector = pd.DataFrame(np.concatenate([np.array([0 for i in np.arange(table.shape[1])])\
                                          .reshape((1,table.shape[1])),
                                          ((table.values[1:]/table.values[0:-1])-1)]),
                          index=table.index,
                          columns=table.columns).fillna(0)

    # Calculate number of days in data
    nDays = data.index.get_level_values(0).unique().shape[0]

    # Getting the number of candles per hour
    candles_per_hour = data.index.get_level_values(1).unique().shape[0]/6.5

    # Setting up the dataframe
    multi_features = pd.DataFrame()
    # print(data.columns)
    for ticker_iter, ticker_name in enumerate(data.Ticker.unique()):

        featuresPD = pd.DataFrame()
        dataPD = data[data.Ticker==ticker_name].copy(deep=True)
#         print(ticker_name)
        for feature in listOfFeatures:

            # Past observations
            if feature.lower() == 'pastobs':
                featuresPD.loc[:,'open'] = dataPD.open
                featuresPD.loc[:,'high'] = dataPD.high
                featuresPD.loc[:,'low'] = dataPD.low
                featuresPD.loc[:,'close'] = dataPD.close

            ## Made it normalised spreads.
            elif feature.lower() == 'spread':
                featuresPD.loc[:,'spread_open'] = dataPD.spread_open / dataPD.open
                featuresPD.loc[:,'spread_high'] = dataPD.spread_high / dataPD.high
                featuresPD.loc[:,'spread_low'] = dataPD.spread_low / dataPD.low
                featuresPD.loc[:,'spread_close'] = dataPD.spread_close / dataPD.close

            elif feature.lower() == 'bidsize':
                featuresPD.loc[:,'bidsize_open'] = dataPD.bidsize_open
                featuresPD.loc[:,'bidsize_high'] = dataPD.bidsize_high
                featuresPD.loc[:,'bidsize_low'] = dataPD.bidsize_low
                featuresPD.loc[:,'bidsize_close'] = dataPD.bidsize_close

            elif feature.lower() == 'ofrsize':
                featuresPD.loc[:,'ofrsize_open'] = dataPD.ofrsize_open
                featuresPD.loc[:,'ofrsize_high'] = dataPD.ofrsize_high
                featuresPD.loc[:,'ofrsize_low'] = dataPD.ofrsize_low
                featuresPD.loc[:,'ofrsize_close'] = dataPD.ofrsize_close

            # Stochastic K
            elif feature.lower() == 'stok':

                tempFeatures= ta.momentum.stoch(dataPD.high,
                                                dataPD.low,
                                                dataPD.close,
                                                fillna=True) / 100

                # Adding the feature
                featuresPD.loc[:,'stok'] = tempFeatures

            # Stochastic D
            elif feature.lower() == 'stod':

                tempFeatures= ta.momentum.stoch_signal(dataPD.high,
                                                       dataPD.low,
                                                       dataPD.close,
                                                       fillna=True) / 100
                # Adding the feature
                featuresPD.loc[:,'stod'] = tempFeatures

            # # Slow Stochastic K
            # elif feature.lower() == 'sstok':
            #
            #     tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
            #                                                                   dataPD.low,
            #                                                                   dataPD.close,
            #                                                                   fillna=True)) / 100
            #     # Adding the feature
            #     featuresPD['sstok'] = tempFeatures
            #
            # ## Removed as it is the same as the above
            # # Slow Stochastic D
            # elif feature.lower() == 'sstod':
            #
            #     tempFeatures= ta.trend.sma_indicator(ta.momentum.stoch_signal(dataPD.high,
            #                                                                   dataPD.low,
            #                                                                   dataPD.close,
            #                                                                   fillna=True)) / 100
            #     # Adding the feature
            #     featuresPD['sstod'] = tempFeatures

            ## Remved as it provides some issues
            # Williams %R
            # elif feature.lower() == 'wilr':
            #
            #     tempFeatures= ta.momentum.wr(dataPD.high,
            #                                  dataPD.low,
            #                                  dataPD.close,
            #                                  fillna=True)
            #     # Adding the feature
            #     featuresPD['wilr'] = tempFeatures

            # Rate Of Change
            elif feature.lower() == 'roc':

                tempFeatures= ta.momentum.roc(dataPD.close)

                # Adding the feature
                featuresPD.loc[:,'roc'] = tempFeatures

            # Relative Strength Index
            elif feature.lower() == 'rsi':

                tempFeatures= ta.momentum.rsi(dataPD.close) / 100

                # Adding the feature
                featuresPD.loc[:,'rsi'] = tempFeatures

            # Average True Range
            elif feature.lower() == 'atr':

                tempFeatures= ta.volatility.average_true_range(dataPD.high,
                                                               dataPD.low,
                                                               dataPD.close)
                # Adding the feature
                featuresPD.loc[:,'atr'] = tempFeatures

            # Commodity Channel Index
            elif feature.lower() == 'cci':

                tempFeatures= ta.trend.cci(dataPD.high,
                                           dataPD.low,
                                           dataPD.close
                                           ,fillna=True) / 100
                # Adding the feature
                featuresPD.loc[:,'cci'] = tempFeatures

             # Detrended Price Ocillator
            elif feature.lower() == 'dpo':

                tempFeatures= ta.trend.dpo(dataPD.close)

                # Adding the feature
                featuresPD.loc[:,'dpo'] = tempFeatures

            ## Changed to return the difference to the currenct close price, instead of simply the SMA
            # Simple Moving Average
            elif feature.lower() == 'sma':

                tempFeatures = dataPD.close - ta.trend.sma_indicator(dataPD.close,10)## Chosen based on ref (Kara et al.)

                # Adding the feature
                featuresPD.loc[:,'sma'] = tempFeatures

            ## Changed to return the difference to the currenct close price, instead of simply the EMA
            # Exponential Moving Average
            elif feature.lower() == 'ema':

                tempFeatures= dataPD.close - ta.trend.ema_indicator(dataPD.close)

                # Adding the feature
                featuresPD.loc[:,'ema'] = tempFeatures

            # Moving Average Convergence Divergence
            elif feature.lower() == 'macd':

                # note: having all 3 causes multicollinearity. Maybe not a problem in ML, let's see :-)
                # macd is the difference between two EMAs
                # macd_signal is an EMA of the above macd line
                # macd_diff is the so-called histogram (just bars really) of the time-wise difference between macd and macd_signal

                # Adding the features
                featuresPD.loc[:,'macd'] = ta.trend.macd(dataPD.close)

            # Moving Average Convergence Divergence Difference
            elif feature.lower() == 'macd_diff':
                # Adding the features
                featuresPD.loc[:,'macd_diff'] = ta.trend.macd_diff(dataPD.close)

            # Moving Average Convergence Divergence Signal
            elif feature.lower() == 'macd_signal':
                # Adding the features
                featuresPD.loc[:,'macd_signal'] = ta.trend.macd_signal(dataPD.close)

             # Disparity 5
            elif feature.lower() == 'dis5':

                tempFeatures= ((dataPD.close/ta.trend.sma_indicator(dataPD.close,5))-1)*100

                # Adding the feature
                featuresPD.loc[:,'dis5'] = tempFeatures

            # Disparity 10
            elif feature.lower() == 'dis10':

                tempFeatures= ((dataPD.close/ta.trend.sma_indicator(dataPD.close,10))-1)*100

                # Adding the feature
                featuresPD.loc[:,'dis10'] = tempFeatures

            # # Bollinger Bands
            # elif feature.lower() == 'bb':
            #
            #     # Define Bollinger Bands function to extract from
            #     bb_function = ta.volatility.BollingerBands(close=dataPD.close, n=20, ndev=2)
            #
            #     # Adding the features
            #     featuresPD['bb_mavg'] = bb_function.bollinger_mavg()
            #     featuresPD['bb_hband'] = bb_function.bollinger_hband()
            #     featuresPD['bb_lband'] = bb_function.bollinger_lband()
            #     featuresPD['bb_pband'] = bb_function.bollinger_pband()
            #     featuresPD['bb_wband'] = bb_function.bollinger_wband()

            # Sector return
            elif feature.lower() == 'sector':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                             dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)
#                 print(returnPD.shape)
                relativeReturns = pd.DataFrame(returnPD.values - tempSector.values,
                                               columns=tempSector.columns,
                                               index=tempSector.index)

                featuresPD[['relReturns_'+i for i in relativeReturns.columns]] = relativeReturns
#                 featuresPD[relativeReturns.columns.str()] = relativeReturns
#                 colnames =
#                 featuresPD['relative_return'] = pd.concat([APPLE,relativeReturns],axis=1)

            # past return
            elif feature.lower() == 'pastreturns':
#                 print(dataPD.shape)
                returnPD = pd.DataFrame({'return':np.concatenate([[0],(((dataPD.close.values[1:]/\
                                                             dataPD.close.values[0:-1]))-1)])},
                                            index=dataPD.index).fillna(0)

                featuresPD.loc[:,'pastreturns'] = returnPD.values

            # time bandit
            elif feature.lower() == 'intradaytime':
#                 print(dataPD.shape)
                intradaytime = pd.DataFrame({'return':9.5+(1/candles_per_hour)+data.index.get_level_values(1).unique()/candles_per_hour - 12.75})  ## <-- to demean (16 + 9.5) / 2
                # print(days)
                fullPeriodIntradayTime = np.tile(intradaytime.values,(nDays,1)).flatten()

                featuresPD.loc[:,'intradaytime'] = fullPeriodIntradayTime

        # if we want any lags:
        if feature_lags > 0:

            # collect names of all raw features (before any lagging)
            all_raw_features = featuresPD.columns

            # loop through each lag and shift all features at once
            for roll_i in np.arange(feature_lags + 1): # + 1 as we treat feature_lags = 1 as having both lag0 and lag1

                #### Original

                # define new column name (feature_name_ + lagX) where X = roll_i is the shifting parameter
                new_col_names = [feature_name + '_lag' + str(roll_i) for feature_name in all_raw_features]

                # Shift/roll all raw features with the shifting parameter roll_i and save as new columns.
                # The shift parameter must be negative (we want lag0 to be the 'newest'/'latest')
                featuresPD[new_col_names] = featuresPD[all_raw_features].shift( - (feature_lags - roll_i))

                if roll_i > 0:
                    featuresPD.index = pd.MultiIndex.from_arrays([pd.Series(featuresPD.index.get_level_values(0)).shift(-1),pd.Series(featuresPD.index.get_level_values(1)).shift(-1)])
                # featuresPD.index = pd.MultiIndex.from_arrays([pd.Series(featuresPD.index.get_level_values(0)).shift(- (feature_lags - roll_i)),pd.Series(featuresPD.index.get_level_values(1)).shift(- (feature_lags - roll_i))])

                # define new column name (feature_name_ + lagX) where X = roll_i is the shifting parameter
                # new_col_names = [feature_name + '_lag' + str(roll_i) for feature_name in all_raw_features]
                #
                # # Shift/roll all raw features with the shifting parameter roll_i and save as new columns.
                # # The shift parameter must be negative (we want lag0 to be the 'newest'/'latest')
                #
                #
                # if roll_i > 0:
                #     featuresPD[new_col_names] = featuresPD[all_raw_features].shift( - (feature_lags - roll_i))
                #     featuresPD.index = pd.MultiIndex.from_arrays([pd.Series(featuresPD.index.get_level_values(0)).shift(-1),pd.Series(featuresPD.index.get_level_values(1)).shift(-1)])
                # else:
                #     featuresPD[new_col_names] = featuresPD[all_raw_features]
            # remove all raw features
            featuresPD = featuresPD.loc[:, ~featuresPD.columns.isin(all_raw_features)]

            # Adjust price feature
        if 'pastobs' in listOfFeatures:
            if feature_lags > 0:
                priceCols = np.concatenate([[c for c in featuresPD.columns if c.startswith(t,0,len(t))] for t in ['open','high','low','close']])#[0:len(t)] == t
                # print(priceCols)
                tempClose = featuresPD.close_lag0.copy(deep=True)
    #             print('\n')

    #             featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols] - featuresPD.close_lag0
                # featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].divide(featuresPD.close_lag0, axis=0)-1
                else:
                    featuresPD.loc[:,priceCols] = featuresPD.loc[:,priceCols].subtract(featuresPD.close_lag0,axis=0)
    #             print('\n')
    #             print([featuresPD.loc[:,priceCols] - featuresPD.close_lag0][0:5])
    #             print(tempClose)
                featuresPD.loc[:,'close_lag0'] = tempClose
            else:
    #             tempClose = copy.deepcopy(featuresPD.close.values)
                tempClose = featuresPD.close.copy(deep=True)

    #             print(tempClose)
    #             featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']] - featuresPD.close
                # featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
                if pastobs_in_percentage:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].divide(featuresPD.close, axis=0)-1
                else:
                    featuresPD.loc[:,['open','high','low','close']] = featuresPD.loc[:,['open','high','low','close']].subtract(featuresPD.close,axis=0)
    #             print('\n')
    #             print(featuresPD.loc[:,['open','high','low','close']])
    #             print(featuresPD.close)
    #             print([featuresPD.loc[:,['open','high','low','close']] - featuresPD.close][0:5])
                featuresPD.loc[:,'close'] = tempClose

        featuresPD.loc[:,'ticker'] = ticker_name

        # append
        multi_features = pd.concat([multi_features, featuresPD])
        # print(ticker_name + " done")
#     print(multi_features.index)
    # Finally adding sector dummies if needed
    # Sector Dummies
    if 'sector' in [ele.lower() for ele in listOfFeatures]:

        ## Adding Sector dummies
        #sectors = data.pop('sector')
        sectors = data.loc[:,'sector'].copy(deep=True)
        tempIndices = multi_features.index
        multi_features = multi_features.reset_index(drop=True)
#         print(multi_features.index)
#         print(pd.get_dummies(sectors
#                             , prefix='d_sector'
#                             , drop_first=False))
        multi_features = pd.concat([multi_features, pd.get_dummies(sectors
                                                                , prefix='d_sector'
                                                                , drop_first=False).reset_index(drop=True)]
                                                                , axis=1)
        multi_features.index = tempIndices


    ## Removing "close" columns
    closeCols = np.concatenate([[c for c in multi_features.columns if c.startswith(t,0,len(t))] for t in ['close']])

    multi_features_wo_close = multi_features.drop(closeCols,axis = 1)

    return multi_features_wo_close
