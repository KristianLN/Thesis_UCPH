import numpy as np
import pandas as pd
import re
import time
import copy
import datetime



def HFDataCleaning(cleaningProcedures,dataToClean,dataType,p3Exchanges = []):

    # There are 11 cleaning procedures, with 3 relevant for both trade and quote data and 4 for either trade or quote data.
    # The cleaning procedures are listed below for simplicity

    # Applicable for both trade and quote data

    # P1. Delete entries with a time stamp outside the 9:30 am to 4 pm window when the exchange is open.
    # P2. Delete entries with a bid, ask or transaction price equal to zero.
    # P3. Retain entries originating from a single exchange. Delete other entries.

    # Applicable for just trade data

    # T1. Delete entries with corrected trades. (Trades with a Correction Indicator, CORR != 0).
    # T2. Delete entries with abnormal Sale Condition. (Trades where COND has a letter code, except for �E� and �F�).
    # T3. If multiple transactions have the same time stamp: use the median price.
    # T4. Delete entries with prices that are above the ask plus the bid-ask spread.
    # Similar for entries with prices below the bid minus the bid-ask spread.

    # Applicable for just quote data

    # Q1. When multiple quotes have the same timestamp, we replace all these with a single entry
    # with the median bid and median ask price.
    # Q2. Delete entries for which the spread is negative.
    # Q3. Delete entries for which the spread is more that 50 times the median spread on that day.
    # Q4. Delete entries for which the mid-quote deviated by more than 5 median absolute deviations from
    # a centered median (excluding the observation under consideration) of 50 observations.

    # Some comments, by (Lunde,2016), on the relative importance of the individual cleaning procedures

    # ? By far the most important rules here are P3, T3 and Q1.
    # ? In our empirical work we will see the impact of suspending P3. It is used to reduce the impact
    # of time-delays in the reporting of trades and quote updates.
    # ? Some form of T3 and Q1 rule seems inevitable here, and it is these rules which lead to the largest deletion of data.
    # ? T4 is an attractive rule, as it disciplines the trade data using quotes. However, it has the disadvantage
    # that it cannot be applied when quote data is not available.
    # ? In situations where quote data is not available, Q4 can be applied to the transaction prices in place of T4.

    dataType = dataType.lower().strip()



    for cp in cleaningProcedures:

        cp = cp.lower().strip()


        # check if cp is sensible
        if (cp.startswith('t')) & (dataType != 'trade'):
            raise ValueError(f'Cleaning procedure {cp} is not compatible with dataType {dataType}')

        elif (cp.startswith('q')) & (dataType != 'quote'):
            raise ValueError(f'Cleaning procedure {cp} is not compatible with dataType {dataType}')


        # if the cleaning procedure in question is p1.
        if cp == 'p1':
            # ((tradeData.Hour+tradeData.Minute/60)>9.5)&((tradeData.Hour+tradeData.Minute/60)<16)
#             dataToClean = dataToClean[(datetime.timedelta(hours = 9,
#                                                          minutes = 30) <= dataToClean.Timestamp)&\
#                                       (dataToClean.Timestamp <= datetime.timedelta(hours = 16,
#                                                                                    minutes = 0))].reset_index(drop=True)

            # dataToClean = dataToClean[((dataToClean.Hour+dataToClean.Minute/60)>=9.5)&\
            #                           ((dataToClean.Hour+dataToClean.Minute/60)<16)]

            # 27-06-2020 addition
            Timestamp_dt = dataToClean['Timestamp'].dt
            Timestamp_float = Timestamp_dt.hour \
                              + Timestamp_dt.minute/60 \
                              + Timestamp_dt.second/(60*60) \
                              + Timestamp_dt.microsecond/(60*60*10**6)
            #dataToClean['Timestamp_float'] = Timestamp_float

            dataToClean = dataToClean[(Timestamp_float>=9.5)&\
                                      (Timestamp_float<16)]

        # if the cleaning procedure in question is p1.
        if cp == 'p1_2':
            # ((tradeData.Hour+tradeData.Minute/60)>9.5)&((tradeData.Hour+tradeData.Minute/60)<16)
#             dataToClean = dataToClean[(datetime.timedelta(hours = 9,
#                                                          minutes = 30) <= dataToClean.Timestamp)&\
#                                       (dataToClean.Timestamp <= datetime.timedelta(hours = 16,
#                                                                                    minutes = 0))].reset_index(drop=True)
            # dataToClean = dataToClean[((dataToClean.Hour+dataToClean.Minute/60)>=9.0)&\
            #                           ((dataToClean.Hour+dataToClean.Minute/60)<16.5)]

            # 27-06-2020 addition
            Timestamp_dt = dataToClean['Timestamp'].dt
            Timestamp_float = Timestamp_dt.hour \
                              + Timestamp_dt.minute/60 \
                              + Timestamp_dt.second/(60*60) \
                              + Timestamp_dt.microsecond/(60*60*10**6)

            dataToClean = dataToClean[(Timestamp_float>=9.0)&\
                                      (Timestamp_float<16.5)]


        # if the cleaning procedure in question is p2.
        elif cp == 'p2':

            # if the cleaning procedure in question is p1.
            if dataType == 'trade':

                dataToClean = dataToClean[dataToClean.price != 0].reset_index(drop=True)

            elif dataType == 'quote':

                dataToClean = dataToClean[(dataToClean.bid != 0) | (dataToClean.ofr != 0)].reset_index(drop=True)


        # if the cleaning procedure in question is p3.
        elif cp == 'p3':

            if len(p3Exchanges) == 0:

                raise ValueError('No exchanges, to filter on, has been provided.\nPlease provide a list with minimum one exchanges to filter on.')

            else:

                # Ensuring correct format
                p3Exchanges = [ele.lower().strip() for ele in p3Exchanges]

                # Filtering on exchanges ### Consider to use "isin" on the dataToClean.ex-Series instead, to improve execution time.
                dataToClean = dataToClean[[True if ele.lower().strip() in p3Exchanges else False for ele in dataToClean.ex]].reset_index(drop=True)


        # if the cleaning procedure in question is t1.
        # T1. Delete entries with corrected trades. (Trades with a Correction Indicator, CORR != 0).
        elif cp == 't1':

            dataToClean = dataToClean[dataToClean['corr'] == '00'].reset_index(drop=True)


        # if the cleaning procedure in question is t2.
        # T2. Delete entries with abnormal Sale Condition. (Trades where COND has a letter code, except for �E� and �F�).
        # FMNS: Most are COND = '@ XX' such as '@ TI', make sure this works properly. Assuming startswith('@') is cool
        elif cp == 't2':

            dataToClean = dataToClean[(dataToClean.cond.startswith('@')) | (dataToClean.cond in ['E', 'F'])].reset_index(drop=True)


        # if the cleaning procedure in question is t3.
        # T3. If multiple transactions have the same time stamp: use the median price.
        # FMNS: Let's consider if these median prices are cheating in relation to OHLC bars
        elif cp == 't3':

            # get unique timestamps
            unique_ts_idx = np.unique(dataToClean.Timestamp, return_index=True)[1]

            # get median prices
            median_price = dataToClean[['Timestamp', 'price']].groupby('Timestamp')['price'].median().values

            # keep only unique timestamps
            dataToClean = dataToClean.iloc[unique_ts_idx, :].reset_index(drop=True)

            # fill the price variable with medians matched on unique_ts
            dataToClean.loc[:,'price'] = median_price

            ### We could add a print to tell how many duplicated values there where? - Kris

            # note that all other variables now hold the first entry for each timestamp!


        # if the cleaning procedure in question is t3.
        # T4. Delete entries with prices that are above the ask plus the bid-ask spread.
        # Similar for entries with prices below the bid minus the bid-ask spread.
        # FMNS: We have no bid/ask/spread in trades-table.
        #       To do this, we would probably need to cross-match timestamps between trades and quotes properly
        elif cp == 't4':

            raise ValueError(f'Cleaning procedure {cp} is on hold')


        # if the cleaning procedure in question is q1.
        # Q1. When multiple quotes have the same timestamp, we replace all these with a single entry
        # with the median bid and median ask price.
        # FMNS: Let's consider if these median prices are cheating in relation to OHLC bars
        elif cp == 'q1':

            if dataType == 'quote':

                # get unique timestamps
                unique_ts_idx = np.unique(dataToClean.Timestamp, return_index=True)[1]

                # get median prices
                median_price = dataToClean[['Timestamp', 'bid', 'ofr']].groupby('Timestamp')['bid', 'ofr'].median().values

                # keep only unique timestamps
                dataToClean = dataToClean.iloc[unique_ts_idx, :].reset_index(drop=True)

                # fill the price variable with medians matched on unique_ts
                dataToClean.loc[:,['bid','ofr']] = median_price

                # note that all other variables now hold the first entry for each timestamp!

            else:

                raise ValueError('The datatype has to be quote, in order to apply this cleaning procedure.\nPlease revisit your request.')


        # if the cleaning procedure in question is q2.
        # Q2. Delete entries for which the spread is negative.
        elif cp == 'q2':

            if dataType == 'quote':

                dataToClean = dataToClean[dataToClean.ofr - dataToClean.bid >= 0].reset_index(drop=True)

            else:
                raise ValueError('The datatype has to be quote, in order to apply this cleaning procedure.\nPlease revisit your request.')

        # if the cleaning procedure in question is q3.
        # Q3. Delete entries for which the spread is more that 50 times the median spread on that day.
        elif cp == 'q3':

            if dataType == 'quote':

                # get all spreads across days, groupby Date and take daily median spreads
                all_spreads = dataToClean[['Date', 'bid', 'ofr']]
                all_spreads['spread'] =  dataToClean.ofr - dataToClean.bid
                all_spreads.drop(['bid','ofr'], axis=1, inplace=True)

                median_spreads = all_spreads.groupby('Date').median().values


                total_keep_idx = []
                # for each unique day ...
                for day in np.unique(dataToClean.Date):

                    # for every spread within this day, check if it's below 50*median
                    # (below_50median is a boolean with all existing index)
                    below_50median = (all_spreads[all_spreads.Date == day].spread <= 50*median_spreads[median_spreads.index == day].values[0][0])

                    # get the indices where below_50median == True (meaning individual spread is within 50*median)
                    below_50median[below_50median].index

                    total_keep_idx.append(below_50median[below_50median].index)


                # after going through all days, flatten the list
                total_keep_idx = [ele for intraday_idx in total_keep_idx for ele in intraday_idx]

                # keep all entries that passed the filter
                dataToClean = dataToClean.iloc[total_keep_idx, :]

            else:

                raise ValueError('The datatype has to be quote, in order to apply this cleaning procedure.\nPlease revisit your request.')

        # if the cleaning procedure in question is q4.
        # Q4. Delete entries for which the mid-quote deviated by more than 5 median absolute deviations from
        # a centered median (excluding the observation under consideration) of 50 observations.
        elif cp == 'q4':

            raise ValueError(f'Cleaning procedure {cp} is on hold')
    return dataToClean
