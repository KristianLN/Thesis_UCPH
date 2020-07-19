import numpy as np
import pandas as pd
import re
import os
import time
import h5py
import copy
import datetime
from utils.data_cleaning import HFDataCleaning
from utils.generate_features import candleCreateNP_vect_final,\
                                    generateFeatures_final


# We create a function to clean the string-type arrays
#f = lambda a: re.split('[\']',a)[1]

# Function to clean the unpacked data from the compressed files.
def strList(ls):
    return list(map(lambda x: x.decode('utf-8'),ls))

# The following function is based on the research of (Lunde, 2016), summarized in the slides found here:
# https://econ.au.dk/fileadmin/site_files/filer_oekonomi/subsites/creates/Diverse_2016/PhD_High-Frequency/HF_TrQuData_v01.pdf

def formatDate(date,timestamps):
    return list(map(lambda x: date[0:4]+'/'+date[4:6]+'/'+date[6:]+' '+str(datetime.timedelta(seconds = int(str(x)[0:5]),
                                                     microseconds = int(str(x)[5:11]))),timestamps))


def load_data(dates, tickers, dataNeeded, path, verbose):
    # Measuring the exraction time
    start = time.time()
    if verbose:
        print(os.listdir())
    #path = 'T:/taqhdf5' #'a:/taqhdf5'
    allFiles = os.listdir(path)
    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

#     # Provide a list of dates of interest (format: yyyymmdd)
#     dates = np.array(['2020040' + str(i) if i < 10 else '202004' + str(i) for i in np.arange(1,32)]).astype(int)
#     # dates = np.array(['20200401']).astype(int)#,'20200402'

#     # Provide a list of tickers of interest
#     tickers = ['GOOG']#'MSFT'

#     # Do we need data on trades, quotes or both?
#     dataNeeded = 'trades' # 'trades', 'quotes' or 'both'

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

    if (dataNeeded == 'both') | (dataNeeded == 'trades'):

    # Lets start out by extracting the trade data
        for i,file in enumerate(trade):

            if (verbose) & (i == 0):
                print('### Trade Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')

            # Store the trade indecies
            TI = raw_data['TradeIndex']

            if (verbose) & (i==0):
                print('The raw H5 trade file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            TIC = np.array([ele[0].astype(str).strip() for ele in TI])

            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                # Getting the specific ticker information
                tickerInfo = TI[TIC==ticker][0]

                if (verbose) & (i == 0):
                        print('Ticker Information: ',tickerInfo,'\n')

                # Raw data - might be up for optimization
                tempData = raw_data['Trades'][list(np.arange(tickerInfo[1],tickerInfo[1]+tickerInfo[2]))]

                # For first file and first ticker.
                if (i == 0) & (j == 0):

                    tradeData = pd.DataFrame(tempData, columns= tempData.dtype.names)

                    tradeData.loc[:,'ex'] = strList(tradeData.ex)
                    tradeData.loc[:,'cond'] = strList(tradeData.cond)
                    tradeData.loc[:,'TradeStopStockIndicator'] = strList(tradeData.TradeStopStockIndicator)
                    tradeData.loc[:,'corr'] = strList(tradeData['corr'])
                    tradeData.loc[:,'TradeID'] = strList(tradeData.TradeID)
                    tradeData.loc[:,'TTE'] = strList(tradeData.TTE)
                    tradeData.loc[:,'TradeReportingFacility'] = strList(tradeData.TradeReportingFacility)
                    tradeData.loc[:,'SourceOfTrade'] = strList(tradeData.SourceOfTrade)

                    # Adding the date of the file to the dataframe.
                    tradeData['Date'] = re.split('[._]',file)[1]

                    # Adding a more readable timestamp - TEST IT
                    tradeData['Timestamp'] = pd.to_datetime(formatDate(re.split('[._]',file)[1],tradeData.utcsec))
                    tradeData['TSRemainder'] = list(map(lambda x: str(x)[11:], tradeData.utcsec))
                    tradeData['Hour'] = tradeData.Timestamp.dt.hour
                    tradeData['Minute'] = tradeData.Timestamp.dt.minute
                    # Adding the ticker
                    tradeData['Ticker'] = ticker

                    if (verbose) & (i==0) & (j==0):
                        print('Sneak peak of the data\n\n',tradeData.head())

                else:

                    # Storing the data on the following tickers in a temporary variable.

                    temp = pd.DataFrame(tempData, columns= tempData.dtype.names)

                    temp.loc[:,'ex'] = strList(temp.ex)
                    temp.loc[:,'cond'] = strList(temp.cond)
                    temp.loc[:,'TradeStopStockIndicator'] = strList(temp.TradeStopStockIndicator)
                    temp.loc[:,'corr'] = strList(temp['corr'])
                    temp.loc[:,'TradeID'] = strList(temp.TradeID)
                    temp.loc[:,'TTE'] = strList(temp.TTE)
                    temp.loc[:,'TradeReportingFacility'] = strList(temp.TradeReportingFacility)
                    temp.loc[:,'SourceOfTrade'] = strList(temp.SourceOfTrade)

                    # Adding the date of the file to the dataframe.
                    temp['Date'] = re.split('[._]',file)[1]

                    # Adding a more readable timestamp - TEST IT
                    temp['Timestamp'] = pd.to_datetime(formatDate(re.split('[._]',file)[1],temp.utcsec))
                    temp['TSRemainder'] = list(map(lambda x: str(x)[11:], temp.utcsec))
                    temp['Hour'] = temp.Timestamp.dt.hour
                    temp['Minute'] = temp.Timestamp.dt.minute

                    # Adding the ticker
                    temp['Ticker'] = ticker

                    # Adding the new data
                    tradeData = pd.concat([tradeData,temp])

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')

            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                # Getting the specific ticker information
                tickerInfo = QI[QIC==ticker][0]

                if (verbose) & (i == 0):
                        print('Ticker Information: ',tickerInfo,'\n')

                # Raw data
                tempData = raw_data['Quotes'][list(np.arange(tickerInfo[1],tickerInfo[1]+tickerInfo[2]))]

                # For first file and first ticker.
                if (i == 0) & (j == 0):

                    quoteData = pd.DataFrame(tempData, columns= tempData.dtype.names)
                    # We remove all unnecessary variables
                    unnecessaryVariables = ['NationalBBOInd',
                                            'FinraBBOInd',
                                            'FinraQuoteIndicator',
                                            'SequenceNumber',
                                            'FinraAdfMpidIndicator',
                                            'QuoteCancelCorrection',
                                            'SourceQuote',
                                            'RPI',
                                            'ShortSaleRestrictionIndicator',
                                            'LuldBBOIndicator',
                                            'SIPGeneratedMessageIdent',
                                            'NationalBBOLuldIndicator',
                                            'ParticipantTimestamp',
                                            'FinraTimestamp',
                                            'FinraQuoteIndicator',
                                            'SecurityStatusIndicator']

                    quoteData = quoteData.drop(columns=unnecessaryVariables)

                    quoteData.loc[:,'ex'] = strList(quoteData.ex)
                    quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                    # Adding the date of the file to the dataframe.
                    quoteData['Date'] = re.split('[._]',file)[1]

                    # Adding a more readable timestamp - TEST IT
                    quoteData['Timestamp'] = pd.to_datetime(formatDate(re.split('[._]',file)[1],quoteData.utcsec))
                    quoteData['TSRemainder'] = list(map(lambda x: str(x)[11:], quoteData.utcsec))
                    quoteData['Hour'] = quoteData.Timestamp.dt.hour
                    quoteData['Minute'] = quoteData.Timestamp.dt.minute
                    # Adding the ticker
                    quoteData['Ticker'] = ticker

                    if (verbose) & (i==0) & (j==0):
                        print('Sneak peak of the data\n\n',quoteData.head())

                else:

                    # Storing the data on the following tickers in a temporary variable.

                    temp = pd.DataFrame(tempData, columns= tempData.dtype.names)
                    # Removing all unnecessary variables
                    temp = temp.drop(columns=unnecessaryVariables)

                    temp.loc[:,'ex'] = strList(temp.ex)
                    temp.loc[:,'mode'] = strList(temp['mode'])

                    # Adding the date of the file to the dataframe.
                    temp['Date'] = re.split('[._]',file)[1]

                    # Adding a more readable timestamp - TEST IT
                    temp['Timestamp'] = pd.to_datetime(formatDate(re.split('[._]',file)[1],temp.utcsec))
                    temp['TSRemainder'] = list(map(lambda x: str(x)[11:], temp.utcsec))
                    temp['Hour'] = temp.Timestamp.dt.hour
                    temp['Minute'] = temp.Timestamp.dt.minute

                    # Adding the ticker
                    temp['Ticker'] = ticker

                    # Adding the new data
                    quoteData = pd.concat([quoteData,temp])

    end = time.time()

    if verbose:
        print('The extraction time was %.3f seconds.' % (end-start))

    if dataNeeded == 'trades':
        return tradeData
    elif dataNeeded == 'quotes':
        quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
        quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid
        return quoteData
    elif dataNeeded == 'both':
        quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
        quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid
        return tradeData, quoteData

# Added cleaning and candle generation, optimized extraction time by masking  across tickers (including ticker marking).
def load_data_v2(dates,
                tickers,
                dataNeeded,
                path,
                verbose,
                extract_candles = False,
                aggHorizon = 1,
                extra_features_from_quotes = None,
                data_sample = 'full'):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

#             if (verbose) & (i == 0):
#                 print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
#             if (verbose) & (i==0):
#                 print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])

            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)

            #tickerNp = np.zeros(raw_data['Quotes'].shape[0], dtype=str)
            tickerNp = np.zeros(raw_data['Quotes'].shape[0], dtype='|S10')

            for t,(pos_start, pos_range) in zip(tickers,selector):
                mask[pos_start : pos_start + pos_range] = True
                tickerNp[pos_start : pos_start + pos_range] = t

            tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            if (i == 0):

                quoteData = pd.DataFrame(tempData, columns= dt2.names)
                # We remove all unnecessary variables
                unnecessaryVariables = ['NationalBBOInd',
                                        'FinraBBOInd',
                                        'FinraQuoteIndicator',
                                        'SequenceNumber',
                                        'FinraAdfMpidIndicator',
                                        'QuoteCancelCorrection',
                                        'SourceQuote',
                                        'RPI',
                                        'ShortSaleRestrictionIndicator',
                                        'LuldBBOIndicator',
                                        'SIPGeneratedMessageIdent',
                                        'NationalBBOLuldIndicator',
                                        'ParticipantTimestamp',
                                        'FinraTimestamp',
                                        'FinraQuoteIndicator',
                                        'SecurityStatusIndicator']

                quoteData = quoteData.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))


                quoteData.loc[:,'ex'] = strList(quoteData.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                quoteData['Date'] = re.split('[._]',
                                             file)[1]
                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
                quoteData['Timestamp'] = dates + times

                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                #quoteData['Hour'] = quoteData.Timestamp.dt.hour
                #quoteData['Minute'] = quoteData.Timestamp.dt.minute
                # Adding the ticker
                quoteData['Ticker'] = tickerNp[mask]

            else:

                # Storing the data on the following tickers in a temporary variable.

                temp = pd.DataFrame(tempData, columns= dt2.names)
                # Removing all unnecessary variables
                temp = temp.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'ex'] = strList(temp.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'mode'] = strList(temp['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                temp['Date'] = re.split('[._]',file)[1]

                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
#                 temp['Timestamp'] = pd.to_datetime(formatDate(re.split('[._]',file)[1],temp.utcsec))
                dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(temp.loc[:,'utcsec'])
                temp['Timestamp'] = dates + times


                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

#                 temp['TSRemainder'] = list(map(lambda x: str(x)[11:], temp.utcsec))
                #temp['Hour'] = temp.Timestamp.dt.hour
                #temp['Minute'] = temp.Timestamp.dt.minute

                # Adding the ticker
                #temp['Ticker'] = ticker
                temp['Ticker'] = tickerNp[mask]

                # Adding the new data
                quoteData = pd.concat([quoteData,temp])

            # Closing the file after having used it.
            raw_data.close()
    end = time.time()

    quoteData = quoteData.reset_index(drop=True)
    kalkun = strList(quoteData.Ticker)
#     quoteData.loc[:,'Ticker'] = kalkun
#     quoteData.loc[:,'Ticker'] = quoteData.loc[:,'Ticker'].str.decode('utf-8','ignore')
    quoteData.loc[:,'ticker'] = strList(quoteData.Ticker)
#     quoteData = quoteData.astype({'Ticker':str})

    quoteData = quoteData.drop(columns=['Ticker']).rename(columns={'ticker':'Ticker'})
    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data
#     DATA_SAMPLE = 'full' # or 'stable'

    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',['q'])
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',['q'])
#     cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',['q'])

    if extract_candles:
        # Creating candles
        candles = candleCreateNP_vect_final(data = cleanedData,
                                           step = aggHorizon,
                                            verbose=False,
                                            fillHoles=True,
                                            sample=data_sample,
                                            numpied=False
                                           ,return_extended=extra_features_from_quotes)
        return candles

    return quoteData

# optimized the way tickers are set on all observations.
def load_data_v3(dates,
                    tickers,
                    dataNeeded,
                    path,
                    verbose,
                    extract_candles = False,
                    aggHorizon = 1,
                    extra_features_from_quotes = None,
                    data_sample = 'full'):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])

            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)

            for t,(pos_s, pos_r) in zip(tickers,selector):
                mask[pos_s : pos_s + pos_r] = True

            tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            if (i == 0):

                quoteData = pd.DataFrame(tempData, columns= dt2.names)
                # We remove all unnecessary variables
                unnecessaryVariables = ['NationalBBOInd',
                                        'FinraBBOInd',
                                        'FinraQuoteIndicator',
                                        'SequenceNumber',
                                        'FinraAdfMpidIndicator',
                                        'QuoteCancelCorrection',
                                        'SourceQuote',
                                        'RPI',
                                        'ShortSaleRestrictionIndicator',
                                        'LuldBBOIndicator',
                                        'SIPGeneratedMessageIdent',
                                        'NationalBBOLuldIndicator',
                                        'ParticipantTimestamp',
                                        'FinraTimestamp',
                                        'FinraQuoteIndicator',
                                        'SecurityStatusIndicator']

                quoteData = quoteData.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))


                quoteData.loc[:,'ex'] = strList(quoteData.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                quoteData['Date'] = re.split('[._]',
                                             file)[1]
                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
                quoteData['Timestamp'] = dates + times

                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    if i == 0:
                        quoteData['Ticker'] = np.nan
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]
                    else:
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

            else:

                # Storing the data on the following tickers in a temporary variable.

                temp = pd.DataFrame(tempData, columns= dt2.names)
                # Removing all unnecessary variables
                temp = temp.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'ex'] = strList(temp.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'mode'] = strList(temp['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                temp['Date'] = re.split('[._]',file)[1]

                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(temp.loc[:,'utcsec'])
                temp['Timestamp'] = dates + times


                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    temp.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

                # Adding the new data
                quoteData = pd.concat([quoteData,temp])

            # Closing the file after having used it.
            raw_data.close()
    end = time.time()

    quoteData = quoteData.reset_index(drop=True)

    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data

    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',['q'])
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',['q'])

    if extract_candles:
        # Creating candles
        candles = candleCreateNP_vect_final(data = cleanedData,
                                           step = aggHorizon,
                                            verbose=False,
                                            fillHoles=True,
                                            sample=data_sample,
                                            numpied=False
                                           ,return_extended=extra_features_from_quotes)
        return candles

    return quoteData


# v4 has saving incorporated
def load_data_v4(dates,
                    tickers,
                    dataNeeded,
                    path,
                    verbose,
                    extract_candles = False,
                    aggHorizon = 1,
                    extra_features_from_quotes = None,
                    data_sample = 'full',
                    save_output = False):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])

            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)

            for t,(pos_s, pos_r) in zip(tickers,selector):
                mask[pos_s : pos_s + pos_r] = True

            tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            if (i == 0):

                quoteData = pd.DataFrame(tempData, columns= dt2.names)
                # We remove all unnecessary variables
                unnecessaryVariables = ['NationalBBOInd',
                                        'FinraBBOInd',
                                        'FinraQuoteIndicator',
                                        'SequenceNumber',
                                        'FinraAdfMpidIndicator',
                                        'QuoteCancelCorrection',
                                        'SourceQuote',
                                        'RPI',
                                        'ShortSaleRestrictionIndicator',
                                        'LuldBBOIndicator',
                                        'SIPGeneratedMessageIdent',
                                        'NationalBBOLuldIndicator',
                                        'ParticipantTimestamp',
                                        'FinraTimestamp',
                                        'FinraQuoteIndicator',
                                        'SecurityStatusIndicator']

                quoteData = quoteData.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))


                quoteData.loc[:,'ex'] = strList(quoteData.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                quoteData['Date'] = re.split('[._]',
                                             file)[1]
                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
                quoteData['Timestamp'] = dates + times

                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    if i == 0:
                        quoteData['Ticker'] = np.nan
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]
                    else:
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

            else:

                # Storing the data on the following tickers in a temporary variable.

                temp = pd.DataFrame(tempData, columns= dt2.names)
                # Removing all unnecessary variables
                temp = temp.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'ex'] = strList(temp.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'mode'] = strList(temp['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                temp['Date'] = re.split('[._]',file)[1]

                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(temp.loc[:,'utcsec'])
                temp['Timestamp'] = dates + times


                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    temp.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

                # Adding the new data
                quoteData = pd.concat([quoteData,temp])

            # Closing the file after having used it.
            raw_data.close()
    end = time.time()

    quoteData = quoteData.reset_index(drop=True)

    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data

    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',['q'])
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',['q'])

    if extract_candles:
        # Creating candles
        if type(aggHorizon) == list:
            for step in aggHorizon:
                candles = candleCreateNP_vect_final(data = cleanedData,
                                                    step = aggHorizon,
                                                    verbose=False,
                                                    fillHoles=True,
                                                    sample=data_sample,
                                                    numpied=False,
                                                    return_extended=extra_features_from_quotes)
                with open('aggregateTAQ_' + int(step*60) + 'min.csv', 'a+') as f:
                    candles.to_csv(f, header=False)


        else:
            candles = candleCreateNP_vect_final(data = cleanedData,
                                                step = aggHorizon,
                                                verbose=False,
                                                fillHoles=True,
                                                sample=data_sample,
                                                numpied=False,
                                                return_extended=extra_features_from_quotes)
            with open('aggregateTAQ_' + int(step*60) + 'min.csv', 'a+') as f:
                candles.to_csv(f, header=False)

        return candles

    return quoteData

# Added the option to specify exchanges
def load_data_v5(dates,
                    tickers,
                    dataNeeded,
                    path,
                    verbose,
                    extract_candles = False,
                    aggHorizon = 1,
                    extra_features_from_quotes = None,
                    data_sample = 'full',
                    exhanges = ['q','t']):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])

            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)

            for t,(pos_s, pos_r) in zip(tickers,selector):
                mask[pos_s : pos_s + pos_r] = True

            tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            if (i == 0):

                quoteData = pd.DataFrame(tempData, columns= dt2.names)
                # We remove all unnecessary variables
                unnecessaryVariables = ['NationalBBOInd',
                                        'FinraBBOInd',
                                        'FinraQuoteIndicator',
                                        'SequenceNumber',
                                        'FinraAdfMpidIndicator',
                                        'QuoteCancelCorrection',
                                        'SourceQuote',
                                        'RPI',
                                        'ShortSaleRestrictionIndicator',
                                        'LuldBBOIndicator',
                                        'SIPGeneratedMessageIdent',
                                        'NationalBBOLuldIndicator',
                                        'ParticipantTimestamp',
                                        'FinraTimestamp',
                                        'FinraQuoteIndicator',
                                        'SecurityStatusIndicator']

                quoteData = quoteData.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))


                quoteData.loc[:,'ex'] = strList(quoteData.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                quoteData['Date'] = re.split('[._]',
                                             file)[1]
                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
                quoteData['Timestamp'] = dates + times

                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    if i == 0:
                        quoteData['Ticker'] = np.nan
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]
                    else:
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

            else:

                # Storing the data on the following tickers in a temporary variable.

                temp = pd.DataFrame(tempData, columns= dt2.names)
                # Removing all unnecessary variables
                temp = temp.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'ex'] = strList(temp.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'mode'] = strList(temp['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                temp['Date'] = re.split('[._]',file)[1]

                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(temp.loc[:,'utcsec'])
                temp['Timestamp'] = dates + times


                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    temp.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

                # Adding the new data
                quoteData = pd.concat([quoteData,temp])

            # Closing the file after having used it.
            raw_data.close()
    end = time.time()

    quoteData = quoteData.reset_index(drop=True)

    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data

    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',exhanges)
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',exhanges)

    if extract_candles:
        # Creating candles
        candles = candleCreateNP_vect_final(data = cleanedData,
                                           step = aggHorizon,
                                            verbose=False,
                                            fillHoles=True,
                                            sample=data_sample,
                                            numpied=False
                                           ,return_extended=extra_features_from_quotes)
        return candles

    return quoteData
# This is v5
def load_data_final(dates,
                    tickers,
                    dataNeeded,
                    path,
                    verbose,
                    extract_candles = False,
                    aggHorizon = 1,
                    extra_features_from_quotes = None,
                    data_sample = 'full',
                    exhanges = ['q','t']):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])

            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)
            # print(pos_start,pos_range)
            for t,(pos_s, pos_r) in zip(tickers,selector):
                mask[pos_s : pos_s + pos_r] = True

            # tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            if (i == 0):

                # quoteData = pd.DataFrame(tempData, columns= dt2.names)
                quoteData = pd.DataFrame(raw_data['Quotes'][mask], columns= dt2.names)
                # We remove all unnecessary variables
                unnecessaryVariables = ['NationalBBOInd',
                                        'FinraBBOInd',
                                        'FinraQuoteIndicator',
                                        'SequenceNumber',
                                        'FinraAdfMpidIndicator',
                                        'QuoteCancelCorrection',
                                        'SourceQuote',
                                        'RPI',
                                        'ShortSaleRestrictionIndicator',
                                        'LuldBBOIndicator',
                                        'SIPGeneratedMessageIdent',
                                        'NationalBBOLuldIndicator',
                                        'ParticipantTimestamp',
                                        'FinraTimestamp',
                                        'FinraQuoteIndicator',
                                        'SecurityStatusIndicator']

                quoteData = quoteData.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))


                quoteData.loc[:,'ex'] = strList(quoteData.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                quoteData['Date'] = re.split('[._]',
                                             file)[1]
                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
                quoteData['Timestamp'] = dates + times

                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    if i == 0:
                        quoteData['Ticker'] = np.nan
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]
                    else:
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

            else:

                # Storing the data on the following tickers in a temporary variable.

                temp = pd.DataFrame(tempData, columns= dt2.names)
                # Removing all unnecessary variables
                temp = temp.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'ex'] = strList(temp.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'mode'] = strList(temp['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                temp['Date'] = re.split('[._]',file)[1]

                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(temp.loc[:,'utcsec'])
                temp['Timestamp'] = dates + times


                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for i,pa in enumerate(pos_range):
                    temp.loc[int(l):int(l+pa),'Ticker'] = tickers[i]

                    l += pa

                # Adding the new data
                quoteData = pd.concat([quoteData,temp])

            # Closing the file after having used it.
            raw_data.close()
    end = time.time()

    quoteData = quoteData.reset_index(drop=True)

    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data

    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',exhanges)
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',exhanges)

    if extract_candles:
        # Creating candles
        candles = candleCreateNP_vect_final(data = cleanedData,
                                           step = aggHorizon,
                                            verbose=False,
                                            fillHoles=True,
                                            sample=data_sample,
                                            numpied=False
                                           ,return_extended=extra_features_from_quotes)
        return candles

    return quoteData

# This is V4 (KRISNI: an updated on right?)
def load_data_and_save_v1(dates,
                    tickers,
                    dataNeeded,
                    path,
                    verbose,
                    extract_candles = False,
                    aggHorizon = 1,
                    extra_features_from_quotes = None,
                    data_sample = 'full',
                    save_output = False):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])

            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)

            for t,(pos_s, pos_r) in zip(tickers,selector):
                mask[pos_s : pos_s + pos_r] = True

            tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            if (i == 0):

                quoteData = pd.DataFrame(tempData, columns= dt2.names)
                # We remove all unnecessary variables
                unnecessaryVariables = ['NationalBBOInd',
                                        'FinraBBOInd',
                                        'FinraQuoteIndicator',
                                        'SequenceNumber',
                                        'FinraAdfMpidIndicator',
                                        'QuoteCancelCorrection',
                                        'SourceQuote',
                                        'RPI',
                                        'ShortSaleRestrictionIndicator',
                                        'LuldBBOIndicator',
                                        'SIPGeneratedMessageIdent',
                                        'NationalBBOLuldIndicator',
                                        'ParticipantTimestamp',
                                        'FinraTimestamp',
                                        'FinraQuoteIndicator',
                                        'SecurityStatusIndicator']

                quoteData = quoteData.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))


                quoteData.loc[:,'ex'] = strList(quoteData.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                quoteData.loc[:,'mode'] = strList(quoteData['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                quoteData['Date'] = re.split('[._]',
                                             file)[1]
                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
                quoteData['Timestamp'] = dates + times

                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for j,pa in enumerate(pos_range):
                    if j == 0:
                        quoteData['Ticker'] = np.nan
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[j]
                    else:
                        quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[j]

                    l += pa

            else:

                # Storing the data on the following tickers in a temporary variable.

                temp = pd.DataFrame(tempData, columns= dt2.names)
                # Removing all unnecessary variables
                temp = temp.drop(columns=unnecessaryVariables)

                if verbose:
                    print('8 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'ex'] = strList(temp.ex)

                if verbose:
                    print('9 Lap time: %.3f' % ((time.time()-start)))

                temp.loc[:,'mode'] = strList(temp['mode'])

                if verbose:
                    print('10 Lap time: %.3f' % ((time.time()-start)))

                # Adding the date of the file to the dataframe.
                temp['Date'] = re.split('[._]',file)[1]

                if verbose:
                    print('11 Lap time: %.3f' % ((time.time()-start)))

                # Adding a more readable timestamp - TEST IT
                dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
                times = pd.to_timedelta(temp.loc[:,'utcsec'])
                temp['Timestamp'] = dates + times


                if verbose:
                    print('12 Lap time: %.3f' % ((time.time()-start)))

                l = 0
                for j,pa in enumerate(pos_range):
                    temp.loc[int(l):int(l+pa),'Ticker'] = tickers[j]

                    l += pa

                # Adding the new data
                # quoteData = pd.concat([quoteData,temp])
                # We cannot append like this with many tickers, too much data
                quoteData = temp

            # Closing the file after having used it.
            raw_data.close()


            if save_output:

                quoteData = quoteData.reset_index(drop=True)
                quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
                quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

                # Cleaning the data
                if data_sample == 'stable':
                    # P1 is used for keeping data within [10, 15.5]
                    cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',['q'])
                elif data_sample == 'full':
                    # P1_2 is used for keeping data within [9.5, 16]
                    cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',['q'])


                # Creating candles
                if type(aggHorizon) == list:
                    # assert save_output == False
                    for step in aggHorizon:
                        for ticker_i in tickers:
                            candles = candleCreateNP_vect_final(data = cleanedData.loc[cleanedData['Ticker'] == ticker_i, :].copy(), #cleanedData[cleanedData['Ticker'] == ticker_i],
                                                                step = step,
                                                                verbose = False,
                                                                fillHoles = True,
                                                                sample = data_sample,
                                                                numpied = False,
                                                                return_extended = extra_features_from_quotes)
                            candles['Ticker'] = ticker_i



                            # with open('aggregateTAQ_' + str(int(step*60)) + 'sec.csv', 'a+') as f:
                            #     candles.to_csv(f, header=False)
                            candles.to_csv('aggregateTAQ_' + str(int(step*60)) + 'sec.csv', mode='a', header=False)


                else:
                    #print(1)
                    #print(cleanedData.loc[cleanedData['Ticker'] == 'GOOG', :].copy().columns)
                    for ticker_i in tickers:
                        #print(2)
                        #print(cleanedData.loc[cleanedData['Ticker'] == ticker_i, :].copy().columns)
                        candles = candleCreateNP_vect_final(data = cleanedData.loc[cleanedData['Ticker'] == ticker_i, :].copy(), #cleanedData[cleanedData['Ticker'] == ticker_i],
                                                            step = aggHorizon,
                                                            verbose = False,
                                                            fillHoles = True,
                                                            sample = data_sample,
                                                            numpied = False,
                                                            return_extended = extra_features_from_quotes)
                        candles['Ticker'] = ticker_i

                        # with open('aggregateTAQ_' + str(int(aggHorizon*60)) + 'sec.csv', 'a+') as f:
                        #     candles.to_csv(f, header=False)
                        candles.to_csv('aggregateTAQ_' + str(int(aggHorizon*60)) + 'sec.csv', mode='a', header=False)






    end = time.time()

    quoteData = quoteData.reset_index(drop=True)

    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data
    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',['q'])
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',['q'])

    if extract_candles:

        candles = candleCreateNP_vect_final(data = cleanedData,
                                            step = aggHorizon,
                                            verbose=False,
                                            fillHoles=True,
                                            sample=data_sample,
                                            numpied=False,
                                            return_extended=extra_features_from_quotes)


        return candles

    return quoteData

def load_data_and_save(dates,
                    tickers,
                    dataNeeded,
                    path,
                    verbose,
                    extract_candles = False,
                    aggHorizon = 1,
                    extra_features_from_quotes = None,
                    data_sample = 'full',
                    exchanges = ['q','t'],
                    save_output = False):

    # Measuring the exraction time
    start = time.time()

    allFiles = os.listdir(path)

    if verbose:
        print(len(allFiles), allFiles[:5], allFiles[-5:])
        print(allFiles[-10:])

    # Extracting just the dates of each file
    allDates = np.array([re.split("[._]",ele)[1] if ("." in ele ) & ("_" in ele) else 0 for ele in allFiles]).astype(int)

    minDate = np.min(dates)
    maxDate = np.max(dates)

    if verbose:
        print('##### Date range #####\n\nDate, Min: %i\nDate, Max: %i\n'%(minDate,maxDate))
        print('\n1 Lap time: %.3f\n' % ((time.time()-start)))

    # Locating what files we need.
    index = np.where((minDate <= allDates) & (allDates <= maxDate))

    relevantFiles = np.array(allFiles)[index[0]]

    # Separating the files into trade and quote files.
    trade = [ele for ele in relevantFiles if 'trade' in ele]
    quote = [ele for ele in relevantFiles if 'quote' in ele]

    if verbose:
        print('##### Data Extraction begins #####\n')

        if dataNeeded.lower() == 'both':
            print('Both trade and quote data is being extracted..\n')
        else:
            print('%s data is being extracted..\n' % dataNeeded[0:5])

        print('\n2 Lap time: %.3f\n' % ((time.time()-start)))

    if (dataNeeded == 'both') | (dataNeeded == 'quotes'):

        # Now to the quote data
        for i,file in enumerate(quote):

            if (verbose) & (i == 0):
                print('### Quote Data ###\n')

            # Reading one file at a time
            raw_data = h5py.File(path+'/'+file,'r')
            dt2 = raw_data['Quotes'].dtype
            if verbose:
                print('3 Lap time: %.3f' % ((time.time()-start)))


            # Store the trade indecies
            QI = raw_data['QuoteIndex']

            if verbose:
                print('4 Lap time: %.3f' % ((time.time()-start)))
            if (verbose) & (i==0):
                print('The raw H5 quote file contains: ',list(raw_data.keys()),'\n')

            # Extracting just the tickers
            QIC = np.array([ele[0].astype(str).strip() for ele in QI])

            if verbose:
                print('5 Lap time: %.3f' % ((time.time()-start)))

            pos_start = []
            pos_range = []
            # Lets get data on each ticker for the file processed at the moment
            for j,ticker in enumerate(tickers):

                tickerInfo = QI[QIC==ticker][0]
                pos_start.append(tickerInfo[1])
                pos_range.append(tickerInfo[2])
            # print(pos_range)
            if verbose:
                print('6 Lap time: %.3f' % ((time.time()-start)))

            # use boolean mask to slice all at once
            selector = zip(pos_start, pos_range)
            mask = np.zeros(raw_data['Quotes'].shape[0], dtype=bool)
            # print(selector)
            for t,(pos_s, pos_r) in zip(tickers,selector):
                mask[pos_s : pos_s + pos_r] = True

            # tempData = raw_data['Quotes'][mask]

            if verbose:
                print('7 Lap time: %.3f' % ((time.time()-start)))

            # For first file and first ticker.
            # if (i == 0):

            # quoteData = pd.DataFrame(tempData, columns= dt2.names)
            quoteData = pd.DataFrame(raw_data['Quotes'][mask], columns= dt2.names)
            # We remove all unnecessary variables
            unnecessaryVariables = ['NationalBBOInd',
                                    'FinraBBOInd',
                                    'FinraQuoteIndicator',
                                    'SequenceNumber',
                                    'FinraAdfMpidIndicator',
                                    'QuoteCancelCorrection',
                                    'SourceQuote',
                                    'RPI',
                                    'ShortSaleRestrictionIndicator',
                                    'LuldBBOIndicator',
                                    'SIPGeneratedMessageIdent',
                                    'NationalBBOLuldIndicator',
                                    'ParticipantTimestamp',
                                    'FinraTimestamp',
                                    'FinraQuoteIndicator',
                                    'SecurityStatusIndicator']

            quoteData = quoteData.drop(columns=unnecessaryVariables)

            if verbose:
                print('8 Lap time: %.3f' % ((time.time()-start)))


            quoteData.loc[:,'ex'] = strList(quoteData.ex)

            if verbose:
                print('9 Lap time: %.3f' % ((time.time()-start)))

            quoteData.loc[:,'mode'] = strList(quoteData['mode'])

            if verbose:
                print('10 Lap time: %.3f' % ((time.time()-start)))

            # Adding the date of the file to the dataframe.
            quoteData['Date'] = re.split('[._]',
                                         file)[1]
            if verbose:
                print('11 Lap time: %.3f' % ((time.time()-start)))

            # Adding a more readable timestamp
            dates = pd.to_datetime(quoteData.loc[:,'Date'], format='%Y%m%d', errors='ignore')
            times = pd.to_timedelta(quoteData.loc[:,'utcsec'])
            quoteData['Timestamp'] = dates + times

            if verbose:
                print('12 Lap time: %.3f' % ((time.time()-start)))

            l = 0
            # print(pos_range)
            for j,pa in enumerate(pos_range):
                if j == 0:
                    quoteData['Ticker'] = np.nan
                    quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[j]
                else:
                    quoteData.loc[int(l):int(l+pa),'Ticker'] = tickers[j]

                l += pa
            # print(pos_range)
            # else:
            #
            #     # Storing the data on the following tickers in a temporary variable.
            #
            #     temp = pd.DataFrame(tempData, columns= dt2.names)
            #     # Removing all unnecessary variables
            #     temp = temp.drop(columns=unnecessaryVariables)
            #
            #     if verbose:
            #         print('8 Lap time: %.3f' % ((time.time()-start)))
            #
            #     temp.loc[:,'ex'] = strList(temp.ex)
            #
            #     if verbose:
            #         print('9 Lap time: %.3f' % ((time.time()-start)))
            #
            #     temp.loc[:,'mode'] = strList(temp['mode'])
            #
            #     if verbose:
            #         print('10 Lap time: %.3f' % ((time.time()-start)))
            #
            #     # Adding the date of the file to the dataframe.
            #     temp['Date'] = re.split('[._]',file)[1]
            #
            #     if verbose:
            #         print('11 Lap time: %.3f' % ((time.time()-start)))
            #
            #     # Adding a more readable timestamp - TEST IT
            #     dates = pd.to_datetime(temp.loc[:,'Date'], format='%Y%m%d', errors='ignore')
            #     times = pd.to_timedelta(temp.loc[:,'utcsec'])
            #     temp['Timestamp'] = dates + times
            #
            #
            #     if verbose:
            #         print('12 Lap time: %.3f' % ((time.time()-start)))
            #
            #     l = 0
            #     for j,pa in enumerate(pos_range):
            #         temp.loc[int(l):int(l+pa),'Ticker'] = tickers[j]
            #
            #         l += pa
            #
            #     # Adding the new data
            #     # quoteData = pd.concat([quoteData,temp])
            #     # We cannot append like this with many tickers, too much data
            #     quoteData = temp

            # Closing the file after having used it.
            raw_data.close()


            if save_output:

                # quoteData = quoteData.reset_index(drop=True)
                quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
                quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid
                # return quoteData

                ## Cleaning the data
                if data_sample == 'stable':
                    # P1 is used for keeping data within [10, 15.5]
                    cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',exchanges)
                elif data_sample == 'full':
                    # P1_2 is used for keeping data within [9.5, 16]
                    cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',exchanges)

                # return cleanedData

                ## Creating candles
                if type(aggHorizon) == list:
                    # assert save_output == False
                    for step in aggHorizon:
                        for ticker_i in tickers:
                            candles = candleCreateNP_vect_final(data = cleanedData.loc[cleanedData['Ticker'] == ticker_i, :].copy(), #cleanedData[cleanedData['Ticker'] == ticker_i],
                                                                step = step,
                                                                verbose = False,
                                                                fillHoles = True,
                                                                sample = data_sample,
                                                                numpied = False,
                                                                return_extended = extra_features_from_quotes)

                            candles['Ticker'] = ticker_i



                            # with open('aggregateTAQ_' + str(int(step*60)) + 'sec.csv', 'a+') as f:
                            #     candles.to_csv(f, header=False)
                            candles.to_csv('aggregateTAQ_' + str(int(step*60)) + 'sec.csv', mode='a', header=False)
                            # return candles

                else:
                    #print(1)
                    #print(cleanedData.loc[cleanedData['Ticker'] == 'GOOG', :].copy().columns)
                    for ticker_i in tickers:
                        #print(2)
                        #print(cleanedData.loc[cleanedData['Ticker'] == ticker_i, :].copy().columns)
                        candles = candleCreateNP_vect_final(data = cleanedData.loc[cleanedData['Ticker'] == ticker_i, :].copy(), #cleanedData[cleanedData['Ticker'] == ticker_i],
                                                            step = aggHorizon,
                                                            verbose = False,
                                                            fillHoles = True,
                                                            sample = data_sample,
                                                            numpied = False,
                                                            return_extended = extra_features_from_quotes)
                        ## Flatten multiIndex columns
                        # candles.columns = np.concatenate([[j+'_'+i for i in candles.columns.get_level_values(1).unique()] for j in candles.columns.get_level_values(0).unique()])

                    #     dataPD = dataPD.loc[:,['price_open','price_high','price_low','price_close']].rename(columns=['open','high','low','close'])
                        # candles = candles.rename(columns={'price_open':'open',
                        #                                 'price_high':'high',
                        #                                 'price_low':'low',
                        #                                 'price_close':'close'})

                        candles['Ticker'] = ticker_i

                        # with open('aggregateTAQ_' + str(int(aggHorizon*60)) + 'sec.csv', 'a+') as f:
                        #     candles.to_csv(f, header=False)
                        candles.to_csv('aggregateTAQ_' + str(int(aggHorizon*60)) + 'sec.csv', mode='a', header=False)
                        # return candles
    #
    if save_output:
        return None

    end = time.time()

    quoteData = quoteData.reset_index(drop=True)

    print('The extraction time was %.3f seconds.' % (end-start))

    quoteData.loc[:,'price'] = (quoteData.bid + quoteData.ofr) / 2
    quoteData.loc[:,'spread'] = quoteData.ofr - quoteData.bid

    # Cleaning the data
    if data_sample == 'stable':
        # P1 is used for keeping data within [10, 15.5]
        cleanedData = HFDataCleaning(['P1','p2','t1','p3'],quoteData,'quote',exchanges)
    elif data_sample == 'full':
        # P1_2 is used for keeping data within [9.5, 16]
        cleanedData = HFDataCleaning(['P1_2','p2', 'q2', 'p3'],quoteData,'quote',exchanges)

    if extract_candles:

        candles = candleCreateNP_vect_final(data = cleanedData,
                                            step = aggHorizon,
                                            verbose=False,
                                            fillHoles=True,
                                            sample=data_sample,
                                            numpied=False,
                                            return_extended=extra_features_from_quotes)


        return candles

    return quoteData


def updateStockInfo(verbose):
    try:
        path = 'a:/taqhdf5'  #'a:/taqhdf5'
        latestFile = os.listdir(path)[-1]
    except:
        path = 't:/taqhdf5'  #'a:/taqhdf5'
        latestFile = os.listdir(path)[-1]

    # Reading the latest file
    rawData = h5py.File(path+'/'+latestFile,'r')

    # Extract all tickers

    TIC = np.array([ele[0].astype(str).strip() for ele in rawData['TradeIndex']])

    start = time.time()

    ########## Lets extract info on all tickers!

    # Set date
    date = str(datetime.date.today())

    ### Initialize PD container
    stockInfo = pd.DataFrame(index=TIC,columns=pd.MultiIndex.from_product([[date],
                                                                          ['sector','exchange','marketCap']],
                                                                          names=['date','ticker']))
    ### Lets track where we are.
    printingRange = np.arange(1000,len(TIC),1000)

    lsContainer = []
    for i,ticker in enumerate(TIC):

        if i in printingRange:
            if verbose:
                print('%i tickers processed, lap timing: %.3f' % (i,(time.time()-start)))

        # Safeguarding
        tick = yf.Ticker(ticker)
        try:
            lsContainer.append([tick.info['sector'],
                                tick.info['exchange'],
                                tick.info['marketCap']])
        except:

            lsContainer.append([np.nan,
                                np.nan,
                                np.nan])

    end = time.time()

    stockInfo.loc[:,date] = lsContainer

    if verbose:
        print('\nTotal processing time: %.3f'%(end-start))
