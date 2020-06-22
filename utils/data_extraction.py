import numpy as np
import pandas as pd
import re
import os
import time
import h5py
import copy
import datetime


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

    print(os.listdir())
    #path = 'T:/taqhdf5' #'a:/taqhdf5'
    allFiles = os.listdir(path)
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

                # Raw data
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
        return quoteData    
    elif dataNeeded == 'both':
        return tradeData, quoteData