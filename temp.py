'''
1) Each timestep
    -check for new positions
    -re-evaluate current positions
    -rebalance maybe
    -

'''


class backtest():
    def __init__(self, X_test, data, preds, weight_scheme, max_steps, max_positions, n_classes):

        self.all_tickers = data.Ticker.unique()
        self.open_long_positions = []#'AAPL','ABT','LFC'
        self.open_short_positions = []#'BAC','KO','ENB'
        self.ticker_dict = {}   # key: ticker, value: [open_price, direction, pnl]
        self.hist_rets = np.ones((max_steps,len(self.all_tickers)+1))
        self.pnl = []
        self.prev_close = pd.DataFrame(0, index=self.all_tickers, columns=['prev_close', 'direction'])
        self.t = 0

        self.X_test = X_test
        self.data = data
        self.preds = preds
        ## New
        self.weight_scheme = weight_scheme
        self.max_steps = max_steps
        self.max_positions = max_positions
        self.n_classes = n_classes

        if verbose >= 1:
            print(f'initial self.open_long_positions: {self.open_long_positions}')
            print(f'initial self.open_short_positions: {self.open_short_positions} \n')



    def run(self):

        unique_timesteps = np.concatenate([[[i,j] for i in np.unique(self.X_test.index.get_level_values(1))] \
                                                  for j in np.unique(self.X_test.index.get_level_values(0))])

        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]

            #print(i)
            try:
                ts_data = self.data.sort_index().loc[(ts[1], ts[0])] ## sort_index() to prevent the performance warning.
            except:
                pass
            if ts_data.shape == 0:
                pass

            #print(ts_data)

            close_info = ts_data[['close','spread_close','Ticker']].reset_index(drop=True)
            ts_preds = self.preds.loc[(ts[1], ts[0])]

            self.update_positions(ts, close_info, ts_preds)

            self.t += 1

        print(f'run function finished at step {self.t}, time: {ts}')

        #for ts in unique_timesteps[:2]:

        # for ts in X_test.reset_index().groupby(['days','timestamps']).size().index[:2]:
        #     #print(i)
        #     ts_data = data.loc[(ts[0], ts[1])]
        #     if ts_data.shape == 0:
        #         pass
        #         #print(i, 'hovhov')


    def update_positions(self, ts, close_info, ts_preds):

        #print(ts_preds, '\n')

        # use predictions to select what to hold
        long_list = ts_preds[ts_preds['class'] == (self.n_classes - 1)].index.values
        short_list = ts_preds[ts_preds['class'] == 0].index.values

        if verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # if any open positions
        # if (len(self.open_long_positions) > 0) or (len(self.open_short_positions) > 0):


        # check if any new positions are made
        new_buy = long_list[~np.isin(long_list, self.open_long_positions)]
        new_sell = short_list[~np.isin(short_list, self.open_short_positions)]




        print(self.ticker_dict)
#             self.hist_rets

        if verbose >= 1:
            print(f'new_buy: {new_buy}')
            print(f'new_sell: {new_sell} \n')

        self.open_long_positions += [long_i for long_i in new_buy]
        self.open_short_positions += [short_i for short_i in new_sell]

        self.ticker_dict

        if verbose >= 1:
            print('Opening new trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

#             print(self.open_long_positions,'\n') #[~np.isin(self.open_long_positions, long_list)])
#             print(np.isin(self.open_long_positions, long_list),'\n')
#             print(long_list,'\n')

        # check if any are closed
        close_buy = np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)]
        close_short = np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)]

        if verbose >= 1:
            print(f'close_buy: {close_buy}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_buy]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        if self.t > 0:

            # update directions for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_buy), 'direction'] = -1
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction'] = 1

#             print(close_info.close.values[21])
#             print(close_info.spread_close.values[21])
#             print(self.prev_close.direction.values[21])
            current_ret = (close_info.close.values * abs(self.prev_close.direction.values) \
                             + (close_info.spread_close.values / 2) * -self.prev_close.direction.values) / self.prev_close.prev_close.values
#             print(self.prev_close.direction.values)
            ## Fixing those not in use, which by the 'direction' equals a return of zero.
#             current_ret[current_ret == 0] = 1
            current_ret[self.prev_close.direction.values == 0] = 1

            # update individual returns for open positions
            self.hist_rets[self.t, :-1] = current_ret * self.hist_rets[self.t - 1, :-1]

            # update total portfolio returns for open positions
#             print(current_ret)
#             print((current_ret-1)[current_ret!=0])
#             current_ret[self.prev_close.direction.values == 0] = 0

            ## if the weights should be based on the probabilities and their distribution.
#             if self.weight_scheme == 'prob_dist':

#             ## if the weights should be based on the probabilities and equally distributed between long and short positions.
#             elif self.weight_scheme == 'prob_equal':

#             else:

            self.hist_rets[self.t, -1] = (1+np.mean(current_ret[self.prev_close.direction.values != 0] - 1)) * self.hist_rets[self.t - 1, -1]

            # update directions for new positions
            self.prev_close.loc[np.isin(self.all_tickers, new_buy), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_sell), 'direction']  = -1

            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_buy), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

        else:



            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_buy), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_sell), 'direction']  = -1


#             self.ticker_dict = {j:[close_info.iloc[i,0], 1, 0] if j in new_buy \
#                                 else j:[close_info.iloc[i,0], -1, 0] if j in new_sell \
#                                 else j:[close_info.iloc[i,0], 0, 0] \
#                                     for i,j in enumerate(close_info.Ticker)}

#             self.ticker_dict = {j.Ticker:[j.close, 1, 0] if j in new_buy \
#                                 else j.Ticker:[j.close, -1, 0] if j in new_sell \
#                                 else j.Ticker:[j.close, 0, 0] \
#                                     for i,j in enumerate(close_info.iterrows())}

            print(close_info.close.copy(deep=True).values)

            self.prev_close.loc[:,'prev_close'] = close_info.close.values
            print(self.prev_close)
            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_buy),'prev_close'] = close_info[np.isin(self.all_tickers, new_buy)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_buy)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_sell),'prev_close'] = close_info[np.isin(self.all_tickers, new_sell)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_sell)].spread_close.values / 2)


        print(close_info.close.copy(deep=True).values)

        self.prev_close.loc[:,'prev_close'] = close_info.close.values

        print(close_info.close.copy(deep=True).values)

        #print(self.prev_close.loc[:,'prev_close'])


backtest_obj = backtest(X_test=X_test,
                        data=data,
                        preds=preds,
                        weight_scheme = None,
                        max_steps=2,
                        max_positions = 50,
                        n_classes=5)
backtest_obj.run()

############################################################################### Check of generate features from Azure ###############################################################################

# Included intraday time and past returns as features
# added option change between past obs in percentage or not.
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
