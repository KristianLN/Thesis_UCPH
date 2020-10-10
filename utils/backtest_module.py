import numpy as np
import pandas as pd


'''
1) Each timestep
    -check for new positions
    -re-evaluate current positions
    -rebalance maybe
    -

'''


class backtest_v1():
    def __init__(self, X_test, data, preds, weight_scheme, max_steps, max_positions, n_classes,verbose=False):

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
        self.verbose = verbose

        if self.verbose >= 1:
            print(f'initial self.open_long_positions: {self.open_long_positions}')
            print(f'initial self.open_short_positions: {self.open_short_positions} \n')



    def run(self):

        unique_timesteps = np.concatenate([[[i,j] for i in np.unique(self.X_test.index.get_level_values(1))] \
                                                  for j in np.unique(self.X_test.index.get_level_values(0))])
        print(self.all_tickers,'\n')

        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]
            print('\n\n################ Period %i ################\n\n' % self.t)

            try:
                ts_data = self.data.sort_index().loc[(ts[1], ts[0])]
            except:
                pass
            if ts_data.shape == 0:
                pass

            close_info = ts_data[['close','spread_close','Ticker']].reset_index(drop=True)
            ts_preds = self.preds.loc[(ts[1], ts[0])]

            self.update_positions(ts, close_info, ts_preds)

            self.t += 1

        print(f'run function finished at step {self.t}, time: {ts}')

    def update_positions(self, ts, close_info, ts_preds):

        # use predictions to select what to hold
        long_list = ts_preds[ts_preds['class'] == (self.n_classes - 1)].index.values
        short_list = ts_preds[ts_preds['class'] == 0].index.values

        if self.verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # check if any new positions are made
        new_buy = long_list[~np.isin(long_list, self.open_long_positions)]
        new_sell = short_list[~np.isin(short_list, self.open_short_positions)]

        if self.verbose >= 1:
            print(f'new_buy: {new_buy}')
            print(f'new_sell: {new_sell} \n')

        self.open_long_positions += [long_i for long_i in new_buy]
        self.open_short_positions += [short_i for short_i in new_sell]

        self.ticker_dict

        if self.verbose >= 1:
            print('Opening new trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        # check if any are closed
        close_buy = np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)]
        close_short = np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)]

        if self.verbose >= 1:
            print(f'close_buy: {close_buy}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_buy]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if self.verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        if self.t > 0:

            # update directions for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_buy), 'direction'] = -1
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction'] = 1

            current_ret = (close_info.close.values / self.prev_close.prev_close.values) - 1
            current_ret = (current_ret * self.prev_close.direction.values) + 1

            ## Correcting those we close
            boolcousin = np.isin(self.all_tickers, close_short) | np.isin(self.all_tickers, close_buy)

            current_ret[boolcousin] = (close_info[boolcousin].close.values\
                                                                 *abs(self.prev_close[boolcousin].direction.values)\
                                                                 +(close_info[boolcousin].spread_close.values / 2)\
                                                                 *self.prev_close[boolcousin].direction.values)\
                                                                /self.prev_close[boolcousin].prev_close.values

            ## Fixing those not in use, which by the 'direction' equals a return of zero.
            current_ret[self.prev_close.direction.values == 0] = 1

            # update individual returns for open positions
            self.hist_rets[self.t, :-1] = current_ret * self.hist_rets[self.t - 1, :-1]

            # update total portfolio returns for open positions
            self.hist_rets[self.t, -1] = (1+np.mean(current_ret[self.prev_close.direction.values != 0] - 1)) * self.hist_rets[self.t - 1, -1]

            # update directions for new positions
            self.prev_close.loc[np.isin(self.all_tickers, new_buy), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_sell), 'direction']  = -1

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_buy),'prev_close'] = close_info[np.isin(self.all_tickers, new_buy)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_buy)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_sell),'prev_close'] = close_info[np.isin(self.all_tickers, new_sell)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_sell)].spread_close.values / 2)

            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_buy), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

        else:

            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_buy), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_sell), 'direction']  = -1

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_buy),'prev_close'] = close_info[np.isin(self.all_tickers, new_buy)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_buy)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_sell),'prev_close'] = close_info[np.isin(self.all_tickers, new_sell)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_sell)].spread_close.values / 2)

'''
1) Each timestep
    -check for new positions
    -re-evaluate current positions
    -rebalance maybe
    -

2) Max_positions: Check if the specified number of max positions results in an uneven split? like max position = 9.
'''


class backtest_v2():
    def __init__(self, X_test, data, preds, weight_scheme, max_steps, max_positions, n_classes,verbose=False):

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
        self.verbose = verbose

        if self.verbose >= 1:
            print(f'initial self.open_long_positions: {self.open_long_positions}')
            print(f'initial self.open_short_positions: {self.open_short_positions} \n')

    def run(self):

        unique_timesteps = np.concatenate([[[i,j] for i in np.unique(self.X_test.index.get_level_values(1))] \
                                                  for j in np.unique(self.X_test.index.get_level_values(0))])
        print(self.all_tickers,'\n')

        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]
            print('\n\n################ Period %i ################\n\n' % self.t)
            #print(i)
            try:
                ts_data = self.data.sort_index().loc[(ts[1], ts[0])] ## sort_index() to prevent the performance warning.
            except:
                pass
            if ts_data.shape == 0:
                pass

            close_info = ts_data[['close','spread_close','Ticker']].reset_index(drop=True)
            ts_preds = self.preds.loc[(ts[1], ts[0])]

            self.update_positions(ts, close_info, ts_preds)

            self.t += 1

        print(f'run function finished at step {self.t}, time: {ts}')

    def update_positions(self, ts, close_info, ts_preds):

#         close_short = []
#         close_buy = []

        # use predictions to select what to hold, order by the confidence
#         print(ts_preds[ts_preds['class'] == (self.n_classes - 1)].sort_values('confidence',
#                                                                               ascending=False))

        # Long positions
        long_pos = ts_preds[ts_preds['class'] == (self.n_classes - 1)].sort_values('confidence',
                                                                                   ascending=False)
        # Short positions
        short_pos = ts_preds[ts_preds['class'] == 0].sort_values('confidence',ascending=False)

        # Open all available long positions if the number of available long positions are less than the
        # intended number of long positions.
        if (long_pos.shape[0] <= (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

            long_list = long_pos.index.values
            short_list = short_pos.index.values

            if self.verbose:
                print("\n NOTE: The number of available positions are %i, compared to the intended %i.\n" % (len(long_list)+\
                                                                                                            len(short_list),self.max_positions))
        ## If both the available long and short exceeds the needed:
        elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] > (self.max_positions / 2)):

            long_list = long_pos.iloc[0:int(self.max_positions/2),:].index.values
            short_list = short_pos.iloc[0:int(self.max_positions/2),:].index.values

        ## If there are less long and excess short positions:
        elif (long_pos.shape[0] <= (self.max_positions / 2)) & (short_pos.shape[0] > (self.max_positions / 2)):

            long_list = long_pos.index.values
            short_list = short_pos.iloc[0:int(self.max_positions-long_pos.shape[0]),:].index.values

        ## If there are less short and excess long positions:
        elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

            long_list = long_pos.iloc[0:int(self.max_positions-short_pos.shape[0]),:].index.values
            short_list = short_pos.index.values

        ## Else return eror
        else:
#             print('Longs: \n\n',long_pos,'\n')
#             print('Shorts: \n\n',short_pos,'\n')
#             long_list = long_pos.iloc[0:int(self.max_positions/2),:].index.values
            raise ValueError('Something is wrong - please investigate!')

#         long_list = ts_preds[ts_preds['class'] == (self.n_classes - 1)].index.values
#         short_list = ts_preds[ts_preds['class'] == 0].index.values

        if self.verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # check if any new positions are made
        new_long = long_list[~np.isin(long_list, self.open_long_positions)]
        new_short = short_list[~np.isin(short_list, self.open_short_positions)]
        long_candidates = self.open_long_positions+list(new_long)
        short_candidates = self.open_short_positions+list(new_short)
#         print(long_candidates)
#         print(ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False))
        print('New shorts:\n\n',new_short,'\n')
        print('Short candidates:\n\n',short_candidates,'\n')
        print('Long candidates:\n\n',long_candidates,'\n')
        if (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) <= (self.max_positions / 2)):

            # Do nothing, sticking with the generated list, consisting of open positions and new buys!
            pass

            if self.verbose:
                print("\n NOTE: The number of available positions are %i, compared to the intended %i.\n" % (len(long_candidates)+\
                                                                                                            len(short_candidates),
                                                                                                             self.max_positions))
            close_long = []
            close_short = []
        ## If both the available long and short exceeds the needed:
        elif (len(long_candidates) > (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

#             long_list = long_pos.iloc[0:int(self.max_positions/2),:].index.values
#             short_list = short_pos.iloc[0:int(self.max_positions/2),:].index.values
            ### Prepping the candidates
            temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values
            temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values
#             print(np.arange(len(temp_long))[np.isin(temp_long,new_long)])

            ## Determining the ranking of the new candidates
            new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]
            new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]
            print('new_short_ranking:\n\n',new_short_ranking,'\n')
            # if some new buy candidates have a ranking that is outside the neeeded long positions,
            # we diregard that new buy.
            if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

            ## Same for sell candidates
            if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

            print('new short:\n\n',new_short,'\n')
            # Locating the close candidates (those that have ranking exceeding the maximum number of intended
            # long positions)
#             print(temp_long)
            close_candidates_long = list(temp_long[int(self.max_positions / 2):])
            close_candidates_short = list(temp_short[int(self.max_positions / 2):])

            # Only close those that are not in new buy (in position to be bought)
            close_long = [i for i in close_candidates_long if i not in new_long]
            close_short = [i for i in close_candidates_short if i not in new_short]
#             print(np.where(new_buy in temp_long,new_buy,temp_long))


        ## If there are less long and excess short positions:
        elif (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

#             long_list = long_pos.index.values

            ## With less long candidates than needed, there is no need to do anything at this point, as all existing
            ## positions should be kept, unless they do not have a buy signal anymore but then they are dropped below,
            ## and new ones should be added.
            close_long = [] # Because of above, the only action we take for buys is initialising the close_buy list.

            ## For the sell candidates we first identify if we need to adjust. If we do not need to adjust, we just
            ## proceed as normally.
#             short_list = short_pos.iloc[0:int(self.max_positions-long_pos.shape[0]),:].index.values
            if len(short_candidates) >= (self.max_positions - len(long_candidates)):

                temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values
#                 print(np.arange(len(temp_long))[np.isin(temp_long,new_buy)])

                ## Determining the ranking of the new candidates
                new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                # if some new sell candidates have a ranking that is outside the neeeded short positions,
                # we diregard that new sell.
                if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                    new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] < ((self.max_positions / 2) - 1)]

                # Locating the close candidates (those that have ranking exceeding the maximum number of intended
                # short positions)
                close_candidates_short = list(temp_short[(self.max_positions / 2):])

                # Only close those that are not in new buy (in position to be bought)
                close_short = [i for i in close_candidates_short if i not in new_short]

        ## If there are less short and excess long positions:
        elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

#             long_list = long_pos.iloc[0:int(self.max_positions-short_pos.shape[0]),:].index.values
#             short_list = short_pos.index.values
            ### Vice versa, compared to the if statement just above.

            ## With less sell candidates than needed, there is no need to do anything at this point, as all existing
            ## positions should be kept, unless they do not have a sell signal anymore but then they are dropped below,
            ## and new ones should be added.
            close_short = [] # Because of above, the only action we take for buys is initialising the close_buy list.

            ## For the short candidates we first identify if we need to adjust. If we do not need to adjust, we just
            ## proceed as normally.
#             short_list = short_pos.iloc[0:int(self.max_positions-long_pos.shape[0]),:].index.values
            if len(long_candidates) >= (self.max_positions - len(short_candidates)):

                temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values
#                 print(np.arange(len(temp_long))[np.isin(temp_long,new_buy)])

                ## Determining the ranking of the new candidates
                new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]

                # if some new sell candidates have a ranking that is outside the neeeded short positions,
                # we diregard that new sell.
                if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                    new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] < ((self.max_positions / 2) - 1)]

                # Locating the close candidates (those that have ranking exceeding the maximum number of intended
                # short positions)
                close_candidates_long = list(temp_long[(self.max_positions / 2):])

                # Only close those that are not in new buy (in position to be bought)
                close_long = [i for i in close_candidates_long if i not in new_long]

        ## Else return eror
        else:
            print('Longs:\n\n',long_pos,'\n')
            print('Shorts:\n\n',short_pos,'\n')
#             long_list = long_pos.iloc[0:int(self.max_positions/2),:].index.values
            raise ValueError('Something is wrong - please investigate!')



        if self.verbose >= 1:
            print(f'new_long: {new_long}')
            print(f'new_short: {new_short} \n')

        self.open_long_positions += [long_i for long_i in new_long]
        self.open_short_positions += [short_i for short_i in new_short]

        self.ticker_dict

        if self.verbose >= 1:
            print('Opening new trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        # check if any are closed
#         print(close_long,close_short)
        close_long = close_long + list(np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)])
        close_short = close_short + list(np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)])

        if self.verbose >= 1:
            print(f'close_long: {close_long}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_long]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if self.verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        if self.t > 0:

            # update directions for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction'] = -1
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction'] = 1

            current_ret = (close_info.close.values / self.prev_close.prev_close.values) - 1
            current_ret = (current_ret * self.prev_close.direction.values) + 1

            ## Correcting those we close
            boolcousin = np.isin(self.all_tickers, close_short) | np.isin(self.all_tickers, close_long)

            current_ret[boolcousin] = (close_info[boolcousin].close.values\
                                                                 *abs(self.prev_close[boolcousin].direction.values)\
                                                                 +(close_info[boolcousin].spread_close.values / 2)\
                                                                 *self.prev_close[boolcousin].direction.values)\
                                                                /self.prev_close[boolcousin].prev_close.values

            ## Fixing those not in use, which by the 'direction' equals a return of zero.
            current_ret[self.prev_close.direction.values == 0] = 1

            # update individual returns for open positions
            self.hist_rets[self.t, :-1] = current_ret * self.hist_rets[self.t - 1, :-1]

            # update total portfolio returns for open positions
            self.hist_rets[self.t, -1] = (1+np.mean(current_ret[self.prev_close.direction.values != 0] - 1)) * self.hist_rets[self.t - 1, -1]

            # update directions for new positions
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)

            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

        else:

            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)
