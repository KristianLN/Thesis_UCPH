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
'''
1) Each timestep
    -check for new positions
    -re-evaluate current positions
    -rebalance maybe
    -

2) Max_positions: Check if the specified number of max positions results in an uneven split? like max position = 9.
'''
class backtest_v3():
    def __init__(self,
                 X_test,
#                  X_train,
                 data,
                 preds,
                 weight_scheme,
                 rebal_scheme,
                 strategy_scheme,
                 rebal_init_data,
                 rebal_last_known_price,
                 rebal_lookback_horizon,
                 rebal_risk_aversion,
                 max_steps,
                 max_positions,
                 n_classes,
                 slpg_warning = False,
                 slpg_input = [],
                 verbose=False):

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
        self.strategy_scheme = strategy_scheme
        self.rebal_scheme = rebal_scheme
        self.rebal_init_data = rebal_init_data
        self.rebal_last_known_price = rebal_last_known_price
        self.rebal_lookback_horizon = rebal_lookback_horizon
        self.rebal_risk_aversion = rebal_risk_aversion
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

        if self.rebal_scheme == 'slpg': # slpg: Stop Loss / Profit Goal

            self.returns_container = returns(self.rebal_init_data,#cp_x_train.iloc[-(self.rebal_lookback_horizon+1):,:]
                                        self.rebal_lookback_horizon)

#             self.output_container = np.zeros((self.max_steps+1,
#                                          len(self.all_tickers)))
            self.output_container = np.zeros((self.max_steps,
                                         len(self.all_tickers)))

#             self.output_container[0] = np.std(self.returns_container,axis=0)

        self.ticker_dict = {i:j for i,j in enumerate(self.all_tickers)}

        print('Size of returns container: ', self.returns_container.shape)
        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]
            print(ts)
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
#             print(close_info)
            self.update_positions(ts, close_info, ts_preds)#,self.rebal_scheme

            self.t += 1

        print(f'run function finished at step {self.t}, time: {ts}')

    def update_positions(self, ts, close_info, ts_preds):#,rebal_scheme

        # Long positions
        long_pos = ts_preds[ts_preds['class'] == (self.n_classes - 1)].sort_values('confidence',
                                                                                   ascending=False)
        # Short positions
        short_pos = ts_preds[ts_preds['class'] == 0].sort_values('confidence',ascending=False)

        ## Open all available long positions if the number of available long positions are less than the
        ## intended number of long positions.
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

            raise ValueError('Something is wrong - please investigate!')

        if self.verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # check if any new positions are made
        new_long = long_list[~np.isin(long_list, self.open_long_positions)]
        new_short = short_list[~np.isin(short_list, self.open_short_positions)]

        ###### check if any needs closed before we determine if any new ones should be
        ###### disregarded.

        ## Closing those that have changed signal.
        close_long = list(np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)])
        close_short = list(np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)])

        ## Check if time to close positions due to stop loss or profit goal.
        if (self.t > 0) & (self.rebal_scheme == 'slpg') & (self.verbose):

            positions_above_pg = np.where(self.hist_rets[self.t-1][0:-1]>(1+self.output_container[self.t-1]*self.rebal_risk_aversion[1]))[0]
            positions_below_sl = np.where(self.hist_rets[self.t-1][0:-1]<(1+self.output_container[self.t-1]*self.rebal_risk_aversion[0]*-1))[0]

            print('Number of positions above profit goal: ',len(positions_above_pg))

            if len(positions_above_pg) > 0:
                print('Positionns exceeding the profit goal: ',[(self.ticker_dict[i],i) for i in positions_above_pg],'\n')

            print('Number of positions below stop loss: ',len(positions_below_sl))

            if len(positions_below_sl) > 0:
                print('Positionns exceeding the stop loss: ',[(self.ticker_dict[i],i) for i in positions_below_sl])

        if self.verbose >= 1:
            print(f'\nclose_long: {close_long}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_long]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if self.verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        long_candidates = self.open_long_positions+list(new_long)
        short_candidates = self.open_short_positions+list(new_short)

        ## Checking if any of the new candidates should be included or any of the existing should be closed
        ## at the expense of a new.

        if (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) <= (self.max_positions / 2)):

            # Do nothing, sticking with the generated list, consisting of open positions and new buys!
            pass

            if self.verbose:
                print("\n NOTE: The number of available positions are %i, compared to the intended %i.\n" % (len(long_candidates)+\
                                                                                                            len(short_candidates),
                                                                                                             self.max_positions))
        ## If both the available long and short exceeds the needed:
        elif (len(long_candidates) > (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

            ### Prepping the candidates
            temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values
            temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

            # Determining the ranking of the new candidates
            new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]
            new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

            # Determining the ranking of the existing positions
            open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]
            open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

            # if some new buy candidates have a ranking that is outside the desired long positions,
            # we diregard that new buy.
            if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

            ## Same for sell candidates
            if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

            ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
            ## we close it.
            if max(open_long_ranking) > ((self.max_positions / 2) - 1):
                self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

            ## Same for sell candidates
            if max(open_short_ranking) > ((self.max_positions / 2) - 1):
                self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]


        ## If there are less long and excess short positions:
        elif (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

            ## With less long candidates than needed, there is no need to do anything at this point, as all existing
            ## positions should be kept, unless they do not have a buy signal anymore but then they are dropped above,
            ## and new ones should be added.
            ## For the sell candidates we first identify if we need to adjust. If we do not need to adjust, we just
            ## proceed as normally.

            if len(short_candidates) >= (self.max_positions - len(long_candidates)):

                temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                ## Determining the ranking of the new candidates
                new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                # Determining the ranking of the existing positions
                open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                # if some new sell candidates have a ranking that is outside the neeeded short positions,
                # we diregard that new sell.
                if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                    new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                ## we close it.
                if max(open_short_ranking) > ((self.max_positions / 2) - 1):
                    self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                    close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]

        ## If there are less short and excess long positions:
        elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

            ### Vice versa, compared to the if statement just above.

            ## With less sell candidates than needed, there is no need to do anything at this point, as all existing
            ## positions should be kept, unless they do not have a sell signal anymore but then they are dropped below,
            ## and new ones should be added.

            ## For the short candidates we first identify if we need to adjust. If we do not need to adjust, we just
            ## proceed as normally.
            if len(long_candidates) >= (self.max_positions - len(short_candidates)):

                temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values

                ## Determining the ranking of the new candidates
                new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]

                # Determining the ranking of the existing positions
                open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]

                # if some new sell candidates have a ranking that is outside the neeeded short positions,
                # we diregard that new sell.
                if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                    new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                ## we close it.
                if max(open_long_ranking) > ((self.max_positions / 2) - 1):
                    self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                    close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

        ## Else return eror
        else:

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

            ## Roll back return matrix, to make room to the return coming in at the end of the period.
            self.returns_container[0:-1] = self.returns_container[1:]

            ## Include the new return.
            self.returns_container[-1] = (close_info.close.values/self.prev_close.prev_close)-1

            ## Updating the standard deviation of the returns
            self.output_container[self.t] = np.std(self.returns_container,axis=0)

            ## Updating the last seen price
            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            ## Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            ## Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)

            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2
            overlap = []
#             print(np.isin(new_long,close_long))
#             print(close_long)
            overlap += list(np.array(close_long)[np.isin(new_long,close_long)])
            if len(overlap) > 0:
                print('New longs in close long: ', overlap)

            overlap += list(np.array(new_long)[np.isin(close_long,new_long)])
            if len(overlap) > 0:
                print('Close longs in new long: ', overlap)

            overlap += list(np.array(close_short)[np.isin(new_short,close_short)])
            if len(overlap) > 0:
                print('New shorts in close short: ', overlap)

            overlap += list(np.array(new_short)[np.isin(close_short,new_short)])
            if len(overlap) > 0:
                print('Close shorts in new short: ', overlap)

            if len(overlap) > 0:
                print('######### Overlap between the new candidates and the close candidates #########')

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

        else:

            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            if self.rebal_scheme == 'slpg':

                ## Place to update returns in the returns container
                # Roll back the returns one period
                self.returns_container[0:-1] = self.returns_container[1:]

                # Include the new return
                print('New prices:\n\n',close_info.close.values)
                print('Old prices:\n\n',self.rebal_last_known_price)
                print('Return:\n\n',(close_info.close.values/self.rebal_last_known_price)-1)
                self.returns_container[-1] = (close_info.close.values/self.rebal_last_known_price)-1

                # Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)
'''
1) Each timestep
    -check for new positions
    -re-evaluate current positions
    -rebalance maybe
    -

2) Max_positions: Check if the specified number of max positions results in an uneven split? like max position = 9.
'''

class backtest_v4():
    def __init__(self,
                 X_test,
#                  X_train,
                 data,
                 preds,
                 weight_scheme,
                 rebal_scheme,
                 strategy_scheme,
                 # rebal_init_data,
                 # rebal_last_known_price,
                 # rebal_lookback_horizon,
                 # rebal_risk_aversion,
                 max_steps,
                 max_positions,
                 n_classes,
                 slpg_warning = False,
                 slpg_input = {},
                 verbose=False):

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
        self.strategy_scheme = strategy_scheme
        self.rebal_scheme = rebal_scheme
        self.slpg_warning = slpg_warning
        self.slpg_init_data = slpg_input['init_data']
        self.slpg_last_known_price = slpg_input['last_known_price']
        self.slpg_lookback_horizon = slpg_input['lookback_horizon']
        self.slpg_risk_aversion = slpg_input['risk_aversion']
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

        if self.slpg_warning: # slpg: Stop Loss / Profit Goal

            self.returns_container = returns(self.slpg_init_data,#cp_x_train.iloc[-(self.rebal_lookback_horizon+1):,:]
                                        self.slpg_lookback_horizon)

#             self.output_container = np.zeros((self.max_steps+1,
#                                          len(self.all_tickers)))
            self.output_container = np.zeros((self.max_steps,
                                         len(self.all_tickers)))

#             self.output_container[0] = np.std(self.returns_container,axis=0)

        self.ticker_dict = {i:j for i,j in enumerate(self.all_tickers)}

        print('Size of returns container: ', self.returns_container.shape)
        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]
            print(ts)
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
#             print(close_info)
            self.update_positions(ts, close_info, ts_preds)#,self.rebal_scheme

            self.t += 1

        print(f'run function finished at step {self.t}, time: {ts}')

    def update_positions(self, ts, close_info, ts_preds):#,rebal_scheme

        # Long positions
        long_pos = ts_preds[ts_preds['class'] == (self.n_classes - 1)].sort_values('confidence',
                                                                                   ascending=False)
        # Short positions
        short_pos = ts_preds[ts_preds['class'] == 0].sort_values('confidence',ascending=False)

        # The choice of strategy scheme determines the actions going forward
        if self.strategy_scheme == 'max_pos':

            ## Open all available long positions if the number of available long positions are less than the
            ## intended number of long positions.
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

                raise ValueError('Something is wrong - please investigate!')

        ## If "None" strategy scheme is chosen, all candicates are chosen.
        elif self.strategy_scheme == None:

            long_list = long_pos.index.values
            short_list = short_pos.index.values

        ## Else return eror
        else:

            raise ValueError('Something is wrong - please investigate!')

        if self.verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # check if any new positions are made
        new_long = long_list[~np.isin(long_list, self.open_long_positions)]
        new_short = short_list[~np.isin(short_list, self.open_short_positions)]

        ###### check if any needs closed before we determine if any new ones should be
        ###### disregarded.

        ## Closing those that have changed signal.
        close_long = list(np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)])
        close_short = list(np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)])

        ## Meant to check if time to close positions due to stop loss or profit goal.
        ## However, at the moment we just inform about positions that exceeds either of the thresholds.
        if (self.t > 0) & (self.slpg_warning):

            positions_above_pg = np.where(self.hist_rets[self.t-1][0:-1]>(1+self.output_container[self.t-1]*self.slpg_risk_aversion[1]))[0]
            positions_below_sl = np.where(self.hist_rets[self.t-1][0:-1]<(1+self.output_container[self.t-1]*self.slpg_risk_aversion[0]*-1))[0]

            print('Number of positions above profit goal: ',len(positions_above_pg))

            if len(positions_above_pg) > 0:
                print('Positionns exceeding the profit goal: ',[(self.ticker_dict[i],i) for i in positions_above_pg],'\n')

            print('Number of positions below stop loss: ',len(positions_below_sl))

            if len(positions_below_sl) > 0:
                print('Positionns exceeding the stop loss: ',[(self.ticker_dict[i],i) for i in positions_below_sl])

        if self.verbose >= 1:
            print(f'\nclose_long: {close_long}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_long]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if self.verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        if self.strategy_scheme == 'max_pos':

            long_candidates = self.open_long_positions+list(new_long)
            short_candidates = self.open_short_positions+list(new_short)

            ## Checking if any of the new candidates should be included or any of the existing should be closed
            ## at the expense of a new.
            if (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) <= (self.max_positions / 2)):

                # Do nothing, sticking with the generated list, consisting of open positions and new buys!
                pass

                if self.verbose:
                    print("\n NOTE: The number of available positions are %i, compared to the intended %i.\n" % (len(long_candidates)+\
                                                                                                                len(short_candidates),
                                                                                                                 self.max_positions))
            ## If both the available long and short exceeds the needed:
            elif (len(long_candidates) > (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

                ### Prepping the candidates
                temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values
                temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                # Determining the ranking of the new candidates
                new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]
                new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                # Determining the ranking of the existing positions
                open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]
                open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                # if some new buy candidates have a ranking that is outside the desired long positions,
                # we diregard that new buy.
                if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                    new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Same for sell candidates
                if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                    new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                ## we close it.
                if max(open_long_ranking) > ((self.max_positions / 2) - 1):
                    self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                    close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

                ## Same for sell candidates
                if max(open_short_ranking) > ((self.max_positions / 2) - 1):
                    self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                    close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]

            ## If there are less long and excess short positions:
            elif (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

                ## With less long candidates than needed, there is no need to do anything at this point, as all existing
                ## positions should be kept, unless they do not have a buy signal anymore but then they are dropped above,
                ## and new ones should be added.
                ## For the sell candidates we first identify if we need to adjust. If we do not need to adjust, we just
                ## proceed as normally.

                if len(short_candidates) >= (self.max_positions - len(long_candidates)):

                    temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                    ## Determining the ranking of the new candidates
                    new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                    # Determining the ranking of the existing positions
                    open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                    # if some new sell candidates have a ranking that is outside the neeeded short positions,
                    # we diregard that new sell.
                    if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                        new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                    ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                    ## we close it.
                    if max(open_short_ranking) > ((self.max_positions / 2) - 1):
                        self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                        close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]

            ## If there are less short and excess long positions:
            elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

                ### Vice versa, compared to the if statement just above.

                ## With less sell candidates than needed, there is no need to do anything at this point, as all existing
                ## positions should be kept, unless they do not have a sell signal anymore but then they are dropped below,
                ## and new ones should be added.

                ## For the short candidates we first identify if we need to adjust. If we do not need to adjust, we just
                ## proceed as normally.
                if len(long_candidates) >= (self.max_positions - len(short_candidates)):

                    temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values

                    ## Determining the ranking of the new candidates
                    new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]

                    # Determining the ranking of the existing positions
                    open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]

                    # if some new sell candidates have a ranking that is outside the neeeded short positions,
                    # we diregard that new sell.
                    if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                        new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                    ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                    ## we close it.
                    if max(open_long_ranking) > ((self.max_positions / 2) - 1):
                        self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                        close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

            ## Else return eror
            else:

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

            if self.slpg_warning:
                ## Roll back return matrix, to make room to the return coming in at the end of the period.
                self.returns_container[0:-1] = self.returns_container[1:]

                ## Include the new return.
                self.returns_container[-1] = (close_info.close.values/self.prev_close.prev_close)-1

                ## Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            ## Updating the last seen price
            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            ## Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            ## Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)
            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2
            overlap = []
#             print(np.isin(new_long,close_long))
#             print(close_long)
            overlap += list(np.array(close_long)[np.isin(close_long, new_long)])
            if len(overlap) > 0:
                print('New longs in close long: ', overlap)

            overlap += list(np.array(new_long)[np.isin(new_long, close_long)])
            if len(overlap) > 0:
                print('Close longs in new long: ', overlap)

            overlap += list(np.array(close_short)[np.isin(close_short, new_short)])
            if len(overlap) > 0:
                print('New shorts in close short: ', overlap)

            overlap += list(np.array(new_short)[np.isin(new_short, close_short)])
            if len(overlap) > 0:
                print('Close shorts in new short: ', overlap)

            if len(overlap) > 0:
                print('######### Overlap between the new candidates and the close candidates #########')

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

        else:

            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            if self.slpg_warning:

                ## Place to update returns in the returns container
                # Roll back the returns one period
                self.returns_container[0:-1] = self.returns_container[1:]

                # Include the new return
                self.returns_container[-1] = (close_info.close.values/self.slpg_last_known_price)-1

                # Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)

'''
1) Each timestep
    -check for new positions
    -re-evaluate current positions
    -rebalance maybe
    -

2) Max_positions: Check if the specified number of max positions results in an uneven split? like max position = 9.
'''


class backtest_v5():
    def __init__(self,
                 X_test,
#                  X_train,
                 data,
                 preds,
                 weight_scheme,
                 rebal_scheme,
                 strategy_scheme,
                 # rebal_init_data,
                 # rebal_last_known_price,
                 # rebal_lookback_horizon,
                 # rebal_risk_aversion,
                 max_steps,
                 max_positions,
                 n_classes,
                 slpg_warning = False,
                 slpg_input = {},
                 verbose=False):

        self.all_tickers = data.Ticker.unique()
        self.open_long_positions = []#'AAPL','ABT','LFC'
        self.open_short_positions = []#'BAC','KO','ENB'

        self.ticker_dict = {}   # key: ticker, value: [open_price, direction, pnl]
        self.hist_rets = np.ones((max_steps,len(self.all_tickers)+1))
        self.hist_directions = np.zeros((max_steps,len(self.all_tickers)))
        self.pnl = []
        self.prev_close = pd.DataFrame(0, index=self.all_tickers, columns=['prev_close', 'direction'])
        self.t = 0

        self.X_test = X_test
        self.data = data
        self.preds = preds
        ## New
        self.weight_scheme = weight_scheme
        self.strategy_scheme = strategy_scheme
        self.rebal_scheme = rebal_scheme
        self.slpg_warning = slpg_warning
        self.slpg_init_data = slpg_input['init_data']
        self.slpg_last_known_price = slpg_input['last_known_price']
        self.slpg_lookback_horizon = slpg_input['lookback_horizon']
        self.slpg_risk_aversion = slpg_input['risk_aversion']
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

        if self.slpg_warning: # slpg: Stop Loss / Profit Goal

            self.returns_container = returns(self.slpg_init_data,#cp_x_train.iloc[-(self.rebal_lookback_horizon+1):,:]
                                        self.slpg_lookback_horizon)

#             self.output_container = np.zeros((self.max_steps+1,
#                                          len(self.all_tickers)))
            self.output_container = np.zeros((self.max_steps,
                                         len(self.all_tickers)))

#             self.output_container[0] = np.std(self.returns_container,axis=0)

        self.ticker_dict = {i:j for i,j in enumerate(self.all_tickers)}

        print('Size of returns container: ', self.returns_container.shape)
        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]
            print(ts)
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
#             print(close_info)
            self.update_positions(ts, close_info, ts_preds)#,self.rebal_scheme

            self.t += 1

        print(f'run function finished at step {self.t}, time: {ts}')

    def update_positions(self, ts, close_info, ts_preds):#,rebal_scheme

        # Long positions
        long_pos = ts_preds[ts_preds['class'] == (self.n_classes - 1)].sort_values('confidence',
                                                                                   ascending=False)
        # Short positions
        short_pos = ts_preds[ts_preds['class'] == 0].sort_values('confidence',ascending=False)

        # The choice of strategy scheme determines the actions going forward
        if self.strategy_scheme == 'max_pos':

            ## Open all available long positions if the number of available long positions are less than the
            ## intended number of long positions.
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

                raise ValueError('Something is wrong - please investigate!')

        ## If "None" strategy scheme is chosen, all candicates are chosen.
        elif self.strategy_scheme == None:

            long_list = long_pos.index.values
            short_list = short_pos.index.values

        ## Else return eror
        else:

            raise ValueError('Something is wrong - please investigate!')

        if self.verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # check if any new positions are made
        new_long = long_list[~np.isin(long_list, self.open_long_positions)]
        new_short = short_list[~np.isin(short_list, self.open_short_positions)]

        ###### check if any needs closed before we determine if any new ones should be
        ###### disregarded.

        ## Closing those that have changed signal.
        close_long = list(np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)])
        close_short = list(np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)])

        ## Meant to check if time to close positions due to stop loss or profit goal.
        ## However, at the moment we just inform about positions that exceeds either of the thresholds.
        if (self.t > 0) & (self.slpg_warning):

            positions_above_pg = np.where(self.hist_rets[self.t-1][0:-1]>(1+self.output_container[self.t-1]*self.slpg_risk_aversion[1]))[0]
            positions_below_sl = np.where(self.hist_rets[self.t-1][0:-1]<(1+self.output_container[self.t-1]*self.slpg_risk_aversion[0]*-1))[0]

            print('Number of positions above profit goal: ',len(positions_above_pg))

            if len(positions_above_pg) > 0:
                print('Positionns exceeding the profit goal: ',[(self.ticker_dict[i],i) for i in positions_above_pg],'\n')

            print('Number of positions below stop loss: ',len(positions_below_sl))

            if len(positions_below_sl) > 0:
                print('Positionns exceeding the stop loss: ',[(self.ticker_dict[i],i) for i in positions_below_sl])

        if self.verbose >= 1:
            print(f'\nclose_long: {close_long}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_long]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if self.verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        if self.strategy_scheme == 'max_pos':

            long_candidates = self.open_long_positions+list(new_long)
            short_candidates = self.open_short_positions+list(new_short)

            ## Checking if any of the new candidates should be included or any of the existing should be closed
            ## at the expense of a new.
            if (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) <= (self.max_positions / 2)):

                # Do nothing, sticking with the generated list, consisting of open positions and new buys!
                pass

                if self.verbose:
                    print("\n NOTE: The number of available positions are %i, compared to the intended %i.\n" % (len(long_candidates)+\
                                                                                                                len(short_candidates),
                                                                                                                 self.max_positions))
            ## If both the available long and short exceeds the needed:
            elif (len(long_candidates) > (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

                ### Prepping the candidates
                temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values
                temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                # Determining the ranking of the new candidates
                new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]
                new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                # Determining the ranking of the existing positions
                open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]
                open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                # if some new buy candidates have a ranking that is outside the desired long positions,
                # we diregard that new buy.
                if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                    new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Same for sell candidates
                if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                    new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                ## we close it.
                if max(open_long_ranking) > ((self.max_positions / 2) - 1):
                    self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                    close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

                ## Same for sell candidates
                if max(open_short_ranking) > ((self.max_positions / 2) - 1):
                    self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                    close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]

            ## If there are less long and excess short positions:
            elif (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

                ## With less long candidates than needed, there is no need to do anything at this point, as all existing
                ## positions should be kept, unless they do not have a buy signal anymore but then they are dropped above,
                ## and new ones should be added.
                ## For the sell candidates we first identify if we need to adjust. If we do not need to adjust, we just
                ## proceed as normally.

                if len(short_candidates) >= (self.max_positions - len(long_candidates)):

                    temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                    ## Determining the ranking of the new candidates
                    new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                    # Determining the ranking of the existing positions
                    open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                    # if some new sell candidates have a ranking that is outside the neeeded short positions,
                    # we diregard that new sell.
                    if max(new_short_ranking) > ((self.max_positions / 2) - 1):
                        new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                    ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                    ## we close it.
                    if max(open_short_ranking) > ((self.max_positions / 2) - 1):
                        self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                        close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]

            ## If there are less short and excess long positions:
            elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

                ### Vice versa, compared to the if statement just above.

                ## With less sell candidates than needed, there is no need to do anything at this point, as all existing
                ## positions should be kept, unless they do not have a sell signal anymore but then they are dropped below,
                ## and new ones should be added.

                ## For the short candidates we first identify if we need to adjust. If we do not need to adjust, we just
                ## proceed as normally.
                if len(long_candidates) >= (self.max_positions - len(short_candidates)):

                    temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values

                    ## Determining the ranking of the new candidates
                    new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]

                    # Determining the ranking of the existing positions
                    open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]

                    # if some new sell candidates have a ranking that is outside the neeeded short positions,
                    # we diregard that new sell.
                    if max(new_long_ranking) > ((self.max_positions / 2) - 1):
                        new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                    ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                    ## we close it.
                    if max(open_long_ranking) > ((self.max_positions / 2) - 1):
                        self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                        close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

            ## Else return eror
            else:

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

        ########## Calculating returns ##########

        if self.t > 0:
            print('\nIndex 37 direction before return calculation: ',self.prev_close.iloc[37, 1],'\n')
            # update directions for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction'] = -1
            print('\nIndex 37 direction, before return but after adjustment: ',self.prev_close.iloc[37, 1],'\n')
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction'] = 1

            current_ret = (close_info.close.values / self.prev_close.prev_close.values) - 1
            current_ret = (current_ret * self.prev_close.direction.values) + 1

            print('Current ret, index 37: ',current_ret[37],'\n')

            ## Correcting those we close
            boolcousin = np.isin(self.all_tickers, close_short) | np.isin(self.all_tickers, close_long)

            print('\nIndex 37: ',boolcousin[37],'\n')
            current_ret[boolcousin] = (close_info[boolcousin].close.values\
                                                                 *abs(self.prev_close[boolcousin].direction.values)\
                                                                 +(close_info[boolcousin].spread_close.values / 2)\
                                                                 *self.prev_close[boolcousin].direction.values)\
                                                                /self.prev_close[boolcousin].prev_close.values - 1

            ## We multiply with the opposite direction to get the correct return.
            current_ret[boolcousin] = (current_ret[boolcousin] * -self.prev_close[boolcousin].direction.values) + 1
            print('Current ret with bool cousin, index 37: ',current_ret[37],'\n')
            ## Fixing those not in use, which by the 'direction' equals a return of zero.
            current_ret[self.prev_close.direction.values == 0] = 1

            # update individual returns for open positions
            self.hist_rets[self.t, :-1] = current_ret * self.hist_rets[self.t - 1, :-1]

            # update total portfolio returns for open positions
            self.hist_rets[self.t, -1] = (1+np.mean(current_ret[self.prev_close.direction.values != 0] - 1)) * self.hist_rets[self.t - 1, -1]

            # update directions for new positions
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            print('\nIndex 37 direction, after return calculation: ',self.prev_close.iloc[37, 1],'\n')
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            if self.slpg_warning:
                ## Roll back return matrix, to make room to the return coming in at the end of the period.
                self.returns_container[0:-1] = self.returns_container[1:]

                ## Include the new return.
                self.returns_container[-1] = (close_info.close.values/self.prev_close.prev_close)-1

                ## Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            ## Updating the last seen price
            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            ## Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            ## Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)
            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2
            overlap = []
#             print(np.isin(new_long,close_long))
#             print(close_long)
            overlap += list(np.array(close_long)[np.isin(close_long, new_long)])
            if len(overlap) > 0:
                print('New longs in close long: ', overlap)

            overlap += list(np.array(new_long)[np.isin(new_long, close_long)])
            if len(overlap) > 0:
                print('Close longs in new long: ', overlap)

            overlap += list(np.array(close_short)[np.isin(close_short, new_short)])
            if len(overlap) > 0:
                print('New shorts in close short: ', overlap)

            overlap += list(np.array(new_short)[np.isin(new_short, close_short)])
            if len(overlap) > 0:
                print('Close shorts in new short: ', overlap)

            if len(overlap) > 0:
                print('######### Overlap between the new candidates and the close candidates #########')

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

            # Save the directions
            self.hist_directions[self.t,:] = self.prev_close.direction.copy(deep=True)

            print('\nIndex 37 hist rets: ',self.hist_rets[:,37],'\n')

        else:

            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            if self.slpg_warning:

                ## Place to update returns in the returns container
                # Roll back the returns one period
                self.returns_container[0:-1] = self.returns_container[1:]

                # Include the new return
                self.returns_container[-1] = (close_info.close.values/self.slpg_last_known_price)-1

                # Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)

            # Save the directions
            self.hist_directions[self.t,:] = self.prev_close.direction.copy(deep=True)

# '''
# 1) Each timestep
#     -check for new positions
#     -re-evaluate current positions
#     -rebalance maybe
#     -

# 2) Max_positions: Check if the specified number of max positions results in an uneven split? like max position = 9.
# '''

class backtest_v6():
    def __init__(self,
                 X_test,
#                  X_train,
                 data,
                 preds,
                 weight_scheme,
                 rebal_scheme,
                 strategy_scheme,
                 # rebal_init_data,
                 # rebal_last_known_price,
                 # rebal_lookback_horizon,
                 # rebal_risk_aversion,
                 max_steps,
                 max_positions,
                 n_classes,
                 slpg_warning = False,
                 slpg_input = {},
                 return_revealer = 1000,
                 zero_spread = False,
                 verbose=False):

        print('#################### Backtest initiated ####################\n')

        self.all_tickers = data.Ticker.unique()
        self.open_long_positions = []#'AAPL','ABT','LFC'
        self.open_short_positions = []#'BAC','KO','ENB'

        self.ticker_dict = {}   # key: ticker, value: [open_price, direction, pnl]
        self.hist_rets = np.ones((max_steps,len(self.all_tickers)+1))
        self.hist_directions = np.zeros((max_steps,len(self.all_tickers)))
        self.hist_prev_weights = np.zeros((max_steps,len(self.all_tickers)))
        self.pnl = []
        self.prev_close = pd.DataFrame(0, index=self.all_tickers, columns=['prev_close', 'direction'])
        self.t = 0

        self.X_test = X_test
        self.data = data
        self.preds = preds
        ## New
        self.weight_scheme = weight_scheme
        self.strategy_scheme = strategy_scheme
        self.rebal_scheme = rebal_scheme
        self.slpg_warning = slpg_warning
        self.slpg_init_data = slpg_input['init_data']
        self.slpg_last_known_price = slpg_input['last_known_price']
        self.slpg_lookback_horizon = slpg_input['lookback_horizon']
        self.slpg_risk_aversion = slpg_input['risk_aversion']
        self.max_steps = max_steps
        self.max_positions = max_positions
        self.n_classes = n_classes
        self.return_revealer = return_revealer
        self.zero_spread = zero_spread
        self.verbose = verbose

        if self.verbose >= 1:
            print(f'initial self.open_long_positions: {self.open_long_positions}')
            print(f'initial self.open_short_positions: {self.open_short_positions} \n')

    def run(self):

        unique_timesteps = np.concatenate([[[i,j] for i in np.unique(self.X_test.index.get_level_values(1))] \
                                                  for j in np.unique(self.X_test.index.get_level_values(0))])
        if self.verbose >= 1: print(self.all_tickers,'\n')

        if self.slpg_warning: # slpg: Stop Loss / Profit Goal

            self.returns_container = returns(self.slpg_init_data,#cp_x_train.iloc[-(self.rebal_lookback_horizon+1):,:]
                                        self.slpg_lookback_horizon)

            self.output_container = np.zeros((self.max_steps,
                                         len(self.all_tickers)))

        self.ticker_dict = {i:j for i,j in enumerate(self.all_tickers)}

        if self.verbose >= 1: print('Size of returns container: ', self.returns_container.shape)

        while self.t < self.max_steps:

            ts = unique_timesteps[self.t]
            if self.verbose >= 1: print(ts)
            if self.verbose >= 1: print('\n\n################ Period %i ################\n\n' % self.t)
            #print(i)
            try:
                ts_data = self.data.sort_index().loc[(ts[1], ts[0])] ## sort_index() to prevent the performance warning.
            except:
                pass
            if ts_data.shape == 0:
                pass

            close_info = ts_data[['close','spread_close','Ticker']].reset_index(drop=True)
            ts_preds = self.preds.loc[(ts[1], ts[0])]

            if self.zero_spread:
                close_info.loc[:,'spread_close'] = 0

            self.update_positions(ts, close_info, ts_preds)#,self.rebal_scheme

            ## Print the current portfolio return if the step are in the sequence.
            if self.t in np.arange(self.return_revealer,self.max_steps+self.return_revealer,self.return_revealer):

                print('Step %i - Current return: %.3f'%(self.t,self.hist_rets[self.t,-1]))

            self.t += 1

        if self.verbose >= 1: print(f'run function finished at step {self.t}, time: {ts}')

    def update_positions(self, ts, close_info, ts_preds):#,rebal_scheme

#         if self.t >= 291:
#             self.verbose = 1

        # Long positions
        long_pos = ts_preds[ts_preds['class'] == (self.n_classes - 1)].sort_values('confidence',
                                                                                   ascending=False)
        # Short positions
        short_pos = ts_preds[ts_preds['class'] == 0].sort_values('confidence',ascending=False)

        # The choice of strategy scheme determines the actions going forward
        if self.strategy_scheme == 'max_pos':

            ## Open all available long positions if the number of available long positions are less than the
            ## intended number of long positions.
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

                raise ValueError('Something is wrong - please investigate!')

        ## If "None" strategy scheme is chosen, all candicates are chosen.
        elif self.strategy_scheme == None:

            long_list = long_pos.index.values
            short_list = short_pos.index.values

        ## Else return eror
        else:

            raise ValueError('Something is wrong - please investigate!')

        if self.verbose >= 1:
            print(f'long_list: {long_list}')
            print(f'short_list: {short_list} \n')

        # check if any new positions are made
        new_long = long_list[~np.isin(long_list, self.open_long_positions)]
        new_short = short_list[~np.isin(short_list, self.open_short_positions)]

        ###### check if any needs closed before we determine if any new ones should be
        ###### disregarded.

        ## Closing those that have changed signal.
        close_long = list(np.array(self.open_long_positions)[~np.isin(self.open_long_positions, long_list)])
        close_short = list(np.array(self.open_short_positions)[~np.isin(self.open_short_positions, short_list)])

        ## Meant to check if time to close positions due to stop loss or profit goal.
        ## However, at the moment we just inform about positions that exceeds either of the thresholds.
        if (self.t > 0) & (self.slpg_warning):

            positions_above_pg = np.where(self.hist_rets[self.t-1][0:-1]>(1+self.output_container[self.t-1]*self.slpg_risk_aversion[1]))[0]
            positions_below_sl = np.where(self.hist_rets[self.t-1][0:-1]<(1+self.output_container[self.t-1]*self.slpg_risk_aversion[0]*-1))[0]

            print('Number of positions above profit goal: ',len(positions_above_pg))

            if len(positions_above_pg) > 0:
                print('Positionns exceeding the profit goal: ',[(self.ticker_dict[i],i) for i in positions_above_pg],'\n')

            print('Number of positions below stop loss: ',len(positions_below_sl))

            if len(positions_below_sl) > 0:
                print('Positionns exceeding the stop loss: ',[(self.ticker_dict[i],i) for i in positions_below_sl])

        if self.verbose >= 1:
            print(f'\nclose_long: {close_long}')
            print(f'close_short: {close_short} \n')

        self.open_long_positions = [long_i for long_i in self.open_long_positions if long_i not in close_long]
        self.open_short_positions = [short_i for short_i in self.open_short_positions if short_i not in close_short]

        if self.verbose >= 1:
            print('Closing trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        if self.strategy_scheme == 'max_pos':

            long_candidates = self.open_long_positions+list(new_long)
            short_candidates = self.open_short_positions+list(new_short)

            ## Checking if any of the new candidates should be included or any of the existing should be closed
            ## at the expense of a new.
            if (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) <= (self.max_positions / 2)):

                # Do nothing, sticking with the generated list, consisting of open positions and new buys!
                pass

                if self.verbose:
                    print("\n NOTE: The number of available positions are %i, compared to the intended %i.\n" % (len(long_candidates)+\
                                                                                                                len(short_candidates),
                                                                                                                 self.max_positions))
            ## If both the available long and short exceeds the needed:
            elif (len(long_candidates) > (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

                ### Prepping the candidates
                temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values
                temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                # Determining the ranking of the new candidates
                new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]
                new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                # Determining the ranking of the existing positions
                open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]
                open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                # if some new buy candidates have a ranking that is outside the desired long positions,
                # we diregard that new buy.
                if len(new_long_ranking) > 0:
                    if (max(new_long_ranking) > ((self.max_positions / 2) - 1)):
                        new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Same for sell candidates
                if len(new_short_ranking) > 0:
                    if (max(new_short_ranking) > ((self.max_positions / 2) - 1)):
                        new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                ## we close it.
                if len(open_long_ranking) > 0:
                    if (max(open_long_ranking) > ((self.max_positions / 2) - 1)):
                        self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                        close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

                ## Same for sell candidates
                if len(open_short_ranking) > 0:
                    if (max(open_short_ranking) > ((self.max_positions / 2) - 1)):
                        self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                        close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]

            ## If there are less long and excess short positions:
            elif (len(long_candidates) <= (self.max_positions / 2)) & (len(short_candidates) > (self.max_positions / 2)):

                ## With less long candidates than needed, there is no need to do anything at this point, as all existing
                ## positions should be kept, unless they do not have a buy signal anymore but then they are dropped above,
                ## and new ones should be added.
                ## For the sell candidates we first identify if we need to adjust. If we do not need to adjust, we just
                ## proceed as normally.

                if len(short_candidates) >= (self.max_positions - len(long_candidates)):

                    temp_short = ts_preds[ts_preds.index.isin(short_candidates)].sort_values('confidence',ascending=False).index.values

                    ## Determining the ranking of the new candidates
                    new_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,new_short)]

                    # Determining the ranking of the existing positions
                    open_short_ranking = np.arange(len(temp_short))[np.isin(temp_short,self.open_short_positions)]

                    # if some new sell candidates have a ranking that is outside the neeeded short positions,
                    # we diregard that new sell.
                    if len(new_short_ranking) > 0:
                        if (max(new_short_ranking) > ((self.max_positions / 2) - 1)):
                            new_short = [j for i,j in enumerate(new_short) if new_short_ranking[i] <= ((self.max_positions / 2) - 1)]

                    ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                    ## we close it.
#                     try:
                    if len(open_short_ranking) > 0:
                        if (max(open_short_ranking) > ((self.max_positions / 2) - 1)):
                            self.open_short_positions = [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] <= ((self.max_positions / 2) - 1)]
                            close_short += [j for i,j in enumerate(self.open_short_positions) if open_short_ranking[i] > ((self.max_positions / 2) - 1)]
#                     except:
#                         print('Step:\n\n'self.t,'\n')
#                         print('open short positions:\n\n',self.open_short_positions,'\n')

            ## If there are less short and excess long positions:
            elif (long_pos.shape[0] > (self.max_positions / 2)) & (short_pos.shape[0] <= (self.max_positions / 2)):

                ### Vice versa, compared to the if statement just above.

                ## With less sell candidates than needed, there is no need to do anything at this point, as all existing
                ## positions should be kept, unless they do not have a sell signal anymore but then they are dropped below,
                ## and new ones should be added.

                ## For the short candidates we first identify if we need to adjust. If we do not need to adjust, we just
                ## proceed as normally.
                if len(long_candidates) >= (self.max_positions - len(short_candidates)):

                    temp_long = ts_preds[ts_preds.index.isin(long_candidates)].sort_values('confidence',ascending=False).index.values

                    ## Determining the ranking of the new candidates
                    new_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,new_long)]

                    # Determining the ranking of the existing positions
                    open_long_ranking = np.arange(len(temp_long))[np.isin(temp_long,self.open_long_positions)]

                    # if some new sell candidates have a ranking that is outside the neeeded short positions,
                    # we diregard that new sell.
                    if len(new_long_ranking) > 0:
                        if (max(new_long_ranking) > ((self.max_positions / 2) - 1)):
                            new_long = [j for i,j in enumerate(new_long) if new_long_ranking[i] <= ((self.max_positions / 2) - 1)]

                    ## Sort of same check as above; if any existing positions have a raking above the intended number of positions
                    ## we close it.
                    if len(open_long_ranking) > 0:
                        if (max(open_long_ranking) > ((self.max_positions / 2) - 1)):
                            self.open_long_positions = [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] <= ((self.max_positions / 2) - 1)]
                            close_long += [j for i,j in enumerate(self.open_long_positions) if open_long_ranking[i] > ((self.max_positions / 2) - 1)]

            ## Else return eror
            else:

                raise ValueError('Something is wrong - please investigate!')

        if self.verbose >= 1:
            print(f'new_long: {new_long}')
            print(f'new_short: {new_short} \n')

        self.open_long_positions += [long_i for long_i in new_long]
        self.open_short_positions += [short_i for short_i in new_short]

        #self.ticker_dict

        if self.verbose >= 1:
            print('Opening new trades, status after:')
            print(f'all long: {self.open_long_positions}')
            print(f'all short: {self.open_short_positions} \n')

        ########## Calculating returns ##########

        if self.t > 0:

            if self.verbose >= 1: print('self.prev_close.direction new step:\n\n',self.prev_close.direction,'\n')

#             print('\nIndex 37 direction before return calculation: ',self.prev_close.iloc[37, 1],'\n')
            # update directions for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction'] = -1
#             print('\nIndex 37 direction, before return but after adjustment: ',self.prev_close.iloc[37, 1],'\n')
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction'] = 1

            current_ret = (close_info.close.values / self.prev_close.prev_close.values) - 1
            current_ret = (current_ret * self.prev_close.direction.values) + 1

#             print('Current ret, index 37: ',current_ret[37],'\n')

            ## Correcting those we close
            boolcousin = np.isin(self.all_tickers, close_short) | np.isin(self.all_tickers, close_long)

#             print('\nIndex 37: ',boolcousin[37],'\n')
            current_ret[boolcousin] = (close_info[boolcousin].close.values\
                                                                 *abs(self.prev_close[boolcousin].direction.values)\
                                                                 +(close_info[boolcousin].spread_close.values / 2)\
                                                                 *self.prev_close[boolcousin].direction.values)\
                                                                /self.prev_close[boolcousin].prev_close.values - 1

            ## We multiply with the opposite direction to get the correct return.
            current_ret[boolcousin] = (current_ret[boolcousin] * -self.prev_close[boolcousin].direction.values) + 1

#             print('portfolio return, accumulated, before:\n\n',current_ret.sum()*self.hist_rets[self.t - 1, -1],'\n')
#             print('Current ret with bool cousin, index 37: ',current_ret[37],'\n')
#             print('Prev close: \n\n',self.prev_close)
            ## Correcting the return of those we rebalance
            #self.prev_close.direction.values != 0&
#             rebal_mask = (self.prev_close.direction.values != 0)&(~(np.isin(self.all_tickers,new_long)|\
#                             np.isin(self.all_tickers,new_short)|np.isin(self.all_tickers,close_long)|
#                             np.isin(self.all_tickers,close_short)))

            # Preparing the generation of the new weights
            future_direction = self.prev_close.direction.copy(deep=True)

            if self.verbose >= 1: print('future_direction 1:\n\n',future_direction,'\n')

            # set directions == 0 for closed positions
            future_direction[np.isin(self.all_tickers, close_long)] = 0
            future_direction[np.isin(self.all_tickers, close_short)] = 0

            if self.verbose >= 1: print('future_direction 2:\n\n',future_direction,'\n')

            # update directions for new positions
            future_direction[np.isin(self.all_tickers, new_long)] = 1
            future_direction[np.isin(self.all_tickers, new_short)] = -1

            if self.verbose >= 1: print('future_direction 3:\n\n',future_direction,'\n')
            if self.verbose >= 1: print('future_direction 3 abs:\n\n',future_direction.abs(),'\n')
            if self.verbose >= 1: print('future_direction 3 abs-sum:\n\n',future_direction.abs().sum(),'\n')

            if self.weight_scheme == 'constant':

                # Set new weights
                new_weights = (1/future_direction.abs().sum())*future_direction.abs()

            elif self.weight_scheme == 'prob':
#                 print('Predictions: \n',ts_preds,'\n')
                new_weights = ((ts_preds.confidence)*future_direction.abs())/((ts_preds.confidence)*future_direction.abs()).sum()

            if self.verbose >= 1: print('New weights:\n\n',new_weights,'\n')
#             if self.verbose >= 1: print('noob')

            if self.verbose >= 1: print('previous weights:\n\n',self.hist_prev_weights[self.t - 1],'\n')

            portfolio_dev = current_ret * self.hist_prev_weights[self.t - 1,:]

            if self.verbose >= 1: print('portfolio development, before correction:\n\n',portfolio_dev,'\n')

            portfolio_share = portfolio_dev / sum(portfolio_dev)

            if self.verbose >= 1: print('portfolio share:\n\n',portfolio_share,'\n')

            correction = np.zeros(len(portfolio_share))

            if self.verbose >= 1: print('correction mask:\n\n',(new_weights != 0)&(self.hist_directions[self.t - 1]==future_direction),'\n')

            correction_mask = ((new_weights != 0)&(self.hist_directions[self.t - 1]==future_direction)).values

            correction[correction_mask] = new_weights[correction_mask] / portfolio_share[correction_mask] - 1

            if self.verbose >= 1: print('correction:\n\n',correction,'\n')

            share_to_keep = 1-abs(correction)
            share_to_correct = abs(correction)
            if self.verbose >= 1: print('share_to_keep: \n\n',share_to_keep,'\n')

#             print(share_to_keep.shape,current_ret.shape,correction_mask.shape)
#             print('rebal_mask:\n\n',rebal_mask,'\n')
#             print('correction_mask:\n\n',correction_mask,'\n')

            if self.verbose >= 1: print('return of share to keep:\n\n',share_to_keep[correction_mask] * current_ret[correction_mask],'\n')
            if self.verbose >= 1: print('return of share to correct:\n\n',share_to_correct[correction_mask] * ((close_info[correction_mask].close.values\
                                                             *abs(self.prev_close[correction_mask].direction.values)\
                                                             +(close_info[correction_mask].spread_close.values / 2)\
                                                             *-1*self.prev_close[correction_mask].direction.values)\
                                                            /self.prev_close[correction_mask].prev_close.values),'<n')
            current_ret[correction_mask] =  share_to_keep[correction_mask] * current_ret[correction_mask] +\
                                            share_to_correct[correction_mask] * ((close_info[correction_mask].close.values\
                                                             *abs(self.prev_close[correction_mask].direction.values)\
                                                             +(close_info[correction_mask].spread_close.values / 2)\
                                                             *-1*self.prev_close[correction_mask].direction.values)\
                                                            /self.prev_close[correction_mask].prev_close.values)#- 1

            if self.verbose >= 1: print('return after: \n\n',current_ret[correction_mask],'\n')

            portfolio_dev[correction_mask] = current_ret[correction_mask] * self.hist_prev_weights[self.t - 1,correction_mask]

            if self.verbose >= 1: print('portfolio development, after correction:\n\n',portfolio_dev,'\n')

            ## Fixing those not in use, which by the 'direction' equals a return of zero.
            current_ret[self.prev_close.direction.values == 0] = 1

            # update individual returns for open positions
            self.hist_rets[self.t, :-1] = current_ret * self.hist_rets[self.t - 1, :-1]

            # update total portfolio returns for open positions
#             self.hist_rets[self.t, -1] = (1+sum(((current_ret*self.prev_close.direction.abs().values) - 1)*self.hist_prev_weights[self.t - 1])) * self.hist_rets[self.t - 1, -1]

            self.hist_rets[self.t, -1] = sum(portfolio_dev)*self.hist_rets[self.t-1, -1]

            if self.verbose >= 1: print('portfolio return, accumulated, after:\n\n',self.hist_rets[self.t, -1],'\n')

            # set directions == 0 for closed positions
            self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction']  = 0
            self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

            # update directions for new positions
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            if self.slpg_warning:
                ## Roll back return matrix, to make room to the return coming in at the end of the period.
                self.returns_container[0:-1] = self.returns_container[1:]

                ## Include the new return.
                self.returns_container[-1] = (close_info.close.values/self.prev_close.prev_close)-1

                ## Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            ## Updating the last seen price
            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            ## Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            ## Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)
            # ************ check: can a ticker be in both closed positions and new positions? then directions change x2
            overlap = []
#             print(np.isin(new_long,close_long))
#             print(close_long)
            overlap += list(np.array(close_long)[np.isin(close_long, new_long)])
            if len(overlap) > 0:
                print('New longs in close long: ', overlap)

            overlap += list(np.array(new_long)[np.isin(new_long, close_long)])
            if len(overlap) > 0:
                print('Close longs in new long: ', overlap)

            overlap += list(np.array(close_short)[np.isin(close_short, new_short)])
            if len(overlap) > 0:
                print('New shorts in close short: ', overlap)

            overlap += list(np.array(new_short)[np.isin(new_short, close_short)])
            if len(overlap) > 0:
                print('Close shorts in new short: ', overlap)

            if len(overlap) > 0:
                print('######### Overlap between the new candidates and the close candidates #########')

#             # set directions == 0 for closed positions
#             self.prev_close.loc[np.isin(self.all_tickers, close_long), 'direction']  = 0
#             self.prev_close.loc[np.isin(self.all_tickers, close_short), 'direction']  = 0

            ## Setting the weights for the open positions in the next period.
            if self.weight_scheme == 'constant':

                # Set new weights
                self.hist_prev_weights[self.t,:] = (1/self.prev_close.direction.abs().sum())*self.prev_close.direction.abs()

            elif self.weight_scheme == 'prob':

                self.hist_prev_weights[self.t,:] = ((ts_preds.confidence)*self.prev_close.direction.abs())/((ts_preds.confidence)*self.prev_close.direction.abs()).sum()

            # Save the directions
            self.hist_directions[self.t,:] = self.prev_close.direction.copy(deep=True)

        else:

            # update directions (first entry)
            self.prev_close.loc[np.isin(self.all_tickers, new_long), 'direction']  = 1
            self.prev_close.loc[np.isin(self.all_tickers, new_short), 'direction']  = -1

            ## Initializing the weights
            if self.weight_scheme == 'constant':
                self.hist_prev_weights[self.t,:] = (1/self.prev_close.direction.abs().sum())*self.prev_close.direction.abs()

            elif self.weight_scheme == 'prob':

                self.hist_prev_weights[self.t,:] = ((ts_preds.confidence)*self.prev_close.direction.abs())/((ts_preds.confidence)*self.prev_close.direction.abs()).sum()

            if self.slpg_warning:

                ## Place to update returns in the returns container
                # Roll back the returns one period
                self.returns_container[0:-1] = self.returns_container[1:]

                # Include the new return
                self.returns_container[-1] = (close_info.close.values/self.slpg_last_known_price)-1

                # Updating the standard deviation of the returns
                self.output_container[self.t] = np.std(self.returns_container,axis=0)

            self.prev_close.loc[:,'prev_close'] = close_info.close.values

            # Those we buy
            self.prev_close.loc[np.isin(self.all_tickers, new_long),'prev_close'] = close_info[np.isin(self.all_tickers, new_long)].close.values +\
                                                                        (close_info[np.isin(self.all_tickers, new_long)].spread_close.values / 2)
            # Those we sell
            self.prev_close.loc[np.isin(self.all_tickers, new_short),'prev_close'] = close_info[np.isin(self.all_tickers, new_short)].close.values -\
                                                                        (close_info[np.isin(self.all_tickers, new_short)].spread_close.values / 2)

            # Save the directions
            self.hist_directions[self.t,:] = self.prev_close.direction.copy(deep=True)
