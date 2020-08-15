from os.path import isfile
import yfinance as yf
import pandas as pd
import numpy as np
import re
import math
import WhiteRealityCheckFor1
import matplotlib.pyplot as plt
from matplotlib import style
import itertools

from computation_helper import *
from data_helper import *
import rotational_momentum as rm


class Portfolio:

    def __init__(self, tickers_str):
        self.tickers_str = tickers_str
        self.tickers_list = tickers_str.split()
        self.tickers_densed_str = ''.join(tickers_str.split())
        self.data_dict = {}

    #TODO: move this to data_helper.py
    def download_data(self, datafile_alias, start_date, end_date):

        #declare price history location
        base_name = datafile_alias + "_" + start_date + "_" + end_date
        file_dir = "Data/" + base_name + ".csv"
        ap_file_dir = "Data/" + base_name + "_AP.csv"
        self.data_dict[datafile_alias] = [file_dir, ap_file_dir]

        #download and save price history if not found locally
        if not (isfile(file_dir)):
            price_df = yf.download(self.tickers_str,
                                   start= start_date,
                                   end= end_date)
            price_df["Adj Close"].to_csv(ap_file_dir, header=True)
            price_df["Close"].to_csv(file_dir, header=True)

        else:
            print("requested data history already exists!")

        return


    def _load_data(self, datafile_alias):

        file_dir = self.data_dict[datafile_alias][0]
        ap_file_dir = self.data_dict[datafile_alias][1]

        dfP = pd.read_csv(file_dir, parse_dates=['Date'])
        dfAP = pd.read_csv(ap_file_dir, parse_dates=['Date'])
        dfP = dfP.sort_values(by='Date')
        dfAP = dfAP.sort_values(by='Date')
        dfP.set_index('Date', inplace = True)
        dfAP.set_index('Date', inplace = True)
        dfP['CASH'] = 1
        dfAP['CASH'] = 1
        return dfP, dfAP


    #TODO: parse frequency_str to return freq, shift and number of
    #sub backtesting
    def _parse_frequency(self, frequency_str):

        size = int(re.search("\d+%", frequency_str).group().strip("%"))
        shift_num = int(int(re.match("\d+", frequency_str).group()) * size / 100)
        shift_unit = re.search("[A-Z]-[A-Z]*", frequency_str).group()
        freq = re.match("\d+[A-Z]-[A-Z]*", frequency_str).group()
        return freq, shift_num, shift_unit, int(100/size)


    def _evaluate(self, df_list):

        #evaluate each sub portfolio
        for df in df_list:
            dfPRR, TotaAnnReturn, CAGR, sharpe, volatility, p_val = evaluate(df)

        #join all result df and evaluate
        n = len(df_list)
        if n > 1:
            dfPRR = df_list[0]
            #aggregate ALL_R and DETREND_ALL_R
            for i in range(1,n):
                dfPRR.ALL_R += df_list[i].ALL_R
                dfPRR.DETREND_ALL_R += df_list[i].DETREND_ALL_R
            dfPRR.ALL_R = dfPRR.ALL_R / n
            dfPRR.DETREND_ALL_R = dfPRR.DETREND_ALL_R / n

            #update I
            dfPRR = dfPRR.assign(I =np.cumprod(1+dfPRR['ALL_R']))
            dfPRR.iat[0,dfPRR.columns.get_loc('I')]= 1

            #update DETREND_I
            dfPRR = dfPRR.assign(DETREND_I =np.cumprod(1+dfPRR['DETREND_ALL_R']))
            dfPRR.iat[0,dfPRR.columns.get_loc('DETREND_I')]= 1
            print("\n==================Overall==================")
            dfPRR, TotaAnnReturn, CAGR, sharpe, volatility, p_val = evaluate(dfPRR)


        return dfPRR, TotaAnnReturn, CAGR, sharpe, volatility

    #entry point of the backtest
    def backtest(self, datafile_alias, lookback, Frequency, ShortTermWeight,
                 LongTermWeight, RSI_weight, v_ratio_weight, Z_weight):

        dfP, dfAP = self._load_data(datafile_alias)

        #TODO: call _parse_frequency
        freq, shift_num, shift_unit, n = self._parse_frequency(Frequency)

        #TODO: loop call the strategy execution function
        #to generate result files
        df_list = []
        for i in range(n):
            #use momentum trading here
            each_shift = str(shift_num * i) + shift_unit
            df = rm.rotational_momentum(dfP, dfAP, lookback, freq, each_shift,
                                        ShortTermWeight,
                                        LongTermWeight, RSI_weight,
                                        v_ratio_weight, Z_weight)

            file_dir = "Results/" + datafile_alias + "_" + str(i+1) + ".csv"
            df.to_csv(file_dir, header = True, index=True)
            df_list.append(df)

        #TODO: call _evaluate
        df, TotaAnnReturn, CAGR, sharpe, volatility = self._evaluate(df_list)

        return df, TotaAnnReturn, CAGR, sharpe, volatility

    #TODO: call backtest to collect performance of each params combo
    def grid_search(self, datafile_alias, lookback, Frequency, ShortTermWeight,
                 LongTermWeight, RSI_weight, v_ratio_weight, Z_weight):

        #create cartesian product of params list
# =============================================================================
#         itertools example of param_1 [1,2] and param_2 [3,4]
#         grid = list(itertools.product([1, 2],[3, 4]))
#         grid should contain (1,3), (1,4), (2,3), (2,4)
# =============================================================================
        results = []
        grid = list(itertools.product(lookback, Frequency,
                                      ShortTermWeight, LongTermWeight,
                                      RSI_weight, v_ratio_weight, Z_weight))

        for combo in grid:
            print(combo)
            #call self.backtest
            df, TotalAnnReturn, CAGR, sharpe, volatility\
            = self.backtest(datafile_alias, combo[0], combo[1], combo[2], combo[3],
                       combo[4], combo[5], combo[6])
            output = [combo[0], combo[1], combo[2], combo[3],combo[4],
                      combo[5], combo[6], TotalAnnReturn, CAGR,
                      sharpe, volatility]
            results.append(output)

        result_df = pd.DataFrame.from_records(data=results,
                                              columns=["lookback",
                                                       "frequency",
                                                       "ShortTermWeight",
                                                       "LongTermWeight",
                                                       "RSI_weight",
                                                       "v_ratio_weight",
                                                       "Z_weight",
                                                       "TotalAnnReturn",
                                                       "CAGR",
                                                       "sharpe",
                                                       "volatility"])

        file_dir = "Results/" + datafile_alias + "_gs" + ".csv"
        result_df.to_csv(file_dir, header = True, index=True)
        return

# =============================================================================
# #example
# portfolio = Portfolio("SPY AAPL")
# portfolio.download_data("train", "2004-01-01", "2015-12-31")
# portfolio.backtest("train")
# =============================================================================
portf = Portfolio("AAPL, WMT, INTC, NFLX")
#print(portf._parse_frequency("4W-FRI-25%"))
portf.download_data("experiment6", "2004-01-01", "2008-12-31")
#portf.backtest("experiment6", 20, "4W-FRI-25%", 1, 2, 1, 0, 0)
portf.grid_search("experiment6",
                  [10, 20],
                  ["1W-FRI-100%", "2W-FRI-100%", "3W-FRI-100%", "4W-FRI-25%"],
                  [1],
                  [2],
                  [1],
                  [0],
                  [0])