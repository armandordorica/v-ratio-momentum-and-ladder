from os.path import isfile
import yfinance as yf
import pandas as pd
import regex as re

class Portfolio:

    def __init__(self, tickers_str):

        self.tickers_str = tickers_str
        self.tickers_list = tickers_str.split()
        self.tickers_densed_str = ''.join(tickers_str.split())
        self.data = None

    def _download_data(self, start_date, end_date):

        #declare price history location
        base_name = self.tickers_densed_str + "_" + start_date + "_" + end_date

        file_dir = "data/" + base_name + ".csv"

        ap_file_dir = "data/" + base_name + "_AP.csv"

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

    #TODO: parse frequency_str to return shift and holding period
    def _parse_frequency(self, frequency_str):


    #TODO: reads result files and evaluate backtest performance
    def _evaluate(self):

        return

    #entry point of the backtest
    def backtest(self, strategy, window, holding, rebalance_ratio=1):

        #TODO: call _parse_frequency

        #TODO: loop call the strategy execution function
        #to generate result files

        #TODO: call _evaluate

# =============================================================================
# #example
# portfolio = Portfolio("SPY AAPL")
# portfolio.download_data("2017-01-01", "2017-04-30")
# =============================================================================
