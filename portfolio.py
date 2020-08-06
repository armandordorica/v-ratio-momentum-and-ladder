from os.path import isfile
import yfinance as yf
import pandas as pd

class Portfolio:
    def __init__(self, tickers_str):

        self.tickers_str = tickers_str
        self.tickers_list = tickers_str.split()
        self.tickers_densed_str = ''.join(tickers_str.split())
        self.data = None

    def download_data(self, start_date, end_date):

        #declare price history location
        file_name = self.tickers_densed_str + "_" + start_date + "_" + end_date
        ap_file_name = file_name + "_AP"

        file_dir = "data/" + file_name
        ap_file_dir = "data/" + ap_file_name

        #download and save price history if not found locally
        if not (isfile(file_dir+".csv")):

            price_df = yf.download(self.tickers_str,
                                   start= start_date,
                                   end= end_date)

            price_df["Adj Close"].to_csv(ap_file_dir+".csv", header=True)
            price_df["Close"].to_csv(file_dir+".csv", header=True)

        else:
            print("data already exist!")

        return

#example
portfolio = Portfolio("SPY AAPL")
portfolio.download_data("2017-01-01", "2017-04-30")