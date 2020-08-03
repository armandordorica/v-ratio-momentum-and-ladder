# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import datetime

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

#Example1
# download dataframe
#data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
# download Panel
#data2 = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
#example2
#start = datetime.datetime(2017, 1, 1)
#symbol = 'SPY'
#data = pdr.get_data_yahoo(symbol, start=start, end=end)
#data.to_csv("C:\\Users\\Rosario\\Documents\\NeuralNetworksMachineLearning\\LSTMReturnPrediction\\data\\YahooSPY.csv")

start_date=datetime.datetime(2015, 1, 1)
#end_date=datetime.datetime(2020, 1, 1)
end_date= datetime.datetime.now()

#Dow Jones Index + TLT:
#StockList = ['AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','UTX','UNH', 'VZ','V','WMT','WBA', 'TLT']
#Dow Jones Index:
#StockList = ['AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','UTX','UNH', 'VZ','V','WMT','WBA']

#21 industrial sectors+ TLT:
#StockList = ["FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ","TLT"]
#21 industrial sectors:
#StockList = ["FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ"]

#10 sectors + TLT:
#stock_list = ["XLB","XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "TLT"] 
#10 sectors:
#StockList = ["XLB","XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XTR"] 

#Bond etfs:
#stock_list = ["BIL","TIP","IEI","IEF","TLH","TLT","SHY"] 

#our ETF list + SHY:
#stock_list = ["FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ","SHY"]



stock_list = ["DIA", "TLT"]  #********************************************************this is good
#stock_list = ["TLT", "SPY"]  #********************************************************this is good


stock_str = ""
for i in range(len(stock_list)):
    stock_str  = stock_str + stock_list[i] + "."

main_df = pd.DataFrame()

for stock in range(len(stock_list)):
     df = pdr.get_data_yahoo(stock_list[stock], start=start_date, end=end_date)
     df.drop(['Close','High', 'Low' , 'Open', 'Volume'], axis=1, inplace=True)
     df.rename(columns={'Adj Close': stock_list[stock]}, inplace=True)
     if main_df.empty:
         main_df = df
     else:
        main_df = main_df.join(df) 
    

#main_df.to_csv(r"C:\Users\Rosario\Documents\PortfolioOptimizationOnline\Markowitz_2\HarrysProblem.SPY.TLT.YFINANCE\\"+stock_str+"AP.csv")
main_df.to_csv(stock_str+"AP.csv")

main_df = pd.DataFrame()

for stock in range(len(stock_list)):
     df = pdr.get_data_yahoo(stock_list[stock], start=start_date, end=end_date)
     df.drop(['Adj Close','High', 'Low' , 'Open', 'Volume'], axis=1, inplace=True)
     df.rename(columns={'Close': stock_list[stock]}, inplace=True)
     if main_df.empty:
         main_df = df
     else:
        main_df = main_df.join(df) 

#main_df.to_csv(r"C:\Users\Rosario\Documents\PortfolioOptimizationOnline\Markowitz_2\HarrysProblem.SPY.TLT.YFINANCE\\"+stock_str+"csv")
main_df.to_csv(stock_str+"csv")

