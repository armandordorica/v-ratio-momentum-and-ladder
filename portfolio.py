from os.path import isfile
import yfinance as yf
import pandas as pd
import re
from RotationalMomentumWFreqFunc import rotational_momentum
import math
import WhiteRealityCheckFor1
from helper import getDate
import matplotlib.pyplot as plt
from matplotlib import style


class Portfolio:

    def __init__(self, tickers_str, datafile_alias):
        self.alias = datafile_alias
        self.tickers_str = tickers_str
        self.tickers_list = tickers_str.split()
        self.tickers_densed_str = ''.join(tickers_str.split())
        self.data = None

    #TODO: move this to data_helper.py
    def _download_data(self, start_date, end_date):

        #declare price history location
        base_name = self.alias + "_" + start_date + "_" + end_date
        file_dir = "Data/" + base_name + ".csv"
        ap_file_dir = "Data/" + base_name + "_AP.csv"

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

    #TODO: parse frequency_str to return freq, shift and number of
    #sub backtesting
    def _parse_frequency(self, frequency_str):

        size = int(re.search("\d+%", frequency_str).group().strip("%"))
        shift = int(int(re.match("\d+", frequency_str).group()) * size / 100)
        unit = re.search("[A-Z]-[A-Z]", frequency_str).group()[0:-2]
        freq = re.match("\d+[A-Z]-[A-Z]*", frequency_str).group()
        return freq, str(shift)+unit, int(100/size)

    #TODO: read result files and evaluate backtest performance
    def _evaluate(self, df_list):

        #TODO: move this function to computation_helper.py
        def evaluate(dfPRR):

            try:
                sharpe = ((dfPRR['ALL_R'].mean() / dfPRR['ALL_R'].std()) * math.sqrt(252))
            except ZeroDivisionError:
                sharpe = 0.0
            if verbose:
                style.use('fivethirtyeight')
                dfPRR['I'].plot()
                plt.legend()
                plt.show()
                #plt.savefig(r'Results\%s.png' %(title))
                #plt.close()

            start = 1
            start_val = start
            end_val = dfPRR['I'].iat[-1]


            start_date = getDate(dfPRR.iloc[0].name)
            end_date = getDate(dfPRR.iloc[-1].name)
            days = (end_date - start_date).days


            TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
            TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
            volatility = dfPRR['ALL_R'].std() * math.sqrt(252)
            volatility_dia = dfPRR['DIA'].std() * math.sqrt(252)
            volatility_tlt = dfPRR['TLT'].std() * math.sqrt(252)

            CAGR_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
            CAGR = round(((float(end_val) / float(start_val)) ** (1/(days/350.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
            if verbose:
                print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
                print ("CAGR = %f" %(CAGR*100))
                print ("Sharpe Ratio = %f" %(round(sharpe,3)))
                print("Volatility= %f" %(round(volatility,3)))
                print("Volatility DIA= %f" %(round(volatility_dia,3)))
                print("Volatility TLT= %f" %(round(volatility_tlt,3)))

                #Detrending Prices and Returns
                p_val = WhiteRealityCheckFor1.bootstrap(dfPRR['DETREND_ALL_R'])
                dfPRR.to_csv(r'Results\dfPRR.csv', header = True,
                             index=True, encoding='utf-8')
                print(dfPRR)
            return dfPRR, TotaAnnReturn, CAGR, sharpe, volatility, p_val

        #evaluate each sub portfolio
        for df in df_list:
            dfPRR, TotaAnnReturn, CAGR, sharpe, volatility, p_val = evaluate(df)

        #join all result files and evaluate
        if len(df_list) > 1:

            #join date, asset price,_NUL, return, all return, etc
            df = pd.DataFrame()

            dfPRR, TotaAnnReturn, CAGR, sharpe, volatility, p_val = evaluate(df)
        return dfPRR, TotaAnnReturn, CAGR, sharpe, volatility

    #entry point of the backtest
    def backtest(self, strategy, frequency, holding, rebalance_ratio=1):

        #TODO: call _parse_frequency
        freq, shift, n = self._parse_frequency(frequency)

        #TODO: loop call the strategy execution function
        #to generate result files
        df_list = []
        for i in range(n):
            #use momentum trading here
            df = rotational_momentum(lookback,shtrm_weight,RSI_weight,
                                     v_ratio_weight, include_cash, verbose)
            file_dir = "Results/" + self.alias + "_" + str(i+1) + ".csv"
            df.to_csv(file_dir, header = True, index=True)
            df_list.append(df)

        #TODO: call _evaluate
        df, TotaAnnReturn, CAGR, sharpe, volatility = self._evaluate(df_list)

        return

    #TODO: call backtest to collect performance of each params combo
    def grid_search(self):

        #create cartesian product of params list

        #for each params combo,

            #call self.backtest

            #collect metrics

        #print and save metrics

        return

# =============================================================================
# #example
# portfolio = Portfolio("SPY AAPL")
# portfolio.download_data("2017-01-01", "2017-04-30")
# =============================================================================
portf = Portfolio("", datafile_alias="train")
print(portf._parse_frequency("4W-FRI-25%"))