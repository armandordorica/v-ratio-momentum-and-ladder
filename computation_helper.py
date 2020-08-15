# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:34:19 2020

@author: hyuan
"""
import math
import matplotlib as plt
from matplotlib import style
import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import detrendPrice
import WhiteRealityCheckFor1
import yfinance as yf
from datetime import *
import datetime
from numpy import *
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import scipy.io as sio
import pandas as pd
from data_helper import *

def compute_rsi(dfP, n=14):

    dfRSI = dfP.drop(labels=None, axis=1, columns=dfP.columns)

    columns = dfP.shape[1]
    for column in range(columns):

        delta = dfP[dfP.columns[column]].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = dUp.rolling(n).mean()
        RolDown = dDown.rolling(n).mean().abs()

        RS = RolUp / RolDown

        dfRSI[dfP.columns[column]] = 100/(100.0 - (100.0 / (1.0 + RS)))
        dfRSI[dfP.columns[column]].fillna(value=0.0, inplace=True)

    return dfRSI

def compute_zscore(dfP, n=20):
    dfZ = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    #Caluclate volatility, z-score and moving average
    columns = dfP.shape[1]
    for column in range(columns):
        dfZ[dfP.columns[column]] = (dfP[dfP.columns[column]]-dfP[dfP.columns[column]].rolling(window=n).mean())/dfP[dfP.columns[column]].rolling(window=n).std()
        dfZ[dfP.columns[column]].fillna(value=float('inf'), inplace=True)

    #force z_score for Cash to be infinity
    dfZ['CASH'] = 0
    return dfZ

def rank_assets(df, order):
    #Ranking each ETF w.r.t. short moving average of returns
    df_ranks = df.copy(deep=True)
    df_ranks[:] = 0

    columns = df_ranks.shape[1]
    rows = df_ranks.shape[0]

    #this loop takes each row of the A dataframe, puts the row into an array,
    #within the array the contents are ranked,
    #then the ranks are placed into the A_ranks dataframe one by one
    for row in range(rows):
        arr_row = df.iloc[row].values
        if order == 1:
            temp = arr_row.argsort() #sort momentum, best is ETF with largest return
        else:
            temp = (-arr_row).argsort()[:arr_row.size] #sort reversion, best is ETF with lowest return
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(1,len(arr_row)+1)
        for column in range(columns):
            df_ranks.iat[row, column] = ranks[column]

    return df_ranks

def choose_asset(df_ranks, Filter, dfZ = None, Zboundary=2.0):
    dfChoice = df_ranks.copy(deep=True)
    dfChoice[:] = 0
    rows = dfChoice.shape[0]

    for row in range(rows):
        arr_row = df_ranks.iloc[row].values
        max_arr_column = np.argmax(arr_row, axis=0) #gets the INDEX of the max
        if Filter == 'z_score':
            try:
                if (dfZ[dfZ.columns[max_arr_column]][row] < Zboundary): #alternative cash filter
                    dfChoice.iat[row, max_arr_column] = 1
                else:
                    dfChoice.iat[row, df_ranks.columns.get_loc('CASH')] = 1
            except:
                print('Specify Z-score dataframe')
        else:
            dfChoice.iat[row, max_arr_column] = 1

    return dfChoice



def calculate_results(dfP, dfAP, dfChoice, Frequency, shift):
    dfDetrend = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    columns = dfP.shape[1]
    for column in range(columns):
        dfDetrend[dfAP.columns[column]] = detrendPrice(dfAP[dfAP.columns[column]]).values

    rows = dfChoice.shape[0]

    #dfPRR is the dataframe containing the log or pct_change returns of the ETFs
    #will be based on adjusted prices rather than straight prices
    dfPRR= dfAP.pct_change()
    dfDetrendRR = dfDetrend.pct_change()

    #T is the dataframe where the trading day is calculated.
    dfT = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    columns = dfP.shape[1]
    for column in range(columns):
        new = dfP.columns[column] + "_CHOICE"
        dfPRR[new] = pd.Series(np.zeros(rows), index=dfPRR.index)
        dfPRR[new] = dfChoice[dfChoice.columns[column]]

    dfT['DateCopy'] = dfT.index
    dfT1 = dfT.asfreq(freq=Frequency, method='pad')
    dfT1.set_index('DateCopy', inplace=True)
    dfTJoin = pd.merge(dfT,
                     dfT1,
                     left_index = True,
                     right_index = True,
                     how='outer',
                     indicator=True)

    dfTJoin = dfTJoin.loc[~dfTJoin.index.duplicated(keep='first')] #eliminates a row with a duplicate index which arises when using kibot data
    dfPRR=pd.merge(dfPRR,dfTJoin, left_index=True, right_index=True, how="inner")
    dfPRR.rename(columns={"_merge": Frequency+"_FREQ"}, inplace=True)
    #dfPRR[Frequency+"_FREQ"] = dfTJoin["_merge"] #better to do this with merge as above


    #shifted ladder
    if int(shift[:-5]) != 0:
        #print(dfPRR.axes)
        dfPRR.drop("DateCopy", axis=1, inplace=True)
        shifted = dfPRR[Frequency+"_FREQ"].shift(freq=shift)
        shifted = shifted[shifted=="both"].to_frame()
        dfPRR[Frequency+"_FREQ"] = pd.Series(np.zeros(rows), index=dfPRR.index)
        for row in shifted.itertuples():
            ind = str(row.Index)
            if ind in dfPRR.index:
                dfPRR.at[ind, Frequency+"_FREQ"] = "both"

    #_LEN means Long entry for that ETF
    #_NUL means number units long of that ETF
    #_LEX means long exit for that ETF
    #_R means returns of that ETF (traded ETF)
    #_ALL_R means returns of all ETFs traded, i.e. portfolio returns
    #CUM_R means commulative returns of all ETFs, i.e. portfolio cummulative returns

    columns = dfP.shape[1]
    for column in range(columns):
        new = dfP.columns[column] + "_LEN"
        dfPRR[new] = ((dfPRR[Frequency+"_FREQ"] =="both") & (dfPRR[dfP.columns[column]+"_CHOICE"] ==1))
        new = dfP.columns[column] + "_LEX"
        dfPRR[new] = ((dfPRR[Frequency+"_FREQ"] =="both") & (dfPRR[dfP.columns[column]+"_CHOICE"] !=1))
        new = dfP.columns[column] + "_NUL"
        dfPRR[new] = np.nan
        dfPRR.loc[dfPRR[dfP.columns[column]+'_LEX'] == True, dfP.columns[column]+'_NUL' ] = 0
        dfPRR.loc[dfPRR[dfP.columns[column]+'_LEN'] == True, dfP.columns[column]+'_NUL' ] = 1 #this order is important
        dfPRR.iat[0,dfPRR.columns.get_loc(dfP.columns[column] + "_NUL")] = 0
        dfPRR[dfP.columns[column] + "_NUL"] = dfPRR[dfP.columns[column] + "_NUL"].fillna(method='pad')
        new = dfP.columns[column] + "_R"
        dfPRR[new] = dfPRR[dfP.columns[column]]*dfPRR[dfP.columns[column]+'_NUL']#.shift(Delay)
        #repeat for detrended returns
        dfDetrendRR[new] = dfDetrendRR[dfP.columns[column]]*dfPRR[dfP.columns[column]+'_NUL']#.shift(Delay)

    #calculating all returns
    dfPRR = dfPRR.assign(ALL_R = pd.Series(np.zeros(rows)).values)
    #repeat for detrended returns
    dfDetrendRR = dfDetrendRR.assign(ALL_R = pd.Series(np.zeros(rows)).values)


    #the return of the portfolio is a sequence of returns made
    #by appending sequences of returns of traded ETFs
    #Since non traded returns are multiplied by zero, we only need to add the columns
    #of the returns of each ETF, traded or not
    columns = dfP.shape[1]
    for column in range(columns):
        dfPRR["ALL_R"] = dfPRR["ALL_R"] + dfPRR[dfP.columns[column]+"_R"]
        #repeat for detrended returns
        dfDetrendRR["ALL_R"] = dfDetrendRR["ALL_R"] + dfDetrendRR[dfP.columns[column]+"_R"]

    dfPRR = dfPRR.assign(DETREND_ALL_R = dfDetrendRR['ALL_R'])

    #dfPRR['CUM_R'] = dfPRR['ALL_R'].cumsum()  #this is good only for log returns
    #dfPRR['CUM_R'] = dfPRR['CUM_R'] + 1 #this is good only for log returns

    #calculating portfolio investment column in a separate dataframe, using 'ALL_R' = portfolio returns

    dfPRR = dfPRR.assign(I =np.cumprod(1+dfPRR['ALL_R'])) #this is good for pct return
    dfPRR.iat[0,dfPRR.columns.get_loc('I')]= 1
    #repeat for detrended returns
    dfDetrendRR = dfDetrendRR.assign(I =np.cumprod(1+dfDetrendRR['ALL_R'])) #this is good for pct return
    dfDetrendRR.iat[0,dfDetrendRR.columns.get_loc('I')]= 1

    dfPRR = dfPRR.assign(DETREND_I = dfDetrendRR['I'])

    return dfPRR


def evaluate(dfPRR):

            try:
                sharpe = ((dfPRR['ALL_R'].mean() / dfPRR['ALL_R'].std()) * math.sqrt(252))
            except ZeroDivisionError:
                sharpe = 0.0

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

            CAGR_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
            CAGR = round(((float(end_val) / float(start_val)) ** (1/(days/350.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
            print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
            print ("CAGR = %f" %(CAGR*100))
            print ("Sharpe Ratio = %f" %(round(sharpe,3)))
            print("Volatility= %f" %(round(volatility,3)))

            #Detrending Prices and Returns
            p_val = WhiteRealityCheckFor1.bootstrap(dfPRR['DETREND_ALL_R'])


            return dfPRR, TotaAnnReturn, CAGR, sharpe, volatility, p_val

def normcdf(X):
    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2*pi)*exp(-L*L/2.) * (a1*K + a2*K*K + a3*pow(K,3) + a4*pow(K,4) + a5*pow(K,5))
    if X < 0:
        w = 1.0-w
    return w

def compute_vratio(a, lag = 2, cor = 'hom'):
    """ the implementation found in the blog Leinenbock
    http://www.leinenbock.com/variance-ratio-test/
    """
    #t = (std((a[lag:]) - (a[1:-lag+1])))**2;
    #b = (std((a[2:]) - (a[1:-1]) ))**2;

    n = len(a)
    mu  = sum(a[1:n]-a[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    #print( mu, m, lag)
    b=sum(square(a[1:n]-a[:n-1]-mu))/(n-1)
    t=sum(square(a[lag:n]-a[:n-lag]-lag*mu))/m
    vratio = t/(lag*b);

    la = float(lag)

    if cor == 'hom':
        varvrt=2*(2*la-1)*(la-1)/(3*la*n)

    elif cor == 'het':
        varvrt=0;
        sum2=sum(square(a[1:n]-a[:n-1]-mu));
        for j in range(lag-1):
            sum1a=square(a[j+1:n]-a[j:n-1]-mu);
            sum1b=square(a[1:n-j]-a[0:n-j-1]-mu)
            sum1=dot(sum1a,sum1b);
            delta=sum1/(sum2**2);
            varvrt=varvrt+((2*(la-j)/la)**2)*delta

    zscore = (vratio - 1) / sqrt(float(varvrt))
    pval = normcdf(zscore);
    if (math.isnan(vratio)) and (math.isnan(zscore)):
        vratio = 0
        zscore = 0
    return vratio, zscore, pval

def hurst2(ts):
    """ the implementation found in the blog Leinenbock
    http://www.leinenbock.com/calculation-of-the-hurst-exponent-to-test-for-trend-and-mean-reversion/
    """
    tau = []; lagvec = []
    #  Step through the different lags
    for lag in range(2,100):
        #  produce price difference with lag
        pp = subtract(ts[lag:],ts[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the differnce vector
        tau.append(sqrt(std(pp)))

    #  linear fit to double-log graph (gives power)
    m = polyfit(log10(lagvec),log10(tau),1)
    # calculate hurst
    hurst = m[0]*2.0
    # plot lag vs variance
    #plt.plot(lagvec,tau,'o')
    #plt.show()

    return hurst

def hurst(ts):
    """ the implewmentation on the blog http://www.quantstart.com
    http://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
    Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def half_life(ts):
    """ this function calculate the half life of mean reversion
    """
    # calculate the delta for each observation.
    # delta = p(t) - p(t-1)
    delta_ts = diff(ts)
        # calculate the vector of lagged prices. lag = 1
    # stack up a vector of ones and transpose
    lag_ts = vstack([ts[1:], ones(len(ts[1:]))]).T

    # calculate the slope (beta) of the deltas vs the lagged values
    beta = linalg.lstsq(lag_ts, delta_ts)

    # compute half life
    half_life = log(2) / beta[0]

    return half_life[0]

def random_walk(seed=1000, mu = 0.0, sigma = 1, length=1000):
    """ this function creates a series of independent, identically distributed values
    with the form of a random walk. Where the best prediction of the next value is the present
    value plus some random variable with mean and variance finite
    We distinguish two types of random walks: (1) random walk without drift (i.e., no constant
    or intercept term) and (2) random walk with drift (i.e., a constant term is present).
    The random walk model is an example of what is known in the literature as a unit root process.
    RWM without drift: Yt = Yt−1 + ut
    RWM with drift: Yt = δ + Yt−1 + ut
    """

    ts = []
    for i in range(length):
        if i == 0:
            ts.append(seed)
        else:
            ts.append(mu + ts[i-1] + random.gauss(0, sigma))

    return ts

def subset_dataframe(data, start_date, end_date):
    start = data.index.searchsorted(start_date)
    end = data.index.searchsorted(end_date)
    return data.ix[start:end]

def cointegration_test(y, x):
    ols_result = sm.OLS(y, x).fit()
    return ts.adfuller(ols_result.resid, maxlag=1)


# =============================================================================
# def get_data_from_matlab(file_url, index, columns, data):
#     """Description:*
#     This function takes a Matlab file .mat and extract some
#     information to a pandas data frame. The structure of the mat
#     file must be known, as the loadmat function used returns a
#     dictionary of arrays and they must be called by the key name
#
#     Args:
#         file_url: the ubication of the .mat file
#         index: the key for the array of string date-like to be used as index
#         for the dataframe
#         columns: the key for the array of data to be used as columns in
#         the dataframe
#         data: the key for the array to be used as data in the dataframe
#     Returns:
#         Pandas dataframe
#
#     """
#
#     import scipy.io as sio
#     import datetime as dt
#     # load mat file to dictionary
#     mat = sio.loadmat(file_url)
#     # define data to import, columns names and index
#     cl = mat[data]
#     stocks = mat[columns]
#     dates = mat[index]
#
#     # extract the ticket to be used as columns name in dataframe
#     # to-do: list compression here
#     columns = []
#     for each_item in stocks:
#         for inside_item in each_item:
#             for ticket in inside_item:
#                 columns.append(ticket)
#     # extract string ins date array and convert to datetimeindex
#     # to-do list compression here
#     df_dates =[]
#     for each_item in dates:
#         for inside_item in each_item:
#             df_dates.append(inside_item)
#     df_dates = pd.Series([pd.to_datetime(date, format= '%Y%m%d') for date in df_dates], name='date')
#
#     # construct the final dataframe
#     data = pd.DataFrame(cl, columns=columns, index=df_dates)
#
#     return data
# =============================================================================


# =============================================================================
# def my_path(loc):
#     if loc == 'PC':
#         root_path = 'C:/Users/javgar119/Documents/Python/Data/'
#     elif loc == 'MAC':
#         root_path = '/Users/Javi/Documents/MarketData/'
#     return root_path
# =============================================================================
