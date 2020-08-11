import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import detrendPrice
import WhiteRealityCheckFor1
import yfinance as yf
yf.pdr_override()

def getDate(dt):
    if type(dt) != str:
        return dt
    try:
        datetime_object = datetime.datetime.strptime(dt, '%Y-%m-%d')
    except Exception:
        datetime_object = datetime.datetime.strptime(dt, '%m/%d/%Y')
        return datetime_object
    else:
        return datetime_object

def load_data(csv_name, include_cash = 0):
    dfP = pd.read_csv(csv_name+'.csv', parse_dates=['Date'])
    dfAP = pd.read_csv(csv_name+'.AP.csv', parse_dates=['Date'])
    dfP = dfP.sort_values(by='Date')
    dfAP = dfAP.sort_values(by='Date')
    dfP.set_index('Date', inplace = True)
    dfAP.set_index('Date', inplace = True)
    if include_cash:
        dfP['CASH'] = 1
        dfAP['CASH'] = 1

    return dfP, dfAP

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

def choose_asset(dfP, df_ranks, Filter, dfZ = None):
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
                    dfChoice.iat[row, dfP.columns.get_loc(StandsForCash)] = 1
            except:
                print('Specify Z-score dataframe')
        else:
            dfChoice.iat[row, max_arr_column] = 1
    #Filter 1 day whips
#     for row in range(1,rows-1,1):
#         if list(dfChoice.iloc[row].values) != list(dfChoice.iloc[row-1].values) and list(dfChoice.iloc[row].values) != list(dfChoice.iloc[row+1].values):
#             dfChoice.iloc[row] = dfChoice.iloc[row-1]

    return dfChoice



def calculate_results(dfP, dfAP, dfChoice, Frequency, Delay, verbose=0):
    dfDetrend = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    columns = dfP.shape[1]
    for column in range(columns):
        dfDetrend[dfAP.columns[column]] =  detrendPrice.detrendPrice(dfAP[dfAP.columns[column]]).values

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
        dfPRR[new] = dfPRR[dfP.columns[column]]*dfPRR[dfP.columns[column]+'_NUL'].shift(Delay)
        #repeat for detrended returns
        dfDetrendRR[new] = dfDetrendRR[dfP.columns[column]]*dfPRR[dfP.columns[column]+'_NUL'].shift(Delay)


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
        WhiteRealityCheckFor1.bootstrap(dfPRR['DETREND_ALL_R'])

        print(dfPRR)

    return dfPRR, TotaAnnReturn, CAGR, sharpe, volatility

