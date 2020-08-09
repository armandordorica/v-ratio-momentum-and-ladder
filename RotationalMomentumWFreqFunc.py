# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import detrendPrice
import WhiteRealityCheckFor1
import yfinance as yf
from helper import *
import functions as ff
yf.pdr_override() 

def rotational_momentum(lookback,shtrm_weight,RSI_weight,v_ratio_weight, include_cash, verbose):

    #Hyper Parameters
    csv_name = "DIA.TLT"
    Aperiods = lookback
    Frequency = "B" #fridays and thursdays are the best trading days
    ShortTermWeight = shtrm_weight
    LongTermWeight = (1-shtrm_weight)
    ShortTermVolatilityWeight = .0
    Delay = 2
    z_score_weight = 0.0
    p_value_weight = 0.0

    #experimental cash filter
    Filter = 0
    MAperiods = 200 
    StandsForCash = "SHY" 
    Zboundary = 2.0
    Zperiods = 20

    #Sorting parameters
    momentum = 1
    volmomentum = 0 


    #lookbacks for longer term period and volatily
    Bperiods = 3*Aperiods+((3*Aperiods)//20)*2 #66 Default
    Speriods = Aperiods #20 Default

    #Load Data
    dfP, dfAP = load_data(csv_name, include_cash)
    

    #Create empty dataframes
    dfA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfB = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfS = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfVR = pd.DataFrame(columns=dfP.columns, index=['vratio','zscore', 'pvalue'])
    
    dfMA = dfP.drop(labels=None, axis=1, columns=dfP.columns)
    dfRSI = compute_rsi(dfP, n=14)
    dfZ = compute_zscore(dfP, n=20)


    #pct_change calculates the percentage change (A[n] - A[n-periods])/A[n-periods]
    dfA = dfP.pct_change(periods=Aperiods-1, fill_method='pad', limit=None, freq=None) #is counting window from 0
    dfB = dfP.pct_change(periods=Bperiods-1, fill_method='pad', limit=None, freq=None) #is counting window from 0
    dfR = dfP.pct_change(periods=1, fill_method='pad', limit=None, freq=None) #is counting window from 0

    #Caluclate volatility, z-score and moving average
    columns = dfP.shape[1]
    for column in range(columns):
        dfS[dfP.columns[column]] = (dfR[dfP.columns[column]].rolling(window=Speriods).std())*math.sqrt(252)
        dfMA[dfP.columns[column]] = (dfP[dfP.columns[column]].rolling(window=MAperiods).mean())
        vratio, zscore, pval = ff.vratio(np.array(dfP[dfP.columns[column]]), lag = Aperiods, cor = 'hom')
        dfVR[dfP.columns[column]] = [vratio, zscore, pval]
        
    #Fill NAN values in dfVR
    dfVR.fillna(1.0,inplace=True)
    
    #Ranking each ETF w.r.t. short moving average of returns
    dfA_ranks = rank_assets(dfA, order=1)
    dfB_ranks = rank_assets(dfB, order=1)
    dfS_ranks = rank_assets(dfS, order=0)
    dfZ_ranks = rank_assets(dfZ, order=0)
    #dfRSI_ranks = rank_assets(dfRSI, order=0)
    

    #Weights of the varous ranks 
    dfA_ranks = dfA_ranks.multiply(ShortTermWeight) 
    dfB_ranks = dfB_ranks.multiply(LongTermWeight) 
    dfS_ranks = dfS_ranks.multiply(ShortTermVolatilityWeight) 
    dfRSI_ranks = dfRSI.multiply(RSI_weight) 
    #dfZ_ranks = dfZ_ranks.multiply(Z_weight) 
    dfAll_ranks = dfA_ranks.add(dfB_ranks, fill_value=0)
    dfAll_ranks = dfAll_ranks.add(dfS_ranks, fill_value=0)
    dfAll_ranks = dfAll_ranks.add(dfRSI_ranks, fill_value=0)
    
    for column in range(columns):
        dfAll_ranks[dfP.columns[column]] = dfAll_ranks[dfP.columns[column]].values + (dfVR[dfP.columns[column]]['vratio'] - 1) * v_ratio_weight    
    
    #dfAll_ranks = dfAll_ranks.add(dfZ_ranks, fill_value=0)

    #Choice is the dataframe where the ETF with the maximum score is identified
    dfChoice = choose_asset(dfP, dfAll_ranks, Filter)

    #calculate results
    TotaAnnReturn, CAGR, sharpe, volatility = calculate_results(dfP, dfAP, dfChoice, Frequency, Delay, verbose)
    
    return TotaAnnReturn, CAGR, sharpe, volatility




