import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import WhiteRealityCheckFor1
import yfinance as yf
yf.pdr_override()
import pandas as pd
from matplotlib import pyplot
import statsmodels.api as sm

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

# =============================================================================
# def load_data(csv_name, include_cash = 0):
#     dfP = pd.read_csv(csv_name+'.csv', parse_dates=['Date'])
#     dfAP = pd.read_csv(csv_name+'.AP.csv', parse_dates=['Date'])
#     dfP = dfP.sort_values(by='Date')
#     dfAP = dfAP.sort_values(by='Date')
#     dfP.set_index('Date', inplace = True)
#     dfAP.set_index('Date', inplace = True)
#     if include_cash:
#         dfP['CASH'] = 1
#         dfAP['CASH'] = 1
#
#     return dfP, dfAP
# =============================================================================

def detrendPrice(series):
    # fit linear model
    length = len(series)
    x = np.arange(length)
    y = np.array(series.values)
    x_const = sm.add_constant(x) #need to add intercept constant
    model = sm.OLS(y,x_const)
    result = model.fit()
    #intercept = result.params[0]
    #beta = result.params[1]
    #print(result.summary())
    df = pd.DataFrame(result.params*x_const)
    y_hat = df[0] + df[1]
    #the residuals are the detrended prices
    resid = y-y_hat
    #add minimum necessary to residuals to avoid negative detrended parices
    resid = resid + abs(resid.min() + 1/10*resid.min())
    return resid