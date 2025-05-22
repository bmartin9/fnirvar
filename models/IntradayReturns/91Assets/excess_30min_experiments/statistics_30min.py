""" 
Script to compute various backtesting statistics given an input file of predicted stock returns values
NOTE: It is assumed that the backtest_design input file is clean: no NA values and has shape (T,N) ) after 
removing the first two columns (date and time) and the header.
"""

#!/usr/bin/env python3 
# USAGE: ./statistics.py <BACKTEST_DESIGN>.csv predictions.csv config.yaml 

import sys 
import yaml 
import numpy as np 
from numpy.random import default_rng 
from scipy import stats 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from fnirvar.modeling.statistics import benchmarking
import plotly.express as px
import plotly.graph_objects as go
import time
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions

with open(sys.argv[3], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
r = config["num_factors"]
lF = config["factor_lags"]
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']
lookback_window = config['lookback_window']
varying_factors = config['varying_factors'] 
save_loadings = config['save_loadings'] 
save_factors = config['save_factors'] 
save_predictions = config['save_predictions']
NIRVAR_embedding_method = config['NIRVAR_embedding_method'] 
use_HPC = config['use_HPC'] 
transaction_cost = config['transaction_cost']
quantile = config['quantile']
weightings_choice = config['weightings']
alpha = config['alpha']
beta = config['beta'] 
beta_Sigma = config['beta_Sigma']
Markowitz_penalty = config['Markowitz_penalty']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)[:, 2:] # Skip first two columns (date and time) and header 
T = Xs.shape[0]
N = Xs.shape[1]  

predictions = np.genfromtxt(sys.argv[2], delimiter=',') #read in predictions. shape = (n_backtest_days_tot,N) 

first_predict_from = first_prediction_day    # first day we do prediction on
last_predict_from = first_predict_from + n_backtest_days_tot # last day we do prediction on
targets = Xs[first_predict_from+1:last_predict_from+1,:] 


####### 30 MINUTELY BACKTESTING STATISTICS ###### 
PnL = np.zeros((n_backtest_days_tot-1))
hit_ratios  = np.zeros((n_backtest_days_tot-1))
long_ratios = np.zeros((n_backtest_days_tot-1)) 
corr_SP = np.zeros((n_backtest_days_tot-1))
half_hourly_rmse = np.zeros((n_backtest_days_tot-1))
half_hourly_turnover = np.zeros((n_backtest_days_tot-1))
half_hourly_r_squared = np.zeros((n_backtest_days_tot-1))

for t in range(1,n_backtest_days_tot): 
    # you must sacrifice the first day of predictions so that you can compute transaction_indicator
    print(t) 
    if weightings_choice == 'equal':
        weightings = np.ones((N)) #equal weightings

    half_hourly_bench = benchmarking(predictions=predictions[t],market_excess_returns=targets[t],yesterdays_predictions=predictions[t-1],transaction_cost=0.0001*transaction_cost)  
    half_hourly_PnL = half_hourly_bench.weighted_PnL_transactions(weights=weightings, quantile=quantile) 
    PnL[t-1] = half_hourly_PnL
    half_hourly_hit_ratio = half_hourly_bench.hit_ratio()
    hit_ratios[t-1] = half_hourly_hit_ratio
    half_hourly_long = half_hourly_bench.long_ratio() 
    long_ratios[t-1] = half_hourly_long
    half_hourly_corr_SP = half_hourly_bench.corr_SP()
    corr_SP[t-1] = half_hourly_corr_SP
    half_hourly_rmse[t-1] = np.sqrt(root_mean_squared_error(predictions[t],targets[t]))
    half_hourly_turnover[t-1] = (1/N)*np.sum(half_hourly_bench.transaction_indicator())
    half_hourly_r_squared = r2_score(y_true=targets[t],y_pred=predictions[t]) 

    print ("\033[A                             \033[A") 

###### DAILY BACKTESTING STATISTICS ######
PnL_30 = PnL[12:] # ignore day 1 due to transaction indicator meaning the first return is sacrificed
num_30min_returns = len(PnL_30)
assert num_30min_returns  % 13 == 0, "Length of x must be divisible by 13"
daily_PnL = PnL_30.reshape(-1, 13).sum(axis=1) 

def deflated_SR(PnL_vector): 
    SR = np.mean(PnL_vector)/np.std(PnL_vector) 
    Ti = len(PnL_vector) 
    g3 = stats.skew(PnL_vector) 
    g4 = stats.kurtosis(PnL_vector) 
    denominator = 1 - g3*SR + (g4-1)*((SR**2)/4)
    test_statistic = SR/(np.sqrt(denominator/(Ti-1))) 
    p1 = stats.norm.cdf(test_statistic) 
    pval = np.minimum(p1, 1 - p1)*2 
    return pval 

deflated_sharpe_ratio = deflated_SR(daily_PnL) 

print(f"PnL Mean: {np.mean(daily_PnL)}")
print(f"PnL Std: {np.std(daily_PnL)}")
sharpe_ratio = (np.mean(daily_PnL)/np.std(daily_PnL))*np.sqrt(252)
mean_spearman_corr = np.mean(corr_SP)
print(f"sum PnL : {np.sum(daily_PnL)}") 
PPT = np.sum(daily_PnL)/((num_30min_returns)*N)
print(f"n_backtest_days_tot : {num_30min_returns}") 
mean_daily_PnL = np.mean(daily_PnL) 
total_hit_ratio = np.mean(hit_ratios)
total_long_ratio = np.mean(long_ratios)
mean_rmse = np.mean(half_hourly_rmse)
mean_turnover = np.mean(half_hourly_turnover)
mean_r_squared = np.mean(half_hourly_r_squared)
MAE = mean_absolute_error(y_true=targets,y_pred=predictions,multioutput="uniform_average")

# Calculate Max draw down
cum_PnL_bpts = 10000*np.cumsum(daily_PnL)/(num_30min_returns) 
def max_drawdown(X):
    #return max draw down in percentage terms
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd 
max_dd = max_drawdown(cum_PnL_bpts)

#Sotino Ratio
sortino_ratio = (np.mean(daily_PnL)/np.std(np.where(daily_PnL<=0,daily_PnL,0)))*np.sqrt(252)

###### WRITE TO OUTPUT FILES ######
np.savetxt('PnL.csv', daily_PnL, delimiter=',', fmt='%.6f')
np.savetxt('hit.csv', hit_ratios, delimiter=',', fmt='%.6f')
np.savetxt('long.csv', daily_PnL, delimiter=',', fmt='%.6f')
np.savetxt('spearman_corr.csv', corr_SP, delimiter=',', fmt='%.6f')

summary_statistics = {'Deflated Sharpe Ratio': deflated_sharpe_ratio,
                    'Sharpe Ratio': sharpe_ratio,
                    'Mean Spearman Correlation' : mean_spearman_corr,
                    'PnL Per Trade' : PPT,
                    'Mean Daily PnL' : mean_daily_PnL,
                    'Hit Ratio' : total_hit_ratio,
                    'Long Ratio' : total_long_ratio,
                    'Max Draw Down' : max_dd,
                    'Sortino Ratio' : sortino_ratio,
                    'Mean RMSE' : mean_rmse,
                    'Mean R Squared' : mean_r_squared,
                    'Mean Turnover' : mean_turnover,
                    'MAE' : MAE
    }

f = open("summary_statistics.txt", "w")
f.write("{\n")
for k in summary_statistics.keys():
    f.write("'{}':'{}'\n".format(k, summary_statistics[k]))
f.write("}")
f.close()
 
