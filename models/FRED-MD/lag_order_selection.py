""" 
Script to compute the number of lags to use in the resticted dynamic factor model. 
The number of factors is taken as an input from the user.
"""

#!/usr/bin/env python3
# USAGE: ./lag_order_selection.py <DESIGN_MATRIX>.csv config.yaml num_factors.csv 

import numpy as np
import sys
import yaml
from fnirvar.modeling.train import FactorAdjustment 
from statsmodels.tsa.api import VAR

with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
varying_factors = config['varying_factors'] 
r = config["num_factors"]
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']
lookback_window = config['lookback_window']
minmax_scaling = config['minmax_scaling']
lF = config['lF']  

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, usecols=range(1,123))
T = Xs.shape[0]
N = Xs.shape[1]
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N) 

###### MINMAX SCALING ######
if minmax_scaling: 
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1,1)) 
    scaler.fit(Xs) 
    Xs_scaled = scaler.transform(Xs)
    Xs_mean = np.mean(Xs_scaled,axis=0)
    Xs = Xs_scaled - Xs_mean 

###### READ IN FACTORS ######
if varying_factors:
    factor_csv = np.genfromtxt(sys.argv[3], delimiter=',',dtype='int')

###### COMPUTE VAR ORDER FOR EACH BACKTESTING DAY ######
days_to_backtest = [int(first_prediction_day + i) for i in range(n_backtest_days_tot)]

estimated_lags = np.zeros(n_backtest_days_tot, dtype=int)

for i, day in enumerate(days_to_backtest):
    X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from

    print(f"day {i} X shape {X.shape}")

    if varying_factors:
            current_r = factor_csv[i]
    else:
        current_r = r 

    factor_model = FactorAdjustment(X, current_r, lF)
    factor_design = factor_model.static_factors()
    
    model = VAR(factor_design) 
    results = model.fit(maxlags=10, ic='aic') 
    lF_estimated = max(results.k_ar,1)
    print(f"day {i} factors {current_r} lags {lF_estimated}") 
    estimated_lags[i] = lF_estimated 

# Save results to CSV
output_filename = 'estimated_lag_order.csv'
np.savetxt(output_filename, estimated_lags, fmt='%d', delimiter=',')

print(f"Estimated lags saved to {output_filename}")
print(f"Summary statistics:")
print(f"Mean number of lags: {np.mean(estimated_lags):.2f}")
print(f"Min number of lags: {np.min(estimated_lags)}")
print(f"Max number of lags: {np.max(estimated_lags)}")

    


