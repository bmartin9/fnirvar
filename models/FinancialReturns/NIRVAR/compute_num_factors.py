""" 
Script to compute the estimated number of factors on each backtesting day.
Outputs csv of integers. 
"""

#!/usr/bin/env python3 
# USAGE: ./compute_num_factors.py <DESIGN_MATRIX>.csv backtesting_config.yaml 

import numpy as np
import sys
import yaml 
from fnirvar.modeling.train import eigenvalue_ratio_test
from fnirvar.modeling.train import baing
import os
from numpy.random import default_rng

with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
n_backtest_days_tot = config['n_backtest_days']
first_prediction_day = config['first_prediction_day']
lookback_window = config['lookback_window']
Q = config['Q']
num_factors_method = config['num_factors_method']
max_num_factors = config['max_num_factors']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N)

Xs = np.reshape(Xs,(T,N,Q),order='F')
Xs = Xs[:,:,1] #pvCLCL  

###### COMPUTE NUMBER OF FACTORS FOR EACH BACKTESTING DAY ######
# Get a list of days to do backtesting on
days_to_backtest = [int(first_prediction_day + i) for i in range(n_backtest_days_tot)]
# print(f"Days to backtest: {days_to_backtest}")

num_factors = np.zeros(n_backtest_days_tot, dtype=int)

for i, day in enumerate(days_to_backtest):
    X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from

    # Compute number of factors
    if num_factors_method == 'ER':
        r, _, _ = eigenvalue_ratio_test(X,kmax=max_num_factors)

    elif num_factors_method == 'PCp1':
        r, _, _, _ = baing(X=X,kmax=max_num_factors,jj=1) 

    elif num_factors_method == 'PCp2':
        r, _, _, _ = baing(X=X,kmax=max_num_factors,jj=2) 
    
    elif num_factors_method == 'PCp3':
        r, _, _, _ = baing(X=X,kmax=max_num_factors,jj=3) 

    # Store the result
    num_factors[i] = r 

# Save results to CSV
output_filename = 'estimated_num_factors.csv'
np.savetxt(output_filename, num_factors, fmt='%d', delimiter=',')

print(f"Estimated factors saved to {output_filename}")
print(f"Summary statistics:")
print(f"Mean number of factors: {np.mean(num_factors):.2f}")
print(f"Min number of factors: {np.min(num_factors)}")
print(f"Max number of factors: {np.max(num_factors)}")