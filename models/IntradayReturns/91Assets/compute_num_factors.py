""" 
Script to compute the estimated number of factors on each backtesting day.
Outputs csv of integers. 
"""

#!/usr/bin/env python3 
# USAGE: ./compute_num_factors.py <DESIGN_MATRIX>.csv config.yaml 

import numpy as np
import sys
import yaml 
from fnirvar.modeling.train import ER
from fnirvar.modeling.train import GR
from fnirvar.modeling.train import ER_kth_biggest
from fnirvar.modeling.train import GR_kth_biggest
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
num_factors_method = config['num_factors_method']
max_num_factors = config['max_num_factors']
kth_eigengap = config['kth_eigengap']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)[:, 2:] # Skip first two columns (date and time) and header 
T = Xs.shape[0]
N = Xs.shape[1] 

###### COMPUTE NUMBER OF FACTORS FOR EACH BACKTESTING DAY ######
# Get a list of days to do backtesting on
days_to_backtest = [int(first_prediction_day + i) for i in range(n_backtest_days_tot)]
# print(f"Days to backtest: {days_to_backtest}")

num_factors = np.zeros(n_backtest_days_tot, dtype=int)

for i, day in enumerate(days_to_backtest):
    print(i) 
    X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from

    # Compute number of factors
    if num_factors_method == 'ER':
        r = ER(X,kmax=max_num_factors)

    elif num_factors_method == 'ER_kth_biggest':
        r = ER_kth_biggest(X,kmax=max_num_factors,k=kth_eigengap)

    elif num_factors_method == 'GR':
        r = GR(X,kmax=max_num_factors)

    elif num_factors_method == 'GR_kth_biggest':
        r = GR_kth_biggest(X,kmax=max_num_factors,k=kth_eigengap)

    elif num_factors_method == 'PCp1':
        r, _, _, _ = baing(X=X,kmax=max_num_factors,jj=1) 

    elif num_factors_method == 'PCp2':
        r, _, _, _ = baing(X=X,kmax=max_num_factors,jj=2) 
    
    elif num_factors_method == 'PCp3':
        r, _, _, _ = baing(X=X,kmax=max_num_factors,jj=3) 

    # Store the result
    num_factors[i] = r 

    print ("\033[A                             \033[A") 


# Save results to CSV
output_filename = 'estimated_num_factors.csv'
np.savetxt(output_filename, num_factors, fmt='%d', delimiter=',')

print(f"Estimated factors saved to {output_filename}")
print(f"Summary statistics:")
print(f"Mean number of factors: {np.mean(num_factors):.2f}")
print(f"Min number of factors: {np.min(num_factors)}")
print(f"Max number of factors: {np.max(num_factors)}")