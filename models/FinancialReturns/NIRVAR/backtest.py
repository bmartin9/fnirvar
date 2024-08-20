""" 
Script to do backtesting. Outputs csv files containing predictions hyperparameters.txt 
NOTE: It is assumed that the backtest_design input file is clean: no NA values and has shape (T,N) 
"""

#!/usr/bin/env python3 
# USAGE: ./backtest.py <DESIGN_MATRIX>.csv backtesting_config.yaml num_factors.csv 

import numpy as np
import sys
import yaml 
from fnirvar.modeling.train import FactorAdjustment 
from fnirvar.modeling.train import NIRVAR
import os
from numpy.random import default_rng

with open(sys.argv[2], "r") as f:
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
do_NIRVAR_estimation = config['do_NIRVAR_estimation']
NIRVAR_embedding_method = config['NIRVAR_embedding_method'] 
use_HPC = config['use_HPC'] 
Q = config['Q']

###### ENVIRONMENT VARIABLES ######  
if use_HPC:
    PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
    NUM_ARRAY_INDICES = int(os.environ['NUM_ARRAY_INDICES'])
else:
    PBS_ARRAY_INDEX = 1
    NUM_ARRAY_INDICES = 1

# Re-define n_backtest_days to be total number of backtesting days divided by the number of array indices 
n_backtest_days = int(n_backtest_days_tot/NUM_ARRAY_INDICES)

# Get a list of days to do backtesting on
days_to_backtest = [int(first_prediction_day + i + (n_backtest_days)*(PBS_ARRAY_INDEX-1)) for i in range(n_backtest_days)]
print(f"Days to backtest: {days_to_backtest}")

random_state = default_rng(seed=SEED)

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N_times_Q = Xs.shape[1]
N = N_times_Q/Q
if N != int(N):
    print("ERROR:Input is not a whole number")
N = int(N)

Xs = np.reshape(Xs,(T,N,Q),order='F')
Xs = Xs[:,:,1] #prCLCL 

###### READ IN FACTORS ######
if varying_factors:
    factor_csv = np.genfromtxt(sys.argv[3], delimiter=',')

###### BACKTESTING ###### 
if do_NIRVAR_estimation:
    predictions = np.zeros((n_backtest_days, N)) 
    for i, day in enumerate(days_to_backtest):
        print(f"Day {day}") 
        X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
        if varying_factors:
            current_r = factor_csv[i]
        else:
            current_r = r
        factor_model = FactorAdjustment(X, current_r, lF)
        Xi = factor_model.get_idiosyncratic_component()
        idiosyncratic_model = NIRVAR(Xi=Xi,
                                     embedding_method=NIRVAR_embedding_method) 
        Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 
        predictions[i, :] = factor_model.predict_common_component()[:,0] + Xi_hat
else:
    predictions = np.zeros((n_backtest_days, N)) 
    for i, day in enumerate(days_to_backtest):
        print(f"Day {day}") 
        X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
        if varying_factors:
            current_r = factor_csv[i]
        else:
            current_r = r
        model = FactorAdjustment(X, current_r, lF)
        predictions[i, :] = model.predict_common_component()[:,0] 

###### OUTPUT TO FILES ###### 
if save_predictions:
    np.savetxt(f"predictions-{PBS_ARRAY_INDEX}.csv", predictions, delimiter=',') 

f = open("backtesting_hyp.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close()




