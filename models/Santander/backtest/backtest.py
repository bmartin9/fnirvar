""" 
Script to do backtesting for the Santander cycles dataset. Outputs csv files containing predictions and hyperparameters.txt 
NOTE: It is assumed that the backtest_design input file is clean: no NA values and has shape (T,N) 
"""

#!/usr/bin/env python3 
# USAGE: ./backtest.py <DESIGN_MATRIX>.csv config.yaml num_factors.csv 

import numpy as np
import sys
import yaml 
from fnirvar.modeling.train import FactorAdjustment 
from fnirvar.modeling.train import NIRVAR
from fnirvar.modeling.train import LASSO
import os
from numpy.random import default_rng
from sklearn.preprocessing import MinMaxScaler

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
save_labels = config['save_labels']
NIRVAR_embedding_method = config['NIRVAR_embedding_method'] 
use_HPC = config['use_HPC'] 
Q = config['Q']
factor_model = config['factor_model']
idiosyncratic_model_name = config['idiosyncratic_model']
minmax_scaling = config['minmax_scaling']
LASSO_penalty = config['LASSO_penalty']
LASSO_hyperparameter_tuning = config['LASSO_hyperparameter_tuning']
clustering_method = config['clustering_method']


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
N = Xs.shape[1]

###### READ IN FACTORS ######
if varying_factors:
    factor_csv = np.genfromtxt(sys.argv[3], delimiter=',',dtype='int')

###### BACKTESTING ###### 
if minmax_scaling:

    if factor_model == 'Static' and idiosyncratic_model_name == 'NIRVAR':
        print("Static Factors + NIRVAR") 
        predictions = np.zeros((n_backtest_days,N)) 
        labels_hat = np.zeros((n_backtest_days,N)) 
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            scaler = MinMaxScaler(feature_range=(-1,1)) 
            scaler.fit(X_diff) 
            X_train = scaler.transform(X_diff)
            X_train_mean = np.mean(X_train,axis=0)
            X_train -= X_train_mean
            if varying_factors:
                current_r = factor_csv[i]
            else:
                current_r = r
            factor_model = FactorAdjustment(X_train, current_r, lF)
            Xi = factor_model.get_idiosyncratic_component()
            idiosyncratic_model = NIRVAR(Xi=Xi,
                                        embedding_method=NIRVAR_embedding_method,
                                        clustering_method=clustering_method)
            Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 
            predictions_scaled = factor_model.predict_common_component()[:,0] + Xi_hat[:]
            predictions_scaled += X_train_mean[:]
            predictions_original_space = scaler.inverse_transform(predictions_scaled.reshape(1,-1))
            predictions[i] = predictions_original_space[0,:] + X[-1] # predict the log number of rides, not the first differences of this

            if save_labels:
                labels_hat[i] = idiosyncratic_model.get_NIRVAR_gmm_labels()

            print ("\033[A                             \033[A") 

    elif factor_model == 'None' and idiosyncratic_model_name == 'NIRVAR':
        print("Only NIRVAR")
        predictions = np.zeros((n_backtest_days,N)) 
        labels_hat = np.zeros((n_backtest_days,N)) 
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            scaler = MinMaxScaler(feature_range=(-1,1)) 
            scaler.fit(X_diff) 
            X_train = scaler.transform(X_diff)
            X_train_mean = np.mean(X_train,axis=0)
            X_train -= X_train_mean
            idiosyncratic_model = NIRVAR(Xi=X_train,
                                        embedding_method=NIRVAR_embedding_method,
                                        clustering_method=clustering_method) 
            Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 
            predictions_scaled =  Xi_hat
            predictions_scaled += X_train_mean[:]
            predictions_original_space = scaler.inverse_transform(predictions_scaled.reshape(1,-1))
            predictions[i] = predictions_original_space[0,:] + X[-1] 

            if save_labels:
                labels_hat[i] = idiosyncratic_model.get_NIRVAR_gmm_labels()

            print ("\033[A                             \033[A") 

    elif factor_model == 'Static' and idiosyncratic_model_name == 'None':
        print("Only Factors")
        predictions = np.zeros((n_backtest_days,N)) 
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            scaler = MinMaxScaler(feature_range=(-1,1)) 
            scaler.fit(X_diff) 
            X_train = scaler.transform(X_diff)
            X_train_mean = np.mean(X_train,axis=0)
            X_train -= X_train_mean
            if varying_factors:
                current_r = factor_csv[i]
            else:
                current_r = r
            model = FactorAdjustment(X_train, current_r, lF)
            predictions_scaled = model.predict_common_component()[:,0]
            predictions_scaled += X_train_mean[:]
            predictions_original_space = scaler.inverse_transform(predictions_scaled.reshape(1,-1))
            predictions[i] = predictions_original_space[0,:] + X[-1]

            print ("\033[A                             \033[A") 
    
    elif factor_model == 'Static' and idiosyncratic_model_name == 'LASSO':
        print("Static Factors + LASSO")  
        predictions = np.zeros((n_backtest_days,N)) 
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            scaler = MinMaxScaler(feature_range=(-1,1)) 
            scaler.fit(X_diff) 
            X_train = scaler.transform(X_diff)
            X_train_mean = np.mean(X_train,axis=0)
            X_train -= X_train_mean
            if varying_factors:
                current_r = factor_csv[i]
            else:
                current_r = r
            factor_model = FactorAdjustment(X_train, current_r, lF)
            Xi = factor_model.get_idiosyncratic_component()
            idiosyncratic_model = LASSO(Xi=Xi) 
            if LASSO_hyperparameter_tuning == 'None' :
                idiosyncratic_model.fit_lasso(alpha=LASSO_penalty[0])
            elif LASSO_hyperparameter_tuning == 'CV' : 
                idiosyncratic_model.fit_lasso_cv(alpha_values=LASSO_penalty)
            elif LASSO_hyperparameter_tuning == 'BIC' : 
                idiosyncratic_model.fit_lasso_bic(alpha_values=LASSO_penalty)
            Xi_hat = idiosyncratic_model.predict_idiosyncratic_component(X_new=Xi[-1,None]) 
            predictions_scaled = factor_model.predict_common_component()[:,0] + Xi_hat[:]
            predictions_scaled += X_train_mean[:]
            predictions_original_space = scaler.inverse_transform(predictions_scaled.reshape(1,-1))
            predictions[i] = predictions_original_space[0,:] + X[-1] # predict the log number of rides, not the first differences of this

            print ("\033[A                             \033[A") 

else: 
    if factor_model == 'Static' and idiosyncratic_model_name == 'NIRVAR':
        print("Static Factors + NIRVAR")
        predictions = np.zeros((n_backtest_days,N)) 
        labels_hat = np.zeros((n_backtest_days,N)) 
        loadings_hat = np.zeros((n_backtest_days,N,r))
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            if varying_factors:
                current_r = factor_csv[i]
            else:
                current_r = r
            factor_model = FactorAdjustment(X_diff, current_r, lF)
            if save_loadings: 
                loadings_i = factor_model.loadings()
                loadings_hat[i] = loadings_i
            Xi = factor_model.get_idiosyncratic_component() 
            idiosyncratic_model = NIRVAR(Xi=Xi,
                                        embedding_method=NIRVAR_embedding_method,
                                        clustering_method=clustering_method) 
            Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 
            predictions[i] = factor_model.predict_common_component()[:,0] + Xi_hat[:] + X[-1]

            if save_labels:
                labels_hat[i] = idiosyncratic_model.get_NIRVAR_gmm_labels()

            print ("\033[A                             \033[A") 

    elif factor_model == 'None' and idiosyncratic_model_name == 'NIRVAR':
        print("Only NIRVAR")
        predictions = np.zeros((n_backtest_days,N)) 
        labels_hat = np.zeros((n_backtest_days,N)) 
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            idiosyncratic_model = NIRVAR(Xi=X_diff,
                                        embedding_method=NIRVAR_embedding_method,
                                        clustering_method=clustering_method) 
            Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 
            predictions[i] =  Xi_hat[:] + X[-1]

            if save_labels:
                labels_hat[i] = idiosyncratic_model.get_NIRVAR_gmm_labels()

            print ("\033[A                             \033[A") 

    elif factor_model == 'Static' and idiosyncratic_model_name == 'None':
        print("Only Factors")
        predictions = np.zeros((n_backtest_days,N))
        loadings_hat = np.zeros((n_backtest_days,N,r))
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            if varying_factors:
                current_r = factor_csv[i]
            else:
                current_r = r
            model = FactorAdjustment(X_diff, current_r, lF)
            if save_loadings: 
                loadings_i = model.loadings()
                loadings_hat[i] = loadings_i
            predictions[i] = model.predict_common_component()[:,0] + X[-1]

            print ("\033[A                             \033[A") 

    elif factor_model == 'Static' and idiosyncratic_model_name == 'LASSO':
        print("Static Factors + LASSO")  
        predictions = np.zeros((n_backtest_days,N))
        loadings_hat = np.zeros((n_backtest_days,N,r))
        for i, day in enumerate(days_to_backtest):
            print(f"Day {day}") 
            X = Xs[day-lookback_window:day+1, :] # day is the day on which you predict tomorrow's returns from 
            X_diff = X[1:] - X[:-1]
            if varying_factors:
                current_r = factor_csv[i]
            else:
                current_r = r
            factor_model = FactorAdjustment(X_diff, current_r, lF)
            if save_loadings: 
                loadings_i = factor_model.loadings()
                loadings_hat[i] = loadings_i
            Xi = factor_model.get_idiosyncratic_component()
            idiosyncratic_model = LASSO(Xi=Xi) 
            if LASSO_hyperparameter_tuning == 'None' :
                idiosyncratic_model.fit_lasso(alpha=LASSO_penalty[0])
            elif LASSO_hyperparameter_tuning == 'CV' : 
                idiosyncratic_model.fit_lasso_cv(alpha_values=LASSO_penalty)
            elif LASSO_hyperparameter_tuning == 'BIC' : 
                idiosyncratic_model.fit_lasso_bic(alpha_values=LASSO_penalty)
            Xi_hat = idiosyncratic_model.predict_idiosyncratic_component(X_new=Xi[-1,None]) 
            predictions_diff = factor_model.predict_common_component()[:,0] + Xi_hat[:]
            predictions[i] = predictions_diff + X[-1] # predict the log number of rides, not the first differences of this

            print ("\033[A                             \033[A") 


###### OUTPUT TO FILES ###### 
if save_predictions:
    np.savetxt(f"predictions-{PBS_ARRAY_INDEX}.csv", predictions, delimiter=',') 

if save_labels and idiosyncratic_model_name == "NIRVAR":
    np.savetxt(f"labels_hat-{PBS_ARRAY_INDEX}.csv", labels_hat, delimiter=',', fmt='%d') 

if save_loadings:
    loadings_hat_2D = np.reshape(loadings_hat,(n_backtest_days*N,r))
    np.savetxt(f"loadings_hat-{PBS_ARRAY_INDEX}.csv", loadings_hat_2D, delimiter=',') 

f = open("backtesting_hyp.txt", "w")
f.write("{\n")
for k in config.keys():
    f.write("'{}':'{}'\n".format(k, config[k]))
f.write("}")
f.close()
