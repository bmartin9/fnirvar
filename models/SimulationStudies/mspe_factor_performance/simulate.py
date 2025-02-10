""" 
Simulation study to assess the effect of mispecification of factor number on prediction error.
"""

#!/usr/bin/env python3 
# USAGE: ./simulate.py hyperparameters.yaml 

from fnirvar.modeling.generativeVAR import GenerateFNIRVAR
from fnirvar.modeling.generativeVAR import GenerateNIRVAR 
from fnirvar.modeling.train import FactorAdjustment
from fnirvar.modeling.train import NIRVAR
import numpy as np 
import sys 
import yaml 
import os 
from numpy.random import default_rng
import ast

with open(sys.argv[1], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
SEED = config["SEED"]
r = config["num_factors"]
r_hat = config["num_estimated_factors"]
T_list = list(config['T_list'])
N_list = list(config['N_list'])
B = config['B']
p_in = config['p_in']
p_out = config['p_out'] 
use_HPC = config['use_HPC'] 
num_common_shocks = config['num_common_shocks']
rho_F = config['rho_F']
l_F = config['l_F']
model = config['model']
num_replicas = int(config['num_replicas'])
NIRVAR_embedding_method = config['NIRVAR_embedding_method'] 
first_prediction_day = config['first_prediction_day']
lookback_window = config['lookback_window']
if model == 'FactorsOnly':
    model = None 

num_N = len(N_list)
num_T = len(T_list)
T_max = max(T_list)

print(model)

random_state = default_rng(seed=SEED)

###### ENVIRONMENT VARIABLES ######  
if use_HPC:
    PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
    NUM_ARRAY_INDICES = int(os.environ['NUM_ARRAY_INDICES'])
else:
    PBS_ARRAY_INDEX = 1
    NUM_ARRAY_INDICES = 1

###### SIMULATION ###### 
mspe_values = np.zeros((num_N,num_T,num_replicas))
for index_N, N in enumerate(N_list ):
    for s in range(num_replicas): 
        loadings_matrix = random_state.normal(size=(N,r))

        phi_dist = np.ones((N,N))

        generate_NIRVAR = GenerateNIRVAR(random_state=random_state,
                                T=T_max,
                                B=B,
                                N=N,
                                Q=1,
                                p_in=p_in,
                                p_out=p_out,
                                phi_distribution=phi_dist )

        Xi = generate_NIRVAR.generate()[:,:,0]

        generate_FNIRVAR = GenerateFNIRVAR(l_F=l_F,
                                   T=T_max,
                                   r=r,
                                   q=num_common_shocks,
                                   rho_F=rho_F,
                                   random_state=random_state,
                                   P=None,
                                   N0=None
                                   )


        if model is None:
            X = generate_FNIRVAR.generate_data(Lambda=loadings_matrix) 

        elif model == 'FNIRVAR':
            X = generate_FNIRVAR.generate_data(Lambda=loadings_matrix,xi=Xi) 

        elif model == 'NIRVAR':
            X = Xi 

        for index_T, T in enumerate(T_list):
            print(f"(N,s,T) : ({N},{s},{T})")
            X_T = X[:T]
            n_backtest_days_tot = T - first_prediction_day -1 
            predictions = np.zeros((n_backtest_days_tot,N))
            for t in range(n_backtest_days_tot):
                X_backtest_data = X_T[first_prediction_day+t-lookback_window:first_prediction_day+t] 

                if model == "FNIRVAR":
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    Xi_estimated = factor_model.get_idiosyncratic_component()
                    idiosyncratic_model = NIRVAR(Xi=Xi_estimated,
                                            embedding_method=NIRVAR_embedding_method) 
                    Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 
                    predictions_t = factor_model.predict_common_component()[:,0] + Xi_hat[:] 
                    predictions[t,:] = predictions_t 

                elif model == "NIRVAR":
                    idiosyncratic_model = NIRVAR(Xi=X_backtest_data,embedding_method=NIRVAR_embedding_method)
                    predictions_t = idiosyncratic_model.predict_idiosyncratic_component() 
                    predictions[t,:] = predictions_t 

                elif model is None:
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    predictions_t = factor_model.predict_common_component()[:,0] 
                    predictions[t,:] = predictions_t 

            targets = X_T[first_prediction_day+1:,:] 
            mspe = np.sum((targets - predictions)**2)/(N*n_backtest_days_tot) 
            mspe_values[index_N,index_T,s] = mspe
            

mspe_mean = np.mean(mspe_values,axis=-1)
mspe_std = np.std(mspe_values,axis=-1)

np.savetxt(f"mspe_mean_r{r}.csv", mspe_mean, delimiter=",", fmt="%.6f")
np.savetxt(f"mspe_std_r{r}.csv", mspe_std, delimiter=",", fmt="%.6f")
