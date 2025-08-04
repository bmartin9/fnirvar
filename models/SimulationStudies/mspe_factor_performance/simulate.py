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
import plotly.express as px
import plotly.graph_objects as go
from fnirvar.modeling.train import LASSO



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
generative_model = config['generative_model']
estimation_model = config['estimation_model']
num_replicas = int(config['num_replicas'])
NIRVAR_embedding_method = config['NIRVAR_embedding_method'] 
first_prediction_day = config['first_prediction_day']
symmetric_phi = config['symmetric_phi']
lookback_window = config['lookback_window']
VAR_spectral_radius = config['VAR_spectral_radius']
if generative_model == 'FactorsOnly':
    generative_model = None 
if estimation_model == 'FactorsOnly':
    estimation_model = None 
LASSO_penalty = config['LASSO_penalty']
LASSO_hyperparameter_tuning = config['LASSO_hyperparameter_tuning']


num_N = len(N_list)
num_T = len(T_list)
T_max = max(T_list)

print(f"generative model: {generative_model}")
print(f"estimation model: {estimation_model}")

random_state = default_rng(seed=SEED)

###### ENVIRONMENT VARIABLES ######  
if use_HPC:
    PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
    NUM_ARRAY_INDICES = int(os.environ['NUM_ARRAY_INDICES'])
else:
    PBS_ARRAY_INDEX = 1
    NUM_ARRAY_INDICES = 1

###### LOADINGS MATRIX GENERATION ######
def loadings(N, r, sigma, rand_state):
    """
    Generate an N x r loadings matrix from a mixture distribution:
       0.5 * N(1, sigma^2) + 0.5 * N(-1, sigma^2).
    
    Parameters
    ----------
    N : int
        Number of rows.
    r : int
        Number of columns (number of factors).
    sigma : float
        Standard deviation for the normal distributions.
    seed : int or None, optional
        If provided, ensures reproducible random draws. Default is None.

    Returns
    -------
    L : np.ndarray of shape (N, r)
        The loadings matrix.
    """
    signs = rand_state.choice([-1, 1], size=(N, r))
    
    base_noise = rand_state.normal(loc=0.0, scale=sigma, size=(N, r))
    
    L = base_noise + signs
    
    return L

###### SIMULATION ###### 
mspe_values = np.zeros((num_N,num_T,num_replicas))
std_error_values = np.zeros((num_N,num_T,num_replicas))
for index_N, N in enumerate(N_list ):
    for s in range(num_replicas): 
        # loadings_matrix = block_loadings(N=N,r=r,normalize=True) 
        loadings_matrix = 0.4*loadings(N=N,r=r,sigma=0.1,rand_state=random_state)

        # phi_dist = np.ones((N,N))
        phi_dist = random_state.normal(1,1,size=(N,N))
        
        generate_NIRVAR = GenerateNIRVAR(random_state=random_state,
                                T=T_max,
                                B=B,
                                N=N,
                                Q=1,
                                p_in=p_in,
                                p_out=p_out,
                                phi_distribution=None,
                                multiplier=VAR_spectral_radius,
                                global_noise=1,
                                symmetrize_phi = symmetric_phi
                                )

        Xi = generate_NIRVAR.generate()[:,:,0]

        custom_colorscale = [
            [0.0, "darkred"],   # lowest values (most negative) are dark red
            [0.5, "white"],     # 0 maps to white (center of the scale)
            [1.0, "darkblue"]]   # highest values (most positive) are dark blue

        fig = go.Figure(data=go.Heatmap(z=generate_NIRVAR.phi_coefficients[:,0,:,0],
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
        fig.show()
        # fig.write_image(f"gt_phi_r_{r}.pdf")

        cov_Xi = Xi.T@Xi/T_max
        cov_Xi = cov_Xi - np.identity(N)
        fig = go.Figure(data=go.Heatmap(z=cov_Xi,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
        # fig.show()


        generate_FNIRVAR = GenerateFNIRVAR(l_F=l_F,
                                   T=T_max,
                                   r=r,
                                   q=num_common_shocks,
                                   rho_F=rho_F,
                                   random_state=random_state,
                                   P=None,
                                   N0=None
                                   )


        if generative_model is None:
            X = generate_FNIRVAR.generate_data(Lambda=loadings_matrix) 
            print(f"Var X : {np.var(X)}")
            fig = px.line(x=np.arange(T_max), y=X[:,0])
            # fig.show()

        elif generative_model == 'FNIRVAR':
            X = generate_FNIRVAR.generate_data(Lambda=loadings_matrix,xi=Xi) 
            print(f"Var Xi : {np.var(Xi)}")
            print(f"Var X : {np.var(X)}")
            fig = px.line(x=np.arange(T_max), y=X[:,0])
            # fig.show()

            fig = go.Figure(data=go.Heatmap(z=X.T@X/T_max,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
            # fig.show()

            min_eval_F = np.sort(np.linalg.eigvals(X.T@X/T_max))[-r] 
            max_eval_Xi = np.sort(np.linalg.eigvals(Xi.T@Xi/T_max))[-1]

            print(f"min eval F: {min_eval_F}")
            print(f"max eval Xi: {max_eval_Xi}")

        elif generative_model == 'NIRVAR':
            X = Xi 
            fig = px.line(x=np.arange(T_max), y=X[:,0])
            # fig.show()

        for index_T, T in enumerate(T_list):
            print(f"(N,s,T) : ({N},{s},{T})")
            X_T = X[:T]
            n_backtest_days_tot = T - first_prediction_day -1 
            predictions = np.zeros((n_backtest_days_tot,N))
            predictions_factors = np.zeros((n_backtest_days_tot,N))

            for t in range(n_backtest_days_tot):
                # print(f"backtest day {t}")
                X_backtest_data = X_T[first_prediction_day+t-lookback_window:first_prediction_day+t+1] 

                if estimation_model == "FNIRVAR":
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    Xi_estimated = factor_model.get_idiosyncratic_component()
                    idiosyncratic_model = NIRVAR(Xi=Xi_estimated,
                                            embedding_method=NIRVAR_embedding_method,
                                            d=B,
                                            K=B) 
                    Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 

                    similarity_matrix, labels = idiosyncratic_model.gmm() 
                    phi_hat = idiosyncratic_model.ols_parameters(similarity_matrix)
                    fig = go.Figure(data=go.Heatmap(z=phi_hat,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
                    # fig.show()
                    # fig.write_image(f"estimated_phi_r_hat_{r_hat}.pdf")

                    estimated_Xi_cov = Xi_estimated.T@Xi_estimated/T - np.identity(N)
                    fig = go.Figure(data=go.Heatmap(z=estimated_Xi_cov,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
                    # fig.show()
                    
                    # print(f"Var Xi_hat : {np.var(Xi_hat)}")
                    # print(f"common component variance : {np.var(factor_model.predict_common_component()[:,0])}")
                    predictions_t = factor_model.predict_common_component()[:,0] + Xi_hat[:] 
                    predictions[t,:] = predictions_t 

                elif estimation_model == "NIRVAR":
                    idiosyncratic_model = NIRVAR(Xi=X_backtest_data,embedding_method=NIRVAR_embedding_method,d=B,K=B)
                    predictions_t = idiosyncratic_model.predict_idiosyncratic_component() 

                    similarity_matrix, labels = idiosyncratic_model.gmm() 
                    phi_hat = idiosyncratic_model.ols_parameters(similarity_matrix)
                    fig = go.Figure(data=go.Heatmap(z=phi_hat,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
                    # fig.show()

                    predictions[t,:] = predictions_t 

                elif estimation_model is None:
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    predictions_t = factor_model.predict_common_component()[:,0] 
                    predictions[t,:] = predictions_t 

                elif estimation_model == "FactorsLASSO":
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    Xi_estimated = factor_model.get_idiosyncratic_component()
                    idiosyncratic_model = LASSO(Xi=Xi_estimated) 
                    if LASSO_hyperparameter_tuning == 'None' :
                        idiosyncratic_model.fit_VARp_LASSO(alpha=LASSO_penalty[0],p=l_F)
                    elif LASSO_hyperparameter_tuning == 'CV' : 
                        idiosyncratic_model.fit_lasso_cv(alpha_values=LASSO_penalty)
                    elif LASSO_hyperparameter_tuning == 'BIC' : 
                        idiosyncratic_model.fit_lasso_bic(alpha_values=LASSO_penalty)

                    Xi_hat = idiosyncratic_model.predict_next_VARp_LASSO(Xi[-1]) 
                    predictions_t = factor_model.predict_common_component()[:,0] + Xi_hat[:] 
                    predictions[t,:] = predictions_t 

                elif estimation_model == "FNIRVAR_vs_FactorsOnly":
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    Xi_estimated = factor_model.get_idiosyncratic_component()
                    idiosyncratic_model = NIRVAR(Xi=Xi_estimated,
                                            embedding_method=NIRVAR_embedding_method,
                                            d=B,
                                            K=B) 
                    Xi_hat = idiosyncratic_model.predict_idiosyncratic_component() 

                    similarity_matrix, labels = idiosyncratic_model.gmm() 
                    phi_hat = idiosyncratic_model.ols_parameters(similarity_matrix)
                    factor_prediction = factor_model.predict_common_component()[:,0]
                    predictions_t = factor_prediction + Xi_hat[:] 
                    predictions[t,:] = predictions_t 
                    predictions_factors[t,:] = factor_prediction


            targets = X_T[first_prediction_day+1:,:] 
            err = targets - predictions
            mspe = np.sum((err)**2)/((N*n_backtest_days_tot))
            print(f"MSPE : {mspe}")
            mspe_values[index_N,index_T,s] = mspe 
            sigma_err2 = np.std((err)**2,mean=mspe,ddof=1)/np.sqrt(n_backtest_days_tot)
            print(f"Standard deviation MSPE : {sigma_err2}") 
            std_error_values[index_N,index_T,s] = sigma_err2


            if estimation_model == "FNIRVAR_vs_FactorsOnly": 
                mspe_factors = np.sum((targets - predictions_factors)**2)/((N*n_backtest_days_tot))
                print(f"MSPE Factors : {mspe_factors}")

            x = np.arange(1, len(targets[:,0]) + 1)

            # Create a figure and add traces for each array
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=targets[:,10], mode='lines', name='Targets'))
            fig.add_trace(go.Scatter(x=x, y=predictions[:,10], mode='lines', name='Predictions'))
            if estimation_model == "FNIRVAR_vs_FactorsOnly": 
                fig.add_trace(go.Scatter(x=x, y=predictions_factors[:,0], mode='lines', name='Predictions Factors Only'))

            fig.show()
            

if num_replicas > 1:
    mspe_mean = np.mean(mspe_values,axis=-1)
    mspe_std = np.std(std_error_values,axis=-1)
else:
    mspe_mean = mspe_values[:,:,0]
    mspe_std = std_error_values[:,:,0]

np.savetxt(f"mspe_mean_rhat{r_hat}_{estimation_model}.csv", mspe_mean, delimiter=",", fmt="%.6f")
np.savetxt(f"mspe_std_rhat{r_hat}_{estimation_model}.csv", mspe_std, delimiter=",", fmt="%.6f")
