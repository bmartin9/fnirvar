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
lookback_window = config['lookback_window']
VAR_spectral_radius = config['VAR_spectral_radius']
if generative_model == 'FactorsOnly':
    generative_model = None 
if estimation_model == 'FactorsOnly':
    estimation_model = None 

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

###### HELPER FUNCTIONS ######
def create_block_loadings(N, r, group_assignments, scale=1.0, normalize=True):
    """
    N: total number of series
    r: number of factors
    group_assignments: list (of length N) that says which group/index each row i belongs to
                      for example, group_assignments[i] = j means row i is assigned to factor j
    scale: standard deviation for random loadings
    normalize: whether to normalize each column to have unit norm

    Returns:
        A (N x r) array, block-structured loading matrix.
    """
    Lambda = np.zeros((N, r))

    # Fill block j with random entries
    for i in range(N):
        # which factor does row i belong to?
        j = group_assignments[i]
        # fill the (i, j) entry with a draw from, say, N(0, scale^2)
        Lambda[i, j] = np.random.normal(loc=0.0, scale=scale)
    
    # Optional: Normalize each column
    if normalize:
        for col in range(r):
            col_norm = np.linalg.norm(Lambda[:, col])
            if col_norm > 0:
                Lambda[:, col] /= col_norm
    
    return Lambda

def create_group_assignments(N, r):
    """
    Partition row indices 0..N-1 into r blocks, each block j is assigned factor j.
    If N is not divisible by r, some blocks will have one extra row.
    
    Returns:
        A list of length N, where group_assignments[i] is the factor index for row i.
    """
    base_size = N // r          # integer division
    extra = N % r               # remainder
    group_assignments = []
    
    current_index = 0
    for j in range(r):
        # Block j size: either base_size or base_size + 1
        block_size_j = base_size + (1 if j < extra else 0)
        
        # Extend group_assignments by 'block_size_j' copies of j
        group_assignments.extend([j] * block_size_j)
        
        current_index += block_size_j
        if current_index >= N:
            break
    
    return group_assignments

def block_loadings(N, r, scale=1.0, normalize=True):
    assignments = create_group_assignments(N, r)
    Lambda = np.zeros((N, r))
    
    for i, factor_idx in enumerate(assignments):
        Lambda[i, factor_idx] = np.random.normal(0, scale)
        # Lambda[i, factor_idx] = 1
    
    # Optional: normalize each column
    if normalize:
        for j in range(r):
            norm_j = np.linalg.norm(Lambda[:, j])
            if norm_j > 0:
                Lambda[:, j] /= norm_j
    return Lambda

def random_loadings(N, r, scale=1.0, normalize=True, random_state=None):
    """
    Generate a random N x r loadings matrix with entries drawn from N(0, scale^2).
    Optionally normalizes each column to have unit Euclidean norm.
    
    Args:
        N (int): Number of rows (series).
        r (int): Number of factors (columns).
        scale (float): Std. dev. for normal draws. Defaults to 1.0.
        normalize (bool): Whether to normalize each column to unit norm. Defaults to True.
        random_state (int or None): Seed for reproducibility. Defaults to None (no fixed seed).

    Returns:
        numpy.ndarray: An N x r array of random loadings.
    """
    rng = np.random.default_rng(random_state)
    # Draw an N x r matrix from Normal(0, scale^2)
    Lambda = rng.normal(loc=0.0, scale=scale, size=(N, r))
    
    if normalize:
        # Normalize each column to have unit Euclidean norm
        for j in range(r):
            col_norm = np.linalg.norm(Lambda[:, j])
            if col_norm > 1e-12:
                Lambda[:, j] /= col_norm
            # If col_norm is very close to zero, the column remains (near) zero.
    
    return Lambda

###### SIMULATION ###### 
mspe_values = np.zeros((num_N,num_T,num_replicas))
for index_N, N in enumerate(N_list ):
    for s in range(num_replicas): 
        # loadings_matrix = block_loadings(N=N,r=r,normalize=True) 
        loadings_matrix = 2.5*random_loadings(N=N,r=r,scale=1,normalize=True,random_state=None)


        phi_dist = np.ones((N,N))

        generate_NIRVAR = GenerateNIRVAR(random_state=random_state,
                                T=T_max,
                                B=B,
                                N=N,
                                Q=1,
                                p_in=p_in,
                                p_out=p_out,
                                phi_distribution=None,
                                multiplier=VAR_spectral_radius,
                                global_noise=1)

        Xi = generate_NIRVAR.generate()[:,:,0]

        custom_colorscale = [
            [0.0, "darkred"],   # lowest values (most negative) are dark red
            [0.5, "white"],     # 0 maps to white (center of the scale)
            [1.0, "darkblue"]]   # highest values (most positive) are dark blue

        # fig = go.Figure(data=go.Heatmap(z=generate_NIRVAR.phi_coefficients[:,0,:,0],
        #                                 colorscale=custom_colorscale,
        #                                 zmid=0  # ensures that 0 is mapped to the middle color (white)
        #                                 ))
                                        
        # fig.show()
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
                print(f"backtest day {t}")
                X_backtest_data = X_T[first_prediction_day+t-lookback_window:first_prediction_day+t] 

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
                                        
                    fig.show()
                    
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
                                        
                    fig.show()

                    predictions[t,:] = predictions_t 

                elif estimation_model is None:
                    factor_model = FactorAdjustment(X_backtest_data, r_hat, l_F)
                    predictions_t = factor_model.predict_common_component()[:,0] 
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
            mspe = np.sum((targets - predictions)**2)/((N*n_backtest_days_tot))
            print(f"MSPE : {mspe}")
            mspe_values[index_N,index_T,s] = mspe 

            if estimation_model == "FNIRVAR_vs_FactorsOnly": 
                mspe_factors = np.sum((targets - predictions_factors)**2)/((N*n_backtest_days_tot))
                print(f"MSPE Factors : {mspe_factors}")

            x = np.arange(1, len(targets[:,0]) + 1)

            # Create a figure and add traces for each array
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=targets[:,0], mode='lines', name='Targets'))
            fig.add_trace(go.Scatter(x=x, y=predictions[:,0], mode='lines', name='Predictions'))
            if estimation_model == "FNIRVAR_vs_FactorsOnly": 
                fig.add_trace(go.Scatter(x=x, y=predictions_factors[:,0], mode='lines', name='Predictions Factors Only'))

            fig.show()
            

mspe_mean = np.mean(mspe_values,axis=-1)
mspe_std = np.std(mspe_values,axis=-1)

np.savetxt(f"mspe_mean_rhat{r_hat}_{estimation_model}.csv", mspe_mean, delimiter=",", fmt="%.6f")
np.savetxt(f"mspe_std_rhat{r_hat}_{estimation_model}.csv", mspe_std, delimiter=",", fmt="%.6f")
