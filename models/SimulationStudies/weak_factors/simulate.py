"Simulation Study to produce the eigenvalues of NIRVAR/FNIRVAR for different values of N"

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
from sklearn.covariance import LedoitWolf


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
eigenvalues_list = config['eigenvalues_list']
NIRVAR_embedding_method = config['NIRVAR_embedding_method'] 
symmetric_phi = config['symmetric_phi']
LedoitWolfEstimator = config['LedoitWolfEstimator']
VAR_spectral_radius = config['VAR_spectral_radius']
if generative_model == 'FactorsOnly':
    generative_model = None 
if estimation_model == 'FactorsOnly':
    estimation_model = None 

num_N = len(N_list)
num_T = len(T_list)
T_max = max(T_list)
num_evals = len(eigenvalues_list)

print(f"generative model: {generative_model}")
print(f"estimation model: {estimation_model}")

random_state = default_rng(seed=SEED)

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
eigenvalues_store = np.zeros((num_N,num_evals,num_replicas))
Gamma_eigenvalues_store = np.zeros((num_N,num_evals,num_replicas))
phi_evals_store = np.zeros((num_N,num_evals,num_replicas))
for index_N, N in enumerate(N_list ):
    for s in range(num_replicas): 
        # loadings_matrix = block_loadings(N=N,r=r,normalize=True) 
        loadings_matrix = 0.3*loadings(N=N,r=r,sigma=0.1,rand_state=random_state)
        # print(loadings_matrix)

        # phi_dist = np.ones((N,N))
        phi_dist = random_state.normal(1,1,size=(N,N))

        generate_NIRVAR = GenerateNIRVAR(random_state=random_state,
                                T=T_max,
                                B=B,
                                N=N,
                                Q=1,
                                p_in=p_in,
                                p_out=p_out,
                                phi_distribution=phi_dist,
                                multiplier=VAR_spectral_radius,
                                global_noise=1,
                                symmetrize_phi = symmetric_phi
                                )

        Xi = generate_NIRVAR.generate()[:,:,0]

        phi_N = generate_NIRVAR.phi_coefficients[:,0,:,0]
        eigenvalues_phi = np.real(np.linalg.eigvals(phi_N))
        idx_phi = np.argsort(eigenvalues_phi)[::-1]
        eigenvalues_phi = eigenvalues_phi[idx_phi] # sort 
        # print(eigenvalues_phi[:10])
        phi_evals_store[index_N,:,s] = eigenvalues_phi[:num_evals]

        custom_colorscale = [
            [0.0, "darkred"],   # lowest values (most negative) are dark red
            [0.5, "white"],     # 0 maps to white (center of the scale)
            [1.0, "darkblue"]]   # highest values (most positive) are dark blue

        fig = go.Figure(data=go.Heatmap(z=phi_N,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
        # fig.show()
        # fig.write_image(f"gt_phi_r_{r}.pdf")

        cov_Xi = Xi.T@Xi/T_max
        # cov_Xi = LedoitWolf().fit(Xi).covariance_
        cov_Xi = cov_Xi - np.identity(N)
        fig = go.Figure(data=go.Heatmap(z=cov_Xi,
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
        # fig.show()
        fig.write_image(f"sample_cov-N{N}.pdf")


        Gamma = np.linalg.inv(np.identity(N) - phi_N@phi_N.T) 
        eigenvalues_factor, eigenvectors_factor = np.linalg.eigh(loadings_matrix@loadings_matrix.T)
        sorted_indices_factor = np.argsort(eigenvalues_factor)[::-1]
        topr_eigenvalues = eigenvalues_factor[sorted_indices_factor[:r]] 
        eigenvalues_Gamma = np.real(np.linalg.eigvals(Gamma)) 
        idx_Gamma = np.argsort(eigenvalues_Gamma)[::-1]
        eigenvalues_Gamma = eigenvalues_Gamma[idx_Gamma] # sort 

        eigenvalues_full_covariance = np.concatenate((topr_eigenvalues,eigenvalues_Gamma))

        fig = go.Figure(data=go.Heatmap(z=Gamma-np.identity(N),
                                        colorscale=custom_colorscale,
                                        zmid=0  # ensures that 0 is mapped to the middle color (white)
                                        ))
                                        
        # fig.show()
        fig.write_image(f"Cov-N{N}.pdf") 

        Gamma_eigenvalues_store[index_N,:,s] = eigenvalues_full_covariance[:num_evals]



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
            eigenvalues = np.linalg.eigvals(X.T@X/T_max) 
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx] # sort 
            eigenvalues_store[index_N,:,s] = eigenvalues[:num_evals]

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

            r_eval_X = np.sort(np.linalg.eigvals(X.T@X/T_max))[-r] 
            max_eval_X = np.sort(np.linalg.eigvals(X.T@X/T_max))[-1] 
            max_eval_Xi = np.sort(np.linalg.eigvals(Xi.T@Xi/T_max))[-1]

            print(f"rth eval X: {r_eval_X}")
            print(f"max eval X: {max_eval_X}")
            print(f"max eval Xi: {max_eval_Xi}")

            eigenvalues = np.linalg.eigvals(X.T@X/T_max) 
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx] # sort 
            eigenvalues_store[index_N,:,s] = eigenvalues[:num_evals]

        elif generative_model == 'NIRVAR':
            X = Xi 
            fig = px.line(x=np.arange(T_max), y=X[:,0])
            # fig.show()

            if LedoitWolfEstimator:
                cov_estimate = LedoitWolf().fit(X).covariance_
            else:
                cov_estimate = X.T@X/T_max

            eigenvalues = np.linalg.eigvals(cov_estimate) 
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx] # sort 
            eigenvalues_store[index_N,:,s] = eigenvalues[:num_evals]

maxN_evals = eigenvalues_store[-1, 0, :] 
eigenvalues_store_scaled = eigenvalues_store/maxN_evals[None, None, :] 

evals_mean_scaled = np.mean(eigenvalues_store_scaled,axis=-1)
evals_std_scaled = np.std(eigenvalues_store_scaled,axis=-1) 

np.savetxt(f"mean_evals_scaled.csv", evals_mean_scaled, delimiter=",", fmt="%.6f")
np.savetxt(f"std_evals_scaled.csv", evals_std_scaled, delimiter=",", fmt="%.6f")

evals_mean = np.mean(eigenvalues_store,axis=-1)
evals_std = np.std(eigenvalues_store,axis=-1) 

np.savetxt(f"mean_evals.csv", evals_mean, delimiter=",", fmt="%.6f")
np.savetxt(f"std_evals.csv", evals_std, delimiter=",", fmt="%.6f")


maxN_evals_phi = phi_evals_store[-1, 0, :] 
eigenvalues_store_scaled_phi = phi_evals_store/maxN_evals_phi[None, None, :]

evals_mean_scaled_phi = np.mean(eigenvalues_store_scaled_phi,axis=-1)
evals_std_scaled_phi = np.std(eigenvalues_store_scaled_phi,axis=-1) 

np.savetxt(f"mean_evals_scaled_phi.csv", evals_mean_scaled_phi, delimiter=",", fmt="%.6f")
np.savetxt(f"std_evals_scaled_phi.csv", evals_std_scaled_phi, delimiter=",", fmt="%.6f")

evals_mean_phi = np.mean(phi_evals_store,axis=-1)
evals_std_phi = np.std(phi_evals_store,axis=-1) 

np.savetxt(f"mean_evals_phi.csv", evals_mean_phi, delimiter=",", fmt="%.6f")
np.savetxt(f"std_evals_phi.csv", evals_std_phi, delimiter=",", fmt="%.6f")

evals_mean_Gamma = np.mean(Gamma_eigenvalues_store,axis=-1)
evals_std_Gamma = np.std(Gamma_eigenvalues_store,axis=-1) 

np.savetxt(f"mean_evals_Gamma.csv", evals_mean_Gamma, delimiter=",", fmt="%.6f")
np.savetxt(f"std_evals_Gamma.csv", evals_std_Gamma, delimiter=",", fmt="%.6f") 

