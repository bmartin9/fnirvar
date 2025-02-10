""" 
Simulate data from the FNIRVAR model. Save to csv file.
"""

#!/usr/bin/env python3 
# USAGE: ./simulate.py generative_hyperparameters.yaml 

from fnirvar.modeling.generativeVAR import GenerateFNIRVAR
from fnirvar.modeling.generativeVAR import GenerateNIRVAR 
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
T = config['T']
N = config['N1']
B = config['B']
p_in = config['p_in']
p_out = config['p_out'] 
use_HPC = config['use_HPC'] 
num_common_shocks = config['num_common_shocks']
rho_F = config['rho_F']
l_F = config['l_F']
model = config['model']
if model == 'FactorsOnly':
    model = None 

print(model)


random_state = default_rng(seed=SEED)

###### ENVIRONMENT VARIABLES ######  
if use_HPC:
    PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
    NUM_ARRAY_INDICES = int(os.environ['NUM_ARRAY_INDICES'])
else:
    PBS_ARRAY_INDEX = 1
    NUM_ARRAY_INDICES = 1

###### SIMULATE FNIRVAR DATA ###### 
loadings_matrix = random_state.normal(size=(N,r))

phi_dist = np.ones((N,N))

generate_NIRVAR = GenerateNIRVAR(random_state=random_state,
                                T=T,
                                B=B,
                                N=N,
                                Q=1,
                                p_in=p_in,
                                p_out=p_out,
                                phi_distribution=phi_dist )

Xi = generate_NIRVAR.generate()[:,:,0]

generate_FNIRVAR = GenerateFNIRVAR(l_F=l_F,
                                   T=T,
                                   r=r,
                                   q=num_common_shocks,
                                   rho_F=rho_F,
                                   random_state=random_state,
                                   P=None,
                                   N0=None
                                   )


if model is None:
    data = generate_FNIRVAR.generate_data(Lambda=loadings_matrix) 

elif model == 'FNIRVAR':
    data = generate_FNIRVAR.generate_data(Lambda=loadings_matrix,xi=Xi) 

elif model == 'NIRVAR':
    data = Xi


np.savetxt(f"generated_data_{model}.csv", data, delimiter=",")