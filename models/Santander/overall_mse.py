""" 
Script to calculate the overall MSE between predicted log number of rides and realised log number of rides for the Santander dataset.
"""

#!/usr/bin/env python3
# USAGE: ./overall-mse.py <DESIGN_MATRIX>.csv config.yaml predictions-1.csv predictions-2.csv ... 

import sys
import numpy as np 
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import time
import yaml

with open(sys.argv[2], "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

###### CONFIG PARAMETERS ###### 
first_prediction_day = config['first_prediction_day']

###### READ IN DATA ######
Xs = np.genfromtxt(sys.argv[1], delimiter=',')
T = Xs.shape[0]
N = Xs.shape[1]

targets = Xs[first_prediction_day+1:,:] 
print(first_prediction_day)

def read_csv_files(argv):
    arrays_list = []

    # Iterate over the system arguments starting from the second argument
    for file_path in argv[2:]:
        try:
            # Read the CSV file as a DataFrame
            df = pd.read_csv(file_path,header=None)

            # Convert the DataFrame to a NumPy array and add it to the list
            arrays_list.append(df.to_numpy())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return arrays_list

predictions_list = read_csv_files(sys.argv) 

overall_mse_list = [] 
for i in range(len(predictions_list)):
    absolute_mse = np.sum((targets[:] - predictions_list[i][:,:])**2)
    overall_mse_list.append(absolute_mse)

print(overall_mse_list[0])

print(f"overall mse list: {overall_mse_list}") 

# Get the proportion of times where the predictions of sys.argv[3] are better than competing models 
proportion_list = [] 
mse_list = [] 
normalised_mse_list = [] 
for i in range(len(predictions_list)):
    absolute_mse = (targets[:] - predictions_list[i][:,:])**2 
    mse_list.append(absolute_mse)
    normalised_mse = np.mean((targets - predictions_list[i])**2)/(np.mean(targets**2))
    normalised_mse_list.append(normalised_mse) 

def calculate_proportion(listA, listB):
    if len(listA) != len(listB):
        raise ValueError("Both lists must have the same length")

    count = sum(a < b for a, b in zip(listA, listB))
    return count / len(listA)

for i in range(1,len(predictions_list)):
    proportion_list.append(calculate_proportion(mse_list[0],mse_list[i]))

# print(proportion_list)

proportion_list = []
station_wide_mse_list = []
# Get proportion of stations where NIRVAR has a lower MSE 
for i in range(len(predictions_list)):
    station_wide_mse = ((targets - predictions_list[i])**2).mean(axis=0)
    station_wide_mse_list.append(station_wide_mse)

for i in range(len(predictions_list)):
    a = np.where(station_wide_mse_list[0]<station_wide_mse_list[i],1,0)
    prop_percentage = 100*np.sum(a)/N
    proportion_list.append(prop_percentage)

summary_statistics = {'Overall MSE List': overall_mse_list,
                      'Normalised MSE': normalised_mse_list,
                      'Proportion of stations where NIRVAR is superior': proportion_list,
                      } 

f = open("summary_statistics.txt", "w")
f.write("{\n")
for k in summary_statistics.keys():
    f.write("'{}':'{}'\n".format(k, summary_statistics[k]))
f.write("}")
f.close()
