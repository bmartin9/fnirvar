""" 
Script to take in raw locations and return the N=774 locations corresponding to the cleaned design matrix. 
"""

#!/usr/bin/env python3
# USAGE: ./get_clean_locations.py commom_start_ids.csv santander_dictionary.pkl santander_locations.csv 

import numpy as np 
import pandas as pd 
import csv
import sys 
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load data
common_start_ids = np.loadtxt(sys.argv[1], delimiter=',')

with open(sys.argv[2], 'rb') as file:
    try:
        while True:
            # Load each object
            santander_dict = pickle.load(file)
    except EOFError:
        # End of file reached
        pass

santander_locations_df = pd.read_csv(sys.argv[3])

# Extract station names corresponding to common_start_ids
station_names = [santander_dict[int(id_)] for id_ in common_start_ids]

# Create a DataFrame with common_start_ids, station names, and labels
common_start_df = pd.DataFrame({
    'StationID': common_start_ids,
    'StationName': station_names,
})

# Merge with santander_locations_df to get latitude and longitude
merged_df = pd.merge(common_start_df, santander_locations_df, on='StationName')

merged_df[["longitude","latitude"]].to_csv("cleaned_locations.csv",sep=",")