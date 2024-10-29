""" 
Script that takes as input a csv with the daily rides count values for each 
station and outputs the multivariate time series for the Santander dataset.
"""


#!/usr/bin/env python
# USAGE: python create_santander_ts.py count_santander_rides.csv 

import pandas as pd
import numpy as np
import sys 

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Create a pivot table with 'start_date' as rows, 'start_id' as columns, and 'count' as values
pivot_table = df.pivot(index='start_date', columns='start_id', values='count')

# Fill missing values with 0
pivot_table = pivot_table.fillna(0)

# Convert the pivot table to a NumPy array
np_array = pivot_table.values

print(f"Shape of the NumPy array: {np_array.shape}")


# Save the array to a CSV file
np.savetxt('output.csv', np_array, delimiter=',', fmt='%s')