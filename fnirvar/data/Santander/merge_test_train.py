""" 
The test and train count files don't have the same number of stations. 
This script finds the intersection of the stations in the test and train files,
and outputs a file with the same number of stations for both.
"""

#!/usr/bin/env python
# USAGE: python merge_test_train.py count_santander_rides_train.csv count_santander_rides_test.csv 

import pandas as pd
import sys

# Load the CSV files
train_df = pd.read_csv(sys.argv[1])
test_df = pd.read_csv(sys.argv[2])

# Get the unique 'start_id' values from the train and test files
train_start_ids = train_df['start_id'].unique()
test_start_ids = test_df['start_id'].unique()

# Find the intersection of the 'start_id' values
common_start_ids = set(train_start_ids).intersection(test_start_ids)

# Filter the train and test DataFrames to only include the common 'start_id' values
train_df_filtered = train_df[train_df['start_id'].isin(common_start_ids)]
test_df_filtered = test_df[test_df['start_id'].isin(common_start_ids)]

# Save the filtered DataFrames to new CSV files
train_output_path = "santander_train_filtered_count.csv"
train_df_filtered.to_csv(train_output_path, index=False)

test_output_path = "santander_test_filtered_count.csv"
test_df_filtered.to_csv(test_output_path, index=False)

# Display the number of common 'start_id' values
print(f"Number of common start_id values: {len(common_start_ids)}")

# Display the number of rows in the filtered train and test DataFrames
print(f"Number of rows in filtered train DataFrame: {len(train_df_filtered)}")
print(f"Number of rows in filtered test DataFrame: {len(test_df_filtered)}")

