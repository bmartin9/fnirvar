""" 
Script that takes santander_train.csv or santander_test.csv as inputs
and counts the number of rides from each station per day.
"""


#!/usr/bin/env python
# USAGE: python count_daily_rides.py santander_train.csv

import pandas as pd
import sys

# Load the CSV file
df = pd.read_csv(sys.argv[1])

print(f"Number of unique start_id values: {df['start_id'].nunique()}")

# Convert 'start_time' to datetime objects and extract the date part
df['start_date'] = pd.to_datetime(df['start_time']).dt.date

print(f"Number of unique start_date values: {df['start_date'].nunique()}")

# Get all unique 'start_id' and 'start_date' values separately
unique_start_ids = df['start_id'].unique()
unique_start_dates = pd.to_datetime(df['start_time']).dt.date.unique()

# Create a multi-index with all combinations of 'start_id' and 'start_date'
multi_index = pd.MultiIndex.from_product([unique_start_ids, unique_start_dates], names=['start_id', 'start_date'])

# Create an empty DataFrame with the multi-index
empty_df = pd.DataFrame(index=multi_index).reset_index()

# Group by 'start_id' and 'start_date' and count the occurrences
grouped_df = df.groupby(['start_id', 'start_date']).size().reset_index(name='count')

# Perform a left join to fill in missing combinations with 0 count
result_df = empty_df.merge(grouped_df, how='left', on=['start_id', 'start_date']).fillna(0).astype({'count': int})

# Rename columns to match the desired output format
result_df = result_df.rename(columns={'start_date': 'start_date', 'count': 'count'})

# Sort the DataFrame first by 'start_id' in ascending order, and then by 'start_date' in ascending order
result_df = result_df.sort_values(['start_id', 'start_date'])

# Save the result to a new CSV file
output_path = "count_santander_rides.csv"
result_df.to_csv(output_path, index=False)

# Display the result
print(result_df)