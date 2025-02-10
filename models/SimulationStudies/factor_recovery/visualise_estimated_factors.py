#!/usr/bin/env python3
"""
Script to read in a variable number of CSV files (passed via command line)
and plot line plots for each using Plotly.

USAGE:
    ./visualise_estimated_factors.py file1.csv file2.csv ...
"""

import sys
import plotly.graph_objs as go
import pandas as pd
import os
import time

# Get CSV file paths from command-line arguments
csv_files = sys.argv[1:]  # everything after the script name

# Manually define line names to match the number of CSV files you have.
# Make sure len(line_names) >= len(csv_files) if you want each trace to have a custom name.
line_names = [
    "Line 1",
    "Line 2",
    "Line 3",
    # ... add as many names as needed ...
]

# Create a figure
fig = go.Figure()

# Read each CSV file and add a trace
for i, file_path in enumerate(csv_files):
    # Read single-column CSV, no header
    df = pd.read_csv(file_path, header=None)
    
    # x-values (assuming 1000 rows, so T ranges from 1 to 1000)
    x_values = list(range(1, len(df) + 1))
    
    # Choose a name from line_names if available, otherwise use the filename
    if i < len(line_names):
        trace_name = line_names[i]
    else:
        trace_name = os.path.basename(file_path)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df[0],
            mode='lines',
            name=trace_name
        )
    )

# Define your custom layout
layout = go.Layout(
    yaxis=dict(
        title='Number of estimated factors',
        showline=True,
        linewidth=1,
        linecolor='black',
        ticks='outside',
        mirror=True
    ),
    xaxis=dict(
        title='T',
        showline=True,
        linewidth=1,
        linecolor='black',
        ticks='outside',
        mirror=True,
        automargin=True,
        # Example range below [0, 1000]; change as needed:
        range=[0, 1000]
    ),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif",
    font_size=11,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)

# Update figure with our layout
fig.update_layout(layout)

# Save figure to a PDF
fig.write_image("plot_estimated_factors.pdf")
# Slight delay to ensure file is fully written before script ends
time.sleep(1)
fig.write_image("plot_estimated_factors.pdf")
