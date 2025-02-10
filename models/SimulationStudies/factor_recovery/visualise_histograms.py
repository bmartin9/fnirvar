#!/usr/bin/env python3
"""
Script to read in a variable number of CSV files (each containing a single column of integer data)
and plot histograms for each on a single Plotly figure.

USAGE:
    ./visualise_histograms.py file1.csv file2.csv ...

Notes:
- Data is assumed to be integer-valued in each CSV (single column, no header).
- Histograms are overlaid by default. Change barmode to 'group' if you prefer side-by-side.
"""

import sys
import plotly.graph_objs as go
import pandas as pd
import os
import time

# Gather CSV filenames from command line
csv_files = sys.argv[1:]  # e.g., file1.csv file2.csv ...

# Manually define histogram names to match the number of CSV files you have.
# Make sure len(histogram_names) >= len(csv_files) if you want a custom name for each file.
histogram_names = [
    "ER",
    "GR",
    "PCp2",
    # ... add more if needed ...
]

# Create a figure
fig = go.Figure()

# Read each CSV file and add a histogram trace
for i, file_path in enumerate(csv_files):
    # Read single-column CSV (no header)
    df = pd.read_csv(file_path, header=None)
    
    # If you have more CSVs than custom names, fall back to the filename
    if i < len(histogram_names):
        trace_name = histogram_names[i]
    else:
        trace_name = os.path.basename(file_path)
    
    # Create a histogram trace for the integer data
    # xbins.size=1 ensures each integer lands in its own bin
    fig.add_trace(
        go.Histogram(
            x=df[0],
            name=trace_name,
            xbins=dict(size=1),  # bin size of 1 (integer bins)
            opacity=0.7
        )
    )

# Define custom layout (using overlay so histograms can be compared)
layout = go.Layout(
    barmode='overlay',  # Use 'group' if you want side-by-side histograms
    xaxis=dict(
        title='Estimated number of factors',
        showline=True,
        linewidth=1,
        linecolor='black',
        ticks='outside',
        mirror=True
    ),
    yaxis=dict(
        title='Count',
        showline=True,
        linewidth=1,
        linecolor='black',
        ticks='outside',
        mirror=True
    ),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=11, 
    margin=dict(l=5, r=5, t=5, b=5),
    width=500, 
    height=350
)

fig.update_layout(layout)

# Export histogram plot to a PDF file
fig.write_image("histogram_plot.pdf")
# Slight delay to ensure file is fully written
time.sleep(1)
fig.write_image("histogram_plot.pdf")
