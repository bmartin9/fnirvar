""" 
Script to get scree plot of eigenvalues on Santander dataset.
"""

#!/usr/bin/env python3 
# USAGE: ./scree_plot.py <DESIGN_MATRIX>.csv 

import numpy as np
import sys
import plotly.express as px
import time
import plotly.graph_objects as go

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: ./backtest.py <DESIGN_MATRIX>.csv")
        sys.exit(1)
    
    ###### READ IN DATA ######
    try:
        # Assuming the first column is an index or non-numeric identifier, skip it
        Xs = np.genfromtxt(sys.argv[1], delimiter=',')
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)
    
    T, N = Xs.shape
    print(f"Data loaded: {T} observations, {N} variables.")

    ###### COMPUTE THE SAMPLE COVARIANCE MATRIX ######
    # By default, np.cov interprets rows as variables. 
    # Set rowvar=False because each row in Xs is an observation and each column is a variable.
    cov_mat = np.cov(Xs, rowvar=False)

    ###### EIGEN-DECOMPOSITION ######
    eigenvalues, _ = np.linalg.eig(cov_mat)
    # Ensure eigenvalues are real (covariance matrices should be positive semi-definite)
    eigenvalues = np.real(eigenvalues)
    # Sort the eigenvalues in descending order for a conventional scree plot
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    top_k = 40
    eigenvalues_top = eigenvalues_sorted[1:top_k]
    indices = list(range(1, top_k + 1))

    ###### CREATE THE SCREE PLOT USING PLOTLY GRAPH OBJECTS ######
    # Create a trace for the eigenvalues
    trace = go.Scatter(
        x=indices,         # Components labeled 1..N
        y=eigenvalues_top, 
        mode='lines+markers',
        marker=dict(size=6),
        line=dict(width=2),
        name='Eigenvalues'
    )

    # Define the layout as specified
    layout = go.Layout(
        yaxis=dict(
            title="Eigenvalue",
            showline=True, 
            linewidth=1, 
            linecolor='black',
            ticks='outside',
            mirror=True
        ),
        xaxis=dict(
            title="Index",
            showline=True, 
            linewidth=1, 
            linecolor='black',
            ticks='outside',
            mirror=True,
            automargin=True
        ),
        paper_bgcolor='white',  # Set background color to white
        plot_bgcolor='white',   # Set plot area color to white
        font=dict(
            family="Serif",
            size=11
        ),
        margin=dict(l=50, r=50, t=50, b=50),  # Increased margins for better readability
        width=500, 
        height=350,
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    ###### SAVE THE PLOT ######

    # Save the plot as a PDF
    try:
        fig.write_image("scree_plot.pdf")
        time.sleep(1)
        fig.write_image("scree_plot.pdf")

    except Exception as e:
        print(f"Error saving the plot as PDF: {e}")
        print("Ensure that the 'kaleido' package is installed.")

if __name__ == "__main__":
    main()