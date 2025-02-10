""" 
Script to output the number of groups of FNIRVAR vs NIRVAR.
Also outpu plot of ARI between FNIRVAR and NIRVAR
"""

#!/usr/bin/env python3
# USAGE: ./compare_groups.py labels_hat1.csv labels_hat2.csv 

import sys
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score
import time

def main():
    if len(sys.argv) != 3:
        print("Usage: ./compare_groups.py labels_hat1.csv labels_hat2.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Read the CSV files (no header, each has T rows and N columns)
    labels1 = pd.read_csv(file1, header=None)
    labels2 = pd.read_csv(file2, header=None)

    # Check that both files have the same shape
    if labels1.shape != labels2.shape:
        print("Error: CSV files must have the same dimensions.")
        sys.exit(1)

    T, N = labels1.shape

    # Compute ARI for each time step (row)
    ari_list = []
    for t in range(T):
        row1 = labels1.iloc[t].values
        row2 = labels2.iloc[t].values
        ari = adjusted_rand_score(row1, row2)
        ari_list.append(ari)

    # Compute the number of clusters per row
    # Assuming 0-based cluster labels => number of clusters = max_label + 1
    num_clusters1 = (labels1.max(axis=1) + 1).tolist()
    num_clusters2 = (labels2.max(axis=1) + 1).tolist()

    layout = go.Layout(
        yaxis=dict(
            title = "ARI",
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True
        ),
        xaxis=dict(
            title = "T",
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True,
            automargin=True
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
    )

    # Plot ARI over time
    fig_ari = go.Figure(layout=layout)
    fig_ari.add_trace(
        go.Scatter(
            x=list(range(T)),
            y=ari_list,
            mode='lines',
            name='ARI'
        )
    )
    fig_ari.write_image("ari_plot.pdf")
    time.sleep(1)
    fig_ari.write_image("ari_plot.pdf")


    # Plot number of clusters over time
    layout = go.Layout(
        yaxis=dict(
            title = "Number of clusters",
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True
        ),
        xaxis=dict(
            title = "T",
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            mirror=True,
            automargin=True
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_family="Serif",
        font_size=11,
        margin=dict(l=5, r=5, t=5, b=5),
        width=500,
        height=350
    )
    fig_clusters = go.Figure(layout=layout)
    fig_clusters.add_trace(
        go.Scatter(
            x=list(range(T)),
            y=num_clusters1,
            mode='lines',
            name='NIRVAR'
        )
    )
    fig_clusters.add_trace(
        go.Scatter(
            x=list(range(T)),
            y=num_clusters2,
            mode='lines',
            name='FNIRVAR'
        )
    )
    fig_clusters.write_image("clusters_plot.pdf")
    time.sleep(1)
    fig_clusters.write_image("clusters_plot.pdf")


if __name__ == "__main__":
    main()