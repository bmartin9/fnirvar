"Script to visualise the scaling of eigenvalues with N of NIRVAR/FNIRVAR"

#!/usr/bin/env python3

import sys
import pandas as pd
import plotly.graph_objects as go
import yaml 
import time

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <means_csv> <stds_csv> hyperparameters.yaml")
        sys.exit(1)

    with open(sys.argv[3], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ###### CONFIG PARAMETERS ###### 
    N_list = list(config['N_list'])

    means_file = sys.argv[1]
    stds_file = sys.argv[2]

    # Read the CSV files into pandas DataFrames
    df_means = pd.read_csv(means_file, header=None)
    df_stds = pd.read_csv(stds_file, header=None)

    # df_means and df_stds should both have shape (num_N, num_evals)
    num_N, num_evals = df_means.shape

    # Loop over each eigenvalue index k
    for k in range(num_evals):
        # -- Create a dynamic y-axis title using k+1 for 1-based indexing -- #
        # If you'd rather keep it 0-based in the label, use just k instead of k+1.
        # y_label = rf'$\lambda_{{{k+1}}}^{{(n)}} / \lambda_{{1}}^{{(N)}}$'
        y_label = rf'$\lambda_{{{k+1}}}(\Gamma) $'

        # Create a layout for each plot individually
        layout_k = go.Layout(
            yaxis=dict(
                title=y_label,
                showline=True,
                linewidth=1,
                linecolor='black',
                ticks='outside',
                mirror=True,
                # range=[0, 1.3]

            ),
            xaxis=dict(
                title='N',
                showline=True,
                linewidth=1,
                linecolor='black',
                ticks='outside',
                mirror=True,
                automargin=True,
                range=[min(N_list)-5, max(N_list)+5]
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font_family="Serif",
            font_size=11,
            margin=dict(l=5, r=5, t=5, b=5),
            width=500,
            height=350
        )

        # Extract the mean and std values for this eigenvalue
        y_means = df_means.iloc[:, k]
        y_stds = df_stds.iloc[:, k]

        # Use your N_list as the x-values
        x_vals = N_list

        # Create the scatter plot with error bars
        trace = go.Scatter(
            x=x_vals,
            y=y_means,
            mode='lines+markers',
            error_y=dict(
                type='data',
                array=y_stds,
                visible=True
            ),
            line=dict(color='blue')
        )

        fig = go.Figure(data=[trace], layout=layout_k)

        # Save each figure as a PDF
        output_filename = f"eigenvalue_{k}.pdf"
        fig.write_image(output_filename)
        time.sleep(1)
        fig.write_image(output_filename)

        print(f"Saved: {output_filename}")


if __name__ == "__main__":
    main()
