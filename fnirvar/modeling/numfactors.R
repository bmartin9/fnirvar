# Script to estimate the number of factors in the design matrix for each backtesting day using the getnfac function
# OUTPUT: A csv file with the number of factors for each backtesting day

#!/usr/bin/env Rscript
# USAGE: Rscript numfactors.R <inputfile>  

library(PANICr)

# PARAMETERS
kmax = 10
criteria = "BIC3"
output_file = "num_factors.csv"
n_backtest_days : 4 
first_prediction_day : 1004
lookback_window : 1004

args <- commandArgs(trailingOnly = TRUE)
design_matrix_file <- args[1]

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) < 1) {
    stop("USAGE: Rscript numfactors.R <inputfile> <outputfile>") 
}

# Read the design matrix
Xs <- read.csv2(design_matrix_file,sep=",",dec = ".",header=FALSE) 

# Create an empty vector to store the number of factors
num_factors <- numeric(n_backtest_days)

# Loop through each backtesting day
for (i in 1:n_backtest_days) {
    # Define the start and end indices of the rolling window
    start_index <- first_prediction_day + i - lookback_window
    end_index <- first_prediction_day + i - 1
    
    # Create the design matrix using the rolling window
    design_matrix <- Xs[start_index:end_index, ]
    
    # Compute the number of factors using the getnfac function
    num_factors[i] <- getnfac(design_matrix, kmax = kmax, criteria = criteria)
}

# Output the number of factors to a csv file
write.csv(num_factors, file = output_file, row.names = FALSE)