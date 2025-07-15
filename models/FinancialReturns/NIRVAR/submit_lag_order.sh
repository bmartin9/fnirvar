#PBS -N backtesting 
#PBS -l walltime=01:30:00 
#PBS -l select=1:ncpus=1:mem=8gb

module load anaconda3/personal
source activate fnirvar

export DESIGN_FILE='../../../data/FinancialReturns/processed/stocks_no_market_cleaned.csv'
export NUM_FACTORS_FILE='num_factors/estimated_num_factors_PCp2.csv'
cd $PBS_O_WORKDIR
python factor_lag_order_selection.py $DESIGN_FILE config.yaml $NUM_FACTORS_FILE
