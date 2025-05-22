#PBS -N backtesting 
#PBS -l walltime=01:30:00 
#PBS -l select=1:ncpus=1:mem=8gb

module load anaconda3/personal
source activate fnirvar

export DESIGN_FILE='../../../data/IntradayReturns/processed/csv_files/excess_returns30min.csv' 
cd $PBS_O_WORKDIR
python compute_num_factors.py $DESIGN_FILE hyperparameters.yaml  
