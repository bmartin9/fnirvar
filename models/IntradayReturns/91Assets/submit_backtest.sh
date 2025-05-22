#PBS -N backtesting 
#PBS -l walltime=02:30:00 
#PBS -l select=1:ncpus=10:mem=8gb
#PBS -J 1-10

export NUM_ARRAY_INDICES=10

module load anaconda3/personal
source activate fnirvar

export DESIGN_FILE='../../../data/IntradayReturns/processed/csv_files/excess_returns30min.csv' 
export NUM_FACTORS='excess_30min_experiments/estimated_num_factorsER.csv' 
export OUTPUT_DIRECTORY='FNIRVAR_30min_excess'
cd $PBS_O_WORKDIR
python backtest.py $DESIGN_FILE hyperparameters.yaml $NUM_FACTORS 

NUM_FILES_CREATED_SO_FAR=$(find . -maxdepth 1 -type f -name 'predictions-*' 2>/dev/null | wc -l)
echo $NUM_FILES_CREATED_SO_FAR 
if [ $NUM_FILES_CREATED_SO_FAR -eq $NUM_ARRAY_INDICES ]; then
        if ls predictions-*.csv 1>/dev/null 2>&1; then
                cat $(ls -v predictions-*.csv) > predictions.csv
                cat $(ls -v phi_hat-*.csv) > phi_hat.csv
                cat $(ls -v labels_hat-*.csv) > labels_hat.csv
                rm predictions-*.csv
                rm phi_hat-*.csv
                rm labels_hat-*.csv
        else
                echo "File predictions-*.csv does not exist."
        fi

fi

# python backtest_statistics.py $DESIGN_FILE predictions-$PBS_ARRAY_INDEX.csv backtesting_config.yaml 

# python 0.3-visualise-backtesting.py PnL.csv

scp predictions.csv  backtesting_hyp.txt  $OUTPUT_DIRECTORY
